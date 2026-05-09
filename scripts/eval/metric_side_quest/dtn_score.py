"""DTN-aware A-Score for proactive-intervention strategies.

The original A-Score in ``generate_timeline_comparison`` assumes a single
GT interval per intervention. With the disjunctive ground truth in
``disjunctive_gt`` each intervention exposes alternative options, and each
option is a conjunction of phases. This module scores a flat prediction
track against that richer GT.

Score components (all in [0, 1]):

* ``A_disj`` — Disjunctive coverage. For each intervention, pick the
  option whose phases are best covered by predictions of the matching
  skill (mean-IoU per phase); average across interventions.
* ``A_act``  — Action correctness on the predictions that overlap any GT
  phase: fraction whose stripped action equals the GT skill.
* ``A_cons`` — Resource consistency. The selected options across
  different interventions should not share time. Computed as
  ``1 - max_pair_iou`` over cross-task pairs of selected phases. A
  conflict-free selection scores 1.0; a fully-overlapping pair scores 0.
* ``A_nec``  — Necessity / no-spurious-mass. Fraction of total prediction
  time that lies inside *some* GT phase (of any option, any task).

``A_score = w_disj * A_disj + w_act * A_act + w_cons * A_cons + w_nec * A_nec``.

Defaults emphasise coverage but require all axes to score well for a
strategy to land near 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from disjunctive_gt import RobotGT, Intervention, Option, Phase


DEFAULT_WEIGHTS: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
"""(w_disj, w_act, w_cons, w_nec)."""


# ── Tiny prediction shape so we don't depend on generate_timeline.Event ──────


@dataclass
class Pred:
    """One predicted intervention interval."""
    skill: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_args(action: str) -> str:
    """``move_resin_to_workplace(arg=1)`` -> ``move_resin_to_workplace``."""
    if "(" in action:
        return action.split("(", 1)[0].strip()
    return action.strip()


def _iou(a_start: float, a_end: float,
         b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _overlap(a_start: float, a_end: float,
             b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ── Per-intervention best-option search ──────────────────────────────────────


def _option_phase_ious(opt: Option, preds: list[Pred]) -> list[float]:
    """For each phase in ``opt``, the best IoU achieved by any prediction."""
    out: list[float] = []
    for ph in opt.phases:
        best = 0.0
        for p in preds:
            best = max(best, _iou(p.start, p.end, ph.t_start, ph.t_end))
        out.append(best)
    return out


def _select_best_option(iv: Intervention,
                        preds_for_skill: list[Pred]
                        ) -> tuple[int, float, list[float]]:
    """Return ``(selected_option_index, mean_iou, per_phase_ious)``."""
    best_idx = 0
    best_mean = -1.0
    best_per_phase: list[float] = []
    for k, opt in enumerate(iv.options):
        per_phase = _option_phase_ious(opt, preds_for_skill)
        mean = sum(per_phase) / len(per_phase) if per_phase else 0.0
        if mean > best_mean:
            best_mean = mean
            best_idx = k
            best_per_phase = per_phase
    if best_mean < 0:
        best_mean = 0.0
    return best_idx, best_mean, best_per_phase


# ── Cross-task overlap penalty ───────────────────────────────────────────────


def _max_cross_task_pair_iou(gt: RobotGT,
                             selected: dict[str, int]) -> float:
    """Largest IoU between phases drawn from *different* selected options.

    The user's DTN scenario assumes a single-resource agent: two distinct
    interventions cannot occupy the same time window. ``A_cons`` penalises
    such overlap. This returns 0.0 when phases are pairwise disjoint.
    """
    iv_list = gt.interventions
    chosen: list[tuple[str, Phase]] = []
    for iv in iv_list:
        opt = iv.options[selected[iv.id]]
        for ph in opt.phases:
            chosen.append((iv.id, ph))

    worst = 0.0
    for i in range(len(chosen)):
        a_id, a_ph = chosen[i]
        for j in range(i + 1, len(chosen)):
            b_id, b_ph = chosen[j]
            if a_id == b_id:
                continue
            worst = max(worst, _iou(a_ph.t_start, a_ph.t_end,
                                    b_ph.t_start, b_ph.t_end))
    return worst


# ── Necessity / action-correctness over predictions ──────────────────────────


def _all_phases_with_skill(gt: RobotGT) -> list[tuple[str, Phase]]:
    """Every (skill, phase) the agent could legitimately fire — across all
    options and phases. Used to define 'predicted time inside any GT phase'.
    """
    out: list[tuple[str, Phase]] = []
    for iv in gt.interventions:
        for opt in iv.options:
            for ph in opt.phases:
                out.append((iv.skill, ph))
    return out


def _necessity(preds: list[Pred],
               all_skill_phases: list[tuple[str, Phase]]) -> float:
    """Fraction of total prediction time that overlaps any GT phase."""
    total = sum(p.duration for p in preds)
    if total <= 0:
        # No predictions → vacuously fine if the GT has no required tasks,
        # but this collapses with A_disj below; we return 1.0 here so this
        # axis never triggers a false penalty. The downstream score still
        # punishes a silent strategy via A_disj.
        return 1.0

    # Build a mask of allowed time across [0, max_end].
    # Simple O(P*Q) sweep is fine for our problem sizes.
    inside = 0.0
    for p in preds:
        # Intersect p with the union of all GT phases.
        rem: list[tuple[float, float]] = [(p.start, p.end)]
        for _, ph in all_skill_phases:
            new_rem: list[tuple[float, float]] = []
            for s, e in rem:
                if e <= ph.t_start or s >= ph.t_end:
                    new_rem.append((s, e))
                    continue
                # Subtract [ph.t_start, ph.t_end] from (s,e).
                if s < ph.t_start:
                    new_rem.append((s, ph.t_start))
                if e > ph.t_end:
                    new_rem.append((ph.t_end, e))
                inside += _overlap(s, e, ph.t_start, ph.t_end)
            rem = new_rem
    return min(1.0, inside / total)


def _action_correctness(preds: list[Pred],
                        all_skill_phases: list[tuple[str, Phase]]) -> float:
    """Of predictions overlapping some GT phase, fraction whose skill
    equals that phase's skill.

    Tie-breaking: if several GT phases are tied at the maximum IoU and any
    of them shares the prediction's skill, that one wins. This avoids
    penalising a correctly-aimed prediction just because a *different*
    task's alternative option happens to occupy the same window.
    """
    if not preds:
        return 1.0
    matched = 0
    correct = 0
    for p in preds:
        pred_skill = _strip_args(p.skill)
        best_iou = 0.0
        tied_skills: set[str] = set()
        for skill, ph in all_skill_phases:
            iou = _iou(p.start, p.end, ph.t_start, ph.t_end)
            if iou > best_iou + 1e-9:
                best_iou = iou
                tied_skills = {skill}
            elif abs(iou - best_iou) <= 1e-9 and best_iou > 0:
                tied_skills.add(skill)
        if best_iou > 0 and tied_skills:
            matched += 1
            if pred_skill in {_strip_args(s) for s in tied_skills}:
                correct += 1
    if matched == 0:
        return 0.0
    return correct / matched


# ── Public API ──────────────────────────────────────────────────────────────


@dataclass
class DTNAScore:
    a_disj: float
    a_act: float
    a_cons: float
    a_nec: float
    a_score: float
    selected_options: dict[str, int] = field(default_factory=dict)
    per_task_disj: dict[str, float] = field(default_factory=dict)
    per_task_phase_ious: dict[str, list[float]] = field(default_factory=dict)
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS


def compute_dtn_a_score(gt: RobotGT,
                        preds: Iterable[Pred],
                        weights: tuple[float, float, float, float]
                            = DEFAULT_WEIGHTS) -> DTNAScore:
    preds = list(preds)
    w_disj, w_act, w_cons, w_nec = weights
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {weights}")

    selected: dict[str, int] = {}
    per_task: dict[str, float] = {}
    per_task_phase_ious: dict[str, list[float]] = {}

    for iv in gt.interventions:
        skill_preds = [p for p in preds
                       if _strip_args(p.skill) == _strip_args(iv.skill)]
        idx, mean_iou, per_phase = _select_best_option(iv, skill_preds)
        selected[iv.id] = idx
        per_task[iv.id] = mean_iou
        per_task_phase_ious[iv.id] = per_phase

    a_disj = (sum(per_task.values()) / len(per_task)) if per_task else 0.0

    all_skill_phases = _all_phases_with_skill(gt)
    a_act = _action_correctness(preds, all_skill_phases)
    a_nec = _necessity(preds, all_skill_phases)

    if gt.interventions:
        a_cons = 1.0 - _max_cross_task_pair_iou(gt, selected)
    else:
        a_cons = 1.0

    a_score = (w_disj * a_disj
               + w_act * a_act
               + w_cons * a_cons
               + w_nec * a_nec)
    return DTNAScore(
        a_disj=a_disj, a_act=a_act, a_cons=a_cons, a_nec=a_nec,
        a_score=a_score,
        selected_options=selected,
        per_task_disj=per_task,
        per_task_phase_ious=per_task_phase_ious,
        weights=weights,
    )


# ── Disjunctive-CSP helper (used by adversaries to build a feasible oracle) ──


def find_consistent_selection(gt: RobotGT) -> dict[str, int] | None:
    """Backtracking search for an option assignment with no cross-task
    phase overlaps. Returns ``None`` if the GT is over-constrained.
    """
    iv_list = gt.interventions
    selection: dict[str, int] = {}

    def phase_overlaps(p1: Phase, p2: Phase) -> bool:
        return max(p1.t_start, p2.t_start) < min(p1.t_end, p2.t_end)

    def consistent(iv: Intervention, opt: Option) -> bool:
        for prev_iv in iv_list:
            if prev_iv.id not in selection or prev_iv.id == iv.id:
                continue
            prev_opt = prev_iv.options[selection[prev_iv.id]]
            for p1 in opt.phases:
                for p2 in prev_opt.phases:
                    if phase_overlaps(p1, p2):
                        return False
        return True

    def backtrack(i: int) -> bool:
        if i == len(iv_list):
            return True
        iv = iv_list[i]
        for k, opt in enumerate(iv.options):
            if consistent(iv, opt):
                selection[iv.id] = k
                if backtrack(i + 1):
                    return True
                del selection[iv.id]
        return False

    return dict(selection) if backtrack(0) else None


# ── Disjunctive TWSS (the time-weighted soft schedule score, lifted) ─────────
#
# Implements eq. (twss-disj) from `twss_metric.tex` §6:
#
#   TWSS_disj = Σ_i [ (L_i*/Z) · Q_i · C_i ]  +  w_w · (T_wait*/Z) · q_wait
#
# with
#   Z            = Σ L_i*  +  w_w · T_wait*
#   q_{i,k,j}    = Gaussian on (Δs, Δe) with σ = α · ℓ_{i,k,j}
#   q̃_{i,k}     = phase-length-weighted mean over j of q_{i,k,j}
#   Q_i, k*_i   = max-option quality and the chosen option index
#   Φ*           = union of selected-option phases
#   L_i*         = total length of the selected option's phases
#   T_wait*      = T − |Φ*|
#   T_fp_eff     = Σ_p √(ℓ_p² + 2δℓ_p),  ℓ_p = pollution length of p
#                  against the same-skill slice of Φ*
#   q_wait       = exp(− T_fp_eff / (β · T_wait*))   (vacuous → 1)
#   C_i          = 1 − max cross-task IoU of option i against any other selected option
#   C            = length-weighted average of C_i (for reporting)
#
# Defaults mirror the tuned single-window TWSS in
# `generate_timeline_comparison.py` so the two scores are directly
# comparable when an intervention has one option / one phase.


DEFAULT_DTWSS_ALPHA = 1.0
DEFAULT_DTWSS_BETA = 1.5
DEFAULT_DTWSS_WAIT_WEIGHT = 0.6
DEFAULT_DTWSS_BLIP_TAX = 4.0


@dataclass
class DisjunctiveTWSS:
    twss: float
    task_score: float           # Σ (L_i*/T) · Q_i  (before C)
    wait_score: float           # w_w · (T_wait*/T) · q_wait  (before C)
    wait_quality: float         # q_wait
    wait_mass: float            # T_wait* / T
    consistency: float          # C
    fp_time: float              # raw pollution sum (s)
    fp_time_eff: float          # blip-tax inflated pollution (s)
    total_wait_time: float      # T_wait*
    total_pred_time: float
    selected_options: dict[str, int] = field(default_factory=dict)
    per_intervention_q: dict[str, float] = field(default_factory=dict)
    alpha: float = DEFAULT_DTWSS_ALPHA
    beta: float = DEFAULT_DTWSS_BETA
    wait_weight: float = DEFAULT_DTWSS_WAIT_WEIGHT
    blip_tax_delta: float = DEFAULT_DTWSS_BLIP_TAX


def _phase_quality(ph: Phase, preds_for_skill: list[Pred],
                   alpha: float) -> float:
    """Best Gaussian (Δs, Δe) decay over same-skill predictions."""
    L = ph.t_end - ph.t_start
    sigma = alpha * L if L > 0 else 0.0
    best = 0.0
    for p in preds_for_skill:
        ds = abs(p.start - ph.t_start)
        de = abs(p.end - ph.t_end)
        if sigma > 0:
            q = math.exp(-((ds / sigma) ** 2 + (de / sigma) ** 2))
        else:
            q = 1.0 if (ds == 0.0 and de == 0.0) else 0.0
        if q > best:
            best = q
    return best


def _option_quality(opt: Option, preds_for_skill: list[Pred],
                    alpha: float) -> float:
    """Phase-length-weighted mean of phase qualities (eq. option-q)."""
    num = 0.0
    denom = 0.0
    for ph in opt.phases:
        L = max(0.0, ph.t_end - ph.t_start)
        q = _phase_quality(ph, preds_for_skill, alpha)
        num += L * q
        denom += L
    return num / denom if denom > 0 else 0.0


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    iv = sorted((s, t) for s, t in intervals if t > s)
    if not iv:
        return 0.0
    total = 0.0
    cur_s, cur_t = iv[0]
    for s, t in iv[1:]:
        if s <= cur_t:
            cur_t = max(cur_t, t)
        else:
            total += cur_t - cur_s
            cur_s, cur_t = s, t
    total += cur_t - cur_s
    return total


def _overlap_with_union(p_start: float, p_end: float,
                        merged: list[tuple[float, float]]) -> float:
    overlap = 0.0
    for s, t in merged:
        a = max(p_start, s)
        b = min(p_end, t)
        if b > a:
            overlap += b - a
    return overlap


def compute_disjunctive_twss(
    gt: RobotGT,
    preds: Iterable[Pred],
    *,
    alpha: float = DEFAULT_DTWSS_ALPHA,
    beta: float = DEFAULT_DTWSS_BETA,
    wait_weight: float = DEFAULT_DTWSS_WAIT_WEIGHT,
    blip_tax_delta: float = DEFAULT_DTWSS_BLIP_TAX,
) -> DisjunctiveTWSS:
    """Time-Weighted Soft Schedule Score on a disjunctive GT (twss_metric §6)."""
    preds = list(preds)
    T = max(0.0, gt.duration_sec)
    total_pred_time = sum(p.duration for p in preds)

    if T <= 0 or not gt.interventions:
        return DisjunctiveTWSS(
            twss=0.0, task_score=0.0, wait_score=0.0,
            wait_quality=1.0, wait_mass=1.0 if T > 0 else 0.0,
            consistency=1.0,
            fp_time=0.0, fp_time_eff=0.0,
            total_wait_time=T, total_pred_time=total_pred_time,
            selected_options={}, per_intervention_q={},
            alpha=alpha, beta=beta,
            wait_weight=wait_weight, blip_tax_delta=blip_tax_delta,
        )

    # Per-intervention: pick the best option (max over options of phase-length
    # weighted mean phase quality). Track which interventions the agent has
    # actually committed to (fired any same-skill prediction): a silent agent
    # has not committed to any plan, so we keep it out of the C calculation
    # later — otherwise an arbitrary default tiebreak would penalise silence.
    selected: dict[str, int] = {}
    Q_per: dict[str, float] = {}
    L_star: dict[str, float] = {}
    committed: set[str] = set()
    selected_phases_by_skill: dict[str, list[tuple[float, float]]] = {}
    selected_phases_by_iv: dict[str, list[tuple[float, float]]] = {}
    for iv in gt.interventions:
        skill_preds = [p for p in preds
                       if _strip_args(p.skill) == _strip_args(iv.skill)]
        best_idx = 0
        best_q = -1.0
        for k, opt in enumerate(iv.options):
            q = _option_quality(opt, skill_preds, alpha)
            if q > best_q:
                best_q = q
                best_idx = k
        if best_q < 0:
            best_q = 0.0
        
        # Only consider the agent committed to an option if it made a prediction
        # that actually matches the option functionally (q > 0). This prevents
        # completely wild predictions (like Far-shifted) from defaulting to 
        # option 0 and causing fake cross-task conflicts.
        if best_q > 0:
            committed.add(iv.id)

        selected[iv.id] = best_idx
        Q_per[iv.id] = best_q
        chosen = iv.options[best_idx]
        L_star[iv.id] = sum(max(0.0, ph.t_end - ph.t_start)
                            for ph in chosen.phases)
        ivs_phases: list[tuple[float, float]] = []
        for ph in chosen.phases:
            s = max(0.0, min(T, ph.t_start))
            t = max(0.0, min(T, ph.t_end))
            if t > s:
                selected_phases_by_skill.setdefault(
                    _strip_args(iv.skill), []).append((s, t))
                ivs_phases.append((s, t))
        selected_phases_by_iv[iv.id] = ivs_phases

    # Wait region = complement of union of selected phases.
    all_selected_intervals = [iv for ivs in selected_phases_by_iv.values()
                              for iv in ivs]
    selected_union_len = _union_length(all_selected_intervals)
    total_wait_time = max(0.0, T - selected_union_len)
    
    L_total = sum(L_star.values())
    Z = L_total + wait_weight * total_wait_time
    if Z <= 0:
        Z = T if T > 0 else 1.0

    wait_mass = total_wait_time / Z

    # Pollution: per-prediction time outside the same-skill selected union.
    union_by_skill: dict[str, list[tuple[float, float]]] = {}
    for sk, ivs in selected_phases_by_skill.items():
        merged: list[tuple[float, float]] = []
        for s, t in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t))
            else:
                merged.append((s, t))
        union_by_skill[sk] = merged

    fp_time = 0.0
    fp_time_eff = 0.0
    for p in preds:
        ps = max(0.0, min(T, p.start))
        pt = max(0.0, min(T, p.end))
        if pt <= ps:
            continue
        sanctioned = union_by_skill.get(_strip_args(p.skill), [])
        sanc_overlap = _overlap_with_union(ps, pt, sanctioned)
        ell = (pt - ps) - sanc_overlap
        if ell <= 0:
            continue
        fp_time += ell
        fp_time_eff += math.sqrt(ell * ell + 2 * blip_tax_delta * ell)

    if total_wait_time <= 0:
        wait_quality = 1.0
    else:
        sigma_w = beta * total_wait_time
        wait_quality = (math.exp(-fp_time_eff / sigma_w)
                        if sigma_w > 0 else 0.0)
    wait_score = wait_weight * wait_mass * wait_quality

    # Consistency: 1 − max cross-task pair IoU, restricted to interventions
    # the agent committed to (i.e. fired at least one same-skill prediction).
    # We compute this *per-intervention* to avoid global task-score wipeout.
    C_per: dict[str, float] = {iid: 1.0 for iid in Q_per}
    if len(committed) >= 2:
        ids = list(committed)
        for i in range(len(ids)):
            C_worst_i = 0.0
            for ph_a in selected_phases_by_iv[ids[i]]:
                for j in range(len(ids)):
                    if i == j:
                        continue
                    for ph_b in selected_phases_by_iv[ids[j]]:
                        C_worst_i = max(C_worst_i, _iou(
                            ph_a[0], ph_a[1], ph_b[0], ph_b[1]))
            C_per[ids[i]] = max(0.0, min(1.0, 1.0 - C_worst_i))

    if L_total > 0:
        consistency = sum(L_star[iid] * C_per[iid] for iid in Q_per) / L_total
    else:
        consistency = 1.0

    # task_score = Σ (L_i*/Z) · Q_i · C_i
    task_score = sum((L_star[iid] / Z) * Q_per[iid] * C_per[iid] for iid in Q_per)

    # The consistency is now baked into task_score locally.
    twss = task_score + wait_score

    return DisjunctiveTWSS(
        twss=twss,
        task_score=task_score,
        wait_score=wait_score,
        wait_quality=wait_quality,
        wait_mass=wait_mass,
        consistency=consistency,
        fp_time=fp_time,
        fp_time_eff=fp_time_eff,
        total_wait_time=total_wait_time,
        total_pred_time=total_pred_time,
        selected_options=selected,
        per_intervention_q=Q_per,
        alpha=alpha, beta=beta,
        wait_weight=wait_weight, blip_tax_delta=blip_tax_delta,
    )

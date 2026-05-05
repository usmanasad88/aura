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

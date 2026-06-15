#!/usr/bin/env python3
"""Generate a single intervention-timeline comparison figure across ablations.

Each ablation is a normal run directory (``logs/<run>/``) with the usual
``decision_engine/`` + ``intent_monitor/`` layout. This script stacks one
prediction track per ablation against a single GT robot track, so a reader
can see at a glance which ablation matches / misses / over-fires.

Default: hand_layup ablations (listed best → worst)
    - hand_layup_skip_intent_perception
        Use GT Intention Results (Skip Perception Monitor) - Use Perception Monitor
    - hand_layup_gt_intent_no_perception
        Use GT Intention Previous State - No perception monitor
    - hand_layup_gt_intent_perception
        Use GT Intention Previous State - Use perception monitor
    - hand_layup_self_intent_perception
        Use Self Created Previous State - Use perception monitor

Usage::

    # Default (hand_layup, four ablations above):
    python scripts/eval/generate_timeline_comparison.py

    # Pick your own run dirs and labels:
    python scripts/eval/generate_timeline_comparison.py \
        --run logs/hand_layup_skip_intent_perception="GT intent (skip perception)" \
        --run logs/hand_layup_gt_intent_perception="GT prev-state (perception)" \
        --output figures/generated/fig_ablation_timeline.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_timeline import (
    AURA_ROOT,
    Event,
    _find_call_dir,
    _read_json,
    _resolve_robot_gt,
    load_predictions,
    load_robot_gt,
    match_predictions,
    plot_multi_model_timeline,
)


# ── A-Score (Appropriateness Score) ──────────────────────────────────────────
# Per the paper (mdpi_main.tex §"The Appropriateness Score (A-Score)"):
#   A_score = w_t * A_time + w_a * A_act + w_n * A_nec
#   A_time  = temporal IoU between matched (GT, prediction) intervals
#   A_act   = 1 if predicted skill matches GT skill, else 0
#   A_nec   = 1 - (FP-time fraction); penalises predictions made while GT == WAIT
# Unmatched GT events (misses) contribute 0 to A_time / A_act, so the score
# reflects coverage as well as quality. Default weights split evenly (1/3 each).

DEFAULT_WEIGHTS = (1.0 / 3, 1.0 / 3, 1.0 / 3)  # (w_t, w_a, w_n)
MATCH_TOLERANCE_SEC = 15.0  # same tolerance the plot uses for TP/FP colouring


@dataclass
class AScore:
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    a_time: float       # IoU averaged over GT events (misses count as 0)
    a_act: float        # action-match averaged over GT events (misses count as 0)
    a_nec: float        # 1 - FP_time / total_pred_time (1.0 if no predictions)
    a_score: float
    weights: tuple[float, float, float]
    matched_iou_mean: float    # diagnostic: IoU averaged only over matched pairs
    matched_act_acc: float     # diagnostic: action accuracy on matched pairs
    psa: float                 # project schedule adherence: LIS length / n_gt


def _strip_args(action: str) -> str:
    """Reduce ``skill(arg=val, ...)`` to bare skill for action comparison."""
    return action.split("(", 1)[0].strip()


def _interval_iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _project_schedule_adherence(gt_events: list[Event],
                                pred_events: list[Event],
                                total_duration: float) -> float:
    """Schedule adherence = (actual time worked or available) / (total time scheduled).

    The GT schedule covers the full project: each instant is either an active
    intervention (Move Resin, Move Hardener, Consolidate, Return Hardener, …)
    or implicit WAIT. Adherence at instant t requires the prediction to match
    the GT at t — same skill during an active interval, or also-WAIT during a
    WAIT interval. Returned as a fraction in [0, 1] (×100 for percentage).

    Computed analytically over the disjoint intervals (no discretisation):
      adherent = total_duration
                 - missed_gt_time            (GT active but pred not matching)
                 - pred_outside_gt_time      (FP into a WAIT slot)
                 - pred_in_gt_wrong_skill    (active during GT but wrong skill)
    """
    if total_duration <= 0:
        return 1.0

    def _overlap(a: Event, b: Event) -> float:
        return max(0.0, min(a.end, b.end) - max(a.start, b.start))

    # Pred time that overlaps a GT interval with the matching skill.
    adherent_active = sum(
        _overlap(g, p)
        for g in gt_events for p in pred_events
        if _strip_args(g.action) == _strip_args(p.action)
    )
    # Pred time that overlaps any GT interval (regardless of skill).
    pred_in_gt = sum(_overlap(g, p) for g in gt_events for p in pred_events)

    gt_total = sum(max(g.end - g.start, 0.0) for g in gt_events)
    pred_total = sum(max(p.end - p.start, 0.0) for p in pred_events)

    missed_gt = gt_total - adherent_active                # GT active, pred silent/wrong
    pred_outside_gt = pred_total - pred_in_gt             # pred fired during WAIT
    pred_wrong_skill = pred_in_gt - adherent_active       # pred during GT, wrong skill

    non_adherent = missed_gt + pred_outside_gt + pred_wrong_skill
    adherent = total_duration - non_adherent
    return max(0.0, min(1.0, adherent / total_duration))


def compute_a_score(gt_events: list[Event],
                    pred_events: list[Event],
                    total_duration: float,
                    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
                    tolerance: float = MATCH_TOLERANCE_SEC) -> AScore:
    """Compute the Appropriateness Score for one ablation vs shared GT.

    Uses the same start-time matching as ``plot_multi_model_timeline`` so
    the numbers and the visual TP/FP/FN colouring stay consistent.
    """
    robot_gt = [e for e in gt_events if e.agent == "robot"]
    matching = match_predictions(robot_gt, pred_events, tolerance=tolerance)
    matched = matching["matched"]            # list[(gt, pred)]
    fps = matching["false_positives"]        # list[pred]
    misses = matching["missed"]              # list[gt]

    # Per-event timeliness/action over the GT population (misses → 0).
    n_gt = len(robot_gt)
    if n_gt > 0:
        a_time = sum(_interval_iou(gt, pr) for gt, pr in matched) / n_gt
        a_act = (sum(1.0 if _strip_args(pr.action) == _strip_args(gt.action) else 0.0
                     for gt, pr in matched) / n_gt)
    else:
        a_time = 0.0
        a_act = 1.0  # nothing to do, vacuously correct

    # Necessity: fraction of predicted time that lines up with a real GT need.
    total_pred_time = sum(max(p.end - p.start, 0.0) for p in pred_events)
    fp_time = sum(max(p.end - p.start, 0.0) for p in fps)
    a_nec = 1.0 - (fp_time / total_pred_time) if total_pred_time > 0 else 1.0

    # Diagnostics over matched pairs only.
    if matched:
        matched_iou_mean = sum(_interval_iou(g, p) for g, p in matched) / len(matched)
        matched_act_acc = (sum(1.0 if _strip_args(p.action) == _strip_args(g.action) else 0.0
                               for g, p in matched) / len(matched))
    else:
        matched_iou_mean = 0.0
        matched_act_acc = 0.0

    w_t, w_a, w_n = weights
    a_score = w_t * a_time + w_a * a_act + w_n * a_nec

    psa = _project_schedule_adherence(robot_gt, pred_events, total_duration)

    return AScore(
        n_gt=n_gt,
        n_pred=len(pred_events),
        tp=len(matched),
        fp=len(fps),
        fn=len(misses),
        a_time=a_time,
        a_act=a_act,
        a_nec=a_nec,
        a_score=a_score,
        weights=weights,
        matched_iou_mean=matched_iou_mean,
        matched_act_acc=matched_act_acc,
        psa=psa,
    )


# ── Time-Weighted Soft Schedule Score (TWSS) ─────────────────────────────────
# Tolerance-free score with smooth Gaussian timing decay. Each GT event
# contributes by its share of the total task duration (re-weighted by γ); the
# implicit WAIT segment gets its own contribution scaled by w_wait. No hard
# match cliff.
#
#   TWSS         = (L_total / Z) · Σ_i (L_i^γ / Σ_j L_j^γ) · q_i
#                  + w_wait · (T_wait / Z) · q_wait
#   Z            = L_total + w_wait · T_wait
#   q_i          = max over same-skill predictions of
#                      exp(−((Δs / σ_i)² + (Δe / σ_i)²))
#   σ_i          = α · L_i               (α tunable, default 1.0)
#   q_wait       = exp(−T_fp_eff / (β · T_wait))
#   T_fp_eff     = Σ over predictions p of  c(ℓ_p^fp)
#                  with c(ℓ) = √(ℓ² + 2δℓ),  ℓ_p^fp = pollution length of p
#   ℓ_p^fp       = duration(p) − overlap(p, ∪ same-skill GT intervals)
#
# Why γ and δ:
#   * γ ∈ (0, 1] re-weights events. γ=1 → time-share (default, original
#     behaviour); γ<1 inflates short events relative to long ones.
#   * δ ≥ 0 is the smooth per-blip "interruption tax", in seconds. δ=0
#     recovers raw T_fp. With δ>0, ten 1s blips cost much more than one 10s
#     blip even though their raw pollution time is identical — each
#     additional spurious interval is something the operator must handle.
#     c(ℓ) is concave near zero (∝ √ℓ for small ℓ), then asymptotes to ℓ + δ.
#
# Pollution semantics: a prediction whose skill is in GT but whose timing is
# off contributes a small q_i (timing-decayed) AND its non-overlapping time
# becomes pollution. A prediction whose skill is absent from GT entirely is
# pure pollution. Multiple GT events of the same skill may share the same
# prediction — timing decay prevents one prediction from earning credit for
# two well-separated events.

DEFAULT_TWSS_ALPHA = 1.0
DEFAULT_TWSS_BETA = 0.5
DEFAULT_TWSS_GAMMA = 1.0           # event-weight exponent: 1.0 = current
DEFAULT_TWSS_WAIT_WEIGHT = 1.0     # multiplier on wait term: 1.0 = current
DEFAULT_TWSS_BLIP_TAX = 0.0        # per-blip smoothing δ (s): 0.0 = current

# CLI defaults: the recommended "tuned" configuration. The vanilla TWSS is
# computed alongside (always with γ=1, w_wait=1, δ=0, plus β=DEFAULT_TWSS_BETA)
# for comparison.
TUNED_TWSS_BETA = 1.5              # gentler wait penalty so high-quality runs aren't crushed by their own FP cost
TUNED_TWSS_GAMMA = 0.5             # √-weighting — short events less crushed
TUNED_TWSS_WAIT_WEIGHT = 0.6       # wait term contributes meaningfully without dominating
TUNED_TWSS_BLIP_TAX = 4.0          # 4 s of equivalent pollution per blip


@dataclass
class TWSSPerEvent:
    gt_action: str
    L_i: float                  # GT event duration (s)
    weight: float               # effective task-mass contribution: (L_total/Z) · L_i^γ / Σ L_j^γ
    q_i: float                  # final per-event quality in [0, 1]
    matched_pred_idx: int | None  # which prediction maximised q_i (None if no same-skill pred)
    dstart: float | None        # |t_start_pred − t_start_gt| at the argmax pred
    dend: float | None          # |t_end_pred − t_end_gt|     at the argmax pred
    sigma: float                # α · L_i, the timing scale used


@dataclass
class TWSS:
    twss: float
    task_score: float           # (L_total/Z) · Σ (L_i^γ / Σ L_j^γ) · q_i
    task_mass_available: float  # L_total / Z (max possible task_score if every q_i=1)
    wait_score: float           # w_wait · (T_wait / Z) · q_wait
    wait_quality: float         # q_wait, computed from T_fp_eff
    wait_mass: float            # T_wait / Z (raw, before w_wait)
    fp_time: float              # T_fp (raw pollution sum, s)
    fp_time_eff: float          # T_fp_eff after blip-tax inflation (s)
    total_wait_time: float      # T_wait (s)
    total_pred_time: float      # diagnostic
    n_gt: int
    n_pred: int
    alpha: float
    beta: float
    gamma: float
    wait_weight: float
    blip_tax_delta: float
    end_weight: float = 1.0
    per_event: list[TWSSPerEvent] = field(default_factory=list)


def _clip_to_window(s: float, t: float, total_duration: float) -> tuple[float, float]:
    return max(0.0, s), min(total_duration, t)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Length of the union of half-open intervals, after merging overlaps."""
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


def _overlap_with_union(p: Event,
                        union_intervals: list[tuple[float, float]],
                        total_duration: float) -> float:
    """Overlap length of prediction ``p`` (clipped to [0, T]) with the given
    interval union (already clipped). Linear scan since lists are small."""
    ps, pt = _clip_to_window(p.start, p.end, total_duration)
    if pt <= ps:
        return 0.0
    overlap = 0.0
    for s, t in union_intervals:
        a = max(ps, s)
        b = min(pt, t)
        if b > a:
            overlap += b - a
    return overlap


def compute_twss(gt_events: list[Event],
                 pred_events: list[Event],
                 total_duration: float,
                 alpha: float = DEFAULT_TWSS_ALPHA,
                 beta: float = DEFAULT_TWSS_BETA,
                 gamma: float = DEFAULT_TWSS_GAMMA,
                 wait_weight: float = DEFAULT_TWSS_WAIT_WEIGHT,
                 blip_tax_delta: float = DEFAULT_TWSS_BLIP_TAX,
                 end_weight: float = 1.0) -> TWSS:
    # ``end_weight`` scales the end-time term of the per-event timing decay:
    #   q_i = exp(−((Δs/σ)² + end_weight·(Δe/σ)²)).
    # end_weight=1.0 is the original symmetric decay; end_weight=0 makes the
    # score depend only on the *start* time (no early-stop penalty), which
    # suits a live robot that runs each triggered skill to completion
    # regardless of how long the LLM kept re-asserting it. Over-extension is
    # still penalised separately through the wait-pollution term.
    """Time-Weighted Soft Schedule Score (see module-level docstring above)."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if blip_tax_delta < 0:
        raise ValueError(f"blip_tax_delta must be ≥ 0, got {blip_tax_delta}")

    robot_gt = [e for e in gt_events if e.agent == "robot"]
    n_gt = len(robot_gt)
    n_pred = len(pred_events)
    total_pred_time = sum(max(p.end - p.start, 0.0) for p in pred_events)

    if total_duration <= 0:
        return TWSS(
            twss=0.0, task_score=0.0, task_mass_available=0.0,
            wait_score=0.0, wait_quality=0.0, wait_mass=0.0,
            fp_time=0.0, fp_time_eff=0.0, total_wait_time=0.0,
            total_pred_time=total_pred_time,
            n_gt=n_gt, n_pred=n_pred,
            alpha=alpha, beta=beta, gamma=gamma,
            wait_weight=wait_weight, blip_tax_delta=blip_tax_delta,
            end_weight=end_weight,
            per_event=[],
        )

    # Pre-pass to compute per-event L_i so we can normalise the γ weights.
    sorted_gt = sorted(robot_gt, key=lambda e: e.start)
    L_list = [max(g.end - g.start, 0.0) for g in sorted_gt]
    L_total = sum(L_list)
    L_pow_total = sum(L ** gamma for L in L_list) if L_total > 0 else 0.0
    
    total_wait_time = max(0.0, total_duration - L_total)
    Z = L_total + wait_weight * total_wait_time
    if Z <= 0:
        Z = total_duration if total_duration > 0 else 1.0

    task_mass_available = L_total / Z  # max task_score if all q_i=1

    # Per-event quality: each GT event picks its argmax same-skill prediction
    # (no claim/lock — multiple GT events may share). Gaussian two-factor decay.
    per_event: list[TWSSPerEvent] = []
    task_score = 0.0
    for g, L_i in zip(sorted_gt, L_list):
        # Effective per-event mass: (L_total/Z) · L_i^γ / Σ L_j^γ.
        # γ=1 collapses this to L_i / Z.
        if L_pow_total > 0:
            w_i = (L_total / Z) * (L_i ** gamma) / L_pow_total
        else:
            w_i = 0.0
        sigma = alpha * L_i if L_i > 0 else 0.0
        gt_skill = _strip_args(g.action)

        best_q = 0.0
        best_idx: int | None = None
        best_dstart: float | None = None
        best_dend: float | None = None
        for j, p in enumerate(pred_events):
            if _strip_args(p.action) != gt_skill:
                continue
            ds = abs(p.start - g.start)
            de = abs(p.end - g.end)
            if sigma > 0:
                q = math.exp(-((ds / sigma) ** 2 + end_weight * (de / sigma) ** 2))
            else:
                q = 1.0 if (ds == 0.0 and (end_weight == 0.0 or de == 0.0)) else 0.0
            if q > best_q:
                best_q = q
                best_idx = j
                best_dstart = ds
                best_dend = de

        q_i = best_q
        task_score += w_i * q_i
        per_event.append(TWSSPerEvent(
            gt_action=gt_skill,
            L_i=L_i,
            weight=w_i,
            q_i=q_i,
            matched_pred_idx=best_idx,
            dstart=best_dstart,
            dend=best_dend,
            sigma=sigma,
        ))

    # Pollution: per-prediction time outside the union of same-skill GT slots.
    # Build per-skill GT unions once (clipped to [0, T]).
    gt_by_skill: dict[str, list[tuple[float, float]]] = {}
    for g in robot_gt:
        s, t = _clip_to_window(g.start, g.end, total_duration)
        if t > s:
            gt_by_skill.setdefault(_strip_args(g.action), []).append((s, t))
    union_by_skill: dict[str, list[tuple[float, float]]] = {}
    for sk, ivs in gt_by_skill.items():
        # Build a merged union explicitly for fast overlap.
        merged: list[tuple[float, float]] = []
        for s, t in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t))
            else:
                merged.append((s, t))
        union_by_skill[sk] = merged

    # Per-blip pollution. T_fp = raw sum; T_fp_eff inflates each blip via
    # c(ℓ) = √(ℓ² + 2δℓ). Smooth in ℓ, c(0)=0, c(ℓ) ≈ √(2δℓ) for small ℓ
    # (concave inflation), c(ℓ) → ℓ + δ for large ℓ.
    fp_time = 0.0
    fp_time_eff = 0.0
    for p in pred_events:
        ps, pt = _clip_to_window(p.start, p.end, total_duration)
        if pt <= ps:
            continue
        sanctioned = union_by_skill.get(_strip_args(p.action), [])
        sanc_overlap = _overlap_with_union(
            Event(action=p.action, start=ps, end=pt, agent=p.agent),
            sanctioned, total_duration,
        )
        ell = (pt - ps) - sanc_overlap
        if ell <= 0:
            continue
        fp_time += ell
        fp_time_eff += math.sqrt(ell * ell + 2 * blip_tax_delta * ell)

    wait_mass = total_wait_time / Z

    if total_wait_time <= 0:
        # No wait segment exists → wait axis is vacuous.
        wait_quality = 1.0
    else:
        sigma_w = beta * total_wait_time
        wait_quality = math.exp(-fp_time_eff / sigma_w) if sigma_w > 0 else 0.0

    wait_score = wait_weight * wait_mass * wait_quality
    twss = task_score + wait_score

    return TWSS(
        twss=twss,
        task_score=task_score,
        task_mass_available=task_mass_available,
        wait_score=wait_score,
        wait_quality=wait_quality,
        wait_mass=wait_mass,
        fp_time=fp_time,
        fp_time_eff=fp_time_eff,
        total_wait_time=total_wait_time,
        total_pred_time=total_pred_time,
        n_gt=n_gt,
        n_pred=n_pred,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        wait_weight=wait_weight,
        blip_tax_delta=blip_tax_delta,
        end_weight=end_weight,
        per_event=per_event,
    )


def format_score_table(scores: dict[str, AScore]) -> str:
    """Pretty-print a fixed-width A-Score table for the terminal."""
    headers = ["Ablation", "GT", "TP", "FP", "FN",
               "A_time", "A_act", "A_nec", "A_score", "PSA"]
    rows = []
    for label, s in scores.items():
        rows.append([
            label,
            str(s.n_gt),
            str(s.tp),
            str(s.fp),
            str(s.fn),
            f"{s.a_time:.3f}",
            f"{s.a_act:.3f}",
            f"{s.a_nec:.3f}",
            f"{s.a_score:.3f}",
            f"{s.psa:.3f}",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    line = fmt.format(*headers)
    out = [line, "-" * len(line)]
    out += [fmt.format(*r) for r in rows]
    return "\n".join(out)


def format_twss_table(twss_exp: dict[str, TWSS]) -> str:
    """Pretty-print a fixed-width TWSS table (exp wait penalty only)."""
    headers = ["Ablation",
               "task", "wait", "qwait", "TWSS",
               "T_fp", "T_fp_eff"]
    rows = []
    for label, e in twss_exp.items():
        rows.append([
            label,
            f"{e.task_score:.3f}",
            f"{e.wait_score:.3f}",
            f"{e.wait_quality:.3f}",
            f"{e.twss:.3f}",
            f"{e.fp_time:.1f}",
            f"{e.fp_time_eff:.1f}",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    line = fmt.format(*headers)
    out = [line, "-" * len(line)]
    out += [fmt.format(*r) for r in rows]
    return "\n".join(out)


# Ablation name -> (run_dir relative to logs/, human-readable label)
DEFAULT_ABLATIONS: list[tuple[str, str]] = [
    ("hand_layup_skip_intent_perception",
     "GT intent results (skip intent monitor) + perception"),
    ("hand_layup_gt_intent_no_perception",
     "GT intent prev-state, no perception"),
    ("hand_layup_gt_intent_perception",
     "GT intent prev-state + perception"),
    ("hand_layup_self_intent_perception",
     "Self intent prev-state + perception"),
]


def _parse_run_arg(s: str) -> tuple[Path, str]:
    """Parse ``path=label`` (label optional). Resolves relative paths under AURA_ROOT."""
    if "=" in s:
        path_str, label = s.split("=", 1)
    else:
        path_str, label = s, ""
    p = Path(path_str)
    if not p.is_absolute():
        p = AURA_ROOT / p
    if not label:
        label = p.name
    return p, label


def _collect_predictions(run_dir: Path) -> list[Event]:
    dec_dir = _find_call_dir(run_dir, "decision_engine")
    if dec_dir is None:
        print(f"  [skip] no decision_engine session in {run_dir}", file=sys.stderr)
        return []
    return load_predictions(dec_dir)


def _resolve_gt_for_runs(runs: list[tuple[Path, str]],
                         task_override: str | None,
                         gt_override: Path | None) -> Path:
    """Pick the robot GT to compare every ablation against."""
    if gt_override:
        return gt_override
    # Use the first run's settings.json to derive task + video.
    for run_dir, _ in runs:
        settings = _read_json(run_dir / "settings.json")
        if not settings:
            continue
        task = task_override or settings.get("task_name") or settings.get("task")
        video = settings.get("video_path") or settings.get("video")
        if not task:
            continue
        gt = _resolve_robot_gt(task, video)
        if gt and gt.exists():
            return gt
    # Last resort: fall back to whatever exists for the override task.
    if task_override:
        gt = _resolve_robot_gt(task_override, None)
        if gt and gt.exists():
            return gt
    raise FileNotFoundError(
        "Could not locate a robot_gt.json for the supplied runs. "
        "Pass --gt explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack ablation prediction tracks against a single GT robot track.")
    parser.add_argument(
        "--run", action="append", default=None,
        help="Ablation run dir, optionally `path=label`. Repeatable. "
             "If omitted, uses the four default hand_layup ablations.")
    parser.add_argument("--task", default=None,
                        help="Task name for GT lookup (default: read from settings.json).")
    parser.add_argument("--gt", type=Path, default=None,
                        help="Explicit robot_gt.json path (skips auto-resolve).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (PNG sibling is also written). "
                             "Default: figures/generated/fig_ablation_timeline_comparison.pdf")
    parser.add_argument("--title", type=str,
                        default="Hand Layup — Ablation Intervention Timeline",
                        help="Figure title.")
    parser.add_argument("--show-legend", action="store_true",
                        help="Include the TP/FP/FN legend.")
    parser.add_argument("--weights", type=str, default=None,
                        help="A-Score weights as 'w_t,w_a,w_n' (must sum to 1). "
                             f"Default: {DEFAULT_WEIGHTS[0]:.3f},"
                             f"{DEFAULT_WEIGHTS[1]:.3f},"
                             f"{DEFAULT_WEIGHTS[2]:.3f}")
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE_SEC,
                        help="Start-time matching tolerance in seconds "
                             f"(default: {MATCH_TOLERANCE_SEC}).")
    parser.add_argument("--twss-alpha", type=float, default=DEFAULT_TWSS_ALPHA,
                        help=f"TWSS Gaussian timing scale: σ_i = α · L_i "
                             f"(default: {DEFAULT_TWSS_ALPHA}).")
    parser.add_argument("--twss-beta", type=float, default=DEFAULT_TWSS_BETA,
                        help=f"TWSS wait-pollution scale (vanilla): σ_w = β · T_wait "
                             f"(default: {DEFAULT_TWSS_BETA}).")
    parser.add_argument("--twss-beta-tuned", type=float, default=TUNED_TWSS_BETA,
                        help=f"TWSS wait-pollution scale used by the tuned "
                             f"variant. Vanilla=0.5, tuned default={TUNED_TWSS_BETA}.")
    parser.add_argument("--twss-gamma", type=float, default=TUNED_TWSS_GAMMA,
                        help=f"TWSS event-weight exponent (γ): w_i ∝ L_i^γ. "
                             f"γ=1 → time-share; γ<1 inflates short events. "
                             f"Vanilla=1.0, tuned default={TUNED_TWSS_GAMMA}.")
    parser.add_argument("--twss-wait-weight", type=float,
                        default=TUNED_TWSS_WAIT_WEIGHT,
                        help=f"Multiplier on the wait term so it cannot "
                             f"dominate the task term. Vanilla=1.0, "
                             f"tuned default={TUNED_TWSS_WAIT_WEIGHT}.")
    parser.add_argument("--twss-blip-tax", type=float,
                        default=TUNED_TWSS_BLIP_TAX,
                        help=f"Smooth per-blip pollution tax δ (s): each FP "
                             f"blip costs c(ℓ)=√(ℓ²+2δℓ) instead of ℓ. "
                             f"Vanilla=0.0, tuned default={TUNED_TWSS_BLIP_TAX}.")
    parser.add_argument("--scores-output", type=Path, default=None,
                        help="JSON path for A-Score / TWSS results. Default: "
                             "alongside the figure as <output>.scores.json")
    args = parser.parse_args()

    if args.weights:
        try:
            ws = tuple(float(x) for x in args.weights.split(","))
        except ValueError:
            print(f"Error: --weights must be three comma-separated floats, "
                  f"got {args.weights!r}", file=sys.stderr)
            sys.exit(1)
        if len(ws) != 3 or abs(sum(ws) - 1.0) > 1e-6:
            print(f"Error: --weights must have 3 values summing to 1, got {ws}",
                  file=sys.stderr)
            sys.exit(1)
        weights = ws
    else:
        weights = DEFAULT_WEIGHTS

    # Build the (run_dir, label) list.
    if args.run:
        runs = [_parse_run_arg(s) for s in args.run]
    else:
        runs = [(AURA_ROOT / "logs" / name, label) for name, label in DEFAULT_ABLATIONS]

    missing = [str(p) for p, _ in runs if not p.is_dir()]
    if missing:
        print("Error: run directories not found:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        sys.exit(1)

    # Resolve the shared GT track.
    gt_path = _resolve_gt_for_runs(runs, args.task, args.gt)
    print(f"Using robot GT: {gt_path}")
    gt_events, total_duration = load_robot_gt(gt_path)
    if total_duration <= 0:
        total_duration = max((e.end for e in gt_events), default=270.0)

    # Load each ablation's predictions.
    model_preds: dict[str, list[Event]] = {}
    for run_dir, label in runs:
        print(f"  loading {run_dir.name} ({label})...")
        preds = _collect_predictions(run_dir)
        model_preds[label] = preds
        # Stretch x-range if any prediction extends past GT duration.
        if preds:
            total_duration = max(total_duration, max(e.end for e in preds))

    if not any(model_preds.values()):
        print("Error: no predictions loaded from any run.", file=sys.stderr)
        sys.exit(1)

    # Plot.
    fig = plot_multi_model_timeline(
        gt_events, model_preds,
        title=args.title,
        total_duration=total_duration,
        show_human_task=False,
        show_legend=args.show_legend,
    )

    # Output.
    out = args.output
    if out is None:
        out_dir = AURA_ROOT / "figures" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_ablation_timeline_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    png_out = str(out)
    if png_out.lower().endswith(".pdf"):
        png_out = png_out[:-4] + ".png"
    else:
        png_out = png_out + ".png"
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Saved comparison timeline to:\n  {out}\n  {png_out}")

    # ── A-Score and TWSS tables ──────────────────────────────────────────
    scores: dict[str, AScore] = {}
    # Vanilla: original TWSS formulation (γ=1, w_wait=1, δ=0).
    vanilla_exp: dict[str, TWSS] = {}
    # Tuned: with γ, w_wait, δ from CLI (defaults to recommended config).
    tuned_exp: dict[str, TWSS] = {}
    for label, preds in model_preds.items():
        scores[label] = compute_a_score(gt_events, preds,
                                        total_duration=total_duration,
                                        weights=weights,
                                        tolerance=args.tolerance)
        vanilla_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta,
            gamma=DEFAULT_TWSS_GAMMA,
            wait_weight=DEFAULT_TWSS_WAIT_WEIGHT,
            blip_tax_delta=DEFAULT_TWSS_BLIP_TAX)
        tuned_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta_tuned,
            gamma=args.twss_gamma,
            wait_weight=args.twss_wait_weight,
            blip_tax_delta=args.twss_blip_tax)

    print("\nA-Score (Appropriateness Score) — weights "
          f"w_t={weights[0]:.3f}, w_a={weights[1]:.3f}, w_n={weights[2]:.3f}, "
          f"match tolerance={args.tolerance:.1f}s")
    print(format_score_table(scores))

    print(f"\nVanilla TWSS — α={args.twss_alpha}, β={args.twss_beta}, "
          f"γ={DEFAULT_TWSS_GAMMA}, w_wait={DEFAULT_TWSS_WAIT_WEIGHT}, "
          f"δ={DEFAULT_TWSS_BLIP_TAX} (exp wait penalty)")
    print(format_twss_table(vanilla_exp))

    print(f"\nTuned TWSS — α={args.twss_alpha}, β={args.twss_beta_tuned}, "
          f"γ={args.twss_gamma}, w_wait={args.twss_wait_weight}, "
          f"δ={args.twss_blip_tax} (exp wait penalty)")
    print(format_twss_table(tuned_exp))

    scores_out = args.scores_output or out.with_suffix(".scores.json")
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_gt": str(gt_path),
        "weights": {"w_t": weights[0], "w_a": weights[1], "w_n": weights[2]},
        "match_tolerance_sec": args.tolerance,
        "twss_alpha": args.twss_alpha,
        "twss_vanilla": {
            "beta": args.twss_beta,
            "gamma": DEFAULT_TWSS_GAMMA,
            "wait_weight": DEFAULT_TWSS_WAIT_WEIGHT,
            "blip_tax_delta": DEFAULT_TWSS_BLIP_TAX,
        },
        "twss_tuned": {
            "beta": args.twss_beta_tuned,
            "gamma": args.twss_gamma,
            "wait_weight": args.twss_wait_weight,
            "blip_tax_delta": args.twss_blip_tax,
        },
        "ablations": {
            label: {
                "ascore": asdict(scores[label]),
                "twss_vanilla": asdict(vanilla_exp[label]),
                "twss_tuned": asdict(tuned_exp[label]),
            } for label in scores
        },
    }
    scores_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved A-Score / TWSS JSON to:\n  {scores_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Construct synthetic adversaries against the hand-layup robot GT and score them.

Each adversary takes the ground-truth robot intervention track and produces a
synthetic prediction track designed to trigger one specific failure mode. We
then score every adversary with the A-Score (and its three components) plus a
set of baseline metrics (event-F1, matched-pair IoU, matched-pair action
accuracy). The point is to show which baselines fail to discriminate which
adversaries — and that A-Score, jointly, does not.

Defaults:
    GT track:        tasks/hand_layup/ground_truth/
                         layup_gesture_demo_stationary_with_overlay.robot_gt.json
    Skill catalogue: tasks/hand_layup/config/robot_skills.json
    Wrong-action #6 picks a different skill from the catalogue (not from the
    GT track), per the validation design.

Outputs:
    figures/generated/fig_ascore_adversaries.{pdf,png}   — stacked timeline
    figures/generated/fig_ascore_adversaries.scores.json — full numbers
    figures/generated/fig_ascore_adversaries.tex         — LaTeX-ready table

Usage::

    python scripts/eval/ascore_adversaries.py
    python scripts/eval/ascore_adversaries.py --tolerance 15
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_timeline import (
    AURA_ROOT,
    Event,
    load_robot_gt,
    match_predictions,
    plot_multi_model_timeline,
)
from generate_timeline_comparison import (
    DEFAULT_TWSS_ALPHA,
    DEFAULT_TWSS_BETA,
    DEFAULT_TWSS_BLIP_TAX,
    DEFAULT_TWSS_GAMMA,
    DEFAULT_TWSS_WAIT_WEIGHT,
    DEFAULT_WEIGHTS,
    MATCH_TOLERANCE_SEC,
    TUNED_TWSS_BETA,
    TUNED_TWSS_BLIP_TAX,
    TUNED_TWSS_GAMMA,
    TUNED_TWSS_WAIT_WEIGHT,
    AScore,
    TWSS,
    _strip_args,
    compute_a_score,
    compute_twss,
)


DEFAULT_GT = (AURA_ROOT / "tasks" / "hand_layup" / "ground_truth"
              / "layup_gesture_demo_stationary_with_overlay.robot_gt.json")
DEFAULT_SKILLS = (AURA_ROOT / "tasks" / "hand_layup" / "config"
                  / "robot_skills.json")

# Skill categories that are not realistic stand-alone proactive interventions
# (low-level motion / utility primitives). The wrong-action adversary should
# pick from intervention-level skills only, otherwise the swap is trivially
# silly and not a meaningful failure case.
EXCLUDED_SKILL_CATEGORIES = {"motion", "gripper", "utility"}

# Adversary parameters.
NEAR_MISS_OFFSET_SEC = 12.0      # < default tolerance (15s), should still match
FAR_SHIFT_OFFSET_SEC = 30.0      # > default tolerance, should miss
PREMATURE_OFFSET_SEC = -15.0     # symmetric direction
OVER_EXTEND_PAD_SEC = 5.0        # added to t_end of every event
SPAMMER_PERIOD_SEC = 5.0         # rapid-fire prediction period
SPAMMER_DURATION_SEC = 4.0       # length of each spam tick
TRUNCATED_INNER_FRAC = 0.25      # keep only the middle 25% of each GT interval
WAIT_BLIP_DURATION_SEC = 5.0     # length of one spurious prediction in wait


# ── Adversary constructors ───────────────────────────────────────────────────

def _shift(events: list[Event], dt: float) -> list[Event]:
    return [Event(action=e.action, start=e.start + dt, end=e.end + dt,
                  agent="robot") for e in events]


def _clip_to_window(events: list[Event], total_duration: float) -> list[Event]:
    """Truncate events to [0, total_duration]; drop those with no remaining mass.

    Offline evaluation has no GT frames past the recorded video duration, so
    predictions outside that window are not scoreable. We clip them at
    construction time so adversary scoring and plotting stay consistent with
    the recorded GT footprint.
    """
    out: list[Event] = []
    for e in events:
        s = max(0.0, e.start)
        t = min(total_duration, e.end)
        if t > s:
            out.append(Event(action=e.action, start=s, end=t, agent=e.agent))
    return out


def _most_common_skill(events: list[Event]) -> str:
    counts = Counter(_strip_args(e.action) for e in events)
    return counts.most_common(1)[0][0]


def _load_intervention_skill_ids(skills_path: Path,
                                 exclude_ids: set[str]) -> list[str]:
    """Read robot_skills.json and return intervention-level skill ids."""
    data = json.loads(skills_path.read_text())
    ids: list[str] = []
    for s in data.get("skills", []):
        sid = s.get("id")
        if not sid or sid in exclude_ids:
            continue
        if s.get("category") in EXCLUDED_SKILL_CATEGORIES:
            continue
        ids.append(sid)
    return ids


def adv_oracle(gt: list[Event], **_) -> list[Event]:
    return copy.deepcopy(gt)


def adv_silent(gt: list[Event], **_) -> list[Event]:
    return []


def adv_spammer(gt: list[Event], total_duration: float, **_) -> list[Event]:
    """Rapid-fire short predictions of the most-common GT skill."""
    skill = _most_common_skill(gt)
    out: list[Event] = []
    t = 0.0
    while t < total_duration:
        out.append(Event(action=skill, start=t,
                         end=min(t + SPAMMER_DURATION_SEC, total_duration),
                         agent="robot"))
        t += SPAMMER_PERIOD_SEC
    return out


def adv_far_shifted(gt: list[Event], **_) -> list[Event]:
    return _shift(gt, FAR_SHIFT_OFFSET_SEC)


def adv_near_miss(gt: list[Event], **_) -> list[Event]:
    return _shift(gt, NEAR_MISS_OFFSET_SEC)


def adv_wrong_action(gt: list[Event],
                     skill_pool: list[str], **_) -> list[Event]:
    """GT intervals, action labels swapped for a different catalogue skill.

    For each GT event, pick the next skill in ``skill_pool`` that is not the
    GT action. If the pool only contains the GT action (degenerate), keep it.
    """
    if not skill_pool:
        return copy.deepcopy(gt)
    out: list[Event] = []
    for i, e in enumerate(gt):
        gt_skill = _strip_args(e.action)
        # Cycle deterministically so the swap is reproducible.
        for k in range(1, len(skill_pool) + 1):
            cand = skill_pool[(i + k) % len(skill_pool)]
            if cand != gt_skill:
                break
        else:
            cand = gt_skill
        out.append(Event(action=cand, start=e.start, end=e.end, agent="robot"))
    return out


def adv_one_shot_blanket(gt: list[Event],
                         total_duration: float, **_) -> list[Event]:
    """A single interval covering the whole task with the most-common GT skill."""
    if not gt:
        return []
    return [Event(action=_most_common_skill(gt),
                  start=0.0, end=total_duration, agent="robot")]


def adv_half_coverage(gt: list[Event], **_) -> list[Event]:
    """Every other GT event, copied perfectly."""
    return [copy.deepcopy(e) for i, e in enumerate(gt) if i % 2 == 0]


def adv_over_extender(gt: list[Event], **_) -> list[Event]:
    """Correct starts and skills, but each end pushed by OVER_EXTEND_PAD_SEC."""
    return [Event(action=e.action, start=e.start, end=e.end + OVER_EXTEND_PAD_SEC,
                  agent="robot") for e in gt]


def adv_premature(gt: list[Event], **_) -> list[Event]:
    return _shift(gt, PREMATURE_OFFSET_SEC)


def adv_truncated(gt: list[Event], **_) -> list[Event]:
    """Right action and centred on the GT interval, but late-starting and
    early-stopping — only the middle ``TRUNCATED_INNER_FRAC`` of each GT
    interval is fired.
    """
    out: list[Event] = []
    for e in gt:
        dur = e.end - e.start
        keep = dur * TRUNCATED_INNER_FRAC
        if keep <= 0:
            continue
        centre = 0.5 * (e.start + e.end)
        out.append(Event(action=e.action,
                         start=centre - keep / 2,
                         end=centre + keep / 2,
                         agent="robot"))
    return out


def adv_wait_blip(gt: list[Event],
                  total_duration: float,
                  skill_pool: list[str], **_) -> list[Event]:
    """Oracle predictions plus one short spurious event during a wait gap.

    Picks the longest wait region (gap between GT intervals, or trailing
    tail) and inserts a single ``WAIT_BLIP_DURATION_SEC``-long prediction
    centred in it. The blip's action is drawn from ``skill_pool`` so it is
    a clean false positive (different skill, fired during wait).
    """
    out = [copy.deepcopy(e) for e in gt]
    sorted_gt = sorted(gt, key=lambda e: e.start)
    gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for e in sorted_gt:
        if e.start > prev_end:
            gaps.append((prev_end, e.start))
        prev_end = max(prev_end, e.end)
    if total_duration > prev_end:
        gaps.append((prev_end, total_duration))
    if not gaps:
        return out
    s, t = max(gaps, key=lambda g: g[1] - g[0])
    centre = 0.5 * (s + t)
    blip_s = max(s, centre - WAIT_BLIP_DURATION_SEC / 2)
    blip_e = min(t, centre + WAIT_BLIP_DURATION_SEC / 2)
    if blip_e <= blip_s:
        return out
    bad_skill = skill_pool[0] if skill_pool else _most_common_skill(gt)
    out.append(Event(action=bad_skill, start=blip_s, end=blip_e, agent="robot"))
    return out


# Order matters for table / figure readability.
ADVERSARIES: list[tuple[str, str, Callable[..., list[Event]]]] = [
    ("Oracle",                  "Copies GT exactly (sanity check)",          adv_oracle),
    ("Silent",                  "Never intervenes",                          adv_silent),
    ("Spammer",                 "Rapid-fire correct skill throughout",       adv_spammer),
    ("One-shot blanket",        "Single interval covering whole task",       adv_one_shot_blanket),
    ("Right-time-wrong-action", "GT intervals, swapped skill from catalogue", adv_wrong_action),
    ("Near-miss timing",        f"GT shifted +{NEAR_MISS_OFFSET_SEC:.0f}s (within tolerance)", adv_near_miss),
    ("Far-shifted",             f"GT shifted +{FAR_SHIFT_OFFSET_SEC:.0f}s (beyond tolerance)", adv_far_shifted),
    ("Premature firer",         f"GT shifted {PREMATURE_OFFSET_SEC:+.0f}s",  adv_premature),
    ("Over-extender",           f"Correct starts, ends padded +{OVER_EXTEND_PAD_SEC:.0f}s", adv_over_extender),
    ("Late-start early-stop",   f"Right action, only middle {TRUNCATED_INNER_FRAC:.0%} fired", adv_truncated),
    ("Wait-period blip",        f"Oracle plus one {WAIT_BLIP_DURATION_SEC:.0f}s spurious event in wait", adv_wait_blip),
    ("Half-coverage perfect",   "Every other GT event copied perfectly",     adv_half_coverage),
]


def plot_compact_monochrome_timeline(gt_events: list[Event],
                                     model_preds: dict[str, list[Event]],
                                     total_duration: float,
                                     title: str | None = None) -> "plt.Figure":
    """Compact, monochrome timeline used for the paper figure.

    Single shade for GT, single shade for predictions, no TP/FP/FN colouring,
    no legend, narrow rows. Adversary tracks are drawn top-to-bottom in the
    order of ``model_preds``.
    """
    import matplotlib.ticker as ticker

    robot_gt = sorted([e for e in gt_events if e.agent == "robot"],
                      key=lambda e: e.start)
    n_models = len(model_preds)
    n_tracks = 1 + n_models  # GT + adversaries

    # Drop Oracle (identical to GT), and the trivially-bad blanket / spammer
    # adversaries — they are uninteresting visually and crowd the figure.
    _hide = {"oracle", "one-shot blanket", "spammer"}
    model_preds = {n: ps for n, ps in model_preds.items()
                   if n.lower() not in _hide}
    n_models = len(model_preds)
    n_tracks = 1 + n_models

    row_h = 0.35
    fig_h = max(2.8, row_h * n_tracks + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h = 0.7
    gt_color = "#000000"
    pred_color = "#cfcfcf"

    def _short(s: str, n: int = 14) -> str:
        s = s.split("(", 1)[0].strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    # Roughly 2.4 chars fit per second of bar width at fontsize 7 in this
    # figure geometry; pick a label length that fits and only skip labels
    # for bars too short to show even a single character.
    def _label_for(action: str, dur: float) -> str | None:
        if dur < 1.0:
            return None
        max_chars = max(2, int(dur * 2.4))
        return _short(action, n=max_chars)

    # GT on top
    gt_y = n_tracks - 1
    for ev in robot_gt:
        dur = ev.end - ev.start
        ax.barh(gt_y, dur, left=ev.start, height=bar_h,
                color=gt_color, edgecolor="white", linewidth=0.4)
        label = _label_for(ev.action, dur)
        if label:
            ax.text(ev.start + dur / 2, gt_y, label,
                    ha="center", va="center", fontsize=7,
                    color="white", clip_on=True)

    # Adversaries below GT, in iteration order
    track_labels = []
    for i, (name, preds) in enumerate(model_preds.items()):
        y = n_tracks - 2 - i
        for p in sorted(preds, key=lambda e: e.start):
            dur = max(p.end - p.start, 0.4)
            ax.barh(y, dur, left=p.start, height=bar_h,
                    color=pred_color, edgecolor="white", linewidth=0.4)
            label = _label_for(p.action, dur)
            if label:
                ax.text(p.start + dur / 2, y, label,
                        ha="center", va="center", fontsize=7,
                        color="black", clip_on=True)
        track_labels.append(name)

    ax.set_yticks(range(n_tracks))
    ax.set_yticklabels(list(reversed(track_labels)) + ["GT"], fontsize=8)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_xlim(-1, total_duration + 2)
    ax.set_ylim(-0.6, n_tracks - 0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis="x", which="major", alpha=0.25)
    ax.grid(axis="x", which="minor", alpha=0.1, linestyle=":")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    fig.tight_layout()
    return fig


# ── Baseline metrics ─────────────────────────────────────────────────────────

@dataclass
class BaselineMetrics:
    event_precision: float
    event_recall: float
    event_f1: float
    matched_iou_mean: float    # IoU averaged over matched pairs only
    matched_act_acc: float     # action accuracy on matched pairs only
    p_agree: float             # temporal action agreement, see below


def compute_temporal_action_agreement(gt: list[Event],
                                      preds: list[Event],
                                      total_duration: float) -> float:
    """Fraction of task time at which the predicted active skill agrees with GT.

    For every instant t in [0, total_duration]:

      gt_skill(t)   = skill of the GT interval covering t, or WAIT if none.
      pred_set(t)   = {skill of every prediction interval covering t}, possibly
                      empty (then treated as WAIT).
      agree(t)      = 1 if (gt_skill is WAIT and pred_set is empty)
                       or (gt_skill in pred_set), else 0.

    Returns (1/T) * integral(agree(t) dt). Computed exactly by sweeping
    breakpoints — no sampling error. Overlapping prediction intervals are
    handled as a set so an agent that has leaked another skill's tail into
    the current GT window still gets credit if the *correct* skill is also
    being emitted at the same instant.
    """
    if total_duration <= 0:
        return 0.0

    bps = {0.0, total_duration}
    for e in list(gt) + list(preds):
        s = max(0.0, min(total_duration, e.start))
        t = max(0.0, min(total_duration, e.end))
        bps.add(s)
        bps.add(t)
    grid = sorted(bps)

    agree_time = 0.0
    for a, b in zip(grid, grid[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        gt_skill: str | None = None
        for e in gt:
            if e.start <= mid < e.end:
                gt_skill = _strip_args(e.action)
                break
        pred_skills: set[str] = set()
        for e in preds:
            if e.start <= mid < e.end:
                pred_skills.add(_strip_args(e.action))
        if gt_skill is None and not pred_skills:
            agree_time += b - a
        elif gt_skill is not None and gt_skill in pred_skills:
            agree_time += b - a

    return agree_time / total_duration


def compute_baselines(gt: list[Event], preds: list[Event],
                      tolerance: float, total_duration: float) -> BaselineMetrics:
    """Event-level baselines that ignore one or more A-Score axes,
    plus the time-domain agreement metric (P_agree)."""
    matching = match_predictions(gt, preds, tolerance=tolerance)
    tp = len(matching["matched"])
    fp = len(matching["false_positives"])
    fn = len(matching["missed"])
    p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    if matching["matched"]:
        ious = []
        accs = []
        for g, pr in matching["matched"]:
            inter = max(0.0, min(g.end, pr.end) - max(g.start, pr.start))
            union = max(g.end, pr.end) - min(g.start, pr.start)
            ious.append(inter / union if union > 0 else 0.0)
            accs.append(1.0 if _strip_args(pr.action) == _strip_args(g.action) else 0.0)
        matched_iou_mean = sum(ious) / len(ious)
        matched_act_acc = sum(accs) / len(accs)
    else:
        matched_iou_mean = 0.0
        matched_act_acc = 0.0

    p_agree = compute_temporal_action_agreement(gt, preds, total_duration)

    return BaselineMetrics(
        event_precision=p,
        event_recall=r,
        event_f1=f1,
        matched_iou_mean=matched_iou_mean,
        matched_act_acc=matched_act_acc,
        p_agree=p_agree,
    )


# ── Reporting ────────────────────────────────────────────────────────────────

def _format_twss_block(rows: list[dict], key: str, title: str) -> str:
    headers = ["Adversary",
               "TWSS", "task", "wait", "qwait",
               "T_fp", "T_fp_eff"]
    cells: list[list[str]] = [headers]
    for r in rows:
        e: TWSS = r[key]
        cells.append([
            r["name"],
            f"{e.twss:.3f}",
            f"{e.task_score:.3f}",
            f"{e.wait_score:.3f}",
            f"{e.wait_quality:.3f}",
            f"{e.fp_time:.1f}",
            f"{e.fp_time_eff:.1f}",
        ])
    widths = [max(len(c[i]) for c in cells) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [title,
             fmt.format(*cells[0]),
             "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines += [fmt.format(*c) for c in cells[1:]]
    return "\n".join(lines)


def format_table(rows: list[dict]) -> str:
    headers = ["Adversary", "F1", "mIoU", "mAcc", "P_agree",
               "A_time", "A_act", "A_nec", "A_score"]
    cells: list[list[str]] = [headers]
    for r in rows:
        cells.append([
            r["name"],
            f"{r['baselines'].event_f1:.3f}",
            f"{r['baselines'].matched_iou_mean:.3f}",
            f"{r['baselines'].matched_act_acc:.3f}",
            f"{r['baselines'].p_agree:.3f}",
            f"{r['ascore'].a_time:.3f}",
            f"{r['ascore'].a_act:.3f}",
            f"{r['ascore'].a_nec:.3f}",
            f"{r['ascore'].a_score:.3f}",
        ])
    widths = [max(len(c[i]) for c in cells) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*cells[0]), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines += [fmt.format(*c) for c in cells[1:]]
    return "\n".join(lines)


def format_per_event_breakdown(rows: list[dict], total_duration: float) -> str:
    """Verbose per-adversary, per-GT-event breakdown of TWSS components."""
    lines: list[str] = []
    for r in rows:
        twss_e: TWSS = r["tuned"]
        lines.append("")
        lines.append(f"── {r['name']} "
                     f"(α={twss_e.alpha}, β={twss_e.beta}, "
                     f"γ={twss_e.gamma}, w_wait={twss_e.wait_weight}, "
                     f"δ={twss_e.blip_tax_delta}, "
                     f"T={total_duration:.1f}s, T_wait={twss_e.total_wait_time:.1f}s, "
                     f"T_pred={twss_e.total_pred_time:.1f}s, "
                     f"T_fp={twss_e.fp_time:.1f}s, T_fp_eff={twss_e.fp_time_eff:.1f}s)")
        lines.append(f"   TWSS = task {twss_e.task_score:.3f} + "
                     f"wait {twss_e.wait_score:.3f}  "
                     f"(q_wait={twss_e.wait_quality:.3f}, mass={twss_e.wait_mass:.3f}) "
                     f"= {twss_e.twss:.3f}")
        lines.append(f"   per-event (skill, L_i, w_i, q_i, |Δs|, |Δe|, σ):")
        for ev in twss_e.per_event:
            ds = "—" if ev.dstart is None else f"{ev.dstart:6.2f}"
            de = "—" if ev.dend is None else f"{ev.dend:6.2f}"
            lines.append(f"     {ev.gt_action:<25s} "
                         f"L={ev.L_i:6.2f}  w={ev.weight:.3f}  "
                         f"q={ev.q_i:.3f}  Δs={ds}  Δe={de}  σ={ev.sigma:6.2f}")
    return "\n".join(lines)


def format_latex_table(rows: list[dict], tolerance: float,
                       weights: tuple[float, float, float]) -> str:
    """A booktabs table block ready to \\input{} into the paper."""
    lines = [
        r"% Generated by scripts/eval/ascore_adversaries.py",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\textbf{Adversary} & \textbf{F1} & \textbf{mIoU} & \textbf{mAcc} & "
        r"$\bm{P_{agree}}$ & "
        r"$\bm{A_{time}}$ & $\bm{A_{act}}$ & $\bm{A_{nec}}$ & $\bm{A_{score}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["name"].replace("_", r"\_")
        lines.append(
            f"{name} & "
            f"{r['baselines'].event_f1:.2f} & "
            f"{r['baselines'].matched_iou_mean:.2f} & "
            f"{r['baselines'].matched_act_acc:.2f} & "
            f"{r['baselines'].p_agree:.2f} & "
            f"{r['ascore'].a_time:.2f} & "
            f"{r['ascore'].a_act:.2f} & "
            f"{r['ascore'].a_nec:.2f} & "
            f"{r['ascore'].a_score:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"% match tolerance = {tolerance:.1f}s, "
        f"weights = ({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f})",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score synthetic adversaries against the hand-layup robot GT.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT,
                        help=f"robot_gt.json path (default: {DEFAULT_GT}).")
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS,
                        help=f"robot_skills.json path (default: {DEFAULT_SKILLS}).")
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE_SEC,
                        help=f"Start-time matching tolerance in seconds "
                             f"(default: {MATCH_TOLERANCE_SEC}).")
    parser.add_argument("--weights", type=str, default=None,
                        help="A-Score weights as 'w_t,w_a,w_n' (must sum to 1). "
                             f"Default: {DEFAULT_WEIGHTS[0]:.3f},"
                             f"{DEFAULT_WEIGHTS[1]:.3f},"
                             f"{DEFAULT_WEIGHTS[2]:.3f}")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path. Default: "
                             "figures/generated/fig_ascore_adversaries.pdf")
    parser.add_argument("--title", type=str,
                        default="Hand Layup — A-Score Adversary Validation",
                        help="Figure title.")
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
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated adversary names to skip "
                             "(matched case-insensitively against the names "
                             "in the ADVERSARIES table).")
    parser.add_argument("--compact-figure", action="store_true",
                        help="Render a compact monochrome timeline (no TP/FP "
                             "colouring, no legend, narrow rows). Used for "
                             "the paper figure.")
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

    if not args.gt.is_file():
        print(f"Error: GT file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)
    if not args.skills.is_file():
        print(f"Error: skills catalogue not found: {args.skills}", file=sys.stderr)
        sys.exit(1)

    gt_events, total_duration = load_robot_gt(args.gt)
    if total_duration <= 0:
        total_duration = max((e.end for e in gt_events), default=270.0)

    # Skills used in the GT track itself are not great wrong-action picks —
    # they could collide with another GT event's action and accidentally read
    # as a correct prediction at a different time. Exclude them.
    gt_skill_ids = {_strip_args(e.action) for e in gt_events}
    skill_pool = _load_intervention_skill_ids(args.skills,
                                              exclude_ids=gt_skill_ids)
    if not skill_pool:
        print("Warning: skill pool empty after excluding GT skills; "
              "wrong-action adversary will fall back to GT actions.",
              file=sys.stderr)

    print(f"GT: {args.gt.name} — {len(gt_events)} interventions, "
          f"duration {total_duration:.1f}s")
    print(f"Skill pool for wrong-action adversary "
          f"({len(skill_pool)} candidates): {skill_pool}")

    excluded = {n.strip().lower() for n in args.exclude.split(",") if n.strip()}
    if excluded:
        print(f"Excluding adversaries: {sorted(excluded)}")

    # Build adversary tracks.
    model_preds: dict[str, list[Event]] = {}
    rows: list[dict] = []
    for name, _desc, ctor in ADVERSARIES:
        if name.lower() in excluded:
            continue
        preds = ctor(gt_events,
                     total_duration=total_duration,
                     skill_pool=skill_pool)
        preds = _clip_to_window(preds, total_duration)
        model_preds[name] = preds
        ascore = compute_a_score(gt_events, preds,
                                 total_duration=total_duration,
                                 weights=weights, tolerance=args.tolerance)
        baselines = compute_baselines(gt_events, preds,
                                      tolerance=args.tolerance,
                                      total_duration=total_duration)
        vanilla = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta,
            gamma=DEFAULT_TWSS_GAMMA,
            wait_weight=DEFAULT_TWSS_WAIT_WEIGHT,
            blip_tax_delta=DEFAULT_TWSS_BLIP_TAX)
        tuned = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta_tuned,
            gamma=args.twss_gamma,
            wait_weight=args.twss_wait_weight,
            blip_tax_delta=args.twss_blip_tax)
        rows.append({
            "name": name, "ascore": ascore, "baselines": baselines,
            "vanilla": vanilla, "tuned": tuned,
        })

    # Print A-Score / baselines table.
    print()
    print(f"Adversary validation — A-Score weights "
          f"w_t={weights[0]:.3f}, w_a={weights[1]:.3f}, w_n={weights[2]:.3f}, "
          f"match tolerance={args.tolerance:.1f}s")
    print(format_table(rows))

    print()
    print(_format_twss_block(
        rows, "vanilla",
        f"Vanilla TWSS — α={args.twss_alpha}, β={args.twss_beta}, "
        f"γ={DEFAULT_TWSS_GAMMA}, w_wait={DEFAULT_TWSS_WAIT_WEIGHT}, "
        f"δ={DEFAULT_TWSS_BLIP_TAX} (exp wait penalty)"))

    print()
    print(_format_twss_block(
        rows, "tuned",
        f"Tuned TWSS — α={args.twss_alpha}, β={args.twss_beta_tuned}, "
        f"γ={args.twss_gamma}, w_wait={args.twss_wait_weight}, "
        f"δ={args.twss_blip_tax} (exp wait penalty)"))

    print()
    print("Per-adversary breakdown (tuned TWSS components and per-event quality):")
    print(format_per_event_breakdown(rows, total_duration))

    # Outputs.
    out = args.output
    if out is None:
        out_dir = AURA_ROOT / "figures" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_ascore_adversaries.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stacked timeline figure: GT vs each adversary track.
    if args.compact_figure:
        fig = plot_compact_monochrome_timeline(
            gt_events, model_preds,
            total_duration=total_duration,
            title=None,
        )
    else:
        fig = plot_multi_model_timeline(
            gt_events, model_preds,
            title=args.title,
            total_duration=total_duration,
            show_human_task=False,
            show_legend=True,
        )
    fig.savefig(str(out))
    png_out = (str(out)[:-4] + ".png") if str(out).lower().endswith(".pdf") \
        else str(out) + ".png"
    fig.savefig(png_out)
    plt.close(fig)
    print(f"\nSaved adversary timeline:\n  {out}\n  {png_out}")

    # JSON dump.
    scores_out = out.with_suffix(".scores.json")
    payload = {
        "gt_track": str(args.gt),
        "skill_catalogue": str(args.skills),
        "skill_pool_for_wrong_action": skill_pool,
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
        "adversaries": {
            r["name"]: {
                "ascore": asdict(r["ascore"]),
                "baselines": asdict(r["baselines"]),
                "twss_vanilla": asdict(r["vanilla"]),
                "twss_tuned": asdict(r["tuned"]),
            } for r in rows
        },
    }
    scores_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved scores JSON:\n  {scores_out}")

    # LaTeX table.
    tex_out = out.with_suffix(".tex")
    tex_out.write_text(format_latex_table(rows, args.tolerance, weights))
    print(f"Saved LaTeX table:\n  {tex_out}")


if __name__ == "__main__":
    main()

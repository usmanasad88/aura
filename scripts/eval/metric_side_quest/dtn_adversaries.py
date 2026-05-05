#!/usr/bin/env python3
"""Construct synthetic adversaries against a *disjunctive* GT and score them.

Sister script to ``scripts/eval/ascore_adversaries.py``, but the GT is
loaded as a ``RobotGT`` with disjunctive options/phases per intervention,
and the score is the DTN-aware A-Score from ``dtn_score`` (axes:
``A_disj``, ``A_act``, ``A_cons``, ``A_nec``).

Default GT is the toy DTN with overlapping options used to validate the
metric. Pass ``--gt`` to point at any v1.0/v2.0 robot GT — v1.0 (single
``t_start``/``t_end`` per task) is auto-promoted to one option / one
phase, in which case the disjunctive axes collapse to the legacy
behaviour.

Outputs (alongside the GT, under ``./out/``):
    fig_dtn_adversaries.{pdf,png}   — stacked timeline (GT options + each adversary)
    fig_dtn_adversaries.scores.json — per-adversary score breakdown
    fig_dtn_adversaries.tex         — LaTeX-ready booktabs table

Usage::

    python scripts/eval/metric_side_quest/dtn_adversaries.py
    python scripts/eval/metric_side_quest/dtn_adversaries.py \
        --gt scripts/eval/metric_side_quest/layup_gesture_demo_stationary_with_overlay.robot_gt.v2.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local-package imports — module lives in metric_side_quest/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from disjunctive_gt import (  # noqa: E402
    Intervention, Option, Phase, RobotGT, load_disjunctive_robot_gt,
)
from dtn_score import (  # noqa: E402
    DEFAULT_WEIGHTS, DTNAScore, Pred,
    compute_dtn_a_score, find_consistent_selection, _strip_args,
)


HERE = Path(__file__).resolve().parent
DEFAULT_GT = HERE / "toy_dtn_three_tasks.robot_gt.json"
DEFAULT_OUT = HERE / "out" / "fig_dtn_adversaries.pdf"

# Adversary parameters.
NEAR_MISS_OFFSET_SEC = 1.5
FAR_SHIFT_OFFSET_SEC = 8.0
PREMATURE_OFFSET_SEC = -1.5
OVER_EXTEND_PAD_SEC = 4.0
SPAMMER_PERIOD_SEC = 3.0
SPAMMER_DURATION_SEC = 2.0


# ── Strategy helpers ─────────────────────────────────────────────────────────


def _strategy_to_preds(gt: RobotGT,
                       selection: dict[str, int]) -> list[Pred]:
    """Materialise an option-selection as one ``Pred`` per phase."""
    out: list[Pred] = []
    for iv in gt.interventions:
        if iv.id not in selection:
            continue
        opt = iv.options[selection[iv.id]]
        for ph in opt.phases:
            out.append(Pred(skill=iv.skill, start=ph.t_start, end=ph.t_end))
    return out


def _shift(preds: list[Pred], dt: float) -> list[Pred]:
    return [Pred(skill=p.skill, start=p.start + dt, end=p.end + dt)
            for p in preds]


def _clip(preds: list[Pred], total: float) -> list[Pred]:
    out: list[Pred] = []
    for p in preds:
        s = max(0.0, p.start)
        e = min(total, p.end)
        if e > s:
            out.append(Pred(skill=p.skill, start=s, end=e))
    return out


def _most_common_skill(gt: RobotGT) -> str:
    if not gt.interventions:
        return "noop"
    return Counter(iv.skill for iv in gt.interventions).most_common(1)[0][0]


def _conflicting_selection(gt: RobotGT) -> dict[str, int]:
    """Pick options that maximise cross-task overlap (worst case for A_cons).

    For each intervention, pick the option whose phases overlap the most
    *previously selected* phases. Tie-break: option 0.
    """
    selection: dict[str, int] = {}
    chosen_phases: list[Phase] = []
    for iv in gt.interventions:
        best_idx = 0
        best_overlap = -1.0
        for k, opt in enumerate(iv.options):
            ov = 0.0
            for p1 in opt.phases:
                for p2 in chosen_phases:
                    ov += max(0.0, min(p1.t_end, p2.t_end)
                                  - max(p1.t_start, p2.t_start))
            if ov > best_overlap:
                best_overlap = ov
                best_idx = k
        selection[iv.id] = best_idx
        chosen_phases.extend(iv.options[best_idx].phases)
    return selection


# ── Adversary constructors ──────────────────────────────────────────────────


def adv_oracle_consistent(gt: RobotGT, **_) -> list[Pred]:
    """Solve the disjunctive CSP for a non-conflicting option assignment."""
    sel = find_consistent_selection(gt)
    if sel is None:
        # Over-constrained GT — fall back to first option each.
        sel = {iv.id: 0 for iv in gt.interventions}
    return _strategy_to_preds(gt, sel)


def adv_oracle_first(gt: RobotGT, **_) -> list[Pred]:
    """Naive greedy: always pick option 0 (may conflict)."""
    sel = {iv.id: 0 for iv in gt.interventions}
    return _strategy_to_preds(gt, sel)


def adv_conflict_prone(gt: RobotGT, **_) -> list[Pred]:
    """Deliberately pick options to maximise cross-task overlap."""
    sel = _conflicting_selection(gt)
    return _strategy_to_preds(gt, sel)


def adv_silent(gt: RobotGT, **_) -> list[Pred]:
    return []


def adv_spammer(gt: RobotGT, total_duration: float, **_) -> list[Pred]:
    skill = _most_common_skill(gt)
    out: list[Pred] = []
    t = 0.0
    while t < total_duration:
        out.append(Pred(skill=skill, start=t,
                        end=min(t + SPAMMER_DURATION_SEC, total_duration)))
        t += SPAMMER_PERIOD_SEC
    return out


def adv_phase_collapsed(gt: RobotGT, **_) -> list[Pred]:
    """Consistent oracle, but only the *first* phase of each chosen option."""
    sel = find_consistent_selection(gt) or {iv.id: 0 for iv in gt.interventions}
    out: list[Pred] = []
    for iv in gt.interventions:
        opt = iv.options[sel[iv.id]]
        first = opt.phases[0]
        out.append(Pred(skill=iv.skill, start=first.t_start, end=first.t_end))
    return out


def adv_half_coverage(gt: RobotGT, **_) -> list[Pred]:
    """Consistent oracle for every other intervention only."""
    sel = find_consistent_selection(gt) or {iv.id: 0 for iv in gt.interventions}
    out: list[Pred] = []
    for i, iv in enumerate(gt.interventions):
        if i % 2 != 0:
            continue
        opt = iv.options[sel[iv.id]]
        for ph in opt.phases:
            out.append(Pred(skill=iv.skill, start=ph.t_start, end=ph.t_end))
    return out


def adv_wrong_skill(gt: RobotGT, **_) -> list[Pred]:
    """Right times (consistent oracle), but each prediction's skill swapped
    for the next intervention's skill (cyclic shift)."""
    sel = find_consistent_selection(gt) or {iv.id: 0 for iv in gt.interventions}
    iv_list = gt.interventions
    if len(iv_list) < 2:
        return _strategy_to_preds(gt, sel)
    out: list[Pred] = []
    for i, iv in enumerate(iv_list):
        wrong_skill = iv_list[(i + 1) % len(iv_list)].skill
        opt = iv.options[sel[iv.id]]
        for ph in opt.phases:
            out.append(Pred(skill=wrong_skill,
                            start=ph.t_start, end=ph.t_end))
    return out


def adv_near_miss(gt: RobotGT, **_) -> list[Pred]:
    return _shift(adv_oracle_consistent(gt), NEAR_MISS_OFFSET_SEC)


def adv_far_shifted(gt: RobotGT, **_) -> list[Pred]:
    return _shift(adv_oracle_consistent(gt), FAR_SHIFT_OFFSET_SEC)


def adv_premature(gt: RobotGT, **_) -> list[Pred]:
    return _shift(adv_oracle_consistent(gt), PREMATURE_OFFSET_SEC)


def adv_over_extender(gt: RobotGT, **_) -> list[Pred]:
    base = adv_oracle_consistent(gt)
    return [Pred(skill=p.skill, start=p.start, end=p.end + OVER_EXTEND_PAD_SEC)
            for p in base]


def adv_one_shot_blanket(gt: RobotGT, total_duration: float, **_) -> list[Pred]:
    if not gt.interventions:
        return []
    return [Pred(skill=_most_common_skill(gt),
                 start=0.0, end=total_duration)]


ADVERSARIES: list[tuple[str, str, Callable[..., list[Pred]]]] = [
    ("Oracle (consistent)",   "Disjunctive-CSP solution; phases as preds",      adv_oracle_consistent),
    ("Oracle (first option)", "Always pick option 0 — may conflict",            adv_oracle_first),
    ("Conflict-prone",        "Pick options that maximise cross-task overlap",  adv_conflict_prone),
    ("Phase-collapsed",       "Consistent oracle, only the first phase fired",  adv_phase_collapsed),
    ("Half-coverage",         "Consistent oracle for every other intervention", adv_half_coverage),
    ("Wrong-skill",           "Right times, skill labels rotated",              adv_wrong_skill),
    ("Near-miss timing",      f"Consistent oracle shifted +{NEAR_MISS_OFFSET_SEC}s", adv_near_miss),
    ("Far-shifted",           f"Consistent oracle shifted +{FAR_SHIFT_OFFSET_SEC}s", adv_far_shifted),
    ("Premature firer",       f"Consistent oracle shifted {PREMATURE_OFFSET_SEC:+.1f}s", adv_premature),
    ("Over-extender",         f"Correct starts, ends padded +{OVER_EXTEND_PAD_SEC}s", adv_over_extender),
    ("One-shot blanket",      "Single interval covering whole task",            adv_one_shot_blanket),
    ("Spammer",               "Rapid-fire most-common skill",                   adv_spammer),
    ("Silent",                "Never intervenes",                               adv_silent),
]


# ── Plotting ─────────────────────────────────────────────────────────────────


_OPTION_COLORS = ["#2980b9", "#16a085", "#c0392b", "#8e44ad", "#d35400"]


def _skill_palette(gt: RobotGT) -> dict[str, str]:
    skills = sorted({iv.skill for iv in gt.interventions})
    cmap = plt.get_cmap("tab10")
    return {sk: cmap(i % 10) for i, sk in enumerate(skills)}


def plot_dtn_timeline(gt: RobotGT,
                      total_duration: float,
                      adversaries: list[tuple[str, list[Pred], DTNAScore]],
                      *, title: str = "DTN A-Score Adversary Validation",
                      ) -> plt.Figure:
    palette = _skill_palette(gt)
    n_rows = len(gt.interventions) + len(adversaries)
    fig, ax = plt.subplots(figsize=(14, 0.55 * n_rows + 1.5))

    row_h = 0.7
    y = 0.0
    yticks: list[float] = []
    ylabels: list[str] = []

    # GT rows: one row per intervention, options stacked with hatch.
    for iv in gt.interventions:
        for k, opt in enumerate(iv.options):
            colour = palette[iv.skill]
            edge = _OPTION_COLORS[k % len(_OPTION_COLORS)]
            for ph in opt.phases:
                ax.add_patch(mpatches.Rectangle(
                    (ph.t_start, y - row_h / 2 + 0.06 * k),
                    ph.duration, row_h * 0.45,
                    facecolor=colour, edgecolor=edge,
                    linewidth=1.4, alpha=0.55,
                    label=None))
        yticks.append(y)
        n_opt = len(iv.options)
        ylabels.append(f"GT: {iv.id} ({iv.skill}) — {n_opt} option(s)")
        y -= 1.0

    # Separator between GT and adversaries.
    ax.axhline(y + 0.5, color="#bdbdbd", linewidth=0.8)

    for name, preds, score in adversaries:
        # Bar per prediction, coloured by the GT skill it most resembles.
        for p in preds:
            colour = palette.get(_strip_args(p.skill), "#7f7f7f")
            ax.add_patch(mpatches.Rectangle(
                (p.start, y - row_h / 2),
                max(0.05, p.end - p.start), row_h,
                facecolor=colour, edgecolor="black",
                linewidth=0.5, alpha=0.9))
        yticks.append(y)
        sel = ", ".join(f"{iid}=opt{score.selected_options[iid]}"
                        for iid in score.selected_options)
        ylabels.append(f"{name}  | A={score.a_score:.2f}  | {sel}")
        y -= 1.0

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, total_duration)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    # Skill legend.
    handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=sk)
               for sk, c in palette.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              title="Skill", title_fontsize=8)
    fig.tight_layout()
    return fig


# ── Reporting ────────────────────────────────────────────────────────────────


def format_table(rows: list[dict]) -> str:
    headers = ["Adversary", "A_disj", "A_act", "A_cons", "A_nec", "A_score"]
    cells: list[list[str]] = [headers]
    for r in rows:
        s: DTNAScore = r["score"]
        cells.append([
            r["name"],
            f"{s.a_disj:.3f}",
            f"{s.a_act:.3f}",
            f"{s.a_cons:.3f}",
            f"{s.a_nec:.3f}",
            f"{s.a_score:.3f}",
        ])
    widths = [max(len(c[i]) for c in cells) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*cells[0]),
             "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines += [fmt.format(*c) for c in cells[1:]]
    return "\n".join(lines)


def format_latex_table(rows: list[dict],
                       weights: tuple[float, float, float, float]) -> str:
    lines = [
        r"% Generated by scripts/eval/metric_side_quest/dtn_adversaries.py",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Adversary} & "
        r"$\bm{A_{disj}}$ & $\bm{A_{act}}$ & $\bm{A_{cons}}$ & "
        r"$\bm{A_{nec}}$ & $\bm{A_{score}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        s: DTNAScore = r["score"]
        name = r["name"].replace("_", r"\_")
        lines.append(
            f"{name} & {s.a_disj:.2f} & {s.a_act:.2f} & "
            f"{s.a_cons:.2f} & {s.a_nec:.2f} & {s.a_score:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"% weights = (w_disj={weights[0]:.2f}, w_act={weights[1]:.2f}, "
        f"w_cons={weights[2]:.2f}, w_nec={weights[3]:.2f})",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def _parse_weights(s: str | None) -> tuple[float, float, float, float]:
    if s is None:
        return DEFAULT_WEIGHTS
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4 or abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(
            f"--weights needs 4 floats summing to 1, got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score synthetic adversaries against a disjunctive GT.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT,
                        help=f"Disjunctive robot_gt path (default: {DEFAULT_GT.name}).")
    parser.add_argument("--weights", type=str, default=None,
                        help="A-Score weights as 'w_disj,w_act,w_cons,w_nec' "
                             "(must sum to 1).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"Output PDF path (default: {DEFAULT_OUT}).")
    parser.add_argument("--title", type=str,
                        default="DTN A-Score Adversary Validation")
    args = parser.parse_args()

    weights = _parse_weights(args.weights)

    if not args.gt.is_file():
        print(f"Error: GT file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)

    gt = load_disjunctive_robot_gt(args.gt)
    total_duration = gt.duration_sec
    print(f"GT: {args.gt.name} (schema {gt.schema_version}) — "
          f"{len(gt.interventions)} interventions, "
          f"duration {total_duration:.1f}s")
    for iv in gt.interventions:
        n_opts = len(iv.options)
        n_phs = [len(o.phases) for o in iv.options]
        print(f"  {iv.id} {iv.skill}: {n_opts} option(s), phases per option = {n_phs}")

    # Build adversary tracks and score them.
    rows: list[dict] = []
    timeline_data: list[tuple[str, list[Pred], DTNAScore]] = []
    for name, _desc, ctor in ADVERSARIES:
        preds = ctor(gt, total_duration=total_duration)
        preds = _clip(preds, total_duration)
        score = compute_dtn_a_score(gt, preds, weights=weights)
        rows.append({"name": name, "preds": preds, "score": score})
        timeline_data.append((name, preds, score))

    print()
    print(f"DTN A-Score adversary validation — weights "
          f"w_disj={weights[0]:.2f} w_act={weights[1]:.2f} "
          f"w_cons={weights[2]:.2f} w_nec={weights[3]:.2f}")
    print(format_table(rows))

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_dtn_timeline(gt, total_duration, timeline_data,
                            title=args.title)
    fig.savefig(str(out))
    png_out = (str(out)[:-4] + ".png") if str(out).lower().endswith(".pdf") \
        else str(out) + ".png"
    fig.savefig(png_out)
    plt.close(fig)
    print(f"\nSaved adversary timeline:\n  {out}\n  {png_out}")

    scores_out = out.with_suffix(".scores.json")
    payload = {
        "gt_file": str(args.gt),
        "schema_version": gt.schema_version,
        "weights": {"w_disj": weights[0], "w_act": weights[1],
                    "w_cons": weights[2], "w_nec": weights[3]},
        "duration_sec": total_duration,
        "interventions": [
            {"id": iv.id, "skill": iv.skill, "n_options": len(iv.options),
             "phases_per_option": [len(o.phases) for o in iv.options]}
            for iv in gt.interventions
        ],
        "adversaries": {
            r["name"]: {
                "score": asdict(r["score"]),
                "preds": [
                    {"skill": p.skill, "start": p.start, "end": p.end}
                    for p in r["preds"]
                ],
            } for r in rows
        },
    }
    scores_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved scores JSON:\n  {scores_out}")

    tex_out = out.with_suffix(".tex")
    tex_out.write_text(format_latex_table(rows, weights))
    print(f"Saved LaTeX table:\n  {tex_out}")


if __name__ == "__main__":
    main()

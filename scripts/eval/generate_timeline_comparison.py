#!/usr/bin/env python3
"""Generate a single intervention-timeline comparison figure across ablations.

Each ablation is a normal run directory (``logs/<run>/``) with the usual
``decision_engine/`` + ``intent_monitor/`` layout. This script stacks one
prediction track per ablation against a single GT robot track, so a reader
can see at a glance which ablation matches / misses / over-fires.

Default: hand_layup ablations
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
import sys
from dataclasses import asdict, dataclass
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


def _strip_args(action: str) -> str:
    """Reduce ``skill(arg=val, ...)`` to bare skill for action comparison."""
    return action.split("(", 1)[0].strip()


def _interval_iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def compute_a_score(gt_events: list[Event],
                    pred_events: list[Event],
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
    )


def format_score_table(scores: dict[str, AScore]) -> str:
    """Pretty-print a fixed-width A-Score table for the terminal."""
    headers = ["Ablation", "GT", "TP", "FP", "FN",
               "A_time", "A_act", "A_nec", "A_score"]
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
    parser.add_argument("--scores-output", type=Path, default=None,
                        help="JSON path for A-Score results. Default: alongside "
                             "the figure as <output>.scores.json")
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

    # ── A-Score table ────────────────────────────────────────────────────
    scores: dict[str, AScore] = {}
    for label, preds in model_preds.items():
        scores[label] = compute_a_score(gt_events, preds,
                                        weights=weights,
                                        tolerance=args.tolerance)

    print("\nA-Score (Appropriateness Score) — weights "
          f"w_t={weights[0]:.3f}, w_a={weights[1]:.3f}, w_n={weights[2]:.3f}, "
          f"match tolerance={args.tolerance:.1f}s")
    print(format_score_table(scores))

    scores_out = args.scores_output or out.with_suffix(".scores.json")
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_gt": str(gt_path),
        "weights": {"w_t": weights[0], "w_a": weights[1], "w_n": weights[2]},
        "match_tolerance_sec": args.tolerance,
        "ablations": {label: asdict(s) for label, s in scores.items()},
    }
    scores_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved A-Score JSON to:\n  {scores_out}")


if __name__ == "__main__":
    main()

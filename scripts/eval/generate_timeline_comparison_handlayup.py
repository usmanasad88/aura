#!/usr/bin/env python3
"""Action-coloured ablation timeline comparison for the hand_layup task.

This is the hand_layup counterpart to
:mod:`generate_timeline_comparison_kettle`. It produces the **same
action-coloured figure scheme** — one track per source (GT + each ablation),
bars coloured by robot action with a colour→action legend, instead of the
TP/FP/FN colouring used by :mod:`generate_timeline_comparison` — plus the
identical A-Score / TWSS tables and per-decision CSV. What it keeps from the
base hand_layup script is the *defaults* and the *data loading*: the four
pre-named ablation run dirs, the original prediction/GT loaders (so wait
handling matches), and the hand_layup-tuned TWSS configuration (symmetric
timing decay, ``end_weight=1``).

Why this is simpler than the kettle variant
--------------------------------------------
hand_layup robot skills carry no meaningful arguments (every intervention's
``args`` is ``{}`` — see ``tasks/hand_layup/ground_truth/*.robot_gt.json``), so
no per-object action splitting is needed and predictions / GT are loaded with
the *original* loaders (``generate_timeline.load_predictions`` /
``load_robot_gt``), not the kettle parameter-aware ones. That preserves a
behaviour the kettle loader deliberately drops: the original loader skips
``wait`` cycles before merging consecutive identical actions, so a wait
sandwiched between two of the same intervention is ignored and the two fuse
into one span (the kettle loader instead keeps each wait as a span
*breakpoint*). Only the action-coloured plot, the per-decision CSV, and the
legend labels are borrowed from the kettle module.

Usage::

    # Default: the four hand_layup ablations (same set as
    # generate_timeline_comparison.py), rendered with the action-coloured scheme:
    python scripts/eval/generate_timeline_comparison_handlayup.py

    # Pick your own run dirs / labels:
    python scripts/eval/generate_timeline_comparison_handlayup.py \
        --run logs/hand_layup_skip_intent_perception="GT intent (skip perception)" \
        --run logs/hand_layup_gt_intent_perception="GT prev-state (perception)" \
        --output figures/generated/fig_ablation_timeline_handlayup.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Borrow the kettle module's action-coloured plotting and per-decision CSV, plus
# its legend hook: we monkeypatch ``gtck.legend_label`` with hand_layup text (the
# same module-global idiom the kettle script uses for ``gtc._strip_args``), and
# ``action_key`` already returns the bare skill for every (non-parametric)
# hand_layup skill. Data, though, is loaded with the ORIGINAL loaders
# (``generate_timeline.load_predictions`` via the base script's
# ``_collect_predictions``, and ``load_robot_gt``) so wait handling matches the
# original hand_layup script: a wait between two identical interventions is
# dropped and the spans fuse, rather than being kept as a span breakpoint.
import generate_timeline_comparison_kettle as gtck
from generate_timeline_comparison_kettle import (
    _resolve_gt_for_runs,
    plot_action_colored_timeline,
    write_decisions_csv,
)
from generate_timeline_comparison import (
    DEFAULT_ABLATIONS,
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
    _collect_predictions,
    _parse_run_arg,
    compute_a_score,
    compute_twss,
    format_score_table,
    format_twss_table,
)
from generate_timeline import AURA_ROOT, Event, load_robot_gt

# hand_layup grades the predicted span's *end* (a triggered skill that ends late
# / early is penalised), unlike kettle where end_weight=0. Keep the base
# hand_layup-tuned config: symmetric Gaussian timing decay.
HANDLAYUP_TWSS_END_WEIGHT = 1.0


def handlayup_legend_label(key: str) -> str:
    """Human-readable legend text for a hand_layup ``action_key`` (bare skill).

    Replaces the kettle module's pick/return-oriented ``legend_label`` for the
    duration of this script (monkeypatched in :func:`main`). Unknown keys fall
    back to a de-underscored skill id, so prediction-only motion primitives
    (``move_to_named_position`` …) still render sensibly.
    """
    return {
        "move_resin_to_workplace": "Move resin → workplace",
        "return_resin_to_storage": "Return resin → storage",
        "move_hardener_to_workplace": "Move hardener → workplace",
        "return_hardener_to_storage": "Return hardener → storage",
        "consolidate_with_roller_force": "Consolidate (roller, force)",
        "clean_table": "Clean table",
        "wait": "Wait",
    }.get(key, key.replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action-coloured ablation timeline comparison (hand_layup).")
    parser.add_argument(
        "--run", action="append", default=None,
        help="Ablation run dir, optionally `path=label`. Repeatable. "
             "If omitted, uses the four default hand_layup ablations.")
    parser.add_argument("--task", default="hand_layup",
                        help="Task name for GT lookup (default: hand_layup).")
    parser.add_argument("--gt", type=Path, default=None,
                        help="Explicit robot_gt.json path (skips auto-resolve).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (PNG sibling also written). Default: "
                             "figures/generated/fig_ablation_timeline_comparison_handlayup.pdf")
    parser.add_argument("--title", type=str,
                        default="Hand Layup — Ablation Intervention Timeline",
                        help="Figure title.")
    parser.add_argument("--weights", type=str, default=None,
                        help="A-Score weights 'w_t,w_a,w_n' (sum to 1).")
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE_SEC,
                        help=f"Start-time match tolerance (default: {MATCH_TOLERANCE_SEC}).")
    parser.add_argument("--twss-alpha", type=float, default=DEFAULT_TWSS_ALPHA)
    parser.add_argument("--twss-beta", type=float, default=DEFAULT_TWSS_BETA)
    parser.add_argument("--twss-beta-tuned", type=float, default=TUNED_TWSS_BETA)
    parser.add_argument("--twss-gamma", type=float, default=TUNED_TWSS_GAMMA)
    parser.add_argument("--twss-wait-weight", type=float, default=TUNED_TWSS_WAIT_WEIGHT)
    parser.add_argument("--twss-blip-tax", type=float, default=TUNED_TWSS_BLIP_TAX)
    parser.add_argument("--twss-end-weight", type=float,
                        default=HANDLAYUP_TWSS_END_WEIGHT,
                        help="Weight on the end-time term of the per-event timing "
                             f"decay. 1 = symmetric (default for hand_layup: "
                             f"{HANDLAYUP_TWSS_END_WEIGHT}); 0 = no early-stop "
                             "penalty (kettle default).")
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--decisions-csv", type=Path, default=None,
                        help="CSV of per-decision rows (decision + intent + "
                             "perception). Default: <output_stem>.decisions.csv")
    args = parser.parse_args()

    # hand_layup legend text for the action-coloured plot. The colour/identity
    # key (action_key) is reused from the kettle module unchanged.
    gtck.legend_label = handlayup_legend_label

    if args.weights:
        try:
            ws = tuple(float(x) for x in args.weights.split(","))
        except ValueError:
            print(f"Error: --weights must be three floats, got {args.weights!r}",
                  file=sys.stderr)
            sys.exit(1)
        if len(ws) != 3 or abs(sum(ws) - 1.0) > 1e-6:
            print(f"Error: --weights must have 3 values summing to 1, got {ws}",
                  file=sys.stderr)
            sys.exit(1)
        weights = ws
    else:
        weights = DEFAULT_WEIGHTS

    # Build the (run_dir, label) list — default to the four hand_layup ablations.
    if args.run:
        runs = [_parse_run_arg(s) for s in args.run]
    else:
        runs = [(AURA_ROOT / "logs" / name, label) for name, label in DEFAULT_ABLATIONS]

    missing = [str(p) for p, _ in runs if not p.is_dir()]
    if missing:
        print("Error: run directories not found:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        sys.exit(1)

    gt_path = _resolve_gt_for_runs(runs, args.task, args.gt)
    print(f"Using robot GT: {gt_path}")
    gt_events, total_duration = load_robot_gt(gt_path)
    if total_duration <= 0:
        total_duration = max((e.end for e in gt_events), default=270.0)

    model_preds: dict[str, list[Event]] = {}
    for run_dir, label in runs:
        print(f"  loading {run_dir.name} ({label})...")
        preds = _collect_predictions(run_dir)
        model_preds[label] = preds
        if preds:
            total_duration = max(total_duration, max(e.end for e in preds))

    if not any(model_preds.values()):
        print("Error: no predictions loaded from any run.", file=sys.stderr)
        sys.exit(1)

    fig = plot_action_colored_timeline(
        gt_events, model_preds,
        total_duration=total_duration,
        title=args.title,
    )

    out = args.output
    if out is None:
        out_dir = AURA_ROOT / "figures" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_ablation_timeline_comparison_handlayup.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    png_out = str(out)
    png_out = png_out[:-4] + ".png" if png_out.lower().endswith(".pdf") else png_out + ".png"
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Saved comparison timeline to:\n  {out}\n  {png_out}")

    scores: dict[str, AScore] = {}
    vanilla_exp: dict[str, TWSS] = {}
    tuned_exp: dict[str, TWSS] = {}
    for label, preds in model_preds.items():
        scores[label] = compute_a_score(gt_events, preds,
                                        total_duration=total_duration,
                                        weights=weights, tolerance=args.tolerance)
        vanilla_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta,
            gamma=DEFAULT_TWSS_GAMMA, wait_weight=DEFAULT_TWSS_WAIT_WEIGHT,
            blip_tax_delta=DEFAULT_TWSS_BLIP_TAX)
        tuned_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta_tuned,
            gamma=args.twss_gamma, wait_weight=args.twss_wait_weight,
            blip_tax_delta=args.twss_blip_tax, end_weight=args.twss_end_weight)

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
          f"δ={args.twss_blip_tax}, end_weight={args.twss_end_weight} "
          f"(exp wait penalty)")
    print(format_twss_table(tuned_exp))

    scores_out = args.scores_output or out.with_suffix(".scores.json")
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_gt": str(gt_path),
        "weights": {"w_t": weights[0], "w_a": weights[1], "w_n": weights[2]},
        "match_tolerance_sec": args.tolerance,
        "twss_alpha": args.twss_alpha,
        "twss_vanilla": {
            "beta": args.twss_beta, "gamma": DEFAULT_TWSS_GAMMA,
            "wait_weight": DEFAULT_TWSS_WAIT_WEIGHT,
            "blip_tax_delta": DEFAULT_TWSS_BLIP_TAX,
        },
        "twss_tuned": {
            "beta": args.twss_beta_tuned, "gamma": args.twss_gamma,
            "wait_weight": args.twss_wait_weight,
            "blip_tax_delta": args.twss_blip_tax,
            "end_weight": args.twss_end_weight,
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

    # Per-decision CSV: decision reasoning ⨉ intent reasoning ⨉ perception.
    csv_out = args.decisions_csv or (out.parent / f"{out.stem}.decisions.csv")
    n_rows = write_decisions_csv(runs, csv_out, set())
    print(f"Saved {n_rows} decision rows to:\n  {csv_out}")


if __name__ == "__main__":
    main()

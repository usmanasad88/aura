#!/usr/bin/env python3
"""Aggregate AURA experiment results across experiments and repetitions.

Reads ``run_results.json`` files from the experiment directory structure
and produces summary tables (JSON, CSV, LaTeX).

Usage::

    python scripts/eval/aggregate_results.py \
        --experiments-dir logs/experiments/ \
        --output logs/experiments/aggregate_results.json

    python scripts/eval/aggregate_results.py \
        --experiments-dir logs/experiments/ \
        --latex results/tables/model_comparison.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_experiment_results(experiments_dir: Path) -> dict[str, list[dict]]:
    """Load all run_results.json files grouped by experiment ID.

    Returns:
        Dict mapping experiment_id to list of per-rep result dicts.
    """
    results: dict[str, list[dict]] = {}

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        exp_id = exp_dir.name
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        rep_results = []

        for rep_dir in sorted(exp_dir.glob("rep_*")):
            result_path = rep_dir / "run_results.json"
            if result_path.exists():
                try:
                    data = json.loads(result_path.read_text())
                    data["_rep"] = rep_dir.name
                    data["_experiment_id"] = exp_id
                    rep_results.append(data)
                except json.JSONDecodeError:
                    continue

        if rep_results:
            results[exp_id] = rep_results

    return results


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and std, returning (0, 0) for empty lists."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return round(float(arr.mean()), 3), round(float(arr.std()), 3)


def aggregate(results: dict[str, list[dict]]) -> dict[str, Any]:
    """Aggregate per-rep results into per-experiment summaries."""
    summaries = {}

    for exp_id, reps in results.items():
        if not reps:
            continue

        # Extract metrics from each rep
        a_scores = [r["a_score"]["mean"] for r in reps if r.get("a_score", {}).get("mean") is not None]
        a_times = [r["a_score"]["a_time"] for r in reps if r.get("a_score", {}).get("a_time") is not None]
        a_mods = [r["a_score"]["a_mod"] for r in reps if r.get("a_score", {}).get("a_mod") is not None]
        a_necs = [r["a_score"]["a_nec"] for r in reps if r.get("a_score", {}).get("a_nec") is not None]
        precisions = [r["precision"] for r in reps if "precision" in r]
        recalls = [r["recall"] for r in reps if "recall" in r]
        f1s = [r["f1"] for r in reps if "f1" in r]

        intent_current = [r["intent_accuracy"]["current_action"] for r in reps
                          if r.get("intent_accuracy", {}).get("current_action") is not None]
        intent_next = [r["intent_accuracy"]["next_action"] for r in reps
                       if r.get("intent_accuracy", {}).get("next_action") is not None]
        det_rates = [r["intent_accuracy"]["detection_rate"] for r in reps
                     if r.get("intent_accuracy", {}).get("detection_rate") is not None]

        n_acts = [r.get("n_act", 0) for r in reps]
        n_waits = [r.get("n_wait", 0) for r in reps]

        # Latency
        intent_latencies = [r["intent_latency"]["mean_sec"] for r in reps
                            if r.get("intent_latency", {}).get("mean_sec") is not None]
        decision_latencies = [r["decision_latency"]["mean_sec"] for r in reps
                              if r.get("decision_latency", {}).get("mean_sec") is not None]

        model = reps[0].get("model", "unknown")
        task = reps[0].get("task", "unknown")

        summaries[exp_id] = {
            "model": model,
            "task": task,
            "n_reps": len(reps),
            "a_score": {"mean": _safe_mean_std(a_scores)[0], "std": _safe_mean_std(a_scores)[1]},
            "a_time": {"mean": _safe_mean_std(a_times)[0], "std": _safe_mean_std(a_times)[1]},
            "a_mod": {"mean": _safe_mean_std(a_mods)[0], "std": _safe_mean_std(a_mods)[1]},
            "a_nec": {"mean": _safe_mean_std(a_necs)[0], "std": _safe_mean_std(a_necs)[1]},
            "precision": {"mean": _safe_mean_std(precisions)[0], "std": _safe_mean_std(precisions)[1]},
            "recall": {"mean": _safe_mean_std(recalls)[0], "std": _safe_mean_std(recalls)[1]},
            "f1": {"mean": _safe_mean_std(f1s)[0], "std": _safe_mean_std(f1s)[1]},
            "intent_current_acc": {"mean": _safe_mean_std(intent_current)[0], "std": _safe_mean_std(intent_current)[1]},
            "intent_next_acc": {"mean": _safe_mean_std(intent_next)[0], "std": _safe_mean_std(intent_next)[1]},
            "detection_rate": {"mean": _safe_mean_std(det_rates)[0], "std": _safe_mean_std(det_rates)[1]},
            "n_act_mean": _safe_mean_std(n_acts)[0],
            "n_wait_mean": _safe_mean_std(n_waits)[0],
            "intent_latency_sec": {"mean": _safe_mean_std(intent_latencies)[0], "std": _safe_mean_std(intent_latencies)[1]},
            "decision_latency_sec": {"mean": _safe_mean_std(decision_latencies)[0], "std": _safe_mean_std(decision_latencies)[1]},
        }

    return summaries


def to_csv(summaries: dict[str, Any]) -> str:
    """Convert summaries to CSV string."""
    headers = [
        "experiment_id", "model", "task", "n_reps",
        "a_score_mean", "a_score_std",
        "a_time_mean", "a_mod_mean", "a_nec_mean",
        "precision_mean", "recall_mean", "f1_mean",
        "intent_current_mean", "intent_next_mean",
        "detection_rate_mean",
        "intent_latency_mean", "decision_latency_mean",
    ]
    lines = [",".join(headers)]

    for exp_id, s in summaries.items():
        row = [
            exp_id, s["model"], s["task"], str(s["n_reps"]),
            str(s["a_score"]["mean"]), str(s["a_score"]["std"]),
            str(s["a_time"]["mean"]), str(s["a_mod"]["mean"]), str(s["a_nec"]["mean"]),
            str(s["precision"]["mean"]), str(s["recall"]["mean"]), str(s["f1"]["mean"]),
            str(s["intent_current_acc"]["mean"]), str(s["intent_next_acc"]["mean"]),
            str(s["detection_rate"]["mean"]),
            str(s["intent_latency_sec"]["mean"]), str(s["decision_latency_sec"]["mean"]),
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def _fmt(mean: float, std: float) -> str:
    """Format mean±std for LaTeX."""
    if std > 0:
        return f"${mean:.3f} \\pm {std:.3f}$"
    return f"${mean:.3f}$"


def to_latex(summaries: dict[str, Any]) -> str:
    """Generate LaTeX table for model comparison."""
    lines = [
        "% Auto-generated by aggregate_results.py",
        "\\begin{table}[H]",
        "\\caption{Model comparison on hand layup task.\\label{tab:model_comparison}}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Model & A-Score & $A_{time}$ & $A_{mod}$ & $A_{nec}$ & Precision & Recall & F1 \\\\",
        "\\midrule",
    ]

    for exp_id, s in summaries.items():
        model = s["model"].replace("_", "\\_")
        lines.append(
            f"{model} & "
            f"{_fmt(s['a_score']['mean'], s['a_score']['std'])} & "
            f"{_fmt(s['a_time']['mean'], s['a_time']['std'])} & "
            f"{_fmt(s['a_mod']['mean'], s['a_mod']['std'])} & "
            f"{_fmt(s['a_nec']['mean'], s['a_nec']['std'])} & "
            f"{_fmt(s['precision']['mean'], s['precision']['std'])} & "
            f"{_fmt(s['recall']['mean'], s['recall']['std'])} & "
            f"{_fmt(s['f1']['mean'], s['f1']['std'])} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate AURA experiment results")
    parser.add_argument("--experiments-dir", type=Path, required=True,
                        help="Root experiments directory")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--csv", type=Path, help="Output CSV path")
    parser.add_argument("--latex", type=Path, help="Output LaTeX table path")
    args = parser.parse_args()

    if not args.experiments_dir.is_dir():
        print(f"Error: {args.experiments_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = load_experiment_results(args.experiments_dir)
    if not results:
        print("No experiment results found.")
        sys.exit(0)

    summaries = aggregate(results)

    # Print summary table
    print(f"\n{'Experiment':<50s}  {'Reps':>4s}  {'A-Score':>8s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  {'IntAcc':>6s}")
    print("-" * 95)
    for exp_id, s in summaries.items():
        print(f"{exp_id:<50s}  {s['n_reps']:>4d}  "
              f"{s['a_score']['mean']:>8.3f}  "
              f"{s['precision']['mean']:>6.3f}  "
              f"{s['recall']['mean']:>6.3f}  "
              f"{s['f1']['mean']:>6.3f}  "
              f"{s['intent_current_acc']['mean']:>6.3f}")

    # Save outputs
    output_data = {"summaries": summaries, "n_experiments": len(summaries)}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nJSON saved to {args.output}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(to_csv(summaries))
        print(f"CSV saved to {args.csv}")

    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(to_latex(summaries))
        print(f"LaTeX saved to {args.latex}")

    if not args.output and not args.csv and not args.latex:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()

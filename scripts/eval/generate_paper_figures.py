#!/usr/bin/env python3
"""Generate all paper figures from evaluation results.

Reads JSON results from the evaluation scripts and produces
publication-quality figures for the MDPI Sensors paper.

Usage:
    python generate_paper_figures.py --output figures/generated/

Prerequisites:
    Run these first:
      python evaluate_intent_accuracy.py --all-sessions
      python compute_a_score.py --demo
      python analyze_latency.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "perfect": "#2ecc71",
    "early_proactive": "#3498db",
    "late_reactive": "#e67e22",
    "noisy": "#9b59b6",
    "wrong_modality": "#e74c3c",
    "reactive_baseline": "#95a5a6",
    "fixed_schedule": "#f39c12",
}


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ── Figure F4: Timeline — GT events vs detected transitions ───────────────
def fig_timeline(intent_results: Dict, output_dir: Path):
    """Horizontal timeline showing GT events and when each session detected them."""
    sessions = intent_results.get("sessions", [])
    if not sessions:
        print("  SKIP fig_timeline: no session data")
        return

    # Use the session with most predictions
    best = max(sessions, key=lambda s: s["n_predictions"])
    confusion = best["current_action"]["confusion_log"]
    transitions = best["transition_timing"]["per_action"]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Plot GT events as ticks on the bottom
    gt_actions = [t["action"] for t in transitions]
    gt_times = [t["gt_timestamp"] for t in transitions]

    ax.scatter(gt_times, [0] * len(gt_times), marker="|", s=200, c="black",
               linewidths=2, zorder=5, label="Ground Truth")

    # Plot detections
    for t in transitions:
        if t["detected"]:
            ax.scatter([t["detected_timestamp"]], [1], marker="^", s=80,
                       c="#2ecc71", zorder=5)
            ax.plot([t["gt_timestamp"], t["detected_timestamp"]], [0, 1],
                    c="#bdc3c7", linestyle="--", linewidth=0.8, alpha=0.7)
        else:
            ax.scatter([t["gt_timestamp"]], [1], marker="x", s=60,
                       c="#e74c3c", zorder=5)

    # Annotate GT actions
    for i, (time, action) in enumerate(zip(gt_times, gt_actions)):
        short = action.replace("_", "\n").replace("to cup", "")
        ax.annotate(short, (time, 0), textcoords="offset points",
                    xytext=(0, -25), ha="center", fontsize=6, rotation=45)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Ground\nTruth", "Detected"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Action Detection Timeline — {best['session']}")
    ax.legend(loc="upper left")
    ax.set_xlim(-5, max(gt_times) + 15)

    detected_patch = mpatches.Patch(color="#2ecc71", label="Detected")
    missed_patch = mpatches.Patch(color="#e74c3c", label="Missed")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="|", color="black", linestyle="None",
                   markersize=10, label="Ground Truth"),
        plt.Line2D([0], [0], marker="^", color="#2ecc71", linestyle="None",
                   markersize=8, label="Detected"),
        plt.Line2D([0], [0], marker="x", color="#e74c3c", linestyle="None",
                   markersize=8, label="Missed"),
    ], loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_timeline.png")
    fig.savefig(output_dir / "fig_timeline.pdf")
    plt.close(fig)
    print("  Generated fig_timeline")


# ── Figure F5: Confusion matrix ──────────────────────────────────────────
def fig_confusion(intent_results: Dict, output_dir: Path):
    """Confusion matrix of predicted vs actual current_action across all sessions."""
    sessions = intent_results.get("sessions", [])
    if not sessions:
        print("  SKIP fig_confusion: no data")
        return

    # Aggregate confusion logs
    all_confusion = []
    for sess in sessions:
        all_confusion.extend(sess["current_action"]["confusion_log"])

    # Get unique labels
    all_labels = sorted(set(
        [c["predicted"] for c in all_confusion] +
        [c["ground_truth"] for c in all_confusion]
    ))
    label_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    matrix = np.zeros((n, n), dtype=int)
    for c in all_confusion:
        gt_i = label_idx[c["ground_truth"]]
        pred_i = label_idx[c["predicted"]]
        matrix[gt_i, pred_i] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    short_labels = [l.replace("_", "\n") for l in all_labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted Action")
    ax.set_ylabel("Ground Truth Action")
    ax.set_title("Intent Prediction Confusion Matrix (All Sessions)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_confusion.png")
    fig.savefig(output_dir / "fig_confusion.pdf")
    plt.close(fig)
    print("  Generated fig_confusion")


# ── Figure F6: A-Score component box plots ────────────────────────────────
def fig_a_score_components(a_score_results: Dict, output_dir: Path):
    """Box plots of A-Score components across scenarios."""
    scenarios = a_score_results.get("scenarios", {})
    if not scenarios:
        print("  SKIP fig_a_score_components: no data")
        return

    names = list(scenarios.keys())
    metrics = ["mean_a_score", "mean_a_time", "mean_a_mod", "mean_a_nec"]
    metric_labels = ["A-Score", "A_time", "A_mod", "A_nec"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(names))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for name in names:
            s = scenarios[name]["summary"]
            val = s.get(metric, s.get(metric.replace("mean_", "mean_") + "_matched", 0))
            values.append(val)
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("A-Score Components Across Prediction Scenarios")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_a_score_components.png")
    fig.savefig(output_dir / "fig_a_score_components.pdf")
    plt.close(fig)
    print("  Generated fig_a_score_components")


# ── Figure F7: A-Score sensitivity to sigma ───────────────────────────────
def fig_sigma_sensitivity(a_score_results: Dict, output_dir: Path):
    """Line plot showing A-Score vs sigma parameter."""
    sens = a_score_results.get("sensitivity", {})
    sigma_data = sens.get("sigma_sensitivity", [])
    if not sigma_data:
        print("  SKIP fig_sigma_sensitivity: no data")
        return

    sigmas = [d["sigma"] for d in sigma_data]
    scores = [d["mean_a_score"] for d in sigma_data]
    a_times = [d["mean_a_time"] for d in sigma_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sigmas, scores, "o-", color="#2c3e50", linewidth=2,
            markersize=6, label="A-Score (composite)")
    ax.plot(sigmas, a_times, "s--", color="#3498db", linewidth=1.5,
            markersize=5, label="A_time component")

    ax.set_xlabel("Sigma (tolerance window, seconds)")
    ax.set_ylabel("Score")
    ax.set_title("A-Score Sensitivity to Timing Tolerance (σ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_sigma_sensitivity.png")
    fig.savefig(output_dir / "fig_sigma_sensitivity.pdf")
    plt.close(fig)
    print("  Generated fig_sigma_sensitivity")


# ── Figure F9: Proactive vs Reactive comparison ──────────────────────────
def fig_scenario_comparison(a_score_results: Dict, output_dir: Path):
    """Bar chart comparing key scenarios: proactive vs reactive vs fixed."""
    scenarios = a_score_results.get("scenarios", {})
    if not scenarios:
        print("  SKIP fig_scenario_comparison: no data")
        return

    # Select the most interesting scenarios for comparison
    compare_keys = ["perfect", "early_proactive", "noisy", "late_reactive",
                    "reactive_baseline", "fixed_schedule", "wrong_modality"]
    compare_keys = [k for k in compare_keys if k in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, (metric, title) in zip(axes, [
        ("mean_a_score", "Composite A-Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]):
        values = [scenarios[k]["summary"][metric] for k in compare_keys]
        colors = [COLORS.get(k, "#7f8c8d") for k in compare_keys]
        labels = [k.replace("_", "\n") for k in compare_keys]

        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.15)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("Score")

    fig.suptitle("Proactive vs Reactive Assistance Strategy Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_scenario_comparison.png")
    fig.savefig(output_dir / "fig_scenario_comparison.pdf")
    plt.close(fig)
    print("  Generated fig_scenario_comparison")


# ── Figure F10: Pipeline latency breakdown ────────────────────────────────
def fig_latency(latency_results: Dict, output_dir: Path):
    """Violin/box plot of LLM generation times + inter-call intervals."""
    gen_times = latency_results.get("raw_generation_times", [])
    if not gen_times:
        print("  SKIP fig_latency: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: LLM inference time distribution
    ax = axes[0]
    parts = ax.violinplot([gen_times], positions=[0], showmeans=True,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#3498db")
        pc.set_alpha(0.7)
    ax.scatter(np.zeros(len(gen_times)), gen_times, alpha=0.5, s=20,
               c="#2c3e50", zorder=5)
    ax.set_xticks([0])
    ax.set_xticklabels(["VLM Inference\n(Gemini Flash)"])
    ax.set_ylabel("Time (seconds)")
    ax.set_title("VLM Inference Latency")
    ax.grid(True, alpha=0.3, axis="y")

    o = latency_results["overall"]
    textstr = (f"Mean: {o['mean_sec']:.1f}s\n"
               f"Median: {o['median_sec']:.1f}s\n"
               f"P95: {o['p95_sec']:.1f}s\n"
               f"n={o['total_calls']}")
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right: Per-session bar chart
    ax = axes[1]
    per_sess = latency_results.get("per_session", {})
    if per_sess:
        sess_names = list(per_sess.keys())
        means = [per_sess[s]["mean_sec"] for s in sess_names]
        stds = [per_sess[s]["std_sec"] for s in sess_names]
        short_names = [s.split("_")[-1] for s in sess_names]

        bars = ax.bar(range(len(means)), means, yerr=stds, capsize=4,
                      color="#3498db", alpha=0.7, edgecolor="white")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(short_names, rotation=45, fontsize=8)
        ax.set_ylabel("Mean Generation Time (s)")
        ax.set_title("Per-Session VLM Latency")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_latency.png")
    fig.savefig(output_dir / "fig_latency.pdf")
    plt.close(fig)
    print("  Generated fig_latency")


# ── Figure F14: Intent accuracy over task phases ──────────────────────────
def fig_accuracy_over_phases(intent_results: Dict, output_dir: Path):
    """Line chart showing intent prediction accuracy at different task phases."""
    sessions = intent_results.get("sessions", [])
    if not sessions:
        print("  SKIP fig_accuracy_over_phases: no data")
        return

    # Aggregate all confusion logs with timestamps
    all_confusion = []
    for sess in sessions:
        for c in sess["current_action"]["confusion_log"]:
            all_confusion.append(c)

    if not all_confusion:
        return

    # Bin by time (every 20s)
    max_t = max(c["timestamp"] for c in all_confusion)
    bin_size = 20.0
    bins = np.arange(0, max_t + bin_size, bin_size)
    bin_correct = []
    bin_total = []
    bin_centers = []

    for i in range(len(bins) - 1):
        in_bin = [c for c in all_confusion
                  if bins[i] <= c["timestamp"] < bins[i+1]]
        if in_bin:
            correct = sum(1 for c in in_bin if c["correct"])
            bin_correct.append(correct)
            bin_total.append(len(in_bin))
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    if not bin_centers:
        return

    accuracies = [c / t if t > 0 else 0 for c, t in zip(bin_correct, bin_total)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, accuracies, width=bin_size * 0.8, alpha=0.7,
           color="#3498db", edgecolor="white")
    ax.plot(bin_centers, accuracies, "o-", color="#2c3e50", markersize=5)

    # Annotate sample counts
    for x, acc, n in zip(bin_centers, accuracies, bin_total):
        ax.annotate(f"n={n}", (x, acc + 0.05), ha="center", fontsize=7,
                    color="#7f8c8d")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Intent Prediction Accuracy Across Task Phases")
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_accuracy_phases.png")
    fig.savefig(output_dir / "fig_accuracy_phases.pdf")
    plt.close(fig)
    print("  Generated fig_accuracy_phases")


# ── Summary table (LaTeX) ─────────────────────────────────────────────────
def generate_latex_tables(intent_results: Dict, a_score_results: Dict,
                          latency_results: Dict, output_dir: Path):
    """Generate LaTeX table fragments for direct inclusion in the paper."""
    lines = []

    # Table T7: Scenario comparison
    scenarios = a_score_results.get("scenarios", {})
    if scenarios:
        lines.append("% Table: A-Score Scenario Comparison")
        lines.append("\\begin{table}[H]")
        lines.append("\\caption{A-Score comparison across prediction scenarios.\\label{tab:a_score_scenarios}}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("Scenario & A-Score & $A_{time}$ & $A_{mod}$ & $A_{nec}$ & Precision & Recall \\\\")
        lines.append("\\midrule")

        for name, data in scenarios.items():
            s = data["summary"]
            display = name.replace("_", " ").title()
            lines.append(
                f"{display} & {s['mean_a_score']:.3f} & {s['mean_a_time']:.3f} & "
                f"{s['mean_a_mod']:.3f} & {s['mean_a_nec']:.3f} & "
                f"{s['precision']:.3f} & {s['recall']:.3f} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    # Table T8: Latency
    if latency_results and "overall" in latency_results:
        o = latency_results["overall"]
        lines.append("% Table: Pipeline Latency")
        lines.append("\\begin{table}[H]")
        lines.append("\\caption{VLM inference latency statistics.\\label{tab:latency}}")
        lines.append("\\begin{tabular}{lc}")
        lines.append("\\toprule")
        lines.append("Metric & Value \\\\")
        lines.append("\\midrule")
        lines.append(f"Mean & {o['mean_sec']:.1f}s \\\\")
        lines.append(f"Median & {o['median_sec']:.1f}s \\\\")
        lines.append(f"Std Dev & {o['std_sec']:.1f}s \\\\")
        lines.append(f"Min & {o['min_sec']:.1f}s \\\\")
        lines.append(f"Max & {o['max_sec']:.1f}s \\\\")
        lines.append(f"P95 & {o['p95_sec']:.1f}s \\\\")
        lines.append(f"Total Calls & {o['total_calls']} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    out_path = output_dir / "latex_tables.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Generated LaTeX tables: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output", type=str, default="figures/generated/")
    parser.add_argument("--results-dir", type=str, default="results/")
    args = parser.parse_args()

    aura_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = aura_root / args.results_dir

    print("Loading results...")

    # Load intent evaluation
    intent_path = results_dir / "intent_evaluation" / "all_sessions_evaluation.json"
    intent_results = load_json(str(intent_path)) if intent_path.exists() else {}

    # Load A-Score
    ascore_path = results_dir / "a_score" / "demo_a_scores.json"
    a_score_results = load_json(str(ascore_path)) if ascore_path.exists() else {}

    # Load latency
    latency_path = results_dir / "latency" / "latency_analysis.json"
    latency_results = load_json(str(latency_path)) if latency_path.exists() else {}

    print("\nGenerating figures...")
    fig_timeline(intent_results, output_dir)
    fig_confusion(intent_results, output_dir)
    fig_a_score_components(a_score_results, output_dir)
    fig_sigma_sensitivity(a_score_results, output_dir)
    fig_scenario_comparison(a_score_results, output_dir)
    fig_latency(latency_results, output_dir)
    fig_accuracy_over_phases(intent_results, output_dir)

    print("\nGenerating LaTeX table fragments...")
    generate_latex_tables(intent_results, a_score_results, latency_results,
                          output_dir)

    print(f"\nAll outputs saved to {output_dir}")
    print(f"Files: {list(output_dir.glob('*'))}")


if __name__ == "__main__":
    main()

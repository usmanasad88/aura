#!/usr/bin/env python3
"""Generate intervention timeline (Gantt chart) comparing human task progress,
ground truth robot interventions, and AURA predicted interventions.

Usage::

    # From experiment rep directory (auto-detects sessions + GT)
    python scripts/eval/generate_timeline.py \
        --task hand_layup \
        --rep-dir logs/experiments/hand_layup__gemini-3.1-pro-preview__fs150__gt/rep_001

    # Explicit paths
    python scripts/eval/generate_timeline.py \
        --gt tasks/hand_layup/config/ground_truth.json \
        --decision-session logs/decision_engine/session_20260404_065941 \
        --intent-session logs/intent_monitor/session_20260404_065938 \
        --output figures/timeline.pdf

    # From aggregate results (multiple models side-by-side)
    python scripts/eval/generate_timeline.py \
        --task hand_layup \
        --experiments-dir logs/experiments/ \
        --output figures/timeline_comparison.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────
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
    "savefig.pad_inches": 0.15,
})

# Colour palette
C_HUMAN = "#3498db"        # blue — human actions
C_ROBOT_GT = "#2ecc71"     # green — ground truth robot
C_ROBOT_PRED = "#e74c3c"   # red — predicted robot (AURA)
C_OVERLAP = "#f39c12"      # orange — human/robot overlap
C_BG_BAND = "#f7f7f7"      # alternating row background
C_MATCH = "#27ae60"        # green tick — matched prediction
C_FP = "#c0392b"           # red cross — false positive
C_MISS = "#7f8c8d"         # grey dashed — missed GT


# ── Data loading ─────────────────────────────────────────────────────────────

@dataclass
class Event:
    action: str
    start: float
    end: float
    agent: str  # "human", "robot", or None


def load_ground_truth(gt_path: Path) -> list[Event]:
    """Load all events from ground_truth.json."""
    data = json.loads(gt_path.read_text())
    events = []
    for ev in data.get("events", []):
        agent = ev.get("agent") or "system"
        events.append(Event(
            action=ev["action"],
            start=ev["start_time"],
            end=ev["end_time"],
            agent=agent,
        ))
    return events


def load_predictions(session_dir: Path) -> list[Event]:
    """Load predicted 'act' decisions from decision engine session."""
    raw: list[tuple[float, str]] = []
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        parsed_path = call_dir / "response_parsed.json"
        if not meta_path.exists() or not parsed_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            parsed = json.loads(parsed_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if parsed is None:
            continue
        decision = parsed.get("decision") or meta.get("decision", "")
        if not decision or decision == "wait":
            continue
        if decision == "act":
            action_id = parsed.get("action_id", "")
        else:
            action_id = decision
        if not action_id:
            continue
        t = meta.get("timestamp_sec", 0.0)
        raw.append((t, action_id))

    if not raw:
        return []

    # Merge consecutive predictions with the same action into single spans
    raw.sort(key=lambda x: x[0])
    events: list[Event] = []
    cur_action = raw[0][1]
    cur_start = raw[0][0]
    prev_t = raw[0][0]

    for t, action in raw[1:]:
        if action != cur_action:
            events.append(Event(action=cur_action, start=cur_start,
                                end=prev_t + 2.0, agent="robot"))
            cur_action = action
            cur_start = t
        prev_t = t
    events.append(Event(action=cur_action, start=cur_start,
                        end=prev_t + 2.0, agent="robot"))
    return events


def load_intent_timeline(session_dir: Path) -> list[Event]:
    """Load intent monitor predictions as a continuous timeline of human actions."""
    raw: list[tuple[float, str]] = []
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        parsed_path = call_dir / "response_parsed.json"
        if not meta_path.exists() or not parsed_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            parsed = json.loads(parsed_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if parsed is None:
            continue
        t = meta.get("timestamp_sec", 0.0)
        action = parsed.get("current_action", "unknown")
        raw.append((t, action))

    if not raw:
        return []

    # Merge consecutive identical actions into spans
    raw.sort(key=lambda x: x[0])
    spans: list[Event] = []
    cur_action = raw[0][1]
    cur_start = raw[0][0]
    prev_t = raw[0][0]

    for t, action in raw[1:]:
        if action != cur_action:
            spans.append(Event(action=cur_action, start=cur_start, end=prev_t + 1.0, agent="human"))
            cur_action = action
            cur_start = t
        prev_t = t
    spans.append(Event(action=cur_action, start=cur_start, end=prev_t + 1.0, agent="human"))
    return spans


# ── Matching predictions to GT ───────────────────────────────────────────────

def match_predictions(gt_robot: list[Event], preds: list[Event],
                      tolerance: float = 10.0) -> dict:
    """Match predicted interventions to GT robot events.

    Returns dict with 'matched', 'false_positives', 'missed' lists.
    """
    gt_remaining = list(gt_robot)
    matched = []
    false_positives = []

    for pred in preds:
        best_gt = None
        best_dist = float("inf")
        for gt in gt_remaining:
            dist = abs(pred.start - gt.start)
            if dist < best_dist:
                best_dist = dist
                best_gt = gt
        if best_gt and best_dist <= tolerance:
            matched.append((best_gt, pred))
            gt_remaining.remove(best_gt)
        else:
            false_positives.append(pred)

    missed = gt_remaining
    return {"matched": matched, "false_positives": false_positives, "missed": missed}


# ── Short label helpers ──────────────────────────────────────────────────────

def _short_label(action: str, max_len: int = 22) -> str:
    """Shorten action name for display."""
    replacements = {
        "consolidate_with_roller_force": "consolidate",
        "move_roller_to_workplace": "move roller",
        "move_hardener_to_workplace": "move hardener",
        "move_resin_to_workplace": "move resin",
        "return_resin_to_storage": "return resin",
        "return_hardener_to_storage": "return hardener",
        "apply_resin_layer_": "apply resin L",
        "place_layer_": "place L",
        "add_resin_to_cup": "add resin",
        "add_hardener_to_cup": "add hardener",
        "mix_resin_hardener": "mix resin",
        "weigh_mixture": "weigh",
        "place_cup_on_scale": "place cup",
        "task_complete": "done",
    }
    for long, short in replacements.items():
        if action.startswith(long):
            suffix = action[len(long):]
            return short + suffix
    label = action.replace("_", " ")
    if len(label) > max_len:
        label = label[:max_len - 1] + "\u2026"
    return label


# ── Single-experiment timeline ───────────────────────────────────────────────

def plot_timeline(gt_events: list[Event],
                  pred_events: list[Event],
                  intent_events: list[Event] | None = None,
                  title: str = "AURA Intervention Timeline",
                  total_duration: float | None = None,
                  show_human_task: bool = False,
                  show_legend: bool = False) -> plt.Figure:
    """Create a Gantt chart comparing AURA predictions to GT robot interventions.

    Tracks (top to bottom):
        1. Human Task Progress (optional, from GT or intent monitor)
        2. Ground Truth Robot Interventions
        3. AURA Predicted Interventions
    """
    human_gt = [e for e in gt_events if e.agent == "human"]
    robot_gt = [e for e in gt_events if e.agent == "robot"]

    if total_duration is None:
        all_ends = ([e.end for e in gt_events] +
                    [e.end for e in pred_events] +
                    ([e.end for e in intent_events] if intent_events else []))
        total_duration = max(all_ends) if all_ends else 270.0

    # Match predictions to GT for colour coding
    matching = match_predictions(robot_gt, pred_events, tolerance=15.0)

    if show_human_task:
        track_labels = ["AURA Predicted", "GT Robot", "Human Task"]
        n_tracks = 3
        fig_height = 4.0
    else:
        track_labels = ["AURA Predicted", "GT Robot"]
        n_tracks = 2
        fig_height = 3.0
    fig, ax = plt.subplots(figsize=(14, fig_height))
    bar_height = 0.6

    # Alternating background bands
    for i in range(n_tracks):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=C_BG_BAND, zorder=0)

    # ── Track 2 (y=2): Human task progress (optional) ────────────────────
    if show_human_task:
        source = intent_events if intent_events else human_gt
        for ev in source:
            if ev.action in ("idle", "task_complete"):
                continue
            duration = max(ev.end - ev.start, 1.5)
            ax.barh(2, duration, left=ev.start, height=bar_height,
                    color=C_HUMAN, alpha=0.75, edgecolor="white", linewidth=0.5)
            if duration > 5:
                ax.text(ev.start + duration / 2, 2, _short_label(ev.action),
                        ha="center", va="center", fontsize=6, color="white",
                        fontweight="bold", clip_on=True)

    # ── Track 1 (y=1): Ground truth robot interventions ──────────────────
    for ev in robot_gt:
        duration = ev.end - ev.start
        ax.barh(1, duration, left=ev.start, height=bar_height,
                color=C_ROBOT_GT, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(ev.start + duration / 2, 1, _short_label(ev.action),
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", clip_on=True)

    # ── Track 0 (y=0): AURA predicted interventions ─────────────────────
    matched_preds = {id(p) for _, p in matching["matched"]}

    for gt, pred in matching["matched"]:
        duration = max(pred.end - pred.start, 2.5)
        ax.barh(0, duration, left=pred.start, height=bar_height,
                color=C_MATCH, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(pred.start + duration / 2, 0, _short_label(pred.action),
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", clip_on=True)
        # Draw connection line from prediction to GT
        ax.annotate("", xy=(gt.start, 1 - bar_height / 2),
                    xytext=(pred.start, 0 + bar_height / 2),
                    arrowprops=dict(arrowstyle="-", color="#95a5a6",
                                    linestyle="--", linewidth=0.8))

    for pred in matching["false_positives"]:
        duration = max(pred.end - pred.start, 2.5)
        ax.barh(0, duration, left=pred.start, height=bar_height,
                color=C_FP, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.text(pred.start + duration / 2, 0, _short_label(pred.action),
                ha="center", va="center", fontsize=6, color="white",
                fontweight="bold", clip_on=True)

    # Show missed GT as dashed outlines on prediction track
    for gt in matching["missed"]:
        ax.barh(0, gt.end - gt.start, left=gt.start, height=bar_height,
                facecolor="none", edgecolor=C_MISS, linewidth=1.5,
                linestyle="--", alpha=0.8)
        ax.text(gt.start + (gt.end - gt.start) / 2, 0 - 0.38,
                f"missed: {_short_label(gt.action)}", ha="center",
                fontsize=5.5, color=C_MISS, style="italic")

    # ── Formatting ───────────────────────────────────────────────────────
    ax.set_yticks(range(n_tracks))
    ax.set_yticklabels(track_labels, fontsize=10)
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(-2, total_duration + 5)
    ax.set_ylim(-0.6, n_tracks - 0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.grid(axis="x", which="major", alpha=0.3, linestyle="-")
    ax.grid(axis="x", which="minor", alpha=0.15, linestyle=":")

    # Legend (optional)
    if show_legend:
        legend_patches = [
            mpatches.Patch(color=C_ROBOT_GT, alpha=0.85, label="GT robot intervention"),
            mpatches.Patch(color=C_MATCH, alpha=0.85, label="Matched prediction (TP)"),
            mpatches.Patch(color=C_FP, alpha=0.7, label="False positive"),
            mpatches.Patch(facecolor="none", edgecolor=C_MISS, linestyle="--",
                           linewidth=1.5, label="Missed (FN)"),
        ]
        if show_human_task:
            legend_patches.insert(0, mpatches.Patch(color=C_HUMAN, alpha=0.75,
                                                   label="Human action"))
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7.5,
                  framealpha=0.9, ncol=3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ── Multi-model comparison ───────────────────────────────────────────────────

def plot_multi_model_timeline(gt_events: list[Event],
                              model_preds: dict[str, list[Event]],
                              title: str = "AURA Intervention Timeline — Model Comparison",
                              total_duration: float | None = None,
                              show_human_task: bool = False,
                              show_legend: bool = False) -> plt.Figure:
    """Timeline with one GT track + one prediction track per model."""
    robot_gt = [e for e in gt_events if e.agent == "robot"]
    human_gt = [e for e in gt_events if e.agent == "human"]

    if total_duration is None:
        all_ends = [e.end for e in gt_events]
        for preds in model_preds.values():
            all_ends.extend(e.end for e in preds)
        total_duration = max(all_ends) if all_ends else 270.0

    n_models = len(model_preds)
    n_tracks = (2 if show_human_task else 1) + n_models
    fig_height = max(3.5, 1.2 * n_tracks)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bar_height = 0.55

    # Background bands
    for i in range(n_tracks):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=C_BG_BAND, zorder=0)

    # Top track: Human (optional)
    if show_human_task:
        top = n_tracks - 1
        for ev in human_gt:
            if ev.action in ("idle", "task_complete"):
                continue
            duration = max(ev.end - ev.start, 1.5)
            ax.barh(top, duration, left=ev.start, height=bar_height,
                    color=C_HUMAN, alpha=0.75, edgecolor="white", linewidth=0.5)
            if duration > 5:
                ax.text(ev.start + duration / 2, top, _short_label(ev.action),
                        ha="center", va="center", fontsize=5.5, color="white",
                        fontweight="bold", clip_on=True)

    # Second track: GT Robot
    gt_track = n_tracks - 2 if show_human_task else n_tracks - 1
    for ev in robot_gt:
        duration = ev.end - ev.start
        ax.barh(gt_track, duration, left=ev.start, height=bar_height,
                color=C_ROBOT_GT, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(ev.start + duration / 2, gt_track, _short_label(ev.action),
                ha="center", va="center", fontsize=6, color="white",
                fontweight="bold", clip_on=True)

    # Model prediction tracks
    model_colors = ["#e74c3c", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"]
    track_labels = []
    for i, (model_name, preds) in enumerate(model_preds.items()):
        y = n_models - 1 - i
        color = model_colors[i % len(model_colors)]
        matching = match_predictions(robot_gt, preds, tolerance=15.0)

        for gt, pred in matching["matched"]:
            duration = max(pred.end - pred.start, 2.5)
            ax.barh(y, duration, left=pred.start, height=bar_height,
                    color=C_MATCH, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.text(pred.start + duration / 2, y, _short_label(pred.action),
                    ha="center", va="center", fontsize=5.5, color="white",
                    fontweight="bold", clip_on=True)

        for pred in matching["false_positives"]:
            duration = max(pred.end - pred.start, 2.5)
            ax.barh(y, duration, left=pred.start, height=bar_height,
                    color=C_FP, alpha=0.7, edgecolor="white", linewidth=0.5)
            ax.text(pred.start + duration / 2, y, _short_label(pred.action),
                    ha="center", va="center", fontsize=5.5, color="white",
                    fontweight="bold", clip_on=True)

        for gt in matching["missed"]:
            ax.barh(y, gt.end - gt.start, left=gt.start, height=bar_height,
                    facecolor="none", edgecolor=C_MISS, linewidth=1.5,
                    linestyle="--", alpha=0.8)

        short_model = model_name.split("__")[0] if "__" in model_name else model_name
        track_labels.append(short_model)

    # Y-axis labels
    gt_labels = ["GT Robot", "Human Task"] if show_human_task else ["GT Robot"]
    all_labels = list(reversed(track_labels)) + gt_labels
    ax.set_yticks(range(n_tracks))
    ax.set_yticklabels(all_labels, fontsize=9)

    # Formatting
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(-2, total_duration + 5)
    ax.set_ylim(-0.6, n_tracks - 0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.grid(axis="x", which="minor", alpha=0.15, linestyle=":")

    if show_legend:
        legend_patches = [
            mpatches.Patch(color=C_ROBOT_GT, alpha=0.85, label="GT robot"),
            mpatches.Patch(color=C_MATCH, alpha=0.85, label="Matched (TP)"),
            mpatches.Patch(color=C_FP, alpha=0.7, label="False positive"),
            mpatches.Patch(facecolor="none", edgecolor=C_MISS, linestyle="--",
                           linewidth=1.5, label="Missed (FN)"),
        ]
        if show_human_task:
            legend_patches.insert(0, mpatches.Patch(color=C_HUMAN, alpha=0.75,
                                                   label="Human action"))
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
                  framealpha=0.9, ncol=3)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ── Path resolution helpers ──────────────────────────────────────────────────

def _find_session(base: Path, component: str) -> Path | None:
    """Find the latest session dir for a component inside a rep directory."""
    comp_dir = base / component
    if not comp_dir.is_dir():
        return None
    sessions = sorted(comp_dir.glob("session_*"))
    return sessions[-1] if sessions else None


def _resolve_gt(task: str, aura_root: Path) -> Path:
    return aura_root / "tasks" / task / "config" / "ground_truth.json"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AURA intervention timeline (Gantt chart)")
    parser.add_argument("--task", default="hand_layup", help="Task name")
    parser.add_argument("--gt", type=Path, help="Ground truth JSON path")
    parser.add_argument("--decision-session", type=Path,
                        help="Decision engine session directory")
    parser.add_argument("--intent-session", type=Path,
                        help="Intent monitor session directory (optional)")
    parser.add_argument("--rep-dir", type=Path,
                        help="Experiment repetition directory (auto-detects sessions)")
    parser.add_argument("--experiments-dir", type=Path,
                        help="Root experiments dir for multi-model comparison")
    parser.add_argument("--output", type=Path, help="Output path (png/pdf)")
    parser.add_argument("--title", type=str, help="Custom figure title")
    parser.add_argument("--show-human-task", action="store_true",
                        help="Include the Human Task progress track (off by default)")
    parser.add_argument("--show-legend", action="store_true",
                        help="Include the TP/FP/FN legend (off by default)")
    args = parser.parse_args()

    aura_root = Path(__file__).resolve().parent.parent.parent

    # Resolve GT path
    gt_path = args.gt or _resolve_gt(args.task, aura_root)
    if not gt_path.exists():
        print(f"Error: ground truth not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    gt_events = load_ground_truth(gt_path)
    gt_data = json.loads(gt_path.read_text())
    total_duration = gt_data.get("total_duration_seconds", 270.0)

    output_dir = aura_root / "figures" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode 1: Multi-model comparison from experiments dir ──────────────
    if args.experiments_dir:
        exp_root = args.experiments_dir
        if not exp_root.is_absolute():
            exp_root = aura_root / exp_root
        model_preds: dict[str, list[Event]] = {}
        for exp_dir in sorted(exp_root.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            # Use first rep's predictions
            rep_dirs = sorted(exp_dir.glob("rep_*"))
            if not rep_dirs:
                continue
            dec_session = _find_session(rep_dirs[0], "decision_engine")
            if dec_session:
                preds = load_predictions(dec_session)
                model_preds[exp_dir.name] = preds

        if not model_preds:
            print("No experiment predictions found.", file=sys.stderr)
            sys.exit(1)

        title = args.title or "AURA Intervention Timeline — Model Comparison"
        fig = plot_multi_model_timeline(gt_events, model_preds, title=title,
                                        total_duration=total_duration,
                                        show_human_task=args.show_human_task,
                                        show_legend=args.show_legend)
        out = args.output or output_dir / "fig_timeline_comparison.pdf"
        fig.savefig(str(out))
        print(f"Saved multi-model timeline to {out}")
        # Also save PNG
        fig.savefig(str(out).replace(".pdf", ".png"))
        plt.close(fig)
        return

    # ── Mode 2: Single experiment from --rep-dir ─────────────────────────
    if args.rep_dir:
        dec_session = args.decision_session or _find_session(args.rep_dir, "decision_engine")
        int_session = args.intent_session or _find_session(args.rep_dir, "intent_monitor")
    else:
        dec_session = args.decision_session
        int_session = args.intent_session

    if dec_session is None:
        print("Error: provide --decision-session or --rep-dir", file=sys.stderr)
        sys.exit(1)

    pred_events = load_predictions(dec_session)
    intent_events = load_intent_timeline(int_session) if int_session else None

    model_name = "unknown"
    # Try to get model name from first call's meta
    first_meta = next(dec_session.glob("call_*/meta.json"), None)
    if first_meta:
        model_name = json.loads(first_meta.read_text()).get("model", "unknown")

    title = args.title or f"AURA Intervention Timeline — {model_name}"
    fig = plot_timeline(gt_events, pred_events, intent_events,
                        title=title, total_duration=total_duration,
                        show_human_task=args.show_human_task,
                        show_legend=args.show_legend)

    out = args.output or output_dir / "fig_timeline.pdf"
    fig.savefig(str(out))
    print(f"Saved timeline to {out}")
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

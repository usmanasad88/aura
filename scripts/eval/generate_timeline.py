#!/usr/bin/env python3
"""Generate intervention timeline (Gantt chart) comparing human task progress,
ground truth robot interventions, and AURA predicted interventions.

Reads the per-video robot ground-truth file
``tasks/<task>/ground_truth/<video_stem>.robot_gt.json`` (schema v1.0:
``interventions: [{skill, args, t_start, t_end, ...}, ...]``).

Usage::

    # Default: walk logs/run_*/ and produce intervention_timeline.png
    # inside every run that has a decision_engine/ folder and a matching
    # robot GT (skips runs whose timeline already exists).
    python scripts/eval/generate_timeline.py
    python scripts/eval/generate_timeline.py --force        # regenerate

    # Single run
    python scripts/eval/generate_timeline.py \
        --rep-dir logs/run_20260425_054919_hand_layup

    # Explicit paths
    python scripts/eval/generate_timeline.py \
        --gt tasks/hand_layup/ground_truth/<stem>.robot_gt.json \
        --decision-session logs/run_.../decision_engine \
        --intent-session   logs/run_.../intent_monitor \
        --output figures/timeline.pdf

    # Multi-model comparison
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


def load_robot_gt(gt_path: Path) -> tuple[list[Event], float]:
    """Load robot interventions from ``<stem>.robot_gt.json`` (schema v1.0).

    Returns ``(events, duration_sec)``. Each intervention becomes an Event
    with ``agent="robot"`` and ``action=skill`` (args, if any, are appended
    in parentheses for display).
    """
    data = json.loads(gt_path.read_text())
    events: list[Event] = []
    for iv in data.get("interventions", []):
        skill = iv.get("skill", "")
        args = iv.get("args") or {}
        if args:
            label = f"{skill}({', '.join(f'{k}={v}' for k, v in args.items())})"
        else:
            label = skill
        events.append(Event(
            action=label,
            start=float(iv["t_start"]),
            end=float(iv["t_end"]),
            agent="robot",
        ))
    return events, float(data.get("duration_sec", 0.0) or 0.0)


def load_intent_gt_humans(gt_path: Path) -> list[Event]:
    """Derive contiguous human-action spans from a sparse ``intent_gt.json``.

    Walks keyframes in order; each transition in ``current_action`` opens a
    new span. Used to plot the optional Human Task track when the user has
    annotated intent GT for the same video.
    """
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text())
    keyframes = data.get("keyframes", [])
    spans: list[Event] = []
    cur_action: str | None = None
    cur_start: float = 0.0
    for kf in keyframes:
        t = float(kf.get("timestamp_sec", 0.0))
        action = (kf.get("state") or {}).get("current_action", "")
        if cur_action is None:
            cur_action = action
            cur_start = t
            continue
        if action != cur_action:
            spans.append(Event(action=cur_action, start=cur_start, end=t, agent="human"))
            cur_action = action
            cur_start = t
    if cur_action is not None and keyframes:
        last_t = float(keyframes[-1].get("timestamp_sec", cur_start))
        if last_t > cur_start:
            spans.append(Event(action=cur_action, start=cur_start, end=last_t, agent="human"))
    return spans


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

def _stagger_offsets(centers_widths: list[tuple[float, float]],
                     levels: tuple[float, ...] = (0.0, -0.15, -0.30, -0.45),
                     padding: float = 1.0) -> list[float]:
    """Assign each label a y-offset so labels don't horizontally overlap.

    ``centers_widths`` is a list of ``(center_x, half_width_x)`` pairs in
    data coordinates. For each item, picks the first level whose most
    recent occupant ends (with ``padding``) before this item starts.
    """
    offsets: list[float] = []
    last_right: list[float] = [float("-inf")] * len(levels)
    for cx, hw in centers_widths:
        left = cx - hw
        right = cx + hw
        chosen = 0
        for i in range(len(levels)):
            if last_right[i] + padding <= left:
                chosen = i
                break
        else:
            # All levels occupied — pick the one that frees up soonest.
            chosen = min(range(len(levels)), key=lambda i: last_right[i])
        offsets.append(levels[chosen])
        last_right[chosen] = right
    return offsets


def _label_half_width(text: str, fontsize: float, ax) -> float:
    """Approximate half-width of rendered text in data (x-axis) coords."""
    fig = ax.get_figure()
    # Rough char width: ~0.55 * fontsize in points; 72pt = 1 inch.
    char_in = 0.55 * fontsize / 72.0
    width_in = max(len(text), 1) * char_in
    fig_w_in, _ = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_in = fig_w_in * bbox.width
    x0, x1 = ax.get_xlim()
    data_per_in = (x1 - x0) / max(ax_w_in, 1e-6)
    return 0.5 * width_in * data_per_in


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

    # Set xlim early so label-width estimation uses the final scale.
    ax.set_xlim(-2, total_duration + 5)

    # Alternating background bands
    for i in range(n_tracks):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=C_BG_BAND, zorder=0)

    # ── Track 2 (y=2): Human task progress (optional) ────────────────────
    if show_human_task:
        source = [e for e in (intent_events if intent_events else human_gt)
                  if e.action not in ("idle", "task_complete")]
        source_sorted = sorted(source, key=lambda e: e.start)
        human_cw = []
        for e in source_sorted:
            duration = max(e.end - e.start, 1.5)
            label = _short_label(e.action)
            human_cw.append((e.start + duration / 2,
                             _label_half_width(label, 6, ax)))
        human_offsets = _stagger_offsets(human_cw)
        for ev, y_off in zip(source_sorted, human_offsets):
            duration = max(ev.end - ev.start, 1.5)
            ax.barh(2, duration, left=ev.start, height=bar_height,
                    color=C_HUMAN, alpha=0.75, edgecolor="white", linewidth=0.5)
            if duration > 5:
                ax.text(ev.start + duration / 2, 2 + y_off, _short_label(ev.action),
                        ha="center", va="center", fontsize=6, color="black",
                        fontweight="bold", clip_on=True)

    # ── Track 1 (y=1): Ground truth robot interventions ──────────────────
    robot_gt_sorted = sorted(robot_gt, key=lambda e: e.start)
    gt_cw = [((e.start + e.end) / 2,
              _label_half_width(_short_label(e.action), 6.5, ax))
             for e in robot_gt_sorted]
    gt_offsets = _stagger_offsets(gt_cw)
    for ev, y_off in zip(robot_gt_sorted, gt_offsets):
        duration = ev.end - ev.start
        ax.barh(1, duration, left=ev.start, height=bar_height,
                color=C_ROBOT_GT, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(ev.start + duration / 2, 1 + y_off, _short_label(ev.action),
                ha="center", va="center", fontsize=6.5, color="black",
                fontweight="bold", clip_on=True)

    # ── Track 0 (y=0): AURA predicted interventions ─────────────────────
    matched_preds = {id(p) for _, p in matching["matched"]}

    # Build a combined, time-sorted list so label staggering accounts for
    # both matched and false-positive predictions on the same track.
    pred_items: list[tuple[Event, Event | None, str]] = []
    for gt, pred in matching["matched"]:
        pred_items.append((pred, gt, "match"))
    for pred in matching["false_positives"]:
        pred_items.append((pred, None, "fp"))
    pred_items.sort(key=lambda it: it[0].start)
    pred_cw = []
    for p, _, kind in pred_items:
        duration = max(p.end - p.start, 2.5)
        fs = 6.5 if kind == "match" else 6
        pred_cw.append((p.start + duration / 2,
                        _label_half_width(_short_label(p.action), fs, ax)))
    pred_offsets = _stagger_offsets(pred_cw)

    for (pred, gt, kind), y_off in zip(pred_items, pred_offsets):
        duration = max(pred.end - pred.start, 2.5)
        if kind == "match":
            ax.barh(0, duration, left=pred.start, height=bar_height,
                    color=C_MATCH, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.text(pred.start + duration / 2, 0 + y_off, _short_label(pred.action),
                    ha="center", va="center", fontsize=6.5, color="black",
                    fontweight="bold", clip_on=True)
            ax.annotate("", xy=(gt.start, 1 - bar_height / 2),
                        xytext=(pred.start, 0 + bar_height / 2),
                        arrowprops=dict(arrowstyle="-", color="#95a5a6",
                                        linestyle="--", linewidth=0.8))
        else:
            ax.barh(0, duration, left=pred.start, height=bar_height,
                    color=C_FP, alpha=0.7, edgecolor="white", linewidth=0.5)
            ax.text(pred.start + duration / 2, 0 + y_off, _short_label(pred.action),
                    ha="center", va="center", fontsize=6, color="black",
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

    # Set xlim early so label-width estimation uses the final scale.
    ax.set_xlim(-2, total_duration + 5)

    # Background bands
    for i in range(n_tracks):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=C_BG_BAND, zorder=0)

    # Top track: Human (optional)
    if show_human_task:
        top = n_tracks - 1
        human_src = sorted(
            [e for e in human_gt if e.action not in ("idle", "task_complete")],
            key=lambda e: e.start)
        human_cw = []
        for e in human_src:
            duration = max(e.end - e.start, 1.5)
            human_cw.append((e.start + duration / 2,
                             _label_half_width(_short_label(e.action), 5.5, ax)))
        human_offsets = _stagger_offsets(human_cw)
        for ev, y_off in zip(human_src, human_offsets):
            duration = max(ev.end - ev.start, 1.5)
            ax.barh(top, duration, left=ev.start, height=bar_height,
                    color=C_HUMAN, alpha=0.75, edgecolor="white", linewidth=0.5)
            if duration > 5:
                ax.text(ev.start + duration / 2, top + y_off, _short_label(ev.action),
                        ha="center", va="center", fontsize=5.5, color="black",
                        fontweight="bold", clip_on=True)

    # Second track: GT Robot
    gt_track = n_tracks - 2 if show_human_task else n_tracks - 1
    robot_gt_sorted = sorted(robot_gt, key=lambda e: e.start)
    gt_cw = [((e.start + e.end) / 2,
              _label_half_width(_short_label(e.action), 6, ax))
             for e in robot_gt_sorted]
    gt_offsets = _stagger_offsets(gt_cw)
    for ev, y_off in zip(robot_gt_sorted, gt_offsets):
        duration = ev.end - ev.start
        ax.barh(gt_track, duration, left=ev.start, height=bar_height,
                color=C_ROBOT_GT, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(ev.start + duration / 2, gt_track + y_off, _short_label(ev.action),
                ha="center", va="center", fontsize=6, color="black",
                fontweight="bold", clip_on=True)

    # Model prediction tracks
    model_colors = ["#e74c3c", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"]
    track_labels = []
    for i, (model_name, preds) in enumerate(model_preds.items()):
        y = n_models - 1 - i
        color = model_colors[i % len(model_colors)]
        matching = match_predictions(robot_gt, preds, tolerance=15.0)

        pred_items: list[tuple[Event, str]] = []
        for _, pred in matching["matched"]:
            pred_items.append((pred, "match"))
        for pred in matching["false_positives"]:
            pred_items.append((pred, "fp"))
        pred_items.sort(key=lambda it: it[0].start)
        pred_cw = []
        for p, _ in pred_items:
            duration = max(p.end - p.start, 2.5)
            pred_cw.append((p.start + duration / 2,
                            _label_half_width(_short_label(p.action), 5.5, ax)))
        pred_offsets = _stagger_offsets(pred_cw)

        for (pred, kind), y_off in zip(pred_items, pred_offsets):
            duration = max(pred.end - pred.start, 2.5)
            color_bar = C_MATCH if kind == "match" else C_FP
            alpha = 0.85 if kind == "match" else 0.7
            ax.barh(y, duration, left=pred.start, height=bar_height,
                    color=color_bar, alpha=alpha, edgecolor="white", linewidth=0.5)
            ax.text(pred.start + duration / 2, y + y_off, _short_label(pred.action),
                    ha="center", va="center", fontsize=5.5, color="black",
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

AURA_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = AURA_ROOT / "logs"
TIMELINE_FILENAME = "intervention_timeline.png"


def _find_call_dir(base: Path, component: str) -> Path | None:
    """Find a callable session dir for ``component`` inside ``base``.

    Handles both layouts:
      * ``base/<component>/call_*``           (run_aura's per-run layout)
      * ``base/<component>/session_*/call_*`` (experiments' per-rep layout)
    """
    comp_dir = base / component
    if not comp_dir.is_dir():
        return None
    if any(comp_dir.glob("call_*")):
        return comp_dir
    sessions = sorted(comp_dir.glob("session_*"))
    for s in reversed(sessions):
        if any(s.glob("call_*")):
            return s
    return None


def _resolve_robot_gt(task: str, video: str | None) -> Path | None:
    """Locate ``tasks/<task>/ground_truth/<video_stem>.robot_gt.json``.

    Falls back to the only ``*.robot_gt.json`` in the directory if the
    video stem doesn't match (handles experiments that don't record the
    video path).
    """
    gt_dir = AURA_ROOT / "tasks" / task / "ground_truth"
    if not gt_dir.is_dir():
        return None
    if video:
        stem = Path(video).stem
        candidate = gt_dir / f"{stem}.robot_gt.json"
        if candidate.exists():
            return candidate
    matches = sorted(gt_dir.glob("*.robot_gt.json"))
    return matches[0] if matches else None


def _resolve_intent_gt(task: str, video: str | None) -> Path | None:
    gt_dir = AURA_ROOT / "tasks" / task / "ground_truth"
    if not gt_dir.is_dir():
        return None
    if video:
        stem = Path(video).stem
        candidate = gt_dir / f"{stem}.intent_gt.json"
        if candidate.exists():
            return candidate
    matches = sorted(gt_dir.glob("*.intent_gt.json"))
    return matches[0] if matches else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ── Batch over logs/run_*/ ───────────────────────────────────────────────────

def _title_for_run(run_dir: Path, settings: dict) -> str:
    task = settings.get("task_name") or settings.get("task") or "?"
    model = settings.get("decision_model") or settings.get("intent_model") or settings.get("model") or ""
    if model:
        return f"{task} — {model} ({run_dir.name})"
    return f"{task} ({run_dir.name})"


def run_batch(logs_dir: Path, force: bool, show_human_task: bool,
              show_legend: bool) -> int:
    """Generate intervention_timeline.png for every run_* under logs_dir."""
    count = 0
    skipped: list[tuple[str, str]] = []

    for run_dir in sorted(logs_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        settings_path = run_dir / "settings.json"
        if not settings_path.exists():
            continue
        settings = _read_json(settings_path)
        if not settings:
            continue

        dec_dir = _find_call_dir(run_dir, "decision_engine")
        if dec_dir is None:
            continue

        output = run_dir / TIMELINE_FILENAME
        if output.exists() and not force:
            continue

        task = settings.get("task_name") or settings.get("task") or ""
        video = settings.get("video_path") or settings.get("video")
        gt_path = _resolve_robot_gt(task, video)
        if gt_path is None:
            skipped.append((run_dir.name, f"No robot_gt.json for task {task!r}"))
            continue

        try:
            gt_events, total_duration = load_robot_gt(gt_path)
        except Exception as e:
            skipped.append((run_dir.name, f"GT parse error: {e}"))
            continue

        pred_events = load_predictions(dec_dir)
        if not pred_events:
            skipped.append((run_dir.name, "No predictions in decision_engine"))
            continue

        intent_events = None
        if show_human_task:
            int_dir = _find_call_dir(run_dir, "intent_monitor")
            if int_dir is not None:
                intent_events = load_intent_timeline(int_dir)
            elif (igt := _resolve_intent_gt(task, video)) is not None:
                intent_events = load_intent_gt_humans(igt)

        if total_duration <= 0:
            all_ends = [e.end for e in gt_events] + [e.end for e in pred_events]
            total_duration = max(all_ends) if all_ends else 270.0

        print(f"[{run_dir.name}] plotting...")
        try:
            fig = plot_timeline(gt_events, pred_events, intent_events,
                                title=_title_for_run(run_dir, settings),
                                total_duration=total_duration,
                                show_human_task=show_human_task,
                                show_legend=show_legend)
            fig.savefig(str(output))
            plt.close(fig)
            count += 1
            print(f"  -> {output}")
        except Exception as e:
            skipped.append((run_dir.name, f"plot error: {e}"))

    if skipped:
        print("\nSkipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    return count


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AURA intervention timeline (Gantt chart)")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR,
                        help="Scan this directory for run_*/ folders (default: logs/)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate timelines even if intervention_timeline.png exists")
    parser.add_argument("--task", default=None, help="Task name (for ad-hoc GT lookup)")
    parser.add_argument("--gt", type=Path, help="robot_gt.json path (overrides --task lookup)")
    parser.add_argument("--decision-session", type=Path,
                        help="Decision engine call dir or session dir")
    parser.add_argument("--intent-session", type=Path,
                        help="Intent monitor call dir or session dir (optional)")
    parser.add_argument("--rep-dir", type=Path,
                        help="Experiment / run directory (auto-detects sessions)")
    parser.add_argument("--experiments-dir", type=Path,
                        help="Root experiments dir for multi-model comparison")
    parser.add_argument("--output", type=Path, help="Output path (png/pdf)")
    parser.add_argument("--title", type=str, help="Custom figure title")
    parser.add_argument("--show-human-task", action="store_true",
                        help="Include the Human Task progress track (off by default)")
    parser.add_argument("--show-legend", action="store_true",
                        help="Include the TP/FP/FN legend (off by default)")
    args = parser.parse_args()

    # ── Default batch mode: no targeting flags → walk logs/run_*/ ────────
    ad_hoc = any([args.task, args.gt, args.decision_session, args.intent_session,
                  args.rep_dir, args.experiments_dir])
    if not ad_hoc:
        if not args.logs_dir.is_dir():
            print(f"No logs dir at {args.logs_dir}", file=sys.stderr)
            sys.exit(1)
        n = run_batch(args.logs_dir, force=args.force,
                      show_human_task=args.show_human_task,
                      show_legend=args.show_legend)
        print(f"\nGenerated {n} timeline(s).")
        return

    output_dir = AURA_ROOT / "figures" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve GT for ad-hoc modes
    if args.gt:
        gt_path: Path | None = args.gt
    else:
        # Try to derive video stem from rep_dir/settings.json if available
        video = None
        if args.rep_dir and (s := args.rep_dir / "settings.json").exists():
            video = _read_json(s).get("video_path") or _read_json(s).get("video")
        gt_path = _resolve_robot_gt(args.task or "hand_layup", video)
    if gt_path is None or not gt_path.exists():
        print(f"Error: robot ground truth not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    gt_events, total_duration = load_robot_gt(gt_path)
    if total_duration <= 0:
        total_duration = max((e.end for e in gt_events), default=270.0)

    # ── Mode 1: Multi-model comparison from experiments dir ──────────────
    if args.experiments_dir:
        exp_root = args.experiments_dir
        if not exp_root.is_absolute():
            exp_root = AURA_ROOT / exp_root
        model_preds: dict[str, list[Event]] = {}
        for exp_dir in sorted(exp_root.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            rep_dirs = sorted(exp_dir.glob("rep_*"))
            if not rep_dirs:
                continue
            dec_dir = _find_call_dir(rep_dirs[0], "decision_engine")
            if dec_dir:
                model_preds[exp_dir.name] = load_predictions(dec_dir)

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
        fig.savefig(str(out).replace(".pdf", ".png"))
        plt.close(fig)
        return

    # ── Mode 2: Single experiment from --rep-dir ─────────────────────────
    if args.rep_dir:
        dec_dir = args.decision_session or _find_call_dir(args.rep_dir, "decision_engine")
        int_dir = args.intent_session or _find_call_dir(args.rep_dir, "intent_monitor")
    else:
        dec_dir = args.decision_session
        int_dir = args.intent_session

    if dec_dir is None:
        print("Error: provide --decision-session or --rep-dir", file=sys.stderr)
        sys.exit(1)

    pred_events = load_predictions(dec_dir)
    intent_events = load_intent_timeline(int_dir) if int_dir else None

    model_name = "unknown"
    first_meta = next(dec_dir.glob("call_*/meta.json"), None)
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

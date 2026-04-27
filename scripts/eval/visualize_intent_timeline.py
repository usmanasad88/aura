#!/usr/bin/env python3
"""Plot GT vs predicted action timelines for AURA intent-monitor runs.

Default behaviour (no arguments): walks ``logs/`` and generates a timeline
PNG inside every ``run_<timestamp>_<task>/`` folder that has both a
``settings.json`` and an ``intent_monitor/`` folder with call data. Skips
runs where ``intent_timeline.png`` already exists unless ``--force``.

Shows, for each action in the task DAG:

* a blue bar for the GT in-progress interval,
* a green fade for the GT completed interval,
* orange tick markers for frames at which the VLM predicted the action
  as in-progress,
* purple tick markers for frames where it was predicted completed.

Usage
-----
::

    # Default: plot every run_*/ under logs/ that isn't already plotted
    python scripts/eval/visualize_intent_timeline.py

    # Redo all
    python scripts/eval/visualize_intent_timeline.py --force

    # Ad-hoc: explicit session + task
    python scripts/eval/visualize_intent_timeline.py \\
        --session logs/run_20260425_031254_tea_making/intent_monitor \\
        --task tea_making \\
        --video demo_data/layup_demo/tea_making.mp4 \\
        --output /tmp/timeline.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session_to_predictions import session_to_predictions
from temporal_f1_state_eval import _normalise_entries


AURA_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = AURA_ROOT / "logs"
TIMELINE_FILENAME = "intent_timeline.png"


# ── Loading ─────────────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    return _normalise_entries(json.loads(Path(path).read_text()))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ── Data shaping ────────────────────────────────────────────────────────────

def _process_ground_truth(entries: list[dict]):
    """Build per-action {start, end, completed} intervals from GT entries."""
    action_timings: dict[str, dict[str, int | None]] = {}
    all_actions: set[str] = set()
    max_frame = 0

    for entry in entries:
        frame = entry["frame_number"]
        max_frame = max(max_frame, frame)
        for a in entry["state"].get("steps_in_progress", []) or []:
            if a == "idle":
                continue
            all_actions.add(a)
            action_timings.setdefault(a, {"start": frame, "end": None, "completed": None})

    previous_completed: set[str] = set()
    for entry in entries:
        frame = entry["frame_number"]
        current_completed = set(entry["state"].get("steps_completed", []) or [])
        for a in current_completed - previous_completed:
            if a not in action_timings:
                action_timings[a] = {"start": frame, "end": frame, "completed": frame}
                all_actions.add(a)
            else:
                if action_timings[a]["completed"] is None:
                    action_timings[a]["completed"] = frame
                if action_timings[a]["end"] is None:
                    action_timings[a]["end"] = frame
        previous_completed = current_completed

    return action_timings, all_actions, max_frame


def _process_predictions(entries: list[dict], frame_skip: int = 1):
    """Build per-action {in_progress, completed} intervals from predictions.

    Each prediction is valid from its frame until the next prediction's
    frame. The last prediction's validity extends by ``frame_skip`` frames.
    Returns per-action lists of ``(start_frame, end_frame)`` intervals.
    """
    predictions: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"in_progress": [], "completed": []},
    )
    all_actions: set[str] = set()
    max_frame = 0

    sorted_entries = sorted(entries, key=lambda e: e["frame_number"])
    for i, entry in enumerate(sorted_entries):
        frame = entry["frame_number"]
        if i + 1 < len(sorted_entries):
            end_frame = sorted_entries[i + 1]["frame_number"]
        else:
            end_frame = frame + max(frame_skip, 1)
        max_frame = max(max_frame, end_frame)

        state = entry["state"] or {}
        for a in state.get("steps_in_progress", []) or []:
            if a == "idle":
                continue
            all_actions.add(a)
            predictions[a]["in_progress"].append((frame, end_frame))
        for a in state.get("steps_completed", []) or []:
            if a == "idle":
                continue
            all_actions.add(a)
            predictions[a]["completed"].append((frame, end_frame))

    return predictions, all_actions, max_frame


def _clean_label(action: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", action).strip().replace("_", " ")


def visualize(
    gt_path: Path,
    pred_entries: list[dict],
    output_path: Path | None,
    title: str | None = None,
    frame_skip: int = 1,
) -> None:
    import matplotlib.pyplot as plt

    gt_entries = _load(gt_path)

    gt_timings, gt_actions, gt_max = _process_ground_truth(gt_entries)
    pred_timings, pred_actions, pred_max = _process_predictions(pred_entries, frame_skip)

    combined = sorted(
        (gt_actions | pred_actions),
        key=lambda a: gt_timings.get(a, {}).get("start", 10**9) or 10**9,
    )
    y_map = {a: i * 0.7 for i, a in enumerate(combined)}
    max_frame = max(gt_max, pred_max, 1)

    fig, ax = plt.subplots(figsize=(20, max(4, 0.5 * len(combined))))

    gt_progress_label_used = False
    gt_completed_label_used = False
    for action, timings in gt_timings.items():
        y = y_map.get(action)
        if y is None:
            continue
        if timings["start"] is not None and timings["end"] is not None:
            duration = max(timings["end"] - timings["start"], 1)
            ax.barh(
                y, duration, left=timings["start"], height=0.4, color="royalblue",
                label=None if gt_progress_label_used else "GT In Progress",
            )
            gt_progress_label_used = True
        if timings["completed"] is not None:
            ax.barh(
                y, max_frame - timings["completed"], left=timings["completed"],
                height=0.4, color="mediumseagreen", alpha=0.5,
                label=None if gt_completed_label_used else "GT Completed",
            )
            gt_completed_label_used = True

    pred_progress_label_used = False
    pred_completed_label_used = False
    for action, t in pred_timings.items():
        y = y_map.get(action)
        if y is None:
            continue
        for start, end in t["in_progress"]:
            ax.barh(
                y + 0.2, max(end - start, 1), left=start, height=0.22,
                color="darkorange", alpha=0.85, edgecolor="none",
                label=None if pred_progress_label_used else "Predicted In Progress",
            )
            pred_progress_label_used = True
        for start, end in t["completed"]:
            ax.barh(
                y - 0.2, max(end - start, 1), left=start, height=0.22,
                color="purple", alpha=0.6, edgecolor="none",
                label=None if pred_completed_label_used else "Predicted Completed",
            )
            pred_completed_label_used = True

    ax.set_yticks([y_map[a] for a in combined])
    ax.set_yticklabels([_clean_label(a) for a in combined], fontsize=10)
    ax.set_xlabel("Frame number", fontsize=14)
    ax.set_ylabel("Action", fontsize=14)
    ax.set_xlim(0, max_frame + max(max_frame * 0.02, 10))
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=12)

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {output_path}")
    else:
        plt.show()


# ── Path resolution ─────────────────────────────────────────────────────────

def resolve_gt(task: str, video: str | None) -> Path | None:
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


def find_intent_dir(run_dir: Path) -> Path | None:
    direct = run_dir / "intent_monitor"
    if direct.is_dir():
        if any(direct.glob("call_*")):
            return direct
        for s in sorted(direct.glob("session_*")):
            if any(s.glob("call_*")):
                return s
    return None


# ── Batch over logs/run_*/ ──────────────────────────────────────────────────

def _title_for_run(run_dir: Path, settings: dict) -> str:
    task = settings.get("task_name") or settings.get("task") or "?"
    model = settings.get("intent_model") or settings.get("model") or ""
    if model:
        return f"{task} — {model} ({run_dir.name})"
    return f"{task} ({run_dir.name})"


def run_batch(logs_dir: Path, force: bool) -> int:
    """Generate timelines for every scorable run under logs_dir. Returns count."""
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

        intent_dir = find_intent_dir(run_dir)
        if intent_dir is None:
            continue

        output = run_dir / TIMELINE_FILENAME
        if output.exists() and not force:
            continue

        task = settings.get("task_name") or settings.get("task") or ""
        video = settings.get("video_path") or settings.get("video")
        gt_path = resolve_gt(task, video)
        if gt_path is None:
            skipped.append((run_dir.name, f"No intent_gt.json for task {task!r}"))
            continue

        print(f"[{run_dir.name}] scoring...")
        try:
            preds = session_to_predictions(intent_dir)
        except Exception as e:
            skipped.append((run_dir.name, f"prediction-extract error: {e}"))
            continue
        if not preds:
            skipped.append((run_dir.name, "No predictions in intent_monitor"))
            continue

        frame_skip = int(settings.get("frame_skip") or 1)
        try:
            visualize(
                gt_path, preds, output,
                title=_title_for_run(run_dir, settings),
                frame_skip=frame_skip,
            )
            count += 1
        except Exception as e:
            skipped.append((run_dir.name, f"plot error: {e}"))

    if skipped:
        print("\nSkipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    return count


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT vs predicted action timelines")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR,
                        help="Scan this directory for run_*/ folders (default: logs/)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate timelines even if intent_timeline.png already exists")

    # Ad-hoc single-run overrides
    parser.add_argument("--session", type=Path,
                        help="Explicit intent_monitor dir (or run_*/ dir containing one)")
    parser.add_argument("--pred", type=Path, help="Pre-computed flat predictions JSON")
    parser.add_argument("--gt", type=Path, help="Intent GT JSON (overrides --task lookup)")
    parser.add_argument("--task", default=None, help="Task name (for GT lookup)")
    parser.add_argument("--video", default=None, help="Video path (for GT stem lookup)")
    parser.add_argument("--output", type=Path,
                        help="Output PNG (default: show interactively for ad-hoc, "
                             "or intent_timeline.png inside the run for batch)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Width (in frames) of each prediction bar past the last "
                             "prediction. Defaults to settings.json's frame_skip, or 1.")
    args = parser.parse_args()

    # Ad-hoc mode if any of these single-run flags were passed
    ad_hoc = any([args.session, args.pred, args.gt, args.task])

    if not ad_hoc:
        if not args.logs_dir.is_dir():
            print(f"No logs dir at {args.logs_dir}", file=sys.stderr)
            sys.exit(1)
        n = run_batch(args.logs_dir, force=args.force)
        print(f"\nGenerated {n} timeline(s).")
        return

    # Resolve GT
    gt_path = args.gt
    settings = {}
    target = args.session
    if target is not None and (target / "settings.json").exists():
        settings = _read_json(target / "settings.json")
        intent_dir = find_intent_dir(target)
        if intent_dir is not None:
            target = intent_dir

    task = args.task or settings.get("task_name") or settings.get("task")
    video = args.video or settings.get("video_path") or settings.get("video")

    if gt_path is None:
        if not task:
            parser.error("--gt or --task (or a run folder with settings.json) is required")
        gt_path = resolve_gt(task, video)
        if gt_path is None:
            parser.error(f"No intent_gt.json found for task '{task}'")

    # Resolve predictions
    if args.pred is not None:
        preds = _normalise_entries(json.loads(args.pred.read_text()))
    else:
        if target is None:
            parser.error("--pred, --session, or no-arg batch mode is required")
        preds = session_to_predictions(target)

    title = args.title
    if title is None and args.session is not None:
        title = _title_for_run(args.session, settings) if settings else None

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = int(settings.get("frame_skip") or 1)

    visualize(gt_path, preds, args.output, title=title, frame_skip=frame_skip)


if __name__ == "__main__":
    main()

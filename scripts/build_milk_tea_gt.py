#!/usr/bin/env python3
"""Build an intent ground-truth file for the ``milk_tea_making`` task
directly from an Ego-Exo4D "Making Milk Tea" rich annotation.

This converts the annotation's ``keystep.segments`` (time-aligned recipe
steps) into the sparse keyframe format consumed by
:class:`aura.monitors.intent_ground_truth.GroundTruthIntentProvider`
(same shape as ``tasks/tea_making/ground_truth/tea_making.intent_gt.json``).

It emits one keyframe per state change. Each keyframe records only the
vision-observable variables from ``state_schema.json`` plus the synthetic
``steps_completed / steps_in_progress / steps_pending / reasoning`` fields.
Boolean state variables are derived from the set of completed DAG steps,
``current_action`` mirrors the in-progress step and
``predicted_next_action`` mirrors the next observed step.

Usage
-----
::

    python scripts/build_milk_tea_gt.py \
        --annotation /home/mani/Repos/annotations/milk_tea_rich/iiith_cooking_03_3_cut.json

The mapping from Ego-Exo4D keysteps to the ``milk_tea_making`` DAG nodes
(``PLAN`` below) is specific to the boiled-milk / doodh-patti flow modelled
by the task. A handful of DAG nodes are *implied* rather than separately
annotated (the saucepan is already on the gas stove when milk is poured in,
the milk heats while tea/sugar are prepped, and the washed sieve is in place
for the final strain), so they are folded into ``steps_completed`` at the
transition where they must have happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent


# ─── DAG → boolean effects ──────────────────────────────────────────────────
# A boolean state variable flips true once its producing step is *completed*.
# stove_on / milk_boiling additionally clear once the stove is turned off.

def derive_booleans(completed: set[str]) -> Dict[str, Any]:
    stove_on = ("turn_on_stove" in completed) and ("turn_off_stove" not in completed)
    milk_boiling = ("heat_milk" in completed) and ("turn_off_stove" not in completed)
    return {
        "milk_in_pot": "pour_milk_into_pot" in completed,
        "water_added": "add_water" in completed,
        "pot_on_stove": "place_pot_on_stove" in completed,
        "stove_on": stove_on,
        "milk_boiling": milk_boiling,
        "tea_added": "add_tea" in completed,
        "sugar_added": "add_sugar" in completed,
        "tea_simmered": "simmer_tea" in completed,
        "stove_off": "turn_off_stove" in completed,
        "tea_strained": "strain_tea_into_cup" in completed,
    }


PHASE_OF = {
    "idle": "initialization",
    "setup_workspace": "setup",
    "pour_milk_into_pot": "setup",
    "add_water": "setup",
    "place_pot_on_stove": "setup",
    "position_sieve": "setup",
    "turn_on_stove": "heating",
    "heat_milk": "heating",
    "add_tea": "brewing",
    "add_sugar": "brewing",
    "simmer_tea": "brewing",
    "adjust_heat": "brewing",
    "stir_pot": "brewing",
    "turn_off_stove": "brewing",
    "strain_tea_into_cup": "straining",
    "cleanup": "serving",
    "task_complete": "complete",
}


# ─── Keystep → DAG plan ─────────────────────────────────────────────────────
# Each entry anchors a keyframe to an annotation segment via (step_id, occ)
# so timings come straight from the annotation. ``completes`` lists the DAG
# nodes that become finished *as this keyframe begins* (folding in the implied
# steps). ``action`` is the step in progress; ``human`` / ``phase`` are the
# observable human/task state.

PLAN: List[Dict[str, Any]] = [
    {"step_id": 681, "occ": 0, "action": "setup_workspace",      "completes": ["idle"],
     "human": "Other"},                       # wash sieve / get cup / get milk
    {"step_id": 849, "occ": 0, "action": "pour_milk_into_pot",   "completes": ["setup_workspace"],
     "human": "pouring_milk"},                # pour milk into the saucepan
    {"step_id": 822, "occ": 0, "action": "turn_on_stove",        "completes": ["pour_milk_into_pot", "place_pot_on_stove"],
     "human": "operating_stove"},             # light the gas stove (pan already on it)
    {"step_id": 590, "occ": 0, "action": "heat_milk",            "completes": ["turn_on_stove"],
     "human": "waiting"},                      # milk heats while tea leaves/spoon are fetched
    {"step_id": 836, "occ": 0, "action": "add_tea",              "completes": ["heat_milk"],
     "human": "adding_ingredient"},           # add tea leaves to the milk
    {"step_id": 585, "occ": 0, "action": "add_sugar",            "completes": ["add_tea"],
     "human": "adding_ingredient"},           # get sugar + add sugar
    {"step_id": 854, "occ": 0, "action": "simmer_tea",           "completes": ["add_sugar"],
     "human": "waiting"},                      # simmer over low heat
    {"step_id": 823, "occ": 0, "action": "adjust_heat",          "completes": [],
     "human": "operating_stove"},             # adjust the stove heat (simmer ongoing)
    {"step_id": 854, "occ": 2, "action": "simmer_tea",           "completes": ["adjust_heat"],
     "human": "waiting"},                      # continue simmering
    {"step_id": 866, "occ": 0, "action": "turn_off_stove",       "completes": ["simmer_tea"],
     "human": "operating_stove"},             # turn off the stove
    {"step_id": 644, "occ": 0, "action": "strain_tea_into_cup",  "completes": ["turn_off_stove", "position_sieve"],
     "human": "straining"},                   # pour the tea through the sieve into the cup
    {"anchor": "end", "action": "cleanup",                       "completes": ["strain_tea_into_cup"],
     "human": "Other"},                        # tea served in the cup; cleanup pending
]


def load_dag_pool(config_dir: Path) -> List[str]:
    dag = json.loads((config_dir / "dag.json").read_text(encoding="utf-8"))
    return [node["id"] for node in dag]


def segment_start(segments: List[Dict[str, Any]], step_id: int, occ: int) -> Optional[float]:
    matches = [s for s in segments if int(s.get("step_id")) == int(step_id)]
    if occ < len(matches):
        return float(matches[occ]["start_time"])
    return None


def build(args: argparse.Namespace) -> int:
    ann_path = Path(args.annotation)
    if not ann_path.exists():
        print(f"Annotation not found: {ann_path}", file=sys.stderr)
        return 2
    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    keystep = ann.get("keystep", {})
    segments = keystep.get("segments", []) or []
    take_name = keystep.get("take_name") or ann_path.stem

    # fps: prefer the annotation's recorded fps, else CLI / default.
    fps = args.fps
    if fps is None:
        fps = (ann.get("relations", {})
                  .get("object_names", {})
                  .get("annotation_metadata", {})
                  .get("annotation_fps"))
    fps = float(fps or 30.0)

    end_time = max((float(s["end_time"]) for s in segments), default=0.0)
    total_frames = int(round(end_time * fps))

    config_dir = _project_root / "tasks" / args.task / "config"
    pool = load_dag_pool(config_dir)

    # ── Build keyframes ──────────────────────────────────────────────
    completed: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for entry in PLAN:
        completed.update(entry["completes"])
        action = entry["action"]

        if entry.get("anchor") == "end":
            start = end_time
        else:
            start = segment_start(segments, entry["step_id"], entry["occ"])
            if start is None:
                print(f"  ! no segment for step_id={entry['step_id']} occ={entry['occ']}; skipping",
                      file=sys.stderr)
                continue

        completed_list = [n for n in pool if n in completed]
        pending = [n for n in pool if n not in completed and n != action]

        state: Dict[str, Any] = {
            "current_phase": PHASE_OF.get(action, "Other"),
            "current_action": action,
            "human_state": entry["human"],
        }
        state.update(derive_booleans(completed))
        state["predicted_next_action"] = action  # back-filled below
        state["prediction_confidence"] = 1.0
        state["steps_completed"] = completed_list
        state["steps_in_progress"] = [action]
        state["steps_pending"] = pending
        state["reasoning"] = "ground_truth"

        rows.append({
            "frame_num": int(round(start * fps)),
            "timestamp_sec": round(start, 5),
            "state": state,
        })

    # predicted_next_action = the next observed action (or task_complete).
    for i, row in enumerate(rows):
        nxt = rows[i + 1]["state"]["current_action"] if i + 1 < len(rows) else "task_complete"
        row["state"]["predicted_next_action"] = nxt

    # GT filename follows GroundTruthIntentProvider.default_gt_path:
    # <video_stem>.intent_gt.json (falling back to the take name).
    video = args.video or take_name
    out_path = (Path(args.output) if args.output else
                config_dir.parent / "ground_truth" /
                f"{Path(video).stem}.intent_gt.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        "video": video,
        "task": args.task,
        "fps": fps,
        "total_frames": total_frames,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotation": str(ann_path),
        "take_uid": ann.get("take_uid", ""),
        "keyframes": rows,
    }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} keyframes → {out_path}")
    print(f"  fps={fps}  total_frames={total_frames}  take={take_name}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--annotation", required=True,
                   help="Path to the Ego-Exo4D milk-tea rich annotation JSON")
    p.add_argument("--task", default="milk_tea_making",
                   help="Task name (dir under tasks/, default: milk_tea_making)")
    p.add_argument("--fps", type=float, default=None,
                   help="Frames per second (default: read from annotation, else 30)")
    p.add_argument("--video", default=None,
                   help="Value for the GT 'video' field (default: take name)")
    p.add_argument("--output", default=None,
                   help="Output path (default: tasks/<task>/ground_truth/<task>.<take>.intent_gt.json)")
    args = p.parse_args()
    sys.exit(build(args))


if __name__ == "__main__":
    main()

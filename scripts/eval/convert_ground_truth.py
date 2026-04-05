#!/usr/bin/env python3
"""Convert task ground_truth.json into normalized intervention events.

Reads ``tasks/<task>/config/ground_truth.json`` and extracts robot-agent
events into the ``InterventionEvent`` format used by the A-Score evaluator.

Usage::

    python scripts/eval/convert_ground_truth.py --task hand_layup
    python scripts/eval/convert_ground_truth.py --task hand_layup --output /tmp/gt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as script from scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_a_score import InterventionEvent

# Patterns to extract target objects from action names
_OBJECT_PATTERNS = {
    "resin": ["resin", "resin_bottle"],
    "hardener": ["hardener", "hardener_bottle"],
    "roller": ["roller"],
    "cup": ["cup"],
    "brush": ["brush"],
    "squeegee": ["squeegee"],
}


def _extract_targets(action: str) -> list[str]:
    """Extract target objects from an action name like 'move_resin_to_workplace'."""
    targets = []
    action_lower = action.lower()
    for key, obj_names in _OBJECT_PATTERNS.items():
        if key in action_lower:
            targets.append(obj_names[0])
    return targets


def convert(gt_path: Path) -> list[InterventionEvent]:
    """Convert ground truth JSON to normalized intervention events."""
    data = json.loads(gt_path.read_text())
    events = []
    for ev in data.get("events", []):
        if ev.get("agent") != "robot":
            continue
        events.append(
            InterventionEvent(
                timestamp=ev["start_time"],
                action_type=ev["action"],
                target_objects=_extract_targets(ev["action"]),
            )
        )
    return events


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Convert ground truth to intervention events")
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    gt_path = repo_root / "tasks" / args.task / "config" / "ground_truth.json"
    if not gt_path.exists():
        print(f"Error: {gt_path} not found", file=sys.stderr)
        sys.exit(1)

    events = convert(gt_path)
    result = {"interventions": [asdict(e) for e in events]}

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Wrote {len(events)} interventions to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract predicted interventions from decision engine session logs.

Walks ``logs/decision_engine/session_*/call_*/`` and collects all
decisions where ``decision == "act"`` into normalized intervention events.

Usage::

    python scripts/eval/extract_predictions.py --session logs/decision_engine/session_20260402_220111
    python scripts/eval/extract_predictions.py --session logs/decision_engine/session_20260402_220111 --output preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_a_score import InterventionEvent, _extract_targets


def extract(session_dir: Path) -> list[InterventionEvent]:
    """Extract 'act' decisions from a decision engine session directory."""
    events = []
    call_dirs = sorted(session_dir.glob("call_*"))

    for call_dir in call_dirs:
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
            # BehaviorTree / gt-intent runs: `decision` is the action_id itself
            action_id = decision

        if not action_id:
            continue

        timestamp = meta.get("timestamp_sec", 0.0)
        target_id = parsed.get("target_id") or parsed.get("target")
        targets = [target_id] if target_id else _extract_targets(action_id)

        events.append(
            InterventionEvent(
                timestamp=timestamp,
                action_type=action_id,
                target_objects=targets,
            )
        )

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract predictions from decision engine logs")
    parser.add_argument("--session", type=Path, required=True, help="Decision engine session directory")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    if not args.session.is_dir():
        print(f"Error: {args.session} is not a directory", file=sys.stderr)
        sys.exit(1)

    events = extract(args.session)
    result = {"interventions": [asdict(e) for e in events]}

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Wrote {len(events)} predicted interventions to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

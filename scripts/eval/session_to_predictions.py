#!/usr/bin/env python3
"""Convert an intent_monitor session log into a flat predictions list.

Walks ``logs/intent_monitor/session_*/call_NNNN/`` and collects each call's
parsed state into a single JSON file formatted like the ground-truth files
produced by ``scripts/annotate_ground_truth.py`` (but flat, not keyframe
nested), so the hcdt-style temporal F1 evaluator can consume both.

Output schema::

    [
      {
        "frame_number": <int>,
        "timestamp_sec": <float>,
        "state": { ... response_parsed.json ... }
      },
      ...
    ]

Usage::

    python scripts/eval/session_to_predictions.py \\
        --session logs/intent_monitor/session_20260424_141603 \\
        --output  results/intent_evaluation/session_20260424_141603.pred.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def session_to_predictions(session_dir: Path) -> list[dict[str, Any]]:
    """Load every call in a session and return predictions sorted by frame."""
    preds: list[dict[str, Any]] = []
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        parsed_path = call_dir / "response_parsed.json"
        if not meta_path.exists() or not parsed_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            state = json.loads(parsed_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if state is None:
            continue

        frame_num = meta.get("frame_num", meta.get("frame_number", 0))
        preds.append({
            "frame_number": int(frame_num),
            "timestamp_sec": float(meta.get("timestamp_sec", 0.0)),
            "state": state,
        })

    preds.sort(key=lambda e: e["frame_number"])
    return preds


def get_session_model(session_dir: Path) -> str:
    """Read the model name from the first call's meta.json."""
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text()).get("model", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an intent_monitor session to a flat predictions file",
    )
    parser.add_argument("--session", type=Path, required=True,
                        help="Intent monitor session directory")
    parser.add_argument("--output", type=Path,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    if not args.session.is_dir():
        print(f"Error: {args.session} is not a directory", file=sys.stderr)
        sys.exit(1)

    preds = session_to_predictions(args.session)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(preds, indent=2))
        print(f"Wrote {len(preds)} predictions to {args.output}")
    else:
        json.dump(preds, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

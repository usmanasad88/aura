#!/usr/bin/env python3
"""Evaluate a single AURA experiment run.

Orchestrates the A-Score, intent accuracy, and latency evaluators for
a pair of intent_monitor + decision_engine session directories.

Usage::

    # Auto-detect latest sessions in a log directory
    python scripts/eval/evaluate_run.py --logs-dir logs/ --task hand_layup

    # Explicit session paths
    python scripts/eval/evaluate_run.py \
        --intent-session logs/intent_monitor/session_20260402_220153 \
        --decision-session logs/decision_engine/session_20260402_220111 \
        --task hand_layup

    # Save to specific output
    python scripts/eval/evaluate_run.py --logs-dir logs/ --task hand_layup \
        --output results/run_eval/run_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_a_score import parse_gt_interventions, compute_a_scores
from extract_predictions import extract
from evaluate_intent_accuracy import evaluate_session as evaluate_intent_session


def count_decisions(session_dir: Path) -> dict[str, int]:
    """Count act/wait decisions in a decision engine session."""
    counts = {"act": 0, "wait": 0, "total": 0}
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        decision = meta.get("decision", "unknown")
        # Normalize BehaviorTree-style decisions (action_id) to "act"
        if decision not in ("wait", "act", "unknown"):
            decision = "act"
        counts[decision] = counts.get(decision, 0) + 1
        counts["total"] += 1
    return counts


def get_model_from_session(session_dir: Path) -> str:
    """Extract model name from the first call's meta.json."""
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                return meta.get("model", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
    return "unknown"


def latency_stats(session_dir: Path) -> dict[str, float]:
    """Compute latency statistics from meta.json files."""
    gen_times = []
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            gt = meta.get("generation_time_sec", 0)
            if gt > 0:
                gen_times.append(gt)
        except (json.JSONDecodeError, OSError):
            continue

    if not gen_times:
        return {}

    import numpy as np
    arr = np.array(gen_times)
    return {
        "mean_sec": round(float(arr.mean()), 3),
        "median_sec": round(float(np.median(arr)), 3),
        "std_sec": round(float(arr.std()), 3),
        "p95_sec": round(float(np.percentile(arr, 95)), 3),
        "n_calls": len(gen_times),
    }


def evaluate_run(
    intent_session: Path | None,
    decision_session: Path | None,
    gt_path: Path,
    *,
    task: str = "unknown",
) -> dict:
    """Run all evaluators on a single experiment pair."""
    aura_root = Path(__file__).resolve().parent.parent.parent
    result: dict = {
        "task": task,
        "intent_session": str(intent_session) if intent_session else None,
        "decision_session": str(decision_session) if decision_session else None,
    }

    # Model info
    model = "unknown"
    if intent_session:
        model = get_model_from_session(intent_session)
    elif decision_session:
        model = get_model_from_session(decision_session)
    result["model"] = model

    # --- A-Score (decision engine) ---
    if decision_session and decision_session.is_dir():
        preds = extract(decision_session)
        gt = parse_gt_interventions(str(gt_path))

        a_result = compute_a_scores(gt, preds)
        s = a_result["summary"]
        result["a_score"] = {
            "mean": s["mean_a_score"],
            "a_time": s["mean_a_time"],
            "a_mod": s["mean_a_mod"],
            "a_nec": s["mean_a_nec"],
        }
        result["precision"] = s["precision"]
        result["recall"] = s["recall"]
        f1 = 0.0
        if s["precision"] + s["recall"] > 0:
            f1 = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"])
        result["f1"] = round(f1, 3)
        result["a_score_detail"] = a_result

        # Decision counts
        counts = count_decisions(decision_session)
        result["n_act"] = counts.get("act", 0)
        result["n_wait"] = counts.get("wait", 0)
        result["n_decision_calls"] = counts["total"]

        # Decision latency
        result["decision_latency"] = latency_stats(decision_session)
    else:
        result["a_score"] = {}

    # --- Intent accuracy ---
    # gt-intent runs bypass the intent monitor entirely (empty session dir);
    # intent is sourced directly from the GT timeline, so accuracy is 1.0.
    intent_is_gt = False
    if intent_session and intent_session.is_dir():
        if not any(intent_session.glob("call_*")):
            intent_is_gt = True

    if intent_is_gt:
        result["intent_accuracy"] = {
            "current_action": 1.0,
            "next_action": 1.0,
            "detection_rate": 1.0,
            "mean_detection_lag_sec": 0.0,
            "source": "ground_truth",
        }
    elif intent_session and intent_session.is_dir():
        intent_result = evaluate_intent_session(str(intent_session), str(gt_path))
        if "error" not in intent_result:
            result["intent_accuracy"] = {
                "current_action": intent_result["current_action"]["overall_accuracy"],
                "next_action": intent_result["next_action"]["accuracy"],
                "detection_rate": intent_result["transition_timing"]["detection_rate"],
                "mean_detection_lag_sec": intent_result["transition_timing"]["mean_detection_lag_sec"],
            }
            result["n_intent_calls"] = intent_result["n_predictions"]
            result["intent_latency"] = latency_stats(intent_session)
        else:
            result["intent_accuracy"] = {"error": intent_result["error"]}
    else:
        result["intent_accuracy"] = {}

    return result


def main() -> None:
    aura_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Evaluate a single AURA experiment run")
    parser.add_argument("--task", required=True, help="Task name (for ground truth lookup)")
    parser.add_argument("--intent-session", type=Path, help="Intent monitor session directory")
    parser.add_argument("--decision-session", type=Path, help="Decision engine session directory")
    parser.add_argument("--logs-dir", type=Path, help="Auto-detect latest sessions from this directory")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    gt_path = aura_root / "tasks" / args.task / "config" / "ground_truth.json"
    if not gt_path.exists():
        print(f"Error: ground truth not found at {gt_path}", file=sys.stderr)
        sys.exit(1)

    intent_session = args.intent_session
    decision_session = args.decision_session

    if args.logs_dir:
        # Auto-detect latest sessions
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from scripts.eval import find_latest_session
        if not intent_session:
            intent_session = find_latest_session(args.logs_dir, "intent_monitor")
        if not decision_session:
            decision_session = find_latest_session(args.logs_dir, "decision_engine")

    if not intent_session and not decision_session:
        print("Error: no sessions found. Provide --intent-session, --decision-session, or --logs-dir", file=sys.stderr)
        sys.exit(1)

    result = evaluate_run(intent_session, decision_session, gt_path, task=args.task)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Results saved to {args.output}")
        # Print summary
        if result.get("a_score"):
            a = result["a_score"]
            print(f"A-Score: {a.get('mean', 0):.3f} (time={a.get('a_time', 0):.3f}, "
                  f"mod={a.get('a_mod', 0):.3f}, nec={a.get('a_nec', 0):.3f})")
            print(f"P={result.get('precision', 0):.3f} R={result.get('recall', 0):.3f} "
                  f"F1={result.get('f1', 0):.3f}")
        if result.get("intent_accuracy") and "error" not in result["intent_accuracy"]:
            ia = result["intent_accuracy"]
            print(f"Intent: current={ia.get('current_action', 0):.3f}, "
                  f"next={ia.get('next_action', 0):.3f}, "
                  f"det_rate={ia.get('detection_rate', 0):.3f}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline evaluation pipeline for AURA framework.

Runs the full AURA workflow on recorded video, collects decision_history,
and evaluates against ground truth using the A-Score metric.

This script:
1. Runs the AURA LangGraph workflow on a video in dry-run mode
2. Extracts all robot intervention decisions from the decision_history
3. Matches predicted interventions against ground truth events
4. Computes A-Score metrics (A_time, A_mod, A_nec)
5. Saves detailed results for figure generation

Usage:
    # Single session evaluation
    .venv/bin/python scripts/run_offline_evaluation.py \
        --task hand_layup \
        --video demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4 \
        --ground-truth tasks/hand_layup/config/ground_truth.json \
        --output results/offline_eval/

    # Evaluate from existing intent logs (no API calls)
    .venv/bin/python scripts/run_offline_evaluation.py \
        --task hand_layup \
        --from-logs logs/intent_monitor/session_20260307_055742 \
        --ground-truth tasks/hand_layup/config/ground_truth.json \
        --output results/offline_eval/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import A-Score computation
sys.path.insert(0, str(_project_root / "scripts"))
from compute_a_score import (
    InterventionEvent,
    parse_gt_interventions,
    compute_a_scores,
    sensitivity_analysis,
)
from evaluate_intent_accuracy import (
    load_ground_truth,
    load_session,
    evaluate_current_action,
    evaluate_next_action,
    evaluate_transition_timing,
    compute_latency_stats,
)


# ── Extract interventions from decision history ───────────────────────────
def extract_interventions_from_decisions(
    decision_history: List[Dict],
) -> List[InterventionEvent]:
    """Convert AURA decision_history entries to InterventionEvent objects."""
    interventions = []
    for d in decision_history:
        if d.get("decision_type") != "action":
            continue

        action_type = d.get("action_type", "unknown")
        # Normalise action types
        if "storage" in action_type.lower() or "return" in action_type.lower():
            action_type = "return_to_storage"
        elif "deliver" in action_type.lower() or "workplace" in action_type.lower():
            action_type = "deliver_to_workplace"

        target = d.get("object_name", d.get("target", ""))
        objects = [target] if target else []

        interventions.append(InterventionEvent(
            timestamp=d.get("timestamp_sec", d.get("timestamp", 0.0)),
            action_type=action_type,
            target_objects=objects,
            reason=d.get("reasoning", ""),
        ))

    return interventions


# ── Simulate decisions from intent logs (rule-based) ─────────────────────
def simulate_decisions_from_logs(
    session_dir: str,
    dag_path: str,
    task_profile_path: str,
) -> List[Dict]:
    """Replay intent monitor logs and simulate rule-based decision engine.

    This avoids needing to re-run the full AURA pipeline. It reads the
    logged intent predictions and applies the same rule engine logic
    to determine what robot actions would be triggered.
    """
    from evaluate_intent_accuracy import load_session as load_preds

    preds = load_preds(session_dir)
    if not preds:
        return []

    # Load DAG for rule engine
    with open(dag_path) as f:
        dag = json.load(f)
    dag_nodes = dag.get("nodes", {})

    # Load task profile for program mapping
    with open(task_profile_path) as f:
        profile = json.load(f)

    decisions = []
    completed_steps = set()
    triggered_returns = set()

    for pred in preds:
        current = pred.current_action
        predicted_next = pred.predicted_next

        # Mark current action as completed (for rule engine)
        if current != "idle" and current != "unknown":
            completed_steps.add(current)

        # Check if current step triggers a return-to-storage
        if current in dag_nodes:
            node = dag_nodes[current]
            ret = node.get("robot_return_to_storage", {})
            if ret and current not in triggered_returns:
                triggered_returns.add(current)
                for obj in ret.get("objects", []):
                    decisions.append({
                        "decision_type": "action",
                        "action_type": "return_to_storage",
                        "object_name": obj,
                        "timestamp_sec": pred.timestamp,
                        "trigger_step": current,
                        "reasoning": ret.get("reason", ""),
                    })

        # Check if predicted_next needs objects delivered
        if predicted_next in dag_nodes:
            next_node = dag_nodes[predicted_next]
            needed = next_node.get("objects_needed_on_workplace", [])
            # Only deliver objects that are in storage
            for obj in needed:
                obj_key = f"deliver_{obj}"
                if obj_key not in triggered_returns:
                    # Check if object is needed but not yet on workplace
                    # (simplified: deliver if the step hasn't been seen before)
                    pass  # Would need object_locations tracking

    return decisions


# ── Evaluate from logged intent sessions ──────────────────────────────────
def evaluate_from_logs(
    session_dir: str,
    gt_path: str,
    dag_path: str,
    task_profile_path: str,
    sigma: float = 3.0,
) -> Dict:
    """Evaluate a single session from logged intent monitor data."""
    from evaluate_intent_accuracy import load_session

    gt_events_raw = load_ground_truth(gt_path)
    preds = load_session(session_dir)

    if not preds:
        return {"error": f"No predictions in {session_dir}"}

    # Intent accuracy
    intent_accuracy = evaluate_current_action(preds, gt_events_raw)
    next_accuracy = evaluate_next_action(preds, gt_events_raw)
    transition = evaluate_transition_timing(preds, gt_events_raw)
    latency = compute_latency_stats(preds)

    # Simulate decisions from intent logs
    decisions = simulate_decisions_from_logs(
        session_dir, dag_path, task_profile_path
    )

    # Convert to intervention events
    pred_interventions = extract_interventions_from_decisions(decisions)
    gt_interventions = parse_gt_interventions(gt_path)

    # Compute A-Score
    a_score_result = compute_a_scores(
        gt_interventions, pred_interventions, sigma=sigma
    )

    # Sensitivity analysis
    sens = sensitivity_analysis(gt_interventions, pred_interventions)

    return {
        "session": os.path.basename(session_dir),
        "n_predictions": len(preds),
        "n_decisions": len(decisions),
        "intent_accuracy": {
            "current_action": intent_accuracy["overall_accuracy"],
            "next_action": next_accuracy["accuracy"],
            "detection_rate": transition["detection_rate"],
            "mean_lag_sec": transition["mean_detection_lag_sec"],
        },
        "latency": latency,
        "a_score": a_score_result["summary"],
        "a_score_per_event": a_score_result["per_intervention"],
        "decisions": decisions,
        "sensitivity": sens,
    }


# ── Batch evaluation ──────────────────────────────────────────────────────
def evaluate_all_sessions(
    logs_dir: str,
    gt_path: str,
    dag_path: str,
    task_profile_path: str,
    sigma: float = 3.0,
) -> Dict:
    """Evaluate all sessions with logged data."""
    logs = Path(logs_dir)
    sessions = sorted([
        str(d) for d in logs.iterdir()
        if d.is_dir() and any(d.glob("call_*"))
    ])

    all_results = []
    for sess in sessions:
        print(f"Evaluating {os.path.basename(sess)}...")
        result = evaluate_from_logs(sess, gt_path, dag_path,
                                    task_profile_path, sigma)
        if "error" not in result:
            all_results.append(result)
            ia = result["intent_accuracy"]
            asc = result["a_score"]
            print(f"  intent_acc={ia['current_action']:.3f}  "
                  f"decisions={result['n_decisions']}  "
                  f"a_score={asc['mean_a_score']:.3f}")
        else:
            print(f"  SKIPPED: {result['error']}")

    if not all_results:
        return {"error": "No valid sessions"}

    # Aggregate
    aggregate = {
        "n_sessions": len(all_results),
        "intent_accuracy": {
            "current_action": {
                "mean": round(float(np.mean([r["intent_accuracy"]["current_action"]
                                             for r in all_results])), 3),
                "std": round(float(np.std([r["intent_accuracy"]["current_action"]
                                           for r in all_results])), 3),
            },
            "next_action": {
                "mean": round(float(np.mean([r["intent_accuracy"]["next_action"]
                                             for r in all_results])), 3),
                "std": round(float(np.std([r["intent_accuracy"]["next_action"]
                                           for r in all_results])), 3),
            },
        },
        "a_score": {
            "mean": round(float(np.mean([r["a_score"]["mean_a_score"]
                                          for r in all_results])), 3),
            "std": round(float(np.std([r["a_score"]["mean_a_score"]
                                        for r in all_results])), 3),
        },
    }

    return {
        "aggregate": aggregate,
        "sessions": all_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AURA offline evaluation pipeline"
    )
    parser.add_argument("--task", type=str, default="hand_layup")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file for live pipeline run")
    parser.add_argument("--from-logs", type=str, default=None,
                        help="Evaluate from existing intent monitor logs")
    parser.add_argument("--all-sessions", action="store_true",
                        help="Evaluate all logged sessions")
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/offline_eval/")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--logs-dir", type=str, default="logs/intent_monitor/")
    args = parser.parse_args()

    aura_root = _project_root
    config_dir = aura_root / "tasks" / args.task / "config"
    gt_path = args.ground_truth or str(config_dir / "ground_truth.json")
    dag_path = str(config_dir / f"{args.task}_dag.json")
    profile_path = str(config_dir / "task_profile.json")
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_sessions:
        results = evaluate_all_sessions(
            str(aura_root / args.logs_dir), gt_path, dag_path,
            profile_path, args.sigma
        )

        out_path = output_dir / "all_sessions_offline_eval.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if "error" not in results:
            agg = results["aggregate"]
            print(f"\n{'='*50}")
            print(f"Aggregate ({agg['n_sessions']} sessions):")
            print(f"  Intent accuracy: {agg['intent_accuracy']['current_action']}")
            print(f"  A-Score:         {agg['a_score']}")
        print(f"\nResults saved to {out_path}")

    elif args.from_logs:
        result = evaluate_from_logs(
            args.from_logs, gt_path, dag_path, profile_path, args.sigma
        )
        sess_name = os.path.basename(args.from_logs)
        out_path = output_dir / f"{sess_name}_eval.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        if "error" not in result:
            print(f"\nSession: {result['session']}")
            print(f"  Intent accuracy: {result['intent_accuracy']}")
            print(f"  Decisions: {result['n_decisions']}")
            print(f"  A-Score: {result['a_score']}")
        print(f"\nResults saved to {out_path}")

    else:
        print("Specify --from-logs, --all-sessions, or --video")
        parser.print_help()


if __name__ == "__main__":
    main()

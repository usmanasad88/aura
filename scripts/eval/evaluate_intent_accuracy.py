#!/usr/bin/env python3
"""Evaluate AURA intent monitor predictions against ground truth.

Loads logged intent monitor sessions (response_parsed.json + meta.json)
and compares current_action / predicted_next_action against the annotated
ground truth timeline.

Outputs:
  - Per-action detection accuracy (precision, recall, F1)
  - Next-action prediction accuracy
  - Temporal detection error (when each step transition is detected vs GT)
  - Confusion matrix data
  - JSON results file for downstream plotting

Usage:
    python evaluate_intent_accuracy.py \
        --session logs/intent_monitor/session_20260307_055742 \
        --ground-truth tasks/hand_layup/config/ground_truth.json \
        --output results/intent_evaluation/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Name normalisation ─────────────────────────────────────────────────────
# Ground truth uses short names; DAG / intent monitor uses longer names.
GT_TO_DAG = {
    "recording_start": "idle",
    "add_resin": "add_resin_to_cup",
    "add_hardener": "add_hardener_to_cup",
}

def normalise_action(name: str) -> str:
    """Canonicalise action name to the DAG vocabulary."""
    return GT_TO_DAG.get(name, name)


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class GTEvent:
    timestamp: float
    action: str  # normalised
    raw_action: str
    robot_action: Optional[str] = None

@dataclass
class Prediction:
    timestamp: float
    current_action: str
    predicted_next: str
    confidence: float
    generation_time: float

@dataclass
class TransitionDetection:
    """Record of when the system detected a specific action starting."""
    action: str
    gt_timestamp: float
    detected_timestamp: Optional[float] = None  # None if never detected
    error_sec: Optional[float] = None
    detected: bool = False


# ── Loading ────────────────────────────────────────────────────────────────
def load_ground_truth(path: str) -> List[GTEvent]:
    with open(path) as f:
        data = json.load(f)
    events = []
    for ev in data["events"]:
        # Support both "timestamp" (legacy) and "start_time" (current) formats
        timestamp = ev.get("start_time", ev.get("timestamp", 0.0))
        events.append(GTEvent(
            timestamp=timestamp,
            action=normalise_action(ev["action"]),
            raw_action=ev["action"],
            robot_action=ev.get("robot_action"),
        ))
    return events


def load_session(session_dir: str) -> List[Prediction]:
    session = Path(session_dir)
    preds = []
    for call_dir in sorted(session.glob("call_*")):
        meta_path = call_dir / "meta.json"
        parsed_path = call_dir / "response_parsed.json"
        if not meta_path.exists() or not parsed_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(parsed_path) as f:
            parsed = json.load(f)

        preds.append(Prediction(
            timestamp=meta["timestamp_sec"],
            current_action=parsed.get("current_action", "unknown"),
            predicted_next=parsed.get("predicted_next_action", "unknown"),
            confidence=parsed.get("prediction_confidence",
                                  parsed.get("action_confidence", 0.0)),
            generation_time=meta.get("generation_time_sec", 0.0),
        ))
    return preds


# ── Evaluation: current-action accuracy ────────────────────────────────────
def get_gt_action_at(gt_events: List[GTEvent], t: float) -> str:
    """Return the ground truth action active at time t.

    The active action is the last GT event whose timestamp <= t.
    """
    active = "idle"
    for ev in gt_events:
        if ev.timestamp <= t:
            active = ev.action
        else:
            break
    return active


def evaluate_current_action(
    preds: List[Prediction],
    gt_events: List[GTEvent],
) -> Dict:
    """Compare each prediction's current_action against the GT action at that time."""
    # Collect all unique actions
    all_actions = sorted(set(
        [ev.action for ev in gt_events] +
        [p.current_action for p in preds]
    ))

    correct = 0
    total = len(preds)
    per_action_tp: Dict[str, int] = {a: 0 for a in all_actions}
    per_action_fp: Dict[str, int] = {a: 0 for a in all_actions}
    per_action_fn: Dict[str, int] = {a: 0 for a in all_actions}
    confusion: List[Dict] = []

    for pred in preds:
        gt_action = get_gt_action_at(gt_events, pred.timestamp)
        pred_action = pred.current_action

        confusion.append({
            "timestamp": pred.timestamp,
            "predicted": pred_action,
            "ground_truth": gt_action,
            "correct": pred_action == gt_action,
        })

        if pred_action == gt_action:
            correct += 1
            per_action_tp[gt_action] = per_action_tp.get(gt_action, 0) + 1
        else:
            per_action_fp[pred_action] = per_action_fp.get(pred_action, 0) + 1
            per_action_fn[gt_action] = per_action_fn.get(gt_action, 0) + 1

    # Per-action precision / recall / F1
    per_action_metrics = {}
    for a in all_actions:
        tp = per_action_tp.get(a, 0)
        fp = per_action_fp.get(a, 0)
        fn = per_action_fn.get(a, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_action_metrics[a] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return {
        "overall_accuracy": round(correct / max(total, 1), 3),
        "total_predictions": total,
        "correct": correct,
        "per_action": per_action_metrics,
        "confusion_log": confusion,
    }


# ── Evaluation: next-action prediction ─────────────────────────────────────
def evaluate_next_action(
    preds: List[Prediction],
    gt_events: List[GTEvent],
) -> Dict:
    """Evaluate predicted_next_action accuracy.

    For each prediction at time t, find the actual next GT action that
    will happen after t and check if predicted_next matches.
    """
    correct = 0
    total = 0
    details = []

    for pred in preds:
        # Find the next GT event after this prediction's timestamp
        next_gt = None
        for ev in gt_events:
            if ev.timestamp > pred.timestamp:
                next_gt = ev
                break
        if next_gt is None:
            continue

        total += 1
        match = pred.predicted_next == next_gt.action
        if match:
            correct += 1

        details.append({
            "pred_timestamp": pred.timestamp,
            "predicted_next": pred.predicted_next,
            "actual_next": next_gt.action,
            "actual_next_at": next_gt.timestamp,
            "lookahead_sec": round(next_gt.timestamp - pred.timestamp, 2),
            "correct": match,
        })

    return {
        "accuracy": round(correct / max(total, 1), 3),
        "total": total,
        "correct": correct,
        "details": details,
    }


# ── Evaluation: transition detection timing ────────────────────────────────
def evaluate_transition_timing(
    preds: List[Prediction],
    gt_events: List[GTEvent],
) -> Dict:
    """For each GT action, find when the intent monitor first detected it.

    Measures temporal lag between GT event timestamp and the first prediction
    whose current_action matches.
    """
    detections: List[TransitionDetection] = []

    for ev in gt_events:
        if ev.action in ("idle", "task_complete"):
            continue

        td = TransitionDetection(
            action=ev.action,
            gt_timestamp=ev.timestamp,
        )

        # Find earliest prediction with this current_action
        for pred in preds:
            if pred.current_action == ev.action:
                td.detected_timestamp = pred.timestamp
                td.error_sec = round(pred.timestamp - ev.timestamp, 2)
                td.detected = True
                break

        detections.append(td)

    detected = [d for d in detections if d.detected]
    errors = [d.error_sec for d in detected if d.error_sec is not None]

    return {
        "total_actions": len(detections),
        "detected": len(detected),
        "missed": len(detections) - len(detected),
        "detection_rate": round(len(detected) / max(len(detections), 1), 3),
        "mean_detection_lag_sec": round(float(np.mean(errors)), 2) if errors else None,
        "std_detection_lag_sec": round(float(np.std(errors)), 2) if errors else None,
        "max_detection_lag_sec": round(float(np.max(errors)), 2) if errors else None,
        "min_detection_lag_sec": round(float(np.min(errors)), 2) if errors else None,
        "per_action": [asdict(d) for d in detections],
    }


# ── Evaluation: latency statistics ─────────────────────────────────────────
def compute_latency_stats(preds: List[Prediction]) -> Dict:
    gen_times = [p.generation_time for p in preds if p.generation_time > 0]
    if not gen_times:
        return {"error": "No generation time data"}
    return {
        "mean_sec": round(float(np.mean(gen_times)), 3),
        "std_sec": round(float(np.std(gen_times)), 3),
        "min_sec": round(float(np.min(gen_times)), 3),
        "max_sec": round(float(np.max(gen_times)), 3),
        "median_sec": round(float(np.median(gen_times)), 3),
        "n_calls": len(gen_times),
    }


# ── Multi-session aggregation ──────────────────────────────────────────────
def aggregate_sessions(session_results: List[Dict]) -> Dict:
    """Aggregate metrics across multiple sessions."""
    accuracies = [r["current_action"]["overall_accuracy"] for r in session_results]
    next_accs = [r["next_action"]["accuracy"] for r in session_results]
    det_rates = [r["transition_timing"]["detection_rate"] for r in session_results]
    lags = [r["transition_timing"]["mean_detection_lag_sec"]
            for r in session_results
            if r["transition_timing"]["mean_detection_lag_sec"] is not None]

    return {
        "n_sessions": len(session_results),
        "current_action_accuracy": {
            "mean": round(float(np.mean(accuracies)), 3),
            "std": round(float(np.std(accuracies)), 3),
        },
        "next_action_accuracy": {
            "mean": round(float(np.mean(next_accs)), 3),
            "std": round(float(np.std(next_accs)), 3),
        },
        "detection_rate": {
            "mean": round(float(np.mean(det_rates)), 3),
            "std": round(float(np.std(det_rates)), 3),
        },
        "mean_detection_lag_sec": {
            "mean": round(float(np.mean(lags)), 2) if lags else None,
            "std": round(float(np.std(lags)), 2) if lags else None,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────
def evaluate_session(session_dir: str, gt_path: str) -> Dict:
    gt_events = load_ground_truth(gt_path)
    preds = load_session(session_dir)

    if not preds:
        return {"error": f"No predictions found in {session_dir}"}

    return {
        "session": os.path.basename(session_dir),
        "n_predictions": len(preds),
        "n_gt_events": len(gt_events),
        "time_span_sec": round(preds[-1].timestamp - preds[0].timestamp, 1),
        "current_action": evaluate_current_action(preds, gt_events),
        "next_action": evaluate_next_action(preds, gt_events),
        "transition_timing": evaluate_transition_timing(preds, gt_events),
        "latency": compute_latency_stats(preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AURA intent predictions")
    parser.add_argument("--session", type=str, help="Path to a single session directory")
    parser.add_argument("--all-sessions", action="store_true",
                        help="Evaluate all sessions in logs/intent_monitor/")
    parser.add_argument("--ground-truth", type=str,
                        default="tasks/hand_layup/config/ground_truth.json")
    parser.add_argument("--output", type=str, default="results/intent_evaluation/")
    parser.add_argument("--logs-dir", type=str,
                        default="logs/intent_monitor/")
    args = parser.parse_args()

    # Resolve paths relative to aura root
    aura_root = Path(__file__).resolve().parent.parent.parent
    gt_path = str(aura_root / args.ground_truth)
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_sessions:
        logs_dir = aura_root / args.logs_dir
        sessions = sorted([
            str(d) for d in logs_dir.iterdir()
            if d.is_dir() and any(d.glob("call_*"))
        ])
        if not sessions:
            print("No sessions with data found.")
            sys.exit(1)

        all_results = []
        for sess in sessions:
            print(f"Evaluating {os.path.basename(sess)}...")
            result = evaluate_session(sess, gt_path)
            if "error" not in result:
                all_results.append(result)
                print(f"  accuracy={result['current_action']['overall_accuracy']}, "
                      f"next_acc={result['next_action']['accuracy']}, "
                      f"det_rate={result['transition_timing']['detection_rate']}")
            else:
                print(f"  SKIPPED: {result['error']}")

        if all_results:
            aggregate = aggregate_sessions(all_results)
            output = {
                "aggregate": aggregate,
                "sessions": all_results,
            }
            out_path = output_dir / "all_sessions_evaluation.json"
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nAggregate results ({aggregate['n_sessions']} sessions):")
            print(f"  Current action accuracy: {aggregate['current_action_accuracy']}")
            print(f"  Next action accuracy:    {aggregate['next_action_accuracy']}")
            print(f"  Detection rate:          {aggregate['detection_rate']}")
            print(f"  Mean detection lag:      {aggregate['mean_detection_lag_sec']}")
            print(f"\nResults saved to {out_path}")

    elif args.session:
        result = evaluate_session(args.session, gt_path)
        out_path = output_dir / f"{os.path.basename(args.session)}_evaluation.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Session: {result['session']}")
            print(f"  Predictions: {result['n_predictions']}")
            print(f"  Current action accuracy: {result['current_action']['overall_accuracy']}")
            print(f"  Next action accuracy:    {result['next_action']['accuracy']}")
            print(f"  Detection rate:          {result['transition_timing']['detection_rate']}")
            print(f"  Mean detection lag:      {result['transition_timing']['mean_detection_lag_sec']}s")
            print(f"  LLM latency (mean):      {result['latency'].get('mean_sec', 'N/A')}s")
            print(f"\nResults saved to {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

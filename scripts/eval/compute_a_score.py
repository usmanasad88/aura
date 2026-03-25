#!/usr/bin/env python3
"""Compute the Appropriateness Score (A-Score) for proactive interventions.

Implements the three-component metric from Section 3 of the paper:
  A_score = w_t * A_time + w_m * A_mod + w_n * A_nec

Components:
  A_time  - Timeliness: Gaussian decay from optimal intervention window
  A_mod   - Modality: Does the predicted action type match ground truth?
  A_nec   - Necessity: Was the intervention actually needed at that time?

Inputs:
  - Ground truth events (with robot_action annotations)
  - AURA predicted interventions (from decision_history or offline eval)

Usage:
    python compute_a_score.py \
        --ground-truth tasks/hand_layup/config/ground_truth.json \
        --predictions results/offline_eval/session_predictions.json \
        --output results/a_score/

    # Or generate synthetic example from ground truth alone:
    python compute_a_score.py \
        --ground-truth tasks/hand_layup/config/ground_truth.json \
        --demo \
        --output results/a_score/
"""

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InterventionEvent:
    """A single ground truth or predicted intervention."""
    timestamp: float
    action_type: str        # e.g. "deliver_to_workplace", "return_to_storage"
    target_objects: List[str]  # e.g. ["resin_bottle"]
    reason: Optional[str] = None

@dataclass
class AScoreResult:
    """A-Score for a single intervention."""
    gt_event: Dict
    pred_event: Optional[Dict]
    a_time: float
    a_mod: float
    a_nec: float
    a_score: float
    timing_error_sec: Optional[float]
    matched: bool


# ── Parse ground truth interventions ───────────────────────────────────────
def parse_gt_interventions(gt_path: str) -> List[InterventionEvent]:
    """Extract robot intervention events from ground truth."""
    with open(gt_path) as f:
        data = json.load(f)

    interventions = []
    for ev in data["events"]:
        ra = ev.get("robot_action")
        if ra is None:
            continue

        # Parse robot_action string: "move_to_storage(obj1, obj2)" or "deliver_to_workplace(obj)"
        action_type = "return_to_storage" if "storage" in ra else "deliver_to_workplace"
        # Extract objects from parentheses
        if "(" in ra and ")" in ra:
            obj_str = ra[ra.index("(") + 1:ra.index(")")]
            objects = [o.strip() for o in obj_str.split(",")]
        else:
            objects = []

        interventions.append(InterventionEvent(
            timestamp=ev["timestamp"],
            action_type=action_type,
            target_objects=objects,
            reason=ev.get("robot_action_reason"),
        ))

    return interventions


# ── A_time: Timeliness ─────────────────────────────────────────────────────
def compute_a_time(
    pred_time: float,
    gt_time: float,
    sigma: float = 3.0,
) -> float:
    """Gaussian decay centered on GT timestamp.

    A_time = exp(-0.5 * ((t_pred - t_gt) / sigma)^2)

    sigma controls the tolerance window width:
      sigma=2.0 → ~95% score within ±4s
      sigma=3.0 → ~95% score within ±6s
      sigma=5.0 → ~95% score within ±10s
    """
    error = pred_time - gt_time
    return math.exp(-0.5 * (error / sigma) ** 2)


# ── A_mod: Modality ────────────────────────────────────────────────────────
def compute_a_mod(
    pred_action_type: str,
    gt_action_type: str,
    pred_objects: List[str],
    gt_objects: List[str],
) -> float:
    """Evaluate whether the type and target of assistance was correct.

    Returns:
      1.0 if action type and objects both match
      0.5 if action type matches but objects differ
      0.0 if action type is wrong
    """
    if pred_action_type != gt_action_type:
        return 0.0

    # Check object overlap
    pred_set = set(pred_objects)
    gt_set = set(gt_objects)
    if not gt_set:
        return 1.0
    overlap = len(pred_set & gt_set) / len(gt_set)
    return 0.5 + 0.5 * overlap


# ── A_nec: Necessity ──────────────────────────────────────────────────────
def compute_a_nec(
    gt_event: InterventionEvent,
    pred_event: Optional[InterventionEvent],
    is_false_positive: bool = False,
) -> float:
    """Evaluate whether the intervention was genuinely needed.

    Returns:
      1.0 for matched GT interventions (by definition needed)
      0.5 for matched but early (proactive, possibly useful)
      0.0 for false positives (no GT event to match)
    """
    if is_false_positive:
        return 0.0

    if pred_event is None:
        # Missed intervention — necessity was high but system failed to act
        return 1.0  # The need was there

    # If the prediction was significantly early, it's still useful but less certain
    if pred_event.timestamp < gt_event.timestamp - 5.0:
        return 0.5

    return 1.0


# ── Matching ───────────────────────────────────────────────────────────────
def match_predictions_to_gt(
    gt_interventions: List[InterventionEvent],
    pred_interventions: List[InterventionEvent],
    max_window: float = 10.0,
) -> List[Tuple[InterventionEvent, Optional[InterventionEvent]]]:
    """Match predicted interventions to ground truth using greedy nearest-first.

    Returns list of (gt_event, matched_pred_or_None).
    Also returns unmatched predictions as false positives.
    """
    matched_preds = set()
    matches: List[Tuple[InterventionEvent, Optional[InterventionEvent]]] = []

    for gt in gt_interventions:
        best_pred = None
        best_error = float("inf")

        for i, pred in enumerate(pred_interventions):
            if i in matched_preds:
                continue
            error = abs(pred.timestamp - gt.timestamp)
            if error < best_error and error <= max_window:
                best_error = error
                best_pred = (i, pred)

        if best_pred is not None:
            matched_preds.add(best_pred[0])
            matches.append((gt, best_pred[1]))
        else:
            matches.append((gt, None))

    # Collect false positives (unmatched predictions)
    false_positives = [
        pred_interventions[i]
        for i in range(len(pred_interventions))
        if i not in matched_preds
    ]

    return matches, false_positives


# ── Full A-Score computation ───────────────────────────────────────────────
def compute_a_scores(
    gt_interventions: List[InterventionEvent],
    pred_interventions: List[InterventionEvent],
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    sigma: float = 3.0,
    max_window: float = 10.0,
) -> Dict:
    """Compute A-Score for all intervention pairs."""
    w_t, w_m, w_n = weights

    matches, false_positives = match_predictions_to_gt(
        gt_interventions, pred_interventions, max_window
    )

    results: List[AScoreResult] = []

    for gt, pred in matches:
        if pred is not None:
            a_time = compute_a_time(pred.timestamp, gt.timestamp, sigma)
            a_mod = compute_a_mod(pred.action_type, gt.action_type,
                                  pred.target_objects, gt.target_objects)
            a_nec = compute_a_nec(gt, pred)
            timing_error = round(pred.timestamp - gt.timestamp, 2)
        else:
            # Missed intervention
            a_time = 0.0
            a_mod = 0.0
            a_nec = 1.0  # Need was there
            timing_error = None

        a_score = w_t * a_time + w_m * a_mod + w_n * a_nec

        results.append(AScoreResult(
            gt_event=asdict(gt),
            pred_event=asdict(pred) if pred else None,
            a_time=round(a_time, 3),
            a_mod=round(a_mod, 3),
            a_nec=round(a_nec, 3),
            a_score=round(a_score, 3),
            timing_error_sec=timing_error,
            matched=pred is not None,
        ))

    # False positive penalties
    fp_results = []
    for fp in false_positives:
        fp_results.append({
            "pred_event": asdict(fp),
            "a_time": 0.0,
            "a_mod": 0.0,
            "a_nec": 0.0,
            "a_score": 0.0,
            "false_positive": True,
        })

    # Aggregate
    matched_results = [r for r in results if r.matched]
    all_scores = [r.a_score for r in results]
    matched_scores = [r.a_score for r in matched_results]
    timing_errors = [r.timing_error_sec for r in matched_results
                     if r.timing_error_sec is not None]

    summary = {
        "weights": {"w_t": w_t, "w_m": w_m, "w_n": w_n},
        "sigma": sigma,
        "n_gt_interventions": len(gt_interventions),
        "n_pred_interventions": len(pred_interventions),
        "n_matched": len(matched_results),
        "n_missed": len(results) - len(matched_results),
        "n_false_positives": len(false_positives),
        "precision": round(len(matched_results) / max(len(pred_interventions), 1), 3),
        "recall": round(len(matched_results) / max(len(gt_interventions), 1), 3),
        "mean_a_score": round(float(np.mean(all_scores)), 3) if all_scores else 0.0,
        "mean_a_score_matched": round(float(np.mean(matched_scores)), 3) if matched_scores else 0.0,
        "mean_a_time": round(float(np.mean([r.a_time for r in matched_results])), 3) if matched_results else 0.0,
        "mean_a_mod": round(float(np.mean([r.a_mod for r in matched_results])), 3) if matched_results else 0.0,
        "mean_a_nec": round(float(np.mean([r.a_nec for r in matched_results])), 3) if matched_results else 0.0,
        "mean_timing_error_sec": round(float(np.mean(timing_errors)), 2) if timing_errors else None,
        "std_timing_error_sec": round(float(np.std(timing_errors)), 2) if timing_errors else None,
    }

    return {
        "summary": summary,
        "per_intervention": [asdict(r) for r in results],
        "false_positives": fp_results,
    }


# ── Sensitivity analysis ──────────────────────────────────────────────────
def sensitivity_analysis(
    gt_interventions: List[InterventionEvent],
    pred_interventions: List[InterventionEvent],
    sigma_values: List[float] = None,
    weight_configs: Dict[str, Tuple[float, float, float]] = None,
) -> Dict:
    """Run A-Score with varying parameters for sensitivity analysis."""
    if sigma_values is None:
        sigma_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    if weight_configs is None:
        weight_configs = {
            "equal": (1/3, 1/3, 1/3),
            "timeliness_focused": (0.5, 0.25, 0.25),
            "modality_focused": (0.25, 0.5, 0.25),
            "necessity_focused": (0.25, 0.25, 0.5),
        }

    sigma_sweep = []
    for sigma in sigma_values:
        result = compute_a_scores(gt_interventions, pred_interventions,
                                  sigma=sigma)
        sigma_sweep.append({
            "sigma": sigma,
            "mean_a_score": result["summary"]["mean_a_score"],
            "mean_a_time": result["summary"]["mean_a_time"],
        })

    weight_sweep = []
    for name, weights in weight_configs.items():
        result = compute_a_scores(gt_interventions, pred_interventions,
                                  weights=weights)
        weight_sweep.append({
            "config": name,
            "weights": {"w_t": weights[0], "w_m": weights[1], "w_n": weights[2]},
            "mean_a_score": result["summary"]["mean_a_score"],
        })

    return {
        "sigma_sensitivity": sigma_sweep,
        "weight_sensitivity": weight_sweep,
    }


# ── Demo mode: generate synthetic predictions with timing offsets ──────────
def generate_demo_predictions(
    gt_interventions: List[InterventionEvent],
    timing_offsets: List[float] = None,
) -> Dict[str, List[InterventionEvent]]:
    """Generate synthetic prediction sets to demonstrate the A-Score metric.

    Returns multiple prediction sets simulating different system behaviors:
    - "perfect": Exact matches
    - "early": Predictions arrive early
    - "late": Predictions arrive late
    - "noisy": Random timing jitter
    - "wrong_modality": Correct timing but wrong action type
    - "reactive": Only acts when prompted (late by 5-15s)
    """
    rng = np.random.default_rng(42)

    scenarios = {}

    # Perfect predictions
    scenarios["perfect"] = [
        InterventionEvent(
            timestamp=gt.timestamp,
            action_type=gt.action_type,
            target_objects=gt.target_objects,
        ) for gt in gt_interventions
    ]

    # Early predictions (proactive, -1 to -4s)
    scenarios["early_proactive"] = [
        InterventionEvent(
            timestamp=max(0, gt.timestamp + rng.uniform(-4.0, -1.0)),
            action_type=gt.action_type,
            target_objects=gt.target_objects,
        ) for gt in gt_interventions
    ]

    # Late predictions (+2 to +8s)
    scenarios["late_reactive"] = [
        InterventionEvent(
            timestamp=gt.timestamp + rng.uniform(2.0, 8.0),
            action_type=gt.action_type,
            target_objects=gt.target_objects,
        ) for gt in gt_interventions
    ]

    # Noisy (±3s Gaussian jitter)
    scenarios["noisy"] = [
        InterventionEvent(
            timestamp=max(0, gt.timestamp + rng.normal(0, 3.0)),
            action_type=gt.action_type,
            target_objects=gt.target_objects,
        ) for gt in gt_interventions
    ]

    # Wrong modality (correct timing, swapped action type)
    scenarios["wrong_modality"] = [
        InterventionEvent(
            timestamp=gt.timestamp + rng.uniform(-1.0, 1.0),
            action_type="deliver_to_workplace" if gt.action_type == "return_to_storage" else "return_to_storage",
            target_objects=gt.target_objects,
        ) for gt in gt_interventions
    ]

    # Reactive baseline (always late by 5-15s, only 60% recall)
    reactive = []
    for gt in gt_interventions:
        if rng.random() < 0.6:
            reactive.append(InterventionEvent(
                timestamp=gt.timestamp + rng.uniform(5.0, 15.0),
                action_type=gt.action_type,
                target_objects=gt.target_objects,
            ))
    scenarios["reactive_baseline"] = reactive

    # Fixed schedule (delivers at fixed task fractions regardless of state)
    total_time = max(gt.timestamp for gt in gt_interventions)
    n_deliveries = len(gt_interventions)
    scenarios["fixed_schedule"] = [
        InterventionEvent(
            timestamp=(i + 1) * total_time / (n_deliveries + 1),
            action_type=gt_interventions[min(i, len(gt_interventions)-1)].action_type,
            target_objects=gt_interventions[min(i, len(gt_interventions)-1)].target_objects,
        ) for i in range(n_deliveries)
    ]

    return scenarios


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute A-Score metrics")
    parser.add_argument("--ground-truth", type=str,
                        default="tasks/hand_layup/config/ground_truth.json")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predicted interventions JSON")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic predictions")
    parser.add_argument("--output", type=str, default="results/a_score/")
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()

    aura_root = Path(__file__).resolve().parent.parent
    gt_path = str(aura_root / args.ground_truth)
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_interventions = parse_gt_interventions(gt_path)
    print(f"Ground truth: {len(gt_interventions)} intervention events")
    for gt in gt_interventions:
        print(f"  t={gt.timestamp:5.1f}s  {gt.action_type:<25s}  objects={gt.target_objects}")

    if args.demo:
        # Generate and evaluate multiple synthetic scenarios
        scenarios = generate_demo_predictions(gt_interventions)
        all_results = {}

        print(f"\n{'Scenario':<22s}  {'A-Score':>8s}  {'A_time':>7s}  {'A_mod':>6s}  {'A_nec':>6s}  {'Prec':>5s}  {'Rec':>5s}  {'ΔT(s)':>6s}")
        print("-" * 80)

        for name, preds in scenarios.items():
            result = compute_a_scores(gt_interventions, preds, sigma=args.sigma)
            s = result["summary"]
            print(f"{name:<22s}  {s['mean_a_score']:>8.3f}  {s['mean_a_time']:>7.3f}  "
                  f"{s['mean_a_mod']:>6.3f}  {s['mean_a_nec']:>6.3f}  "
                  f"{s['precision']:>5.3f}  {s['recall']:>5.3f}  "
                  f"{s['mean_timing_error_sec'] or 0:>6.2f}")
            all_results[name] = result

        # Sensitivity analysis
        print("\n--- Sigma sensitivity (perfect predictions) ---")
        sens = sensitivity_analysis(gt_interventions, scenarios["noisy"])
        for sv in sens["sigma_sensitivity"]:
            print(f"  sigma={sv['sigma']:5.1f}  A-Score={sv['mean_a_score']:.3f}  A_time={sv['mean_a_time']:.3f}")

        print("\n--- Weight sensitivity (noisy predictions) ---")
        for wv in sens["weight_sensitivity"]:
            print(f"  {wv['config']:<25s}  A-Score={wv['mean_a_score']:.3f}")

        output = {
            "scenarios": {name: result for name, result in all_results.items()},
            "sensitivity": sens,
        }
        out_path = output_dir / "demo_a_scores.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    elif args.predictions:
        # Load actual predictions
        with open(args.predictions) as f:
            pred_data = json.load(f)

        pred_interventions = [
            InterventionEvent(**p) for p in pred_data["interventions"]
        ]
        result = compute_a_scores(gt_interventions, pred_interventions,
                                  sigma=args.sigma)
        s = result["summary"]
        print(f"\nA-Score Results:")
        print(f"  Mean A-Score:     {s['mean_a_score']:.3f}")
        print(f"  Mean A_time:      {s['mean_a_time']:.3f}")
        print(f"  Mean A_mod:       {s['mean_a_mod']:.3f}")
        print(f"  Mean A_nec:       {s['mean_a_nec']:.3f}")
        print(f"  Precision:        {s['precision']:.3f}")
        print(f"  Recall:           {s['recall']:.3f}")
        print(f"  Timing error:     {s['mean_timing_error_sec']}s")

        out_path = output_dir / "a_score_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

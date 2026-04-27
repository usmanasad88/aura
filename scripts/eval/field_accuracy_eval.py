"""Per-field accuracy metric over intent-monitor predictions vs. ground truth.

Complements ``temporal_f1_state_eval`` (which scores the step lists and
``predicted_next_action``) by grading the *categorical* and *boolean* fields
of the tea_making schema. Each scored field contributes one binary "item"
per prediction frame; overall accuracy is the fraction of items correct.

Scored fields
-------------
Categorical (string equality):
    current_phase, current_action, human_state, robot_engagement
Boolean (equality):
    water_in_pot, pot_on_stove, stove_on, water_boiling,
    chai_added, milk_added, tea_poured, sugar_added, tea_served

Skipped fields
--------------
steps_completed, steps_in_progress (covered by temporal_f1_state_eval),
steps_pending, predicted_next_action, prediction_confidence, reasoning

Usage
-----
::

    # Default: score every predictions.json under
    # logs/Tea_Making_Results/ against tasks/tea_making/ground_truth/tea_making.intent_gt.json
    python scripts/eval/field_accuracy_eval.py

    # Score one specific predictions file
    python scripts/eval/field_accuracy_eval.py \\
        --gt   tasks/tea_making/ground_truth/tea_making.intent_gt.json \\
        --pred logs/run_.../predictions.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from temporal_f1_state_eval import get_state_at, load_entries


AURA_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRED_DIR = AURA_ROOT / "logs" / "Tea_Making_Results"
DEFAULT_GT_PATH = AURA_ROOT / "tasks" / "tea_making" / "ground_truth" / "tea_making.intent_gt.json"


CATEGORICAL_FIELDS: tuple[str, ...] = (
    "current_phase",
    "current_action",
    "human_state",
    "robot_engagement",
)

BOOLEAN_FIELDS: tuple[str, ...] = (
    "water_in_pot",
    "pot_on_stove",
    "stove_on",
    "water_boiling",
    "chai_added",
    "milk_added",
    "tea_poured",
    "sugar_added",
    "tea_served",
)

# ── Helpers ────────────────────────────────────────────────────────────────

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_values[int(k)])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def compute_latency_stats(pred_path: str | Path) -> dict[str, Any]:
    """Compile per-call latency stats from sibling ``intent_monitor/call_*/meta.json``.

    Returns counts, total/mean/median/min/max/std and p90/p95/p99 (seconds).
    Empty dict if no meta files are found.
    """
    monitor_dir = Path(pred_path).parent / "intent_monitor"
    if not monitor_dir.is_dir():
        return {}

    latencies: list[float] = []
    missing = 0
    for meta_path in sorted(monitor_dir.glob("call_*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue
        v = meta.get("generation_time_sec")
        if v is None:
            missing += 1
            continue
        latencies.append(float(v))

    if not latencies:
        return {}

    sorted_lat = sorted(latencies)
    n = len(latencies)
    mean = sum(latencies) / n
    var = sum((x - mean) ** 2 for x in latencies) / n
    std = math.sqrt(var)
    median = _percentile(sorted_lat, 50)

    return {
        "n_calls": n,
        "missing": missing,
        "total_sec": round(sum(latencies), 4),
        "mean_sec": round(mean, 4),
        "median_sec": round(median, 4),
        "std_sec": round(std, 4),
        "min_sec": round(sorted_lat[0], 4),
        "max_sec": round(sorted_lat[-1], 4),
        "p90_sec": round(_percentile(sorted_lat, 90), 4),
        "p95_sec": round(_percentile(sorted_lat, 95), 4),
        "p99_sec": round(_percentile(sorted_lat, 99), 4),
    }


def _matches_with_grace(
    pred_value: Any,
    gt_state: dict[str, Any],
    grace_state: dict[str, Any],
    field: str,
) -> bool:
    """Equality with look-ahead: pred matches if it equals GT now or at grace."""
    if pred_value == gt_state.get(field):
        return True
    if grace_state and pred_value == grace_state.get(field):
        return True
    return False


# ── Core ───────────────────────────────────────────────────────────────────

def evaluate(
    gt_path: str | Path,
    pred_path: str | Path,
    grace_frames: int = 30,
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute per-field and overall accuracy.

    For each prediction frame we look up the GT state at that frame and at
    ``frame + grace_frames``. A categorical/boolean field is correct if the
    predicted value equals the GT value at *either* of those frames. For a
    set field, every step in the GT vocabulary is one item: correct iff
    "step is in pred set" matches "step is in GT set" (effective GT being
    the union of the now-state and the grace-state).

    Returns a dict with::

        {
          "per_field": { field: {"correct": int, "total": int, "accuracy": float}, ... },
          "overall": {"correct": int, "total": int, "accuracy": float},
          "n_predictions": int,
          "grace_frames": int,
          "latency": { "n_calls", "mean_sec", "median_sec", "p95_sec", ... },
        }
    """
    gt_entries = load_entries(gt_path)

    with open(pred_path, "r") as f:
        raw_preds = json.load(f)
    pred_entries: list[dict[str, Any]]
    if isinstance(raw_preds, list):
        pred_entries = sorted(
            (
                {
                    "frame_number": int(
                        e.get("frame_number") or e.get("frame_num") or e.get("frame") or 0
                    ),
                    "state": e.get("state") or {},
                }
                for e in raw_preds
            ),
            key=lambda e: e["frame_number"],
        )
    else:
        raise ValueError(f"Predictions must be a JSON list; got {type(raw_preds).__name__}")

    selected = list(fields) if fields else list(CATEGORICAL_FIELDS + BOOLEAN_FIELDS)

    counts: dict[str, dict[str, int]] = {f: {"correct": 0, "total": 0} for f in selected}

    for pred in pred_entries:
        frame = pred["frame_number"]
        pred_state = pred.get("state") or {}
        gt_state = get_state_at(gt_entries, frame)
        grace_state = get_state_at(gt_entries, frame + grace_frames) if grace_frames else {}

        for fld in selected:
            if fld not in gt_state:
                # GT doesn't have this field at this frame — skip.
                continue
            counts[fld]["total"] += 1
            if _matches_with_grace(pred_state.get(fld), gt_state, grace_state, fld):
                counts[fld]["correct"] += 1

    per_field: dict[str, dict[str, float]] = {}
    overall_correct = 0
    overall_total = 0
    for fld, c in counts.items():
        total = c["total"]
        correct = c["correct"]
        per_field[fld] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
        overall_correct += correct
        overall_total += total

    return {
        "per_field": per_field,
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": round(overall_correct / overall_total, 4) if overall_total else 0.0,
        },
        "n_predictions": len(pred_entries),
        "grace_frames": grace_frames,
        "latency": compute_latency_stats(pred_path),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print(f"n_predictions: {result['n_predictions']}")
    print(f"grace_frames:  {result['grace_frames']}")
    print()
    print(f"{'field':<25s} {'correct':>9s} {'total':>9s} {'accuracy':>10s}")
    print("-" * 56)

    groups = [
        ("Categorical", CATEGORICAL_FIELDS),
        ("Boolean",     BOOLEAN_FIELDS),
    ]
    pf = result["per_field"]
    for label, group in groups:
        printed_header = False
        for fld in group:
            if fld not in pf:
                continue
            if not printed_header:
                print(f"[{label}]")
                printed_header = True
            v = pf[fld]
            print(f"  {fld:<23s} {v['correct']:>9d} {v['total']:>9d} {v['accuracy']:>10.4f}")

    o = result["overall"]
    print("-" * 56)
    print(f"  {'overall':<23s} {o['correct']:>9d} {o['total']:>9d} {o['accuracy']:>10.4f}")

    lat = result.get("latency") or {}
    if lat:
        print()
        print("[Latency (sec)]")
        print(f"  n_calls={lat['n_calls']}  missing={lat['missing']}  total={lat['total_sec']:.2f}")
        print(
            f"  mean={lat['mean_sec']:.3f}  median={lat['median_sec']:.3f}  "
            f"std={lat['std_sec']:.3f}  min={lat['min_sec']:.3f}  max={lat['max_sec']:.3f}"
        )
        print(
            f"  p90={lat['p90_sec']:.3f}  p95={lat['p95_sec']:.3f}  p99={lat['p99_sec']:.3f}"
        )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-field accuracy of intent-monitor predictions vs. ground truth",
    )
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT_PATH,
                        help=f"Ground-truth JSON (default: {DEFAULT_GT_PATH})")
    parser.add_argument("--pred", type=Path, default=None,
                        help="Predictions JSON. If omitted, every predictions.json "
                             f"under {DEFAULT_PRED_DIR} is scored.")
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_DIR,
                        help=f"Root dir to scan for predictions.json when --pred "
                             f"is not given (default: {DEFAULT_PRED_DIR})")
    parser.add_argument("--grace-frames", type=int, default=30,
                        help="Look-ahead tolerance window in frames (default 30)")
    parser.add_argument("--output", type=Path, default=None,
                        help="If given, also write the full result(s) as JSON here")
    parser.add_argument("--fields", nargs="*", default=None,
                        help="Restrict scoring to a subset of fields")
    args = parser.parse_args()

    if args.pred is not None:
        pred_paths = [args.pred]
    else:
        if not args.pred_root.is_dir():
            raise SystemExit(f"Predictions root does not exist: {args.pred_root}")
        pred_paths = sorted(args.pred_root.rglob("predictions.json"))
        if not pred_paths:
            raise SystemExit(f"No predictions.json found under {args.pred_root}")

    multi = len(pred_paths) > 1
    all_results: list[dict[str, Any]] = []

    for pred_path in pred_paths:
        if multi:
            label = pred_path.relative_to(args.pred_root).parent
            print(f"\n=== {label} ===")
        result = evaluate(args.gt, pred_path, grace_frames=args.grace_frames, fields=args.fields)
        result["pred_path"] = str(pred_path)
        result["gt_path"] = str(args.gt)
        _print_report(result)
        all_results.append(result)

    if multi:
        print("\n" + "=" * 78)
        print(
            f"{'run':<35s} {'overall acc':>10s} {'n_pred':>8s} "
            f"{'lat mean':>9s} {'lat p95':>9s}"
        )
        print("-" * 78)
        for r in all_results:
            label = str(Path(r["pred_path"]).relative_to(args.pred_root).parent)
            lat = r.get("latency") or {}
            mean = f"{lat['mean_sec']:.3f}" if lat else "-"
            p95 = f"{lat['p95_sec']:.3f}" if lat else "-"
            print(
                f"{label:<35s} {r['overall']['accuracy']:>10.4f} "
                f"{r['n_predictions']:>8d} {mean:>9s} {p95:>9s}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = all_results[0] if not multi else {"results": all_results}
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    _main()

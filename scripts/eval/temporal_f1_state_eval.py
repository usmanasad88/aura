"""Temporal F1 metric over per-frame task-state predictions.

Port of the hcdt ``temporal_f1_state_eval`` module, adapted for the AURA
intent-monitor log format and the ``<task>.intent_gt.json`` ground-truth
schema produced by ``scripts/annotate_ground_truth.py``.

The evaluator scores each prediction's set of state members (e.g.
``steps_completed``, ``steps_in_progress``) against the ground-truth state
active at that frame, granting a configurable look-ahead window so that
slightly-early predictions are not penalised.

Expected shapes
---------------
Ground truth: either the keyframe-wrapped form

    {"keyframes": [{"frame_num": N, "timestamp_sec": T, "state": {...}}, ...]}

or a flat list

    [{"frame": N, "state": {...}}, ...]

Predictions: flat list in the same shape as the flat GT (produced by
:func:`session_to_predictions.session_to_predictions`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


# Per-key weights used in the combined F1 score. Mirrors the hcdt weights
# but maps ``steps_available`` -> ``steps_pending`` and
# ``immediate_next_step`` -> ``predicted_next_action`` to match the AURA
# intent monitor's output schema.
DEFAULT_WEIGHTS: dict[str, float] = {
    "steps_completed": 0.6,
    "steps_in_progress": 0.2,
    "steps_pending": 0.1,
    "predicted_next_action": 0.1,
}


# ── Ground-truth loading ────────────────────────────────────────────────────

def _normalise_entries(data: Any) -> list[dict[str, Any]]:
    """Return a list of ``{frame_number, state}`` entries from any known shape."""
    if isinstance(data, dict) and "keyframes" in data:
        entries = data["keyframes"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unrecognised GT/predictions shape: {type(data).__name__}")

    out: list[dict[str, Any]] = []
    for e in entries:
        frame = e.get("frame_number") or e.get("frame_num") or e.get("frame") or 0
        out.append({
            "frame_number": int(frame),
            "state": e.get("state") or {},
        })
    out.sort(key=lambda e: e["frame_number"])
    return out


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load a GT or predictions file and normalise its entries."""
    with open(path, "r") as f:
        return _normalise_entries(json.load(f))


def get_state_at(entries: list[dict[str, Any]], frame_number: int) -> dict[str, Any]:
    """Return the GT state active at ``frame_number`` (last entry ≤ frame)."""
    active: dict[str, Any] = {}
    for e in entries:
        if e["frame_number"] <= frame_number:
            active = e["state"]
        else:
            break
    return active


# ── Action extraction ───────────────────────────────────────────────────────

def extract_actions(state: dict[str, Any] | None, key: str | Iterable[str]) -> set[str]:
    """Pull a set of action strings from a state dict for the given key(s)."""
    if not state:
        return set()

    keys = [key] if isinstance(key, str) else list(key)
    actions: set[str] = set()
    for k in keys:
        value = state.get(k)
        if value is None or value == "null":
            continue
        if isinstance(value, list):
            actions.update(str(v) for v in value if v not in (None, "null"))
        else:
            actions.add(str(value))
    return actions


# ── F1 computation ──────────────────────────────────────────────────────────

def calculate_segmental_f1(
    gt_path: str | Path,
    pred_path: str | Path,
    key: str | Iterable[str],
    grace_frames: int = 30,
) -> tuple[float, float, float]:
    """F1 of predicted state[key] against GT, with a look-ahead grace window.

    Parameters
    ----------
    gt_path, pred_path
        Paths to ground-truth and predictions JSON files (any supported shape).
    key
        State field name(s) to compare. Passing an iterable unions the values.
    grace_frames
        Frames to look ahead in the GT when scoring a prediction. A predicted
        action counts as a true positive if it is present in either the GT at
        the same frame or the GT at ``frame + grace_frames``.

    Returns
    -------
    (f1, precision, recall)
    """
    gt_entries = load_entries(gt_path)
    pred_entries = load_entries(pred_path)

    tp = fp = fn = 0
    for pred in pred_entries:
        frame = pred["frame_number"]
        pred_state = pred["state"]

        gt_state = get_state_at(gt_entries, frame)
        grace_state = get_state_at(gt_entries, frame + grace_frames)

        pred_actions = extract_actions(pred_state, key)
        gt_actions = extract_actions(gt_state, key)
        grace_actions = extract_actions(grace_state, key)
        effective_gt = gt_actions | grace_actions

        for a in pred_actions:
            if a in effective_gt:
                tp += 1
            else:
                fp += 1

        for a in gt_actions:
            if a not in pred_actions:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def combined_f1(
    gt_path: str | Path,
    pred_path: str | Path,
    grace_frames: int = 30,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-key F1s plus a weighted combined score.

    Returns a dict with one entry per weighted key (``f1_<key>``, plus
    ``precision_<key>`` / ``recall_<key>``) and an overall ``combined_f1``.
    """
    w = weights or DEFAULT_WEIGHTS
    out: dict[str, float] = {}
    combined = 0.0
    for k, weight in w.items():
        f1, p, r = calculate_segmental_f1(gt_path, pred_path, k, grace_frames)
        out[f"f1_{k}"] = round(f1, 4)
        out[f"precision_{k}"] = round(p, 4)
        out[f"recall_{k}"] = round(r, 4)
        combined += weight * f1
    out["combined_f1"] = round(combined, 4)
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Temporal F1 for AURA intent predictions")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth JSON")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSON")
    parser.add_argument("--grace-frames", type=int, default=30,
                        help="Look-ahead tolerance window in frames (default 30)")
    args = parser.parse_args()

    scores = combined_f1(args.gt, args.pred, grace_frames=args.grace_frames)
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    _main()

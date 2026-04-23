"""Ground-truth intent provider.

Loads sparse keyframe annotations produced by
``scripts/annotate_ground_truth.py`` and returns an ``IntentResult`` for
any frame — the last keyframe whose ``frame_num`` is <= the requested
one. Shape matches ``AURAIntentMonitor.predict()`` so it is a drop-in
substitute for ``run_intent_node`` in evaluation mode.

Only vision-observable variables are served; ``source: system`` and
``source: perception`` variables are intentionally absent and should be
filled by their respective providers upstream (same contract as the
real intent monitor).
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

from aura.monitors.intent_monitor import IntentResult


logger = logging.getLogger(__name__)


def default_gt_path(task_config_dir: str | Path, video_path: str | Path) -> Path:
    """Resolve the canonical GT file path for a (task, video) pair."""
    cfg = Path(task_config_dir)
    task_dir = cfg.parent if cfg.name == "config" else cfg
    stem = Path(video_path).stem
    return task_dir / "ground_truth" / f"{stem}.intent_gt.json"


class GroundTruthIntentProvider:
    """Frame-indexed lookup over sparse intent annotations.

    Parameters
    ----------
    gt_path:
        Path to a ``*.intent_gt.json`` produced by the annotator.
    """

    def __init__(self, gt_path: str | Path):
        self.gt_path = Path(gt_path)
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground-truth file not found: {self.gt_path}")

        data = json.loads(self.gt_path.read_text(encoding="utf-8"))
        self.video: str = data.get("video", "")
        self.task: str = data.get("task", "")
        self.fps: float = float(data.get("fps", 0.0) or 0.0)

        keyframes: List[Dict[str, Any]] = data.get("keyframes", []) or []
        keyframes.sort(key=lambda k: int(k["frame_num"]))
        self._frames: List[int] = [int(k["frame_num"]) for k in keyframes]
        self._states: List[Dict[str, Any]] = [
            dict(k.get("state", {})) for k in keyframes
        ]
        logger.info(
            "Loaded %d intent GT keyframes from %s", len(self._frames), self.gt_path
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _state_at_frame(self, frame_num: int) -> Optional[Dict[str, Any]]:
        if not self._frames:
            return None
        idx = bisect_right(self._frames, int(frame_num)) - 1
        if idx < 0:
            return None
        return self._states[idx]

    def get_at_frame(
        self,
        frame_num: int,
        timestamp_sec: Optional[float] = None,
    ) -> IntentResult:
        """Return the most recent annotation at or before ``frame_num``.

        If no annotation exists at or before the frame, a default
        ``IntentResult`` (all defaults, ``reasoning='no_ground_truth'``)
        is returned so callers don't have to handle ``None``.
        """
        state = self._state_at_frame(frame_num)
        ts = (
            float(timestamp_sec)
            if timestamp_sec is not None
            else (frame_num / self.fps if self.fps > 0 else 0.0)
        )

        if state is None:
            r = IntentResult(timestamp=ts, frame_num=int(frame_num))
            r.reasoning = "no_ground_truth"
            return r

        r = IntentResult(timestamp=ts, frame_num=int(frame_num))
        r.state = dict(state)
        r.current_phase = state.get("current_phase", "initialization")
        r.current_action = state.get("current_action", "idle")
        r.human_state = state.get("human_state", "idle")
        r.steps_completed = list(state.get("steps_completed", []) or [])
        r.steps_in_progress = list(state.get("steps_in_progress", []) or [])
        r.steps_pending = list(state.get("steps_pending", []) or [])
        r.predicted_next_action = state.get("predicted_next_action", "unknown")
        try:
            r.prediction_confidence = float(
                state.get("prediction_confidence", 1.0) or 0.0
            )
        except (TypeError, ValueError):
            r.prediction_confidence = 1.0
        r.reasoning = state.get("reasoning", "ground_truth")
        return r

    def get_at_timestamp(self, timestamp_sec: float) -> IntentResult:
        """Convenience: convert timestamp → frame_num via ``fps``."""
        if self.fps <= 0:
            raise ValueError("fps not set in GT file; use get_at_frame instead")
        frame = int(round(timestamp_sec * self.fps))
        return self.get_at_frame(frame, timestamp_sec=timestamp_sec)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_keyframes(self) -> int:
        return len(self._frames)

    @property
    def keyframe_numbers(self) -> List[int]:
        return list(self._frames)

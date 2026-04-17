"""Activity gate — keypoint-based motion thresholding for intent dispatch.

Tracks 2D pose keypoints across cycles and decides whether the observed
human motion is significant enough to warrant an intent prediction.

Policy:
  * **First detection**: any human presence triggers activity (no prior
    keypoints to compare against).
  * **Subsequent cycles**: compute mean Euclidean displacement of valid
    keypoints between the current and previous pose snapshot.  Activity
    is flagged only when displacement exceeds a configurable threshold
    (in pixels).

This keeps the expensive VLM intent call from firing when the human is
stationary or only exhibiting noise-level jitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default pixel-displacement threshold.  For a 1080p frame a standing
# person is ~500 px tall; breathing / jitter is ~2-5 px, a deliberate
# reach is 30-100 px.  15 px is a reasonable default that filters noise
# while catching meaningful movements.
_DEFAULT_THRESHOLD_PX = 15.0


@dataclass
class ActivityResult:
    """Outcome of a single activity-gate evaluation."""
    detected: bool
    reason: str
    kind: str  # "first_detection" | "significant_motion" | "insufficient_motion" | "no_human" | ...
    displacement_px: Optional[float] = None
    threshold_px: Optional[float] = None


class ActivityGate:
    """Stateful gate that tracks keypoint displacement across cycles.

    One instance is created per config key (lazy-singleton in nodes.py).
    """

    def __init__(self, threshold_px: float = _DEFAULT_THRESHOLD_PX) -> None:
        self.threshold_px = threshold_px
        self._prev_keypoints: Optional[List[np.ndarray]] = None  # [N,2] per person
        self._first_seen = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, pose_output: Dict[str, Any]) -> ActivityResult:
        """Decide whether the current pose snapshot constitutes activity.

        Parameters
        ----------
        pose_output:
            The ``monitor_outputs["pose"]`` dict produced by
            ``run_pose_node``.  Expected keys: ``available``,
            ``num_persons``, ``persons`` (list of dicts with optional
            ``keypoints_2d`` numpy arrays).

        Returns
        -------
        ActivityResult with detection flag, reason, and displacement.
        """
        # Pose server down → bypass gate (always trigger).
        if pose_output.get("available") is False:
            return ActivityResult(
                detected=True,
                reason="pose unavailable — assume activity",
                kind="pose_unavailable",
            )

        num_persons = int(pose_output.get("num_persons", 0) or 0)
        if num_persons == 0:
            return ActivityResult(
                detected=False,
                reason="no human",
                kind="no_human",
            )

        current_kpts = self._extract_keypoints(pose_output)

        # First time we see a person → always trigger.
        if not self._first_seen:
            self._first_seen = True
            self._prev_keypoints = current_kpts
            return ActivityResult(
                detected=True,
                reason=f"first detection: {num_persons} person(s)",
                kind="first_detection",
            )

        # Compute displacement against previous snapshot.
        displacement = self._compute_displacement(self._prev_keypoints, current_kpts)
        self._prev_keypoints = current_kpts  # update baseline

        if displacement is None:
            # Can't compare (missing keypoints on one side) → trigger
            # to be safe rather than silently blocking intent.
            return ActivityResult(
                detected=True,
                reason="no keypoint baseline for comparison",
                kind="no_baseline",
            )

        if displacement >= self.threshold_px:
            return ActivityResult(
                detected=True,
                reason=f"keypoint displacement {displacement:.1f}px >= {self.threshold_px:.0f}px",
                kind="significant_motion",
                displacement_px=displacement,
                threshold_px=self.threshold_px,
            )

        return ActivityResult(
            detected=False,
            reason=f"keypoint displacement {displacement:.1f}px < {self.threshold_px:.0f}px",
            kind="insufficient_motion",
            displacement_px=displacement,
            threshold_px=self.threshold_px,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keypoints(pose_output: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        """Pull keypoints_2d arrays from the pose persons list."""
        persons = pose_output.get("persons") or []
        if not persons:
            return None
        kpts_list: List[np.ndarray] = []
        for p in persons:
            kpts = p.get("keypoints_2d")
            if kpts is not None:
                arr = np.asarray(kpts, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                kpts_list.append(arr)
        return kpts_list if kpts_list else None

    @staticmethod
    def _compute_displacement(
        prev: Optional[List[np.ndarray]],
        curr: Optional[List[np.ndarray]],
    ) -> Optional[float]:
        """Mean L2 displacement of valid keypoints for the primary person.

        Uses person index 0.  If person counts differ between frames
        (someone enters/leaves), returns ``None`` which the caller
        treats as "can't compare → trigger".
        """
        if prev is None or curr is None:
            return None
        if len(prev) == 0 or len(curr) == 0:
            return None

        p = prev[0]
        c = curr[0]
        if p.shape != c.shape:
            return None

        # Valid = both frames have non-zero coordinates for this keypoint.
        valid = ((p[:, 0] != 0) | (p[:, 1] != 0)) & ((c[:, 0] != 0) | (c[:, 1] != 0))
        if valid.sum() == 0:
            return None

        displacements = np.linalg.norm(c[valid] - p[valid], axis=1)
        return float(displacements.mean())

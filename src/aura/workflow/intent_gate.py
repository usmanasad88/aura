"""Intent-monitor gating and frame-sampling policy.

The intent monitor issues long (multi-second) LLM calls. To keep the
fast perception loop responsive we run it asynchronously in the
background, and we only issue a ``predict()`` when it's actually useful.

This module owns two concerns:

* :func:`should_run_intent` — the policy that decides *whether* the
  intent monitor should fire on a given snapshot of state. The current
  rule is "run the intent monitor when the pose monitor sees at least
  one person", but the function is the single hook to extend later
  (e.g. also require gesture activity, SSG change, robot idle, …).

* :func:`sample_intent_frames` — selects which frames from the shared
  buffer get sent to the LLM. For offline video sources the buffer is
  already decimated by ``frame_skip`` at source level so we just take
  the tail uniformly; for realtime / webcam / GoPro / screen sources
  the buffer contains every captured frame, so we walk backwards picking
  frames spaced ``frame_skip`` apart by frame number.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


# ─── Gate ───────────────────────────────────────────────────────────────────

def should_run_intent(state: Dict[str, Any]) -> Tuple[bool, str]:
    """Return ``(run, reason)`` — whether to invoke the intent monitor now.

    Parameters
    ----------
    state:
        The current ``AuraGraphState``.

    Rules (in order):
      1. If no frames have been captured yet → don't run.
      2. If an activity monitor (``activity`` in ``active_monitors``) is
         configured, require ``activity_detected`` — i.e. the activity
         node saw a human (or, in future, classified meaningful motion).
      3. If pose is configured but activity is not, fall back to the raw
         pose count so we still gate on human presence.
      4. If neither gate is configured (or the pose server is down) we
         always run — better to call intent blindly than block forever.
    """
    buf = state.get("frames_buffer") or []
    if not buf:
        return False, "no frames yet"

    config = state.get("config", {}) or {}
    active = set(config.get("active_monitors") or [])
    monitor_outs = state.get("monitor_outputs") or {}

    # Activity monitor takes precedence — it's the new gate.
    if "activity" in active:
        if state.get("activity_detected"):
            activity_out = monitor_outs.get("activity") or {}
            return True, activity_out.get("reason", "activity detected")
        return False, "no activity detected"

    # Pose-only fallback (legacy configs).
    if "pose" in active:
        pose_out = monitor_outs.get("pose") or {}
        if pose_out and pose_out.get("available") is False:
            return True, "pose unavailable — gate bypassed"
        num_persons = int(pose_out.get("num_persons", 0) or 0)
        if num_persons <= 0:
            return False, "no human detected"
        return True, f"{num_persons} person(s) present"

    return True, "no gate configured"


# ─── Frame sampling ─────────────────────────────────────────────────────────

def sample_intent_frames(
    frames: Sequence[Any],
    frame_nums: Sequence[int],
    timestamps: Sequence[float],
    *,
    n: int,
    frame_skip: int,
    realtime: bool,
) -> Tuple[List[Any], List[int], List[float]]:
    """Pick up to ``n`` frames from the buffer for the intent prompt.

    Parameters
    ----------
    frames, frame_nums, timestamps:
        Parallel sequences — the rolling buffer maintained by
        ``capture_frame_node``.
    n:
        Maximum number of frames to return.
    frame_skip:
        Target spacing between selected frames, measured in **source frame
        numbers** (not buffer indices).
    realtime:
        When False, the source is a pre-recorded video file — decimation
        already happened at source open time, so the buffer is uniform and
        we just take the tail. When True, the source delivers every raw
        frame and we must re-decimate by ``frame_skip``.

    Returns
    -------
    ``(frames_out, frame_nums_out, timestamps_out)`` in chronological
    order (oldest first), ready to feed into ``monitor.predict``.
    """
    if n <= 0 or not frames:
        return [], [], []

    # Defensive length harmonisation — the three buffers should always move
    # in lockstep but we clip to the shortest just in case.
    count = min(len(frames), len(frame_nums), len(timestamps))
    if count == 0:
        return [], [], []
    frames = list(frames)[-count:]
    frame_nums = list(frame_nums)[-count:]
    timestamps = list(timestamps)[-count:]

    if not realtime:
        # Video file: buffer is already frame-skip-decimated at source.
        return frames[-n:], frame_nums[-n:], timestamps[-n:]

    # Realtime: walk backwards, pick frames spaced >= frame_skip apart.
    picked_idx: List[int] = []
    last_fn = None
    step = max(int(frame_skip), 1)
    for i in range(count - 1, -1, -1):
        fn = int(frame_nums[i])
        if last_fn is None or (last_fn - fn) >= step:
            picked_idx.append(i)
            last_fn = fn
            if len(picked_idx) >= n:
                break

    picked_idx.reverse()
    return (
        [frames[i] for i in picked_idx],
        [frame_nums[i] for i in picked_idx],
        [timestamps[i] for i in picked_idx],
    )

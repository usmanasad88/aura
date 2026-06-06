"""Background runner for the slow intent (RCWPS) monitor.

The LangGraph fast loop (capture → gesture → perception → pose →
update_ssg → check_ssg_change → decide …) must stay responsive so that
every new frame actually reaches the SSG. The intent monitor, which
runs a multi-second VLM call, used to sit inline on that path and
starved everything downstream of it.

:class:`BackgroundIntentRunner` pulls intent out of the graph into its
own asyncio task. It owns no state of its own — on each iteration it
reads the latest :class:`AuraGraphState` snapshot published by the
graph event loop, asks :func:`aura.workflow.intent_gate.should_run_intent`
whether to fire, samples frames via
:func:`aura.workflow.intent_gate.sample_intent_frames`, invokes the
monitor, and hands the result back to the graph via the shared slot
in :mod:`aura.workflow.nodes`.

The runner never mutates the SSG directly; all writes happen from the
graph thread in ``update_ssg_node``, which preserves the existing
single-writer invariant on the SSG.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from .intent_gate import sample_intent_frames, should_run_intent
from .nodes import _get_intent_monitor, push_intent_result, resolve_source_mode

logger = logging.getLogger(__name__)


class BackgroundIntentRunner:
    """Owns the background asyncio task that drives the intent monitor.

    Usage::

        runner = BackgroundIntentRunner(initial_state)
        task = asyncio.create_task(runner.run())
        # … each time LangGraph emits an update, call:
        runner.update_state(latest_state)
        # … on shutdown:
        runner.stop()
        await task
    """

    def __init__(
        self,
        initial_state: Dict[str, Any],
        *,
        idle_poll_sec: float = 0.2,
        min_interval_sec: Optional[float] = None,
    ):
        self._state: Dict[str, Any] = initial_state
        self._stop_flag = False
        self._idle_poll = float(idle_poll_sec)
        # If not provided, fall back to the legacy predict_interval so the
        # runner can't hammer the LLM faster than the task needs.
        self._min_interval = (
            float(min_interval_sec)
            if min_interval_sec is not None
            else float(initial_state.get("config", {}).get("predict_interval", 0.0))
        )
        self._last_call_ts = 0.0

    # ── public API ────────────────────────────────────────────────────

    def update_state(self, state: Dict[str, Any]) -> None:
        """Replace the snapshot the runner reads from.

        Called by the graph driver after each ``astream`` update. We keep
        only a reference — the snapshot itself is a plain dict and is
        immutable from the runner's perspective.
        """
        if state is not None:
            self._state = state

    def stop(self) -> None:
        """Request the runner to exit after the current iteration."""
        self._stop_flag = True

    async def run(self) -> None:
        """Main loop — runs until :meth:`stop` is called or the state
        reports the task as complete.
        """
        logger.info("BackgroundIntentRunner started")
        try:
            while not self._stop_flag:
                if self._state.get("is_complete") or self._state.get("error"):
                    break

                await self._tick()
        except asyncio.CancelledError:
            logger.info("BackgroundIntentRunner cancelled")
            raise
        except Exception:
            logger.exception("BackgroundIntentRunner crashed")
        finally:
            logger.info("BackgroundIntentRunner stopped")

    # ── one iteration ─────────────────────────────────────────────────

    async def _tick(self) -> None:
        state = self._state
        config = state.get("config", {}) or {}

        # Gate: should we run at all right now?
        run_now, reason = should_run_intent(state)
        if not run_now:
            await asyncio.sleep(self._idle_poll)
            return

        # Minimum-interval throttle (optional — defaults to predict_interval).
        now = time.monotonic()
        if self._min_interval > 0 and (now - self._last_call_ts) < self._min_interval:
            await asyncio.sleep(
                max(0.05, self._min_interval - (now - self._last_call_ts))
            )
            return

        frames, frame_nums, timestamps = sample_intent_frames(
            state.get("frames_buffer") or [],
            state.get("frames_buffer_frame_nums") or [],
            state.get("frames_buffer_timestamps") or [],
            n=int(config.get("intent_num_frames", 5)),
            frame_skip=int(config.get("frame_skip", 30)),
            redecimate=resolve_source_mode(config) != "offline_eval",
        )
        if not frames:
            await asyncio.sleep(self._idle_poll)
            return

        try:
            monitor = _get_intent_monitor(state)
        except Exception as exc:
            logger.warning("Intent monitor init failed: %s", exc)
            await asyncio.sleep(self._idle_poll)
            return

        window_duration = (
            timestamps[-1] - timestamps[0] if len(timestamps) >= 2
            else (timestamps[-1] if timestamps else 0.0)
        )
        last_frame_num = frame_nums[-1] if frame_nums else state.get("current_frame_num", 0)

        self._last_call_ts = time.monotonic()
        logger.info(
            "BackgroundIntentRunner predict — %d frames, span=%.2fs (gate: %s)",
            len(frames), window_duration, reason,
        )

        # Inject externally-sourced state (robot status, perception results)
        # so the intent monitor's RCWPS context includes them in its prompt.
        # The monitor's schema determines which workflow state keys are
        # relevant — no task-specific keys are hardcoded here.
        #
        # robot_state / robot_active_program live in ssg.task_state (written
        # by update_ssg_node), not as top-level AuraGraphState keys.
        task_state: dict = dict(state.get("task_state") or {})
        external_vars = monitor.collect_external_state_from_workflow(
            robot_state=task_state.get("robot_state"),
            robot_active_program=task_state.get("robot_active_program"),
            object_locations=dict(state.get("object_locations") or {}),
        )
        if external_vars:
            monitor.inject_external_state(external_vars)
            logger.debug("Intent monitor external state injected: %s", list(external_vars.keys()))

        # Run the (potentially slow) LLM call in a worker thread so we
        # don't block the event loop. The monitor itself is synchronous.
        def _call():
            return monitor.predict(
                frames=frames,
                timestamp=timestamps[-1] if timestamps else state.get("current_timestamp_sec", 0.0),
                frame_num=last_frame_num,
                window_duration_sec=window_duration,
            )

        try:
            result = await asyncio.get_running_loop().run_in_executor(None, _call)
        except Exception as exc:
            logger.warning("Intent predict failed: %s", exc)
            return

        if result is None:
            return

        result_dict = {
            "timestamp": result.timestamp,
            "frame_num": result.frame_num,
            "state": result.state,
            "current_phase": result.current_phase,
            "current_action": result.current_action,
            "human_state": result.human_state,
            "steps_completed": list(result.steps_completed),
            "steps_in_progress": list(result.steps_in_progress),
            "steps_pending": list(result.steps_pending),
            "predicted_next_action": result.predicted_next_action,
            "prediction_confidence": result.prediction_confidence,
            "reasoning": result.reasoning,
            "generation_time_sec": result.generation_time_sec,
        }
        config_dir = config.get("config_dir", "") or "default"
        push_intent_result(config_dir, result_dict)

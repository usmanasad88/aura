"""Reusable LangGraph node functions for AURA workflows.

Each function has the signature ``f(state: AuraGraphState) -> dict``
and returns a *partial* state update.  Functions are stateless —
all mutable context lives in ``AuraGraphState`` fields.

Nodes in this module:

* **capture_frame_node** — read next frame from video/webcam
* **run_gesture_node** — continuous gesture recognition (sets help-requested flag)
* **run_intent_node** — RCWPS intent prediction via ``AURAIntentMonitor`` (time-throttled)
* **update_ssg_node** — sync intent result into the SSG
* **decide_action_node** — hybrid LLM/rule-based decision via ``DecisionEngine``
* **execute_action_node** — dispatch robot program (or dry-run log)
* **check_complete_node** — check task completion / loop condition
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from aura.core.scene_graph import SemanticSceneGraph
    from aura.monitors.intent_monitor import AURAIntentMonitor
    from aura.monitors.gesture_monitor import GestureMonitor
    from aura.brain.decision_engine import DecisionEngine

from .state import AuraGraphState

logger = logging.getLogger(__name__)


# ── Lazy-loaded singletons ──────────────────────────────────────────────────
# These are expensive to create (Gemini client, MediaPipe model, video cap).
# We instantiate them once on first call and cache in module-level dicts keyed
# by a hashable config identity.  Each node function calls a ``_get_*()``
# helper that handles lazy init.

_intent_monitors: Dict[str, Any] = {}
_gesture_monitors: Dict[str, Any] = {}
_perception_monitors: Dict[str, Any] = {}
_decision_engines: Dict[str, Any] = {}
_video_sources: Dict[str, Any] = {}
_robot_clients: Dict[str, Any] = {}
_ssg_instances: Dict[str, Any] = {}
_ground_truth_data_cache: Dict[str, Any] = {}

# Audio bridge singleton — set externally by run_aura.py when audio is enabled.
# The bridge is NOT part of LangGraph state (it's not serialisable); it lives
# here as a module-level reference that nodes can access.
_audio_bridge: Any = None


def set_audio_bridge(bridge) -> None:
    """Register the AudioWorkflowBridge for use by workflow nodes.

    Called once from run_aura.py after the bridge and sound monitor
    are initialised.
    """
    global _audio_bridge
    _audio_bridge = bridge


def get_audio_bridge():
    """Get the registered AudioWorkflowBridge (or None)."""
    return _audio_bridge


def _get_ssg(state: AuraGraphState) -> "SemanticSceneGraph":
    """Get or create the live SSG instance for this task, restoring from
    the serialised ``ssg`` snapshot in state on first call."""
    from aura.core.scene_graph import SemanticSceneGraph

    config = state.get("config", {})
    task_name = config.get("task_name", "default")

    if task_name not in _ssg_instances:
        ssg_data = state.get("ssg")
        if ssg_data and ssg_data.get("nodes"):
            _ssg_instances[task_name] = SemanticSceneGraph.from_dict(cast(Dict[str, Any], ssg_data))
        else:
            _ssg_instances[task_name] = SemanticSceneGraph(name=task_name)
    return _ssg_instances[task_name]


def _get_intent_monitor(state: AuraGraphState) -> "AURAIntentMonitor":
    from aura.monitors.intent_monitor import AURAIntentMonitor

    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    if config_dir not in _intent_monitors:
        # Per-component overrides fall back to shared defaults
        intent_backend = config.get("intent_backend") or config.get("llm_backend", "gemini")
        intent_model = config.get("intent_model") or config.get("model", "gemini-3.1-pro-preview")
        _intent_monitors[config_dir] = AURAIntentMonitor(
            config_dir=config_dir,
            model=intent_model,
            realtime=config.get("realtime", True),
            enable_logging=True,
            llm_backend=intent_backend,
            sglang_base_url=config.get("sglang_base_url", "http://localhost:8100/v1"),
            max_tokens=config.get("intent_max_tokens", 4096),
        )
    return _intent_monitors[config_dir]


def _get_gesture_monitor(state: AuraGraphState) -> "GestureMonitor":
    from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

    config = state.get("config", {})
    task_profile = state.get("task_profile", {})
    wf = task_profile.get("workflow_config", {})

    key = config.get("config_dir", "default")
    if key not in _gesture_monitors:
        _gesture_monitors[key] = GestureMonitor(GestureMonitorConfig(
            resume_gestures=set(wf.get("resume_gestures", ["Thumb_Up"])),
            stop_gestures=set(wf.get("stop_gestures", ["Open_Palm", "Pointing_Up"])),
            gesture_hold_frames=wf.get("gesture_hold_frames", 3),
        ))
    return _gesture_monitors[key]


def _get_perception_monitor(state: AuraGraphState):
    """Lazy-init a task-specific perception monitor (or None)."""
    config = state.get("config", {})
    key = config.get("config_dir", "default")
    if key not in _perception_monitors:
        _perception_monitors[key] = _create_perception_monitor(
            config.get("task_name", ""), config,
        )
    return _perception_monitors[key]


def _create_perception_monitor(task_name: str, config: dict):
    """Factory: import the right perception monitor for *task_name*."""
    normalised = task_name.lower().replace(" ", "_")
    if normalised == "hand_layup":
        try:
            from tasks.hand_layup.perception.layup_perception_monitor import (
                LayupPerceptionMonitor,
            )
            return LayupPerceptionMonitor()
        except ImportError:
            logger.warning("hand_layup perception monitor not found")
            return None
    if normalised == "sorting":
        try:
            from tasks.sorting.perception.sorting_perception_monitor import (
                SortingPerceptionMonitor,
            )
            return SortingPerceptionMonitor()
        except ImportError:
            logger.warning("sorting perception monitor not found")
            return None
    logger.info("No task-specific perception monitor for '%s'", task_name)
    return None


def _get_decision_engine(state: AuraGraphState) -> "DecisionEngine":
    """Lazy-init the Brain DecisionEngine with SSG + SkillRegistry."""
    from aura.brain.decision_engine import DecisionEngine, DecisionEngineConfig

    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    decision_mode = config.get("decision_mode", "hybrid")

    if config_dir not in _decision_engines:
        # Per-component overrides fall back to shared defaults
        decision_backend = config.get("decision_backend") or config.get("llm_backend", "gemini")
        decision_model = config.get("decision_model") or config.get("model", "gemini-2.5-pro-preview-06-05")
        task_profile = state.get("task_profile", {})
        engine_config = DecisionEngineConfig(
            gemini_model=decision_model,
            enable_llm_reasoning=(decision_mode in ("llm", "hybrid")),
            proactive_threshold=0.6,
            max_completion_tokens=config.get("decision_max_tokens", 1024),
            llm_backend=decision_backend,
            sglang_base_url=config.get("sglang_base_url", "http://localhost:8100/v1"),
            task_system_instruction=task_profile.get("system_instruction", ""),
        )
        engine = DecisionEngine(config_dir=config_dir, config=engine_config)

        # Wire in the shared SSG
        engine.graph = _get_ssg(state)

        # Re-create reasoner against the shared SSG
        from aura.core.scene_graph import GraphReasoner
        engine.reasoner = GraphReasoner(engine.graph)

        # Load task artefacts
        p = Path(config_dir)
        dag_path = p / "dag.json"
        state_path = p / "state_schema.json"
        skills_path = p / "robot_skills.json"
        scene_path = p / "initial_scene.json"

        engine.load_task(
            dag_path=str(dag_path),
            state_path=str(state_path) if state_path.exists() else "",
            skills_path=str(skills_path) if skills_path.exists() else "",
            initial_scene_path=str(scene_path) if scene_path.exists() else "",
        )
        _decision_engines[config_dir] = engine
    return _decision_engines[config_dir]


def _get_video_source(state: AuraGraphState):
    """Get or open the video/webcam source."""
    config = state.get("config", {})
    video_path = config.get("video_path")
    webcam_device = config.get("webcam_device")
    realtime = config.get("realtime", True)

    gopro_stream = config.get("gopro_stream", False)
    gopro_ip = config.get("gopro_ip", "172.29.170.51")
    gopro_lens = config.get("gopro_lens", "front")

    screen_capture = config.get("screen_capture", False)
    screen_monitor = config.get("screen_monitor", 1)
    screen_region = config.get("screen_region")

    key = video_path or (
        f"gopro:{gopro_ip}:{gopro_lens}" if gopro_stream
        else f"screen:{screen_monitor}" if screen_capture
        else f"webcam:{webcam_device}"
    )
    if key not in _video_sources:
        if gopro_stream:
            from aura.sources.gopro_stream_source import GoProStreamSource
            source = GoProStreamSource(camera_ip=gopro_ip, lens=gopro_lens)
        elif screen_capture:
            from aura.sources.screen_capture import ScreenCaptureSource
            source = ScreenCaptureSource(
                monitor=screen_monitor,
                region=tuple(screen_region) if screen_region else None,
            )
        elif webcam_device is not None:
            from aura.sources.webcam import WebcamSource
            source = WebcamSource(device=webcam_device)
        elif video_path is None:
            raise ValueError("video_path must be set when webcam_device is not provided")
        elif realtime:
            from aura.sources.realtime_video import RealtimeVideoSource
            source = RealtimeVideoSource(
                path=video_path,
                speed=config.get("speed", 1.0),
            )
        else:
            from aura.sources.video_file import VideoFileSource
            source = VideoFileSource(
                path=video_path,
                frame_skip=config.get("frame_skip", 30),
            )
        source.open()
        _video_sources[key] = source
    return _video_sources[key]


def _get_robot_client(state: AuraGraphState):
    """Get or create the RobotControlClient (None if dry-run)."""
    config = state.get("config", {})
    if config.get("dry_run", True):
        return None

    url = config.get("robot_url", "http://localhost:5050")
    if url not in _robot_clients:
        from aura.interfaces.robot_control_client import RobotControlClient
        client = RobotControlClient(url)
        _robot_clients[url] = client if client.is_available() else None
    return _robot_clients.get(url)


def _get_ground_truth_data(state: AuraGraphState) -> Dict[str, Any] | None:
    """Load and cache the task's ground-truth file.

    Expected format: ``tasks/<task>/config/ground_truth.json`` with an
    ``events`` array and optional ``total_duration_seconds``.
    """
    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    if not config_dir:
        return None

    path = str(Path(config_dir) / "ground_truth.json")
    if path in _ground_truth_data_cache:
        return _ground_truth_data_cache[path]

    gt_path = Path(path)
    if not gt_path.exists():
        logger.warning("Ground-truth robot status requested but file missing: %s", gt_path)
        _ground_truth_data_cache[path] = None
        return None

    try:
        with open(gt_path, "r", encoding="utf-8") as handle:
            gt_data = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read ground truth file %s: %s", gt_path, exc)
        _ground_truth_data_cache[path] = None
        return None

    events = gt_data.get("events", [])
    if not isinstance(events, list):
        logger.warning("Invalid ground truth format (events not a list): %s", gt_path)
        _ground_truth_data_cache[path] = None
        return None

    events = sorted(
        [event for event in events if isinstance(event, dict)],
        key=lambda event: float(event.get("timestamp", 0.0)),
    )
    cached = {
        "events": events,
        "total_duration_seconds": float(gt_data.get("total_duration_seconds", 0.0) or 0.0),
    }
    _ground_truth_data_cache[path] = cached
    return cached


def _robot_status_from_ground_truth(state: AuraGraphState, timestamp_sec: float) -> Dict[str, Any]:
    """Derive robot status at ``timestamp_sec`` from ground-truth events.

    Rules:
    - when the latest robot-tagged event at/under timestamp has a non-null
      ``robot_action``, robot is BUSY running that action;
    - if a completion marker ``<action>_complete`` is observed later, robot
      returns to IDLE with empty active action;
    - otherwise defaults to unknown.
    """
    gt_data = _get_ground_truth_data(state)
    if not gt_data:
        return {
            "robot_state": "unknown",
            "robot_active_program": "",
        }

    events: List[Dict[str, Any]] = gt_data.get("events", [])
    if not events:
        return {
            "robot_state": "unknown",
            "robot_active_program": "",
        }

    active_action = ""
    robot_state = "idle"

    for event in events:
        event_ts = float(event.get("timestamp", 0.0) or 0.0)
        if event_ts > float(timestamp_sec):
            break

        robot_action = event.get("robot_action")
        action_name = str(event.get("action", "") or "")

        if isinstance(robot_action, str) and robot_action.strip():
            active_action = robot_action.strip()
            robot_state = "busy"
            continue

        if active_action and action_name == f"{active_action}_complete":
            active_action = ""
            robot_state = "idle"

    return {
        "robot_state": robot_state,
        "robot_active_program": active_action,
    }


# ── Live robot status via External Control API ────────────────────────────

# Track the last active program so we can detect transitions to idle.
_last_active_program: str = ""


def _robot_status_from_api(state: AuraGraphState) -> Dict[str, Any]:
    """Poll ``GET /api/status`` and derive ``robot_state`` / ``robot_active_program``.

    Heuristics:
    * ``executor_running=False`` → ``idle``  (no executor process)
    * Any joint velocity > threshold → ``busy``
    * Otherwise → ``idle``

    Also returns gripper state and joint positions for downstream use.
    """
    global _last_active_program

    robot = _get_robot_client(state)
    if robot is None:
        return {"robot_state": "unknown", "robot_active_program": ""}

    try:
        status = robot.get_status()
    except Exception as exc:
        logger.debug("Robot API status poll failed: %s", exc)
        return {"robot_state": "unknown", "robot_active_program": ""}

    executor_running = status.get("executor_running", False)
    joint_state = status.get("joint_state") or {}
    gripper_state = status.get("gripper_state") or {}
    velocities = joint_state.get("velocities", [])

    # Determine busy/idle from joint velocities
    velocity_threshold = 0.01  # rad/s — above this we consider the robot moving
    is_moving = any(abs(v) > velocity_threshold for v in velocities) if velocities else False

    if not executor_running:
        robot_state = "idle"
        active_program = ""
    elif is_moving:
        robot_state = "busy"
        active_program = _last_active_program  # retain from execute_action_node
    else:
        robot_state = "idle"
        active_program = ""

    _last_active_program = active_program

    return {
        "robot_state": robot_state,
        "robot_active_program": active_program,
        "executor_running": executor_running,
        "joint_positions": joint_state.get("positions", []),
        "joint_velocities": velocities,
        "gripper_position": gripper_state.get("position", 0.0),
        "speed": status.get("speed"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node functions
# ═══════════════════════════════════════════════════════════════════════════


def _maybe_stream_frame_to_live(state: AuraGraphState, image) -> None:
    """If the intent monitor uses a GeminiLiveClient, stream the frame."""
    try:
        from aura.utils.llm_client import GeminiLiveClient
        monitor = _get_intent_monitor(state)
        client = getattr(monitor, "_llm_client", None)
        if isinstance(client, GeminiLiveClient):
            import cv2
            from PIL import Image
            if image.ndim == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb = image
            pil = Image.fromarray(rgb)
            if max(pil.size) > 768:
                scale = 768 / max(pil.size)
                pil = pil.resize(
                    (int(pil.width * scale), int(pil.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            client.send_frame(pil)
    except Exception as exc:
        logger.debug("Live frame stream skipped: %s", exc)


def capture_frame_node(state: AuraGraphState) -> dict:
    """Read the next frame from the video/webcam source.

    Populates ``frames_buffer``, ``current_frame_num`` and
    ``current_timestamp_sec``.  Sets ``is_complete=True`` when the
    source is exhausted.
    """
    try:
        source = _get_video_source(state)
    except Exception as e:
        return {"error": f"Video source error: {e}", "is_complete": True}

    frame_obj = source.read()
    if frame_obj is None:
        return {"is_complete": True}

    # Keep a rolling buffer of the last 10 images (with parallel timestamp/frame-num lists)
    buf: list = list(state.get("frames_buffer") or [])
    ts_buf: list = list(state.get("frames_buffer_timestamps") or [])
    fn_buf: list = list(state.get("frames_buffer_frame_nums") or [])

    buf.append(frame_obj.image)
    ts_buf.append(frame_obj.timestamp)
    fn_buf.append(frame_obj.frame_number)

    if len(buf) > 10:
        buf = buf[-10:]
        ts_buf = ts_buf[-10:]
        fn_buf = fn_buf[-10:]

    # Stream frame to Gemini Live session (if active) for continuous
    # visual context between generate() calls.
    _maybe_stream_frame_to_live(state, frame_obj.image)

    return {
        "frames_buffer": buf,
        "frames_buffer_timestamps": ts_buf,
        "frames_buffer_frame_nums": fn_buf,
        "current_frame_num": frame_obj.frame_number,
        "current_timestamp_sec": frame_obj.timestamp,
    }


def run_gesture_node(state: AuraGraphState) -> dict:
    """Run gesture recognition every cycle (cheap MediaPipe call).

    Sets ``human_requesting_help=True`` when a resume gesture (e.g.
    ``Thumb_Up``) is detected.  This flag signals the decision engine
    that the human is requesting intervention — it does **not** gate
    any downstream nodes.
    """
    buf = state.get("frames_buffer") or []
    if not buf:
        return {"human_requesting_help": False}

    latest_frame = buf[-1]
    try:
        gm = _get_gesture_monitor(state)
        gesture_output = asyncio.get_event_loop().run_until_complete(
            gm.update(frame=latest_frame)
        )
    except RuntimeError:
        # No running event loop; create one
        gesture_output = asyncio.run(
            _get_gesture_monitor(state).update(frame=latest_frame)
        )

    config = state.get("config", {})
    resume_gestures = config.get("resume_gestures") or ["Thumb_Up"]
    is_help_requested = (
        gesture_output is not None
        and getattr(gesture_output, "dominant_gesture", None) in resume_gestures
    )

    monitor_out = state.get("monitor_outputs") or {}
    gesture_dict = {
        "dominant_gesture": getattr(gesture_output, "dominant_gesture", None),
        "safety_triggered": getattr(gesture_output, "safety_triggered", False),
    } if gesture_output else {}

    return {
        "human_requesting_help": is_help_requested,
        "monitor_outputs": {**monitor_out, "gesture": gesture_dict},
    }


def run_perception_node(state: AuraGraphState) -> dict:
    """Run task-specific perception on the latest frame.

    Detects tracked objects (tables, bottles) via SAM3 and uses mask
    heuristics to update ``object_locations`` with bottle positions.
    """
    buf = state.get("frames_buffer") or []
    if not buf:
        return {}

    monitor = _get_perception_monitor(state)
    if monitor is None:
        return {}

    latest_frame = buf[-1]
    try:
        result = asyncio.get_event_loop().run_until_complete(
            monitor.process_frame(latest_frame)
        )
    except RuntimeError:
        result = asyncio.run(monitor.process_frame(latest_frame))

    if not result:
        return {}

    # Merge perception-derived locations (only overwrite when definitive).
    # Monitors may return locations under "object_locations" or
    # task-specific keys like "bottle_locations".
    obj_locs = dict(state.get("object_locations") or {})
    for key in ("object_locations", "bottle_locations"):
        for obj_id, region in result.get(key, {}).items():
            if region != "unknown":
                obj_locs[obj_id] = region

    monitor_out = state.get("monitor_outputs") or {}
    return {
        "object_locations": obj_locs,
        "monitor_outputs": {**monitor_out, "perception": result},
    }


def run_intent_node(state: AuraGraphState) -> dict:
    """Run RCWPS intent prediction using ``AURAIntentMonitor.predict()``.

    Runs on a time-based throttle (``predict_interval`` seconds between
    calls) to avoid excessive LLM usage.  When the interval has not
    elapsed, returns the existing ``intent_result`` unchanged.
    """
    config = state.get("config", {})
    predict_interval = config.get("predict_interval", 3.0)
    last_predict = state.get("last_predict_time", 0.0)
    now = time.time()

    if (now - last_predict) < predict_interval:
        # Keep existing intent_result; not time to predict yet
        return {}

    buf = state.get("frames_buffer") or []
    if not buf:
        return {"intent_result": None}

    try:
        monitor = _get_intent_monitor(state)

        # Use at most max_frames from the tail of the buffer.
        frames_to_send = buf[-5:]
        ts_buf = list(state.get("frames_buffer_timestamps") or [])[-5:]
        fn_buf = list(state.get("frames_buffer_frame_nums") or [])[-5:]

        # Window duration = actual timestamp span of the selected frames.
        if len(ts_buf) >= 2:
            window_duration = ts_buf[-1] - ts_buf[0]
        elif ts_buf:
            window_duration = ts_buf[-1]
        else:
            window_duration = 0.0

        # Report the frame number of the last frame in the window.
        last_frame_num = fn_buf[-1] if fn_buf else state.get("current_frame_num", 0)

        result = monitor.predict(
            frames=frames_to_send,
            timestamp=state.get("current_timestamp_sec", 0.0),
            frame_num=last_frame_num,
            window_duration_sec=window_duration,
        )
    except Exception as e:
        logger.error("Intent prediction failed: %s", e)
        return {"intent_result": None, "error": f"Intent error: {e}"}

    # Serialise IntentResult to dict
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

    monitor_out = state.get("monitor_outputs") or {}
    return {
        "intent_result": result_dict,
        "last_predict_time": now,
        "monitor_outputs": {**monitor_out, "intent": result_dict},
    }


def update_ssg_node(state: AuraGraphState) -> dict:
    """Sync monitor outputs into the live SSG and snapshot.

    1. Calls ``graph.update_from_intent_result()`` on the live SSG
       (when a new intent result is available).
    2. Syncs perception-detected object locations into SSG task_state
       so the decision engine prompt includes them.
    3. Updates ``completed_steps`` and ``object_locations`` in flat state.
    4. Takes a snapshot for serialisation back into ``AuraGraphState.ssg``.
    """
    ssg = _get_ssg(state)

    # ── Sync intent result into SSG ────────────────────────────────
    intent = state.get("intent_result")
    if intent:
        ssg.update_from_intent_result(intent)

    # ── Sync robot status into SSG ───────────────────────────────
    config = state.get("config", {})
    dry_run = config.get("dry_run", True)

    if config.get("use_ground_truth_robot_status", False):
        # Offline evaluation: derive from annotated ground-truth file
        robot_status = _robot_status_from_ground_truth(
            state,
            state.get("current_timestamp_sec", 0.0),
        )
    elif not dry_run:
        # Live mode: poll the real robot API
        robot_status = _robot_status_from_api(state)
    else:
        robot_status = None

    if robot_status:
        ssg.set_task_state("robot_state", robot_status["robot_state"])
        ssg.set_task_state("robot_active_program", robot_status["robot_active_program"])

        robot = ssg.get_node("robot")
        if robot and hasattr(robot, "state"):
            from aura.core.scene_graph.nodes import AgentState
            setattr(robot, "state",
                    AgentState.BUSY if robot_status["robot_state"] == "busy"
                    else AgentState.IDLE)

    # ── Sync perception-detected object locations ──────────────────
    # Task-specific perception monitors return detected locations
    # (e.g. bottle_0 → storage).  Store them in SSG task_state so the
    # decision engine can see them even though perception cannot
    # distinguish semantically named objects (resin vs hardener).
    perception_output = (state.get("monitor_outputs") or {}).get("perception")
    if isinstance(perception_output, dict):
        detected_locs = perception_output.get("bottle_locations")
        if detected_locs:
            ssg.set_task_state("detected_bottle_locations", detected_locs)

    # ── Sync audio events into SSG ───────────────────────────────
    # Drain events from the AudioWorkflowBridge (if active).
    # The sound monitor runs as a background task and pushes events
    # into thread-safe queues; we consume them here, once per cycle.
    audio_bridge = get_audio_bridge()
    sound_dict = {}
    if audio_bridge is not None and audio_bridge.is_active:
        # 1. Utterances → SSG task_state["recent_utterances"]
        utterance_events = audio_bridge.drain_utterances()
        if utterance_events:
            recent: list = list(ssg.task_state.get("recent_utterances", []))
            for evt in utterance_events:
                recent.append({
                    "text": evt.data.get("text", ""),
                    "speaker": evt.data.get("speaker", "human"),
                    "timestamp": evt.timestamp,
                })
            ssg.set_task_state("recent_utterances", recent[-20:])
            sound_dict["utterances"] = [e.data for e in utterance_events]

        # 2. SSG updates from human speech (e.g. "the resin is on the table")
        ssg_updates = audio_bridge.drain_ssg_updates()
        for evt in ssg_updates:
            key = evt.data.get("key", "")
            value = evt.data.get("value", "")
            if key:
                parts = key.split(".")
                if len(parts) == 2 and parts[1] == "location":
                    try:
                        ssg.set_location(parts[0], value)
                    except Exception:
                        pass
                ssg.set_task_state(key, value)
                logger.info("Audio SSG update: %s = %s", key, value)

        # 3. Context messages → SSG task_state["human_context_messages"]
        context_events = audio_bridge.drain_context_messages()
        if context_events:
            existing = list(ssg.task_state.get("human_context_messages", []))
            for evt in context_events:
                existing.append({
                    "text": evt.data.get("text", ""),
                    "timestamp": evt.timestamp,
                })
            ssg.set_task_state("human_context_messages", existing[-10:])

    # ── Update flat tracking fields ────────────────────────────────
    completed_steps = list(set(state.get("completed_steps") or []))
    if intent:
        completed_steps = list(set(
            completed_steps + (intent.get("steps_completed") or [])
        ))

    # Derive object locations from SSG / intent state
    obj_locs: Dict[str, str] = dict(state.get("object_locations") or {})
    if intent:
        for key, val in intent.get("state", {}).items():
            if key.endswith("_location") and isinstance(val, str):
                obj_locs[key.removesuffix("_location")] = val

    ssg.take_snapshot()

    monitor_out = state.get("monitor_outputs") or {}
    if sound_dict:
        monitor_out = {**monitor_out, "sound": sound_dict}

    return {
        "ssg": ssg.to_dict(),
        "task_state": dict(ssg.task_state),
        "completed_steps": completed_steps,
        "object_locations": obj_locs,
        "monitor_outputs": monitor_out,
    }


async def decide_action_node(state: AuraGraphState) -> dict:
    """Decide what the robot should do.

    Queries the Brain ``DecisionEngine`` which reasons over the current
    state, DAG, monitors, and robot_skills.json to select actions.

    If the decision engine decides to communicate with the human
    (``decision == "ask"`` or ``"communicate"``), the message is
    pushed to the AudioWorkflowBridge for the SoundMonitor to speak.
    """
    intent_result = state.get("intent_result")
    if not intent_result:
        return {}

    timestamp = state.get("current_timestamp_sec", 0.0)
    help_requested = state.get("human_requesting_help", False)

    actions: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []

    if help_requested:
        reasoning_parts.append("Human requesting help (gesture detected)")

    # ── Brain decision engine ───────────────────────────────────────
    speech_message: Optional[str] = None
    try:
        engine = _get_decision_engine(state)
        prediction = await engine.decide_action(current_time_sec=timestamp)
        if prediction:
            # Check if this is a communication action (ask/communicate)
            if prediction.action_id in ("ask_preference", "ask_question",
                                         "communicate", "speak"):
                speech_message = prediction.reasoning
                reasoning_parts.append(f"Brain: communicate — {prediction.reasoning[:80]}")
            else:
                actions.append({
                    "action_type": prediction.action_id,
                    "object_name": prediction.target_id,
                    "trigger_step": "brain",
                    "reason": prediction.reasoning,
                    "confidence": prediction.confidence,
                    "timestamp": timestamp,
                })
                reasoning_parts.append(
                    f"Brain: {prediction.action_id}"
                )
    except Exception as e:
        logger.warning("Brain engine error: %s", e)

    # ── Push speech to human via audio bridge ─────────────────────
    audio_bridge = get_audio_bridge()
    if speech_message and audio_bridge is not None and audio_bridge.is_active:
        audio_bridge.push_speech(speech_message)
        logger.info("Decision engine → human: %s", speech_message[:100])

    decision_record = {
        "timestamp_sec": timestamp,
        "frame_num": state.get("current_frame_num", 0),
        "actions": actions,
        "speech_message": speech_message,
        "reasoning": " | ".join(reasoning_parts) or "No action needed",
        "decided_at": datetime.now().isoformat(),
    }

    return {
        "pending_actions": actions,
        "last_decision": decision_record,
        "decision_history": [decision_record],
    }


def execute_action_node(state: AuraGraphState) -> dict:
    """Execute pending robot actions (or log them in dry-run mode).

    Updates ``object_locations`` based on action effects and records
    results in ``decision_history``.
    """
    actions = state.get("pending_actions") or []
    if not actions:
        return {"pending_actions": []}

    config = state.get("config", {})
    dry_run = config.get("dry_run", True)

    # Get skill registry from the decision engine for api_call lookup
    engine = _get_decision_engine(state)
    skills = engine.skills

    robot = _get_robot_client(state)
    obj_locs: Dict[str, str] = dict(state.get("object_locations") or {})
    executed: List[Dict[str, Any]] = []

    # ── Set robot BUSY in SSG before execution ──────────────────
    from aura.core.scene_graph.nodes import AgentState, AgentNode

    ssg = _get_ssg(state)
    robot_node = ssg.get_node("robot")
    if robot_node and isinstance(robot_node, AgentNode):
        robot_node.state = AgentState.BUSY
        robot_node.current_action = actions[0].get("action_type", "")

    for action in actions:
        action_type = action.get("action_type", "")
        obj_name = action.get("object_name", "")
        skill = skills.get(action_type)
        api_call = skill.metadata.get("api_call") if skill else None

        # Track active program for live status polling
        global _last_active_program
        _last_active_program = action_type

        result = {**action, "skill_id": action_type, "executed": True}

        if dry_run or robot is None:
            result["success"] = True
            result["mode"] = "dry_run"
            logger.info(
                "[DRY-RUN] Would execute: %s %s (api_call=%s) — SSG updated, robot API skipped",
                action_type, obj_name, api_call,
            )
        else:
            try:
                resp: Dict[str, Any] = {"success": False, "error": "no handler"}
                params = action.get("parameters") or {}
                if api_call:
                    # Dispatch using the skill's api_call definition
                    endpoint = api_call.get("endpoint", "")
                    body = dict(api_call.get("body", {}))
                    # Substitute any LLM-provided parameters into the body
                    for k, v in params.items():
                        if f"<{k}>" in str(body.get(k, "")):
                            body[k] = v
                        elif k not in body:
                            body[k] = v
                    resp = robot._post(endpoint, body)
                elif skill and skill.category == "gripper":
                    # Infer gripper action from skill id
                    if "open" in action_type:
                        resp = robot.gripper_open()
                    elif "close" in action_type:
                        resp = robot.gripper_close()
                elif skill and skill.category == "motion":
                    # Infer named position from skill description or parameters
                    # Sorting skills encode the position name in the description:
                    # "Move to Pick_White_Ball position via /api/move/named {name: Pick_White_Ball}"
                    named_pos = params.get("position_name") or params.get("named_position")
                    if not named_pos:
                        # Parse "via /api/move/named {name: <Pos>}" from description
                        import re
                        m = re.search(r'\{name:\s*(\S+)\}', skill.description) if skill.description else None
                        named_pos = m.group(1) if m else None
                    if named_pos:
                        resp = robot.move_to_named(named_pos)
                    else:
                        logger.warning("Cannot infer named position for skill '%s'", action_type)
                else:
                    logger.warning(
                        "No api_call found for skill '%s' — action skipped",
                        action_type,
                    )
                result["success"] = resp.get("success", False)
                result["api_response"] = resp
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

        # Update object locations from skill effects on success
        if result.get("success") and skill:
            for effect_key, effect_val in skill.effects.items():
                # Effects like "resin_bottle.location": "workplace"
                parts = effect_key.split(".")
                if len(parts) == 2 and parts[1] == "location":
                    obj_locs[parts[0]] = effect_val

        executed.append(result)

    # ── Sync execution results back into live SSG ─────────────
    # Apply skill effects to SSG (e.g., "resin_bottle.location": "workplace")
    for ex in executed:
        if not ex.get("success"):
            continue
        ex_skill = skills.get(ex.get("action_type", ""))
        if not ex_skill:
            continue
        for effect_key, effect_val in ex_skill.effects.items():
            parts = effect_key.split(".")
            if len(parts) == 2 and parts[1] == "location" and ssg.has_node(parts[0]):
                ssg.set_location(parts[0], effect_val)

    # Reset robot state after execution
    _last_active_program = ""
    if robot_node and isinstance(robot_node, AgentNode):
        robot_node.state = AgentState.IDLE
        robot_node.current_action = None

    ssg.take_snapshot()

    return {
        "pending_actions": [],
        "object_locations": obj_locs,
        "decision_history": executed,
        "ssg": ssg.to_dict(),
    }


def check_complete_node(state: AuraGraphState) -> dict:
    """Check whether the task is complete or the source has ended.

    Increments ``cycle_count`` and resets ``human_requesting_help``
    for the next iteration (gesture must be re-detected each cycle).
    """
    cycle = (state.get("cycle_count") or 0) + 1
    max_cycles = state.get("config", {}).get("max_cycles", 500)

    # Check for explicit completion — find terminal nodes (no step depends on them)
    dag = state.get("dag") or []
    all_ids = {step["id"] for step in dag if isinstance(step, dict)}
    depended_on = {d for step in dag if isinstance(step, dict) for d in step.get("dependencies", [])}
    end_nodes = all_ids - depended_on if all_ids else {"task_complete"}
    completed = set(state.get("completed_steps") or [])

    is_complete = bool(completed & end_nodes) or state.get("is_complete", False)

    if state.get("error"):
        is_complete = True

    # Safety: stop after max cycles to prevent infinite loops
    if cycle >= max_cycles:
        logger.warning(f"Reached max cycles ({max_cycles}), stopping workflow")
        is_complete = True

    return {
        "cycle_count": cycle,
        "human_requesting_help": False,
        "is_complete": is_complete,
    }

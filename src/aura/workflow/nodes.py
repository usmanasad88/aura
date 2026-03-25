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
from typing import Any, Dict, List, TYPE_CHECKING, cast

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
_decision_engines: Dict[str, Any] = {}
_video_sources: Dict[str, Any] = {}
_robot_clients: Dict[str, Any] = {}
_ssg_instances: Dict[str, Any] = {}
_ground_truth_data_cache: Dict[str, Any] = {}


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
        _intent_monitors[config_dir] = AURAIntentMonitor(
            config_dir=config_dir,
            model=config.get("model", "gemini-3.1-pro-preview"),
            realtime=config.get("realtime", True),
            enable_logging=True,
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


def _get_decision_engine(state: AuraGraphState) -> "DecisionEngine":
    """Lazy-init the Brain DecisionEngine with SSG + SkillRegistry."""
    from aura.brain.decision_engine import DecisionEngine, DecisionEngineConfig

    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    decision_mode = config.get("decision_mode", "hybrid")

    if config_dir not in _decision_engines:
        engine_config = DecisionEngineConfig(
            gemini_model=config.get("model", "gemini-2.5-pro-preview-06-05"),
            enable_llm_reasoning=(decision_mode in ("llm", "hybrid")),
            proactive_threshold=0.6,
        )
        engine = DecisionEngine(config=engine_config)

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

    key = video_path or f"webcam:{webcam_device}"
    if key not in _video_sources:
        if webcam_device is not None:
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


# ═══════════════════════════════════════════════════════════════════════════
#  Node functions
# ═══════════════════════════════════════════════════════════════════════════


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

    # Keep a rolling buffer of the last 10 images
    buf: list = list(state.get("frames_buffer") or [])
    buf.append(frame_obj.image)
    if len(buf) > 10:
        buf = buf[-10:]

    return {
        "frames_buffer": buf,
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
        result = monitor.predict(
            frames=buf[-5:],
            timestamp=state.get("current_timestamp_sec", 0.0),
            frame_num=state.get("current_frame_num", 0),
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
    """Sync the latest intent result into the live SSG and snapshot.

    1. Calls ``graph.update_from_intent_result()`` on the live SSG.
    2. Updates ``completed_steps`` and ``object_locations`` in flat state.
    3. Takes a snapshot for serialisation back into ``AuraGraphState.ssg``.
    """
    intent = state.get("intent_result")
    if not intent:
        return {}

    ssg = _get_ssg(state)
    ssg.update_from_intent_result(intent)

    config = state.get("config", {})
    if config.get("use_ground_truth_robot_status", False):
        robot_status = _robot_status_from_ground_truth(
            state,
            state.get("current_timestamp_sec", 0.0),
        )
        ssg.set_task_state("robot_state", robot_status["robot_state"])
        ssg.set_task_state("robot_active_program", robot_status["robot_active_program"])

        robot = ssg.get_node("robot")
        if robot and hasattr(robot, "state"):
            setattr(robot, "state", "BUSY" if robot_status["robot_state"] == "busy" else "IDLE")

    # Update flat tracking fields
    completed_steps = list(set(
        (state.get("completed_steps") or [])
        + (intent.get("steps_completed") or [])
    ))

    # Derive object locations from SSG / intent state
    obj_locs: Dict[str, str] = dict(state.get("object_locations") or {})
    for key, val in intent.get("state", {}).items():
        if key.endswith("_location") and isinstance(val, str):
            obj_locs[key.removesuffix("_location")] = val

    ssg.take_snapshot()

    return {
        "ssg": ssg.to_dict(),
        "task_state": dict(ssg.task_state),
        "completed_steps": completed_steps,
        "object_locations": obj_locs,
    }


async def decide_action_node(state: AuraGraphState) -> dict:
    """Decide what the robot should do.

    Queries the Brain ``DecisionEngine`` which reasons over the current
    state, DAG, monitors, and robot_skills.json to select actions.
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
    try:
        engine = _get_decision_engine(state)
        prediction = await engine.decide_action(current_time_sec=timestamp)
        if prediction:
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

    decision_record = {
        "timestamp_sec": timestamp,
        "frame_num": state.get("current_frame_num", 0),
        "actions": actions,
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
    task_profile = state.get("task_profile", {})
    program_map_raw = task_profile.get("program_map", {})

    # Parse program_map from "action|object" keys
    program_map: Dict[tuple, str] = {}
    for k, v in program_map_raw.items():
        parts = k.split("|")
        if len(parts) == 2:
            program_map[(parts[0], parts[1])] = v

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
        prog = program_map.get((action_type, obj_name))

        result = {**action, "program": prog, "executed": True}

        if dry_run or robot is None:
            result["success"] = True
            result["mode"] = "dry_run"
            logger.info(
                "[DRY-RUN] Would execute: %s %s (program=%s) — SSG updated, robot API skipped",
                action_type, obj_name, prog,
            )
        else:
            try:
                resp = robot.execute_program(prog) if prog else {"success": False, "error": "no program"}
                result["success"] = resp.get("success", False)
                result["api_response"] = resp
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

        # Update locations on success
        if result.get("success"):
            if action_type == "return_to_storage":
                obj_locs[obj_name] = "storage"
            elif action_type == "deliver_to_workplace":
                obj_locs[obj_name] = "workplace"

        executed.append(result)

    # ── Sync execution results back into live SSG ─────────────
    # Map action outcomes to SSG region IDs (flat state uses "storage",
    # but SSG region nodes are "storage_area" and "workplace").
    for ex in executed:
        if not ex.get("success"):
            continue
        obj_name = ex.get("object_name", "")
        action_type = ex.get("action_type", "")
        if action_type == "deliver_to_workplace" and ssg.has_node(obj_name):
            ssg.set_location(obj_name, "workplace")
        elif action_type == "return_to_storage" and ssg.has_node(obj_name):
            ssg.set_location(obj_name, "storage_area")

    # Reset robot state after execution
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

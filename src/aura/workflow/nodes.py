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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aura.core.scene_graph import SemanticSceneGraph
    from aura.assistant.intent_monitor import AURAIntentMonitor
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


def _get_ssg(state: dict) -> "SemanticSceneGraph":
    """Get or create the live SSG instance for this task, restoring from
    the serialised ``ssg`` snapshot in state on first call."""
    from aura.core.scene_graph import SemanticSceneGraph

    config = state.get("config", {})
    task_name = config.get("task_name", "default")

    if task_name not in _ssg_instances:
        ssg_data = state.get("ssg")
        if ssg_data and ssg_data.get("nodes"):
            _ssg_instances[task_name] = SemanticSceneGraph.from_dict(ssg_data)
        else:
            _ssg_instances[task_name] = SemanticSceneGraph(name=task_name)
    return _ssg_instances[task_name]


def _get_intent_monitor(state: dict) -> "AURAIntentMonitor":
    from aura.assistant.intent_monitor import AURAIntentMonitor

    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    if config_dir not in _intent_monitors:
        _intent_monitors[config_dir] = AURAIntentMonitor(
            config_dir=config_dir,
            model=config.get("model", "gemini-2.5-flash"),
            realtime=True,
            enable_logging=True,
        )
    return _intent_monitors[config_dir]


def _get_gesture_monitor(state: dict) -> "GestureMonitor":
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


def _get_decision_engine(state: dict) -> "DecisionEngine":
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
            state_path=str(state_path) if state_path.exists() else None,
            skills_path=str(skills_path) if skills_path.exists() else None,
            initial_scene_path=str(scene_path) if scene_path.exists() else None,
        )
        _decision_engines[config_dir] = engine
    return _decision_engines[config_dir]


def _get_video_source(state: dict):
    """Get or open the video/webcam source."""
    config = state.get("config", {})
    video_path = config.get("video_path")
    webcam_device = config.get("webcam_device")

    key = video_path or f"webcam:{webcam_device}"
    if key not in _video_sources:
        if webcam_device is not None:
            from aura.sources.webcam import WebcamSource
            source = WebcamSource(device=webcam_device)
        else:
            from aura.sources.realtime_video import RealtimeVideoSource
            source = RealtimeVideoSource(
                path=video_path,
                speed=config.get("speed", 1.0),
            )
        source.open()
        _video_sources[key] = source
    return _video_sources[key]


def _get_robot_client(state: dict):
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


def decide_action_node(state: AuraGraphState) -> dict:
    """Decide what the robot should do using hybrid LLM + rule engine.

    1. Feeds the latest intent result into the old rule-based
       ``AURADecisionEngine`` for backward-compatible DAG-driven
       delivery/return actions.
    2. When ``human_requesting_help`` is True (gesture detected),
       always queries the Brain ``DecisionEngine`` for LLM-based
       proactive opportunities, even if rule actions already exist.
    3. Merges results into ``pending_actions`` and ``last_decision``.
    """
    intent_result = state.get("intent_result")
    if not intent_result:
        return {}

    config = state.get("config", {})
    decision_mode = config.get("decision_mode", "hybrid")
    timestamp = state.get("current_timestamp_sec", 0.0)
    help_requested = state.get("human_requesting_help", False)

    actions: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []

    if help_requested:
        reasoning_parts.append("Human requesting help (gesture detected)")

    # ── Phase 1: Rule-based (always) ────────────────────────────────
    rule_actions = _run_rule_engine(state, intent_result, timestamp)
    actions.extend(rule_actions)
    if rule_actions:
        reasoning_parts.append(
            f"Rule engine: {len(rule_actions)} action(s)"
        )

    # ── Phase 2: Brain LLM ─────────────────────────────────────────
    # When human requests help, always consult the brain regardless
    # of existing rule actions.  Otherwise, only if no rule actions.
    should_consult_brain = (
        decision_mode in ("llm", "hybrid")
        and (help_requested or not actions)
    )
    if should_consult_brain:
        brain_action = _run_brain_engine(state, timestamp)
        if brain_action:
            actions.append(brain_action)
            reasoning_parts.append(
                f"Brain LLM: {brain_action.get('action_id', '?')}"
            )

    decision_record = {
        "timestamp_sec": timestamp,
        "frame_num": state.get("current_frame_num", 0),
        "actions": actions,
        "reasoning": " | ".join(reasoning_parts) or "No action needed",
        "decision_mode": decision_mode,
        "decided_at": datetime.now().isoformat(),
    }

    return {
        "pending_actions": actions,
        "last_decision": decision_record,
        "decision_history": [decision_record],
    }


def _run_rule_engine(
    state: dict,
    intent_result: dict,
    timestamp: float,
) -> List[Dict[str, Any]]:
    """Execute rule-based logic from the existing AURADecisionEngine.

    Instead of instantiating AURADecisionEngine (which has its own
    internal state), we replicate the core logic inline using the DAG
    and task_profile from the state.
    """
    dag = state.get("dag", {})
    task_profile = state.get("task_profile", {})
    nodes_def = dag.get("nodes", {})

    env = task_profile.get("environment", {})
    movable_objects = set(env.get("movable_objects", []))
    initial_delivery = set(env.get("initial_delivery_objects", []))

    obj_locs: Dict[str, str] = dict(state.get("object_locations") or {})
    completed: set = set(state.get("completed_steps") or [])
    cycle_count = state.get("cycle_count", 0)

    actions: List[Dict[str, Any]] = []

    # ── Initial delivery (first cycle only) ─────────────────────────
    if cycle_count == 0:
        for obj in initial_delivery:
            if obj_locs.get(obj, "storage") == "storage":
                actions.append({
                    "action_type": "deliver_to_workplace",
                    "object_name": obj,
                    "trigger_step": "idle",
                    "reason": f"Initial setup — delivering {obj} to workplace",
                    "timestamp": timestamp,
                })
                obj_locs[obj] = "workplace"

    # ── Return-to-storage triggers ──────────────────────────────────
    new_completed = set(intent_result.get("steps_completed", [])) - completed
    for step_name in new_completed:
        node_def = nodes_def.get(step_name, {})
        rts = node_def.get("robot_return_to_storage", {})
        for obj in rts.get("objects", []):
            if obj_locs.get(obj) == "workplace":
                actions.append({
                    "action_type": "return_to_storage",
                    "object_name": obj,
                    "trigger_step": step_name,
                    "reason": rts.get("reason", f"{obj} no longer needed"),
                    "timestamp": timestamp,
                })
                obj_locs[obj] = "storage"

    # ── Proactive delivery based on predicted next action ───────────
    predicted = intent_result.get("predicted_next_action", "")
    if predicted and predicted != "unknown":
        needed = nodes_def.get(predicted, {}).get("objects_needed_on_workplace", [])
        for obj in needed:
            if obj in movable_objects and obj_locs.get(obj) == "storage":
                actions.append({
                    "action_type": "deliver_to_workplace",
                    "object_name": obj,
                    "trigger_step": predicted,
                    "reason": f"Proactively needed for {predicted}",
                    "timestamp": timestamp,
                })
                obj_locs[obj] = "workplace"

    return actions


def _run_brain_engine(state: dict, timestamp: float) -> Optional[Dict[str, Any]]:
    """Query the Brain DecisionEngine for LLM-based action selection."""
    try:
        engine = _get_decision_engine(state)
        # Use rule-based fallback if in sync context (LLM is async)
        prediction = engine._rule_based_decide(
            engine.reasoner.get_available_actions("robot"),
            engine.reasoner.get_proactive_opportunities("robot"),
            timestamp,
        )
        if prediction:
            return {
                "action_type": prediction.action_id,
                "object_name": prediction.target_id,
                "trigger_step": "brain_proactive",
                "reason": prediction.reasoning,
                "confidence": prediction.confidence,
                "timestamp": timestamp,
            }
    except Exception as e:
        logger.warning("Brain engine error: %s", e)
    return None


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

    for action in actions:
        action_type = action.get("action_type", "")
        obj_name = action.get("object_name", "")
        prog = program_map.get((action_type, obj_name))

        result = {**action, "program": prog, "executed": True}

        if dry_run or robot is None:
            result["success"] = True
            result["mode"] = "dry_run"
            logger.info(
                "[DRY-RUN] %s %s (program=%s, trigger=%s)",
                action_type, obj_name, prog, action.get("trigger_step"),
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

    return {
        "pending_actions": [],
        "object_locations": obj_locs,
        "decision_history": executed,
    }


def check_complete_node(state: AuraGraphState) -> dict:
    """Check whether the task is complete or the source has ended.

    Increments ``cycle_count`` and resets ``human_requesting_help``
    for the next iteration (gesture must be re-detected each cycle).
    """
    cycle = (state.get("cycle_count") or 0) + 1

    # Check for explicit completion
    dag = state.get("dag", {})
    end_nodes = set(dag.get("end_nodes", ["task_complete"]))
    completed = set(state.get("completed_steps") or [])

    is_complete = bool(completed & end_nodes) or state.get("is_complete", False)

    if state.get("error"):
        is_complete = True

    return {
        "cycle_count": cycle,
        "human_requesting_help": False,
        "is_complete": is_complete,
    }

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
_intent_gt_providers: Dict[str, Any] = {}
_gesture_monitors: Dict[str, Any] = {}
_perception_monitors: Dict[str, Any] = {}
_pose_monitors: Dict[str, Any] = {}
_pose_failures: Dict[str, int] = {}
_pose_disabled: Dict[str, bool] = {}
_activity_gates: Dict[str, Any] = {}
_decision_engines: Dict[str, Any] = {}
_video_sources: Dict[str, Any] = {}
_robot_clients: Dict[str, Any] = {}
_ssg_instances: Dict[str, Any] = {}
_ground_truth_data_cache: Dict[str, Any] = {}

# Intent dispatcher state.
#
# The intent VLM call is slow (multi-second). To keep the fast perception loop
# responsive in realtime mode we dispatch the call onto a single-worker thread
# pool and let the LangGraph node return immediately. When the future resolves
# the result is published to ``_intent_slot``; ``update_ssg_node`` drains it
# and applies the change to the SSG.
#
# In offline / eval mode (``config["intent_blocking"]=True``) the same
# dispatcher is used, but ``run_intent_node`` blocks on the future before
# returning so every cycle has a fresh intent prediction — same code path,
# different timing.
import threading as _threading
import concurrent.futures as _futures
_intent_slot_lock = _threading.Lock()
_intent_slot: Dict[str, Any] = {}   # {config_dir: intent_result_dict}
_intent_executor = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-intent")
_intent_futures: Dict[str, "_futures.Future"] = {}
_intent_last_dispatch: Dict[str, float] = {}
# Tracks the (frame_num, timestamp) of the most recent predict() dispatch so
# that the next cycle can look up the GT "previous state" at that point when
# ``intent_previous_state_source=ground_truth`` is set.
_intent_last_dispatched_frame: Dict[str, int] = {}
_intent_last_dispatched_ts: Dict[str, float] = {}

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


# ── Background-intent slot accessors ────────────────────────────────────────

def push_intent_result(config_dir: str, result: Dict[str, Any]) -> None:
    """Publish a new intent_result dict from the background runner.

    Called by :class:`BackgroundIntentRunner` after each ``predict()`` call.
    The fast loop picks it up on the next ``update_ssg_node`` tick.
    """
    with _intent_slot_lock:
        _intent_slot[config_dir] = result


def pop_intent_result(config_dir: str) -> Optional[Dict[str, Any]]:
    """Atomically take the latest pending intent_result (or None)."""
    with _intent_slot_lock:
        return _intent_slot.pop(config_dir, None)


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


def _apply_skill_effect(ssg: "SemanticSceneGraph", key: str, value: Any) -> None:
    """Apply a single skill effect to the SSG.

    Effect keys use dotted notation ``"<node_id>.<attr>"``. ``.location``
    updates the spatial edge via ``ssg.set_location``; any other attribute
    is written into ``ssg.task_state`` under the flat key ``"<node>_<attr>"``
    (matching the precondition lookup convention in
    :func:`aura.brain.bt_policy._check_preconditions`). A bare key with no
    dot is written directly into ``task_state``.

    Task-state keys are auto-created on first write: skill authors do not
    need to pre-declare every effect variable in ``state_schema.json``.
    """
    if "." in key:
        node_id, attr = key.split(".", 1)
        if attr == "location" and ssg.has_node(node_id):
            try:
                ssg.set_location(node_id, value)
                return
            except Exception as exc:
                logger.debug("set_location(%s, %s) failed: %s", node_id, value, exc)
        # Non-location dotted effect → flat task_state key ``<node>_<attr>``.
        ssg.set_task_state(f"{node_id}_{attr}", value)
    else:
        ssg.set_task_state(key, value)


def _get_intent_monitor(state: AuraGraphState) -> "AURAIntentMonitor":
    from aura.monitors.intent_monitor import AURAIntentMonitor

    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    if config_dir not in _intent_monitors:
        # Per-component overrides fall back to shared defaults
        intent_backend = config.get("intent_backend") or config.get("llm_backend", "gemini")
        intent_model = config.get("intent_model") or config.get("model", "gemini-3.1-pro-preview")
        run_log_dir = config.get("run_log_dir")
        intent_log_dir = (
            str(Path(run_log_dir) / "intent_monitor") if run_log_dir else None
        )
        _intent_monitors[config_dir] = AURAIntentMonitor(
            config_dir=config_dir,
            model=intent_model,
            max_frames=int(config.get("intent_num_frames", 5)),
            realtime=config.get("realtime", True),
            enable_logging=True,
            log_dir=intent_log_dir,
            llm_backend=intent_backend,
            sglang_base_url=config.get("sglang_base_url", "http://localhost:8100/v1"),
            max_tokens=config.get("intent_max_tokens", 4096),
            include_previous_state=config.get("intent_include_previous_state", True),
        )
    return _intent_monitors[config_dir]


def _get_intent_gt_provider(state: AuraGraphState):
    """Lazy-init the :class:`GroundTruthIntentProvider` for the current video.

    Resolves the GT file from ``tasks/<task>/ground_truth/<video_stem>.intent_gt.json``
    unless ``config["intent_gt_path"]`` is set explicitly. Returns ``None``
    if no GT file is available (callers fall back to the live monitor).
    """
    from aura.monitors.intent_ground_truth import (
        GroundTruthIntentProvider, default_gt_path,
    )

    config = state.get("config", {}) or {}
    config_dir = config.get("config_dir", "")
    video_path = config.get("video_path")
    explicit = config.get("intent_gt_path")
    key = f"{config_dir}::{explicit or video_path or ''}"

    if key in _intent_gt_providers:
        return _intent_gt_providers[key]

    if explicit:
        gt_path = Path(explicit)
    elif video_path and config_dir:
        gt_path = default_gt_path(config_dir, video_path)
    else:
        raise RuntimeError(
            "intent_source=ground_truth requires either config['intent_gt_path'] "
            "or both config_dir + video_path to locate the GT file."
        )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Intent GT file not found: {gt_path}. "
            "Create it with scripts/annotate_ground_truth.py."
        )

    provider = GroundTruthIntentProvider(gt_path)
    _intent_gt_providers[key] = provider
    return provider


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
    """Factory: import the right perception monitor for *task_name*.

    Dispatch keys off the task's directory slug (the basename of
    ``config_dir``'s parent, e.g. ``kettle_tea_making``) when available,
    falling back to the normalised display name. The display ``task_name``
    is a human label (e.g. "Collaborative Kettle Tea Making") that does not
    reliably normalise to the directory slug, so it must not be the only key.
    """
    normalised = task_name.lower().replace(" ", "_")
    config_dir = config.get("config_dir")
    if config_dir:
        # ``config_dir`` points at ``tasks/<slug>/config`` — its parent name
        # is the canonical task slug used for the package import paths below.
        slug = Path(config_dir).parent.name
        if slug:
            normalised = slug
    if normalised == "hand_layup":
        try:
            from tasks.hand_layup.perception.layup_perception_monitor import (
                LayupPerceptionMonitor,
            )
            return LayupPerceptionMonitor()
        except ImportError:
            logger.warning("hand_layup perception monitor not found")
            return None
    if normalised == "cuboid_manipulation":
        try:
            from tasks.cuboid_manipulation.perception.cuboid_perception_monitor import (
                CuboidPerceptionMonitor,
            )
            return CuboidPerceptionMonitor.from_task_profile(config)
        except ImportError as exc:
            logger.warning("cuboid_manipulation perception monitor not found: %s", exc)
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
    if normalised == "kettle_tea_making":
        try:
            from tasks.kettle_tea_making.perception.kettle_perception_monitor import (
                KettlePerceptionMonitor,
            )
            return KettlePerceptionMonitor()
        except ImportError:
            logger.warning("kettle_tea_making perception monitor not found")
            return None
    logger.info("No task-specific perception monitor for '%s'", task_name)
    return None


def _get_pose_monitor(state: AuraGraphState):
    """Lazy-init the SAM-3D-Body pose monitor (ZMQ client).

    Returns ``None`` if pyzmq is unavailable or the server socket can't be
    created — the fast loop then skips ``run_pose_node`` silently so the
    rest of the graph keeps running.
    """
    from aura.monitors.body_pose_monitor import BodyPoseMonitor, BodyPoseMonitorConfig

    config = state.get("config", {})
    key = config.get("config_dir", "default")
    if key not in _pose_monitors:
        endpoint = config.get("pose_server_endpoint", "tcp://localhost:5556")
        try:
            _pose_monitors[key] = BodyPoseMonitor(BodyPoseMonitorConfig(
                server_endpoint=endpoint,
                # Short timeout so a missing pose server doesn't stall the
                # fast perception loop — on failure we disable the gate
                # and fall back to always-run intent (see run_pose_node).
                timeout_sec=float(config.get("pose_timeout_sec", 2.0)),
            ))
        except Exception as exc:
            logger.warning("BodyPoseMonitor init failed: %s", exc)
            _pose_monitors[key] = None
    return _pose_monitors[key]


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
        # decision_mode may also be set inside task_profile.workflow_config;
        # if that override exists and the caller didn't supply one, use it.
        wf_cfg = task_profile.get("workflow_config") or {}
        wf_mode = wf_cfg.get("decision_mode")
        if wf_mode in ("llm", "bt", "hybrid"):
            decision_mode = wf_mode

        # Easy toggle: on idle ticks where no BT branch fires, defer to the
        # LLM instead of waiting. Settable via runtime config or the task
        # profile's workflow_config (runtime config takes precedence).
        llm_fallback_on_idle = config.get("llm_fallback_on_idle")
        if llm_fallback_on_idle is None:
            llm_fallback_on_idle = wf_cfg.get("llm_fallback_on_idle", False)

        sys_instr = task_profile.get("system_instruction", "")
        goal_policy = task_profile.get("goal_policy", {})
        if "description" in goal_policy:
            sys_instr += f"\n\nGoal Policy: {goal_policy['description']}"

        run_log_dir = config.get("run_log_dir")
        decision_log_dir = (
            str(Path(run_log_dir) / "decision_engine") if run_log_dir else None
        )
        engine_config = DecisionEngineConfig(
            gemini_model=decision_model,
            enable_llm_reasoning=(decision_mode in ("llm", "hybrid")),
            proactive_threshold=0.6,
            max_completion_tokens=config.get("decision_max_tokens", 1024),
            llm_backend=decision_backend,
            sglang_base_url=config.get("sglang_base_url", "http://localhost:8100/v1"),
            task_system_instruction=sys_instr,
            decision_mode=decision_mode,
            llm_fallback_on_idle=bool(llm_fallback_on_idle),
            log_dir=decision_log_dir,
        )
        engine = DecisionEngine(config_dir=config_dir, config=engine_config)

        # Wire in the shared SSG
        engine.graph = _get_ssg(state)

        # Re-create reasoner against the shared SSG
        from aura.core.scene_graph import GraphReasoner
        engine.reasoner = GraphReasoner(engine.graph, skills=engine.skills)

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


def _resolve_robot_gt_path(state: AuraGraphState) -> Path | None:
    """Resolve the per-video ``<stem>.robot_gt.json`` file.

    Path: ``tasks/<task>/ground_truth/<video_stem>.robot_gt.json``, mirroring
    the intent-GT layout (see ``intent_ground_truth.default_gt_path``).
    """
    config = state.get("config", {})
    config_dir = config.get("config_dir", "")
    video_path = config.get("video_path") or ""
    if not config_dir or not video_path:
        return None
    cfg = Path(config_dir)
    task_dir = cfg.parent if cfg.name == "config" else cfg
    return task_dir / "ground_truth" / f"{Path(video_path).stem}.robot_gt.json"


def _get_robot_gt_data(state: AuraGraphState) -> Dict[str, Any] | None:
    """Load and cache the task's per-video robot ground-truth file.

    Schema: ``{interventions: [{skill, args, t_start, t_end, ...}, ...],
    duration_sec: float}``.
    """
    gt_path = _resolve_robot_gt_path(state)
    if gt_path is None:
        return None

    key = str(gt_path)
    if key in _ground_truth_data_cache:
        return _ground_truth_data_cache[key]

    if not gt_path.exists():
        logger.warning("Robot ground-truth requested but file missing: %s", gt_path)
        _ground_truth_data_cache[key] = None
        return None

    try:
        with gt_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read robot ground truth file %s: %s", gt_path, exc)
        _ground_truth_data_cache[key] = None
        return None

    interventions = raw.get("interventions", [])
    if not isinstance(interventions, list):
        logger.warning("Invalid robot GT format (interventions not a list): %s", gt_path)
        _ground_truth_data_cache[key] = None
        return None

    parsed: List[Dict[str, Any]] = []
    for iv in interventions:
        if not isinstance(iv, dict):
            continue
        try:
            t_start = float(iv["t_start"])
            t_end = float(iv["t_end"])
            skill = str(iv["skill"])
        except (KeyError, TypeError, ValueError):
            continue
        args = iv.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        parsed.append({
            "skill": skill,
            "args": args,
            "t_start": t_start,
            "t_end": t_end,
        })
    parsed.sort(key=lambda iv: iv["t_start"])

    cached = {
        "interventions": parsed,
        "duration_sec": float(raw.get("duration_sec", 0.0) or 0.0),
    }
    _ground_truth_data_cache[key] = cached
    return cached


def _format_skill_call(skill: str, args: Dict[str, Any]) -> str:
    """Render ``skill(k=v, k=v)`` for the active-program field. Empty args → ``skill``."""
    if not args:
        return skill
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{skill}({inner})"


def _command_announcement(
    skill: Any, action_type: str, obj_name: str, params: Dict[str, Any]
) -> str:
    """Build a short spoken sentence describing a dispatched robot command.

    Task-agnostic: uses the skill's configured human-readable ``name`` (or
    the raw action id, de-underscored) and appends the primary target — the
    ``item`` parameter if the skill is parametric, else the action's
    ``object_name``. No task-specific vocabulary is assumed.
    """
    params = params or {}
    name = (getattr(skill, "name", None) or action_type.replace("_", " ")).strip()
    target = params.get("item") or obj_name or ""
    target_h = str(target).replace("_", " ").strip()
    return f"{name}: {target_h}." if target_h else f"{name}."


def _robot_status_from_ground_truth(state: AuraGraphState, timestamp_sec: float) -> Dict[str, Any]:
    """Derive robot status at ``timestamp_sec`` from the per-video robot GT.

    Rule: robot is BUSY iff some intervention has ``t_start <= t < t_end``;
    otherwise IDLE. The active program string is ``skill(args)``. If multiple
    intervals overlap the timestamp (rare), the latest-starting one wins.
    """
    gt_data = _get_robot_gt_data(state)
    if not gt_data:
        return {"robot_state": "unknown", "robot_active_program": ""}

    interventions: List[Dict[str, Any]] = gt_data.get("interventions", [])
    t = float(timestamp_sec)
    active = None
    for iv in interventions:
        if iv["t_start"] <= t < iv["t_end"]:
            if active is None or iv["t_start"] > active["t_start"]:
                active = iv

    if active is None:
        return {"robot_state": "idle", "robot_active_program": ""}
    return {
        "robot_state": "busy",
        "robot_active_program": _format_skill_call(active["skill"], active["args"]),
    }


# ── Live robot status via External Control API ────────────────────────────

# Track the last active program so we can detect transitions to idle.
_last_active_program: str = ""


def _robot_status_from_api(state: AuraGraphState) -> Dict[str, Any]:
    """Poll ``GET /api/status`` and derive ``robot_state`` / ``robot_active_program``.

    Heuristics:
    * ``executor_running=True`` → ``busy``
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
    executor_state = status.get("executor_state", "IDLE").upper()
    joint_state = status.get("joint_state") or {}
    gripper_state = status.get("gripper_state") or {}
    velocities = joint_state.get("velocities", [])

    # Determine busy/idle from joint velocities
    velocity_threshold = 0.01  # rad/s — above this we consider the robot moving
    is_moving = any(abs(v) > velocity_threshold for v in velocities) if velocities else False

    if executor_state in ("EXECUTING", "PAUSED"):
        robot_state = "busy"
        active_program = _last_active_program  # retain from execute_action_node
    elif is_moving:
        robot_state = "busy"
        active_program = _last_active_program
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

    # Keep a rolling buffer sized for the slowest consumer (the background
    # intent runner, which may sample frames spread across ~frame_skip * N
    # frames). The default of 300 covers ~10 s at 30 fps; override via the
    # ``frame_buffer_size`` runtime config.
    config = state.get("config", {})
    buf_cap = int(config.get("frame_buffer_size", 300))
    buf: list = list(state.get("frames_buffer") or [])
    ts_buf: list = list(state.get("frames_buffer_timestamps") or [])
    fn_buf: list = list(state.get("frames_buffer_frame_nums") or [])

    buf.append(frame_obj.image)
    ts_buf.append(frame_obj.timestamp)
    fn_buf.append(frame_obj.frame_number)

    if len(buf) > buf_cap:
        buf = buf[-buf_cap:]
        ts_buf = ts_buf[-buf_cap:]
        fn_buf = fn_buf[-buf_cap:]

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

    gesture_dict = {
        "dominant_gesture": getattr(gesture_output, "dominant_gesture", None),
        "safety_triggered": getattr(gesture_output, "safety_triggered", False),
    } if gesture_output else {}

    return {
        "human_requesting_help": is_help_requested,
        "monitor_outputs": {"gesture": gesture_dict},
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

    # Merge perception-derived locations into a single generic dict.
    # Any key ending in "_locations" that maps to a {str: str} dict is
    # treated as object-location data — monitors use task-specific names
    # (e.g. "bottle_locations", "tool_locations") and this handles them
    # all without enumerating them here.
    obj_locs = dict(state.get("object_locations") or {})
    for key, val in result.items():
        if key.endswith("_locations") and isinstance(val, dict):
            for obj_id, region in val.items():
                if region != "unknown":
                    obj_locs[obj_id] = region

    # Perception monitors may also emit a "task_state" dict carrying
    # state-schema variables they own (e.g. live xy coords, held flags).
    # Mirror them into the SSG so the decision engine prompt sees them.
    perception_task_state = result.get("task_state") if isinstance(result, dict) else None
    if isinstance(perception_task_state, dict) and perception_task_state:
        ssg = _get_ssg(state)
        for k, v in perception_task_state.items():
            ssg.set_task_state(k, v)

    return {
        "object_locations": obj_locs,
        "monitor_outputs": {"perception": result},
    }


_POSE_FAIL_LIMIT = 2  # after this many consecutive failures, stop invoking


def run_pose_node(state: AuraGraphState) -> dict:
    """Run SAM-3D-Body pose inference on the latest frame.

    Writes summary stats into ``monitor_outputs["pose"]``. Activity
    detection (whether a human is present, etc.) lives in
    ``run_activity_node``, which consumes this output.

    When the pose server is unreachable or repeatedly times out, the
    node marks the monitor as unavailable and stops invoking it. In
    that case ``monitor_outputs["pose"]["available"]`` becomes False
    and the activity / intent gate treats pose as disabled — i.e. the
    intent monitor runs regardless, rather than being blocked forever.
    """
    buf = state.get("frames_buffer") or []
    if not buf:
        return {}

    config = state.get("config", {}) or {}
    key = config.get("config_dir", "default")

    if _pose_disabled.get(key):
        return {
            "monitor_outputs": {
                "pose": {
                    "available": False,
                    "num_persons": 0,
                    "error": "pose server unavailable — activity gate bypassed",
                },
            },
        }

    monitor = _get_pose_monitor(state)
    if monitor is None:
        _pose_disabled[key] = True
        logger.warning("Pose monitor could not be constructed — disabling pose gate")
        return {
            "monitor_outputs": {
                "pose": {"available": False, "num_persons": 0, "error": "init failed"},
            },
        }

    latest_frame = buf[-1]
    error_msg: Optional[str] = None
    result = None
    try:
        try:
            result = asyncio.get_event_loop().run_until_complete(
                monitor.update(frame=latest_frame)
            )
        except RuntimeError:
            result = asyncio.run(monitor.update(frame=latest_frame))
    except Exception as exc:
        error_msg = str(exc)
        logger.debug("Pose monitor call failed: %s", exc)

    is_valid = result is not None and getattr(result, "is_valid", False)
    if not is_valid and error_msg is None and result is not None:
        error_msg = getattr(result, "error", "invalid pose result")

    if not is_valid:
        fails = _pose_failures.get(key, 0) + 1
        _pose_failures[key] = fails
        if fails >= _POSE_FAIL_LIMIT:
            _pose_disabled[key] = True
            logger.warning(
                "Pose monitor failed %d times (last error: %s) — disabling pose gate; "
                "intent monitor will run without pose pre-check",
                fails, error_msg,
            )
        return {
            "monitor_outputs": {
                "pose": {
                    "available": not _pose_disabled.get(key, False),
                    "num_persons": 0,
                    "error": error_msg or "unknown pose error",
                    "consecutive_failures": fails,
                },
            },
        }

    _pose_failures[key] = 0
    assert result is not None  # narrowed by is_valid

    persons_summary = []
    for p in result.persons:
        bbox = p.bbox
        try:
            if hasattr(bbox, "shape") and len(bbox.shape) > 1:
                bbox = bbox[0]
            person_data: Dict[str, Any] = {
                "bbox": [float(v) for v in bbox[:4]],
            }
            # Include 2D keypoints so the activity gate can track motion.
            kpts_2d = p.keypoints_2d
            if kpts_2d is not None:
                if len(kpts_2d.shape) == 3:
                    kpts_2d = kpts_2d[0]
                person_data["keypoints_2d"] = kpts_2d
            persons_summary.append(person_data)
        except Exception:
            continue

    pose_dict = {
        "available": True,
        "num_persons": int(result.num_persons),
        "persons": persons_summary,
        "inference_time_sec": float(getattr(result, "inference_time_sec", 0.0) or 0.0),
    }
    return {"monitor_outputs": {"pose": pose_dict}}


def _serialise_intent_result(result: Any) -> Dict[str, Any]:
    """Convert an :class:`IntentResult` dataclass into a plain dict."""
    return {
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


def _drain_intent_future(key: str) -> Optional[Dict[str, Any]]:
    """If a dispatched intent future has finished, push its result to the slot."""
    fut = _intent_futures.get(key)
    if fut is None or not fut.done():
        return None
    _intent_futures.pop(key, None)
    try:
        result = fut.result()
    except Exception as exc:
        logger.warning("Intent prediction failed: %s", exc)
        return None
    if result is None:
        return None
    serialised = _serialise_intent_result(result)
    push_intent_result(key, serialised)
    return serialised


def _get_activity_gate(state: AuraGraphState):
    """Lazy-init the per-config ActivityGate."""
    from .activity_gate import ActivityGate

    config = state.get("config", {}) or {}
    key = config.get("config_dir", "default")
    if key not in _activity_gates:
        threshold = float(config.get("activity_keypoint_threshold", 15.0))
        _activity_gates[key] = ActivityGate(threshold_px=threshold)
    return _activity_gates[key]


def run_activity_node(state: AuraGraphState) -> dict:
    """Decide whether there is significant human activity.

    Delegates to :class:`~aura.workflow.activity_gate.ActivityGate`
    which tracks 2D keypoint displacement across cycles:

    * **First detection** — any human presence triggers activity.
    * **Subsequent cycles** — mean keypoint displacement must exceed
      ``activity_keypoint_threshold`` (default 15 px) to count as
      significant motion worth running intent on.
    * **Pose unavailable** — gate is bypassed so intent keeps running.
    """
    monitor_outs = state.get("monitor_outputs") or {}
    pose_out = monitor_outs.get("pose") or {}

    gate = _get_activity_gate(state)
    result = gate.evaluate(pose_out)

    activity_dict: Dict[str, Any] = {
        "detected": result.detected,
        "reason": result.reason,
        "kind": result.kind,
    }
    if result.displacement_px is not None:
        activity_dict["keypoint_displacement_px"] = result.displacement_px
    if result.threshold_px is not None:
        activity_dict["threshold_px"] = result.threshold_px

    return {
        "activity_detected": result.detected,
        "monitor_outputs": {"activity": activity_dict},
    }


def run_intent_node(state: AuraGraphState) -> dict:
    """Run RCWPS intent prediction without stalling the fast loop.

    A single-worker thread pool owns the actual ``predict()`` call. Each
    invocation:

    1. Drains the previous future if it has finished, publishing the result
       to the shared slot (consumed by ``update_ssg_node``).
    2. If no call is currently in flight and the gate + min-interval
       throttle agree, samples frames and dispatches a new ``predict()``.
    3. In **realtime** mode (default) returns immediately — the fast loop
       continues while the VLM thinks; the result lands a few cycles later
       via the slot and ``update_ssg_node``.
    4. In **eval** mode (``config["intent_blocking"]=True``) blocks on the
       in-flight future before returning, so every cycle has a fresh
       prediction. Same code path; the only difference is the wait.

    The node returns a small status dict for the dashboard but never writes
    to ``intent_result`` directly — that remains the SSG single-writer's
    responsibility (``update_ssg_node``).
    """
    from .intent_gate import sample_intent_frames, should_run_intent

    config = state.get("config", {}) or {}
    key = config.get("config_dir", "default")
    realtime = bool(config.get("realtime", True))
    blocking = bool(config.get("intent_blocking", not realtime))

    # ── Ground-truth short-circuit ─────────────────────────────────
    # When ``intent_source == "ground_truth"``, skip the VLM entirely
    # and serve pre-annotated state from the GT file. Same output shape
    # (dict pushed onto the intent slot), consumed identically by
    # ``update_ssg_node``. No future, no throttle, no blocking — the
    # lookup is O(log N) on frame_num.
    if (config.get("intent_source") or "llm") == "ground_truth":
        provider = _get_intent_gt_provider(state)
        if provider is not None:
            frame_num = int(state.get("current_frame_num") or 0)
            ts = float(state.get("current_timestamp_sec") or 0.0)
            gt_result = provider.get_at_frame(frame_num, timestamp_sec=ts)
            serialised = _serialise_intent_result(gt_result)
            push_intent_result(key, serialised)
            return {
                "monitor_outputs": {
                    "intent_status": {
                        "source": "ground_truth",
                        "gt_path": str(provider.gt_path),
                        "num_keyframes": provider.num_keyframes,
                        "drained_this_cycle": True,
                        "dispatched_this_cycle": False,
                        "in_flight": False,
                        "blocking": True,
                        "skip_reason": "",
                    }
                },
                "last_predict_time": time.time(),
            }
        raise RuntimeError(
            "intent_source=ground_truth but GT provider unavailable. "
            "Check config_dir / video_path / intent_gt_path and ensure the "
            "GT file exists (see scripts/annotate_ground_truth.py). "
            "Refusing to silently fall back to the live LLM monitor."
        )

    # 1. Pick up any completed future (so the slot has the freshest result).
    drained = _drain_intent_future(key)

    # 2. Maybe dispatch a new call.
    in_flight = key in _intent_futures
    dispatched = False
    skip_reason = ""
    if not in_flight:
        run_now, gate_reason = should_run_intent(state)
        if not run_now:
            skip_reason = gate_reason
        else:
            min_interval = float(config.get("predict_interval", 0.0) or 0.0)
            now = time.monotonic()
            last_ts = _intent_last_dispatch.get(key, 0.0)
            num_frames = int(config.get("intent_num_frames", 5))
            intent_backend = config.get("intent_backend") or config.get("llm_backend", "gemini")
            if realtime:
                num_frames = min(num_frames, 3)
            if intent_backend == "sglang":
                num_frames = min(num_frames, 2)
            frame_skip = int(config.get("frame_skip", 30))
            stride = max(num_frames, 1) * max(frame_skip, 1)
            cur_fn = int(state.get("current_frame_num") or 0)
            last_fn = _intent_last_dispatched_frame.get(key)
            if min_interval > 0 and (now - last_ts) < min_interval:
                # Sleep instead of skip — keeps us under provider rate limits
                # (e.g. Gemini Flash Lite free tier: 15 RPM → 4s spacing) while
                # still dispatching this cycle so non-realtime stride sampling
                # picks up the next [last_fn+frame_skip, …] window rather than
                # racing ahead to the buffer tail.
                wait_s = min_interval - (now - last_ts)
                time.sleep(wait_s)
                now = time.monotonic()
            if last_fn is not None and (cur_fn - last_fn) < stride:
                skip_reason = (
                    f"frame stride ({cur_fn - last_fn} < {stride}; "
                    f"each frame sent to VLM at most once)"
                )
            else:
                frames, frame_nums, timestamps = sample_intent_frames(
                    state.get("frames_buffer") or [],
                    state.get("frames_buffer_frame_nums") or [],
                    state.get("frames_buffer_timestamps") or [],
                    n=num_frames,
                    frame_skip=frame_skip,
                    realtime=realtime,
                )
                if not frames:
                    skip_reason = "no frames sampled"
                else:
                    try:
                        monitor = _get_intent_monitor(state)
                    except Exception as exc:
                        logger.warning("Intent monitor init failed: %s", exc)
                        monitor = None
                    if monitor is not None:
                        window_duration = (
                            timestamps[-1] - timestamps[0] if len(timestamps) >= 2
                            else (timestamps[-1] if timestamps else 0.0)
                        )
                        last_frame_num = (
                            frame_nums[-1] if frame_nums
                            else state.get("current_frame_num", 0)
                        )
                        last_ts_val = (
                            timestamps[-1] if timestamps
                            else state.get("current_timestamp_sec", 0.0)
                        )
                        # Inject externally-sourced state (robot status,
                        # perception results) so the intent monitor's RCWPS
                        # context includes them in its prompt.
                        task_state_snap: Dict[str, Any] = dict(
                            state.get("task_state") or {}
                        )
                        external_vars = monitor.collect_external_state_from_workflow(
                            robot_state=task_state_snap.get("robot_state"),
                            robot_active_program=task_state_snap.get(
                                "robot_active_program"
                            ),
                            object_locations=dict(
                                state.get("object_locations") or {}
                            ),
                        )
                        if external_vars:
                            monitor.inject_external_state(external_vars)
                            logger.debug(
                                "Intent monitor external state injected: %s",
                                list(external_vars.keys()),
                            )

                        # Optionally override the "previous-cycle" state in
                        # the prompt with ground-truth annotations, keyed on
                        # the frame_num of the last dispatched predict().
                        prev_state_source = (
                            config.get("intent_previous_state_source") or "self"
                        )
                        include_prev = bool(
                            config.get("intent_include_previous_state", True)
                        )
                        if prev_state_source == "ground_truth" and include_prev:
                            prev_frame = _intent_last_dispatched_frame.get(key)
                            prev_ts = _intent_last_dispatched_ts.get(key)
                            if prev_frame is not None:
                                try:
                                    gt_provider = _get_intent_gt_provider(state)
                                except Exception as exc:
                                    logger.warning(
                                        "intent_previous_state_source=ground_truth "
                                        "but GT provider unavailable (%s) — "
                                        "falling back to self-tracked previous state",
                                        exc,
                                    )
                                    gt_provider = None
                                if gt_provider is not None:
                                    gt_res = gt_provider.get_at_frame(
                                        prev_frame, timestamp_sec=prev_ts,
                                    )
                                    if gt_res.reasoning != "no_ground_truth":
                                        monitor.set_previous_state(
                                            gt_res.state, timestamp=prev_ts,
                                        )
                                        logger.debug(
                                            "Intent monitor previous state "
                                            "overridden from GT at frame=%d",
                                            prev_frame,
                                        )

                        _intent_last_dispatch[key] = now
                        _intent_last_dispatched_frame[key] = int(last_frame_num)
                        _intent_last_dispatched_ts[key] = float(last_ts_val)
                        _intent_futures[key] = _intent_executor.submit(
                            monitor.predict,
                            frames=frames,
                            timestamp=last_ts_val,
                            frame_num=last_frame_num,
                            window_duration_sec=window_duration,
                        )
                        dispatched = True
                        logger.info(
                            "run_intent_node dispatched — %d frames, span=%.2fs (gate: %s)",
                            len(frames), window_duration, gate_reason,
                        )

    # 3. Eval mode: wait for the in-flight call so this cycle sees fresh intent.
    if blocking and key in _intent_futures:
        fut = _intent_futures[key]
        try:
            result = fut.result()
            if result is not None:
                drained = _serialise_intent_result(result)
                push_intent_result(key, drained)
        except Exception as exc:
            logger.warning("Intent prediction failed (blocking): %s", exc)
        finally:
            _intent_futures.pop(key, None)

    status = {
        "in_flight": key in _intent_futures,
        "dispatched_this_cycle": dispatched,
        "drained_this_cycle": drained is not None,
        "skip_reason": skip_reason,
        "blocking": blocking,
    }
    update: Dict[str, Any] = {
        "monitor_outputs": {"intent_status": status},
    }
    if dispatched:
        update["last_predict_time"] = time.time()

    # Pause gate: when the dashboard pause toggle is on, block here until the
    # user resumes (or stops the workflow). Placed after intent runs/returns
    # so the latest prediction is visible while the rest of the cycle is
    # held — pausing the entire workflow downstream of this node.
    try:
        from aura.dashboard import get_dashboard
        dash = get_dashboard()
    except Exception:
        dash = None
    if dash is not None:
        while getattr(dash, "paused", False) and not getattr(dash, "stop_requested", False):
            time.sleep(0.1)

    return update


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

    # ── Drain any new intent_result from the background runner ───────
    config_dir = state.get("config", {}).get("config_dir", "")
    bg_intent = pop_intent_result(config_dir) if config_dir else None
    intent = bg_intent if bg_intent is not None else state.get("intent_result")

    # ── Sync intent result into SSG ────────────────────────────────
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

        # Apply deferred skill effects once the robot has completed
        # the dispatched program. A completion is: we observed the
        # executor busy at least once since dispatch, and it is now
        # idle. Pending effects were recorded by execute_action_node
        # and withheld until this point.
        pending = list(ssg.task_state.get("pending_skill_effects") or [])
        if pending:
            if robot_status["robot_state"] == "busy":
                for entry in pending:
                    entry["observed_busy"] = True
                ssg.set_task_state("pending_skill_effects", pending)
            elif robot_status["robot_state"] == "idle":
                ready = [e for e in pending if e.get("observed_busy")]
                remaining = [e for e in pending if not e.get("observed_busy")]
                for entry in ready:
                    for effect_key, effect_val in (entry.get("effects") or {}).items():
                        _apply_skill_effect(ssg, effect_key, effect_val)
                if ready:
                    ssg.set_task_state("pending_skill_effects", remaining)

    # ── Sync perception-detected object locations ──────────────────
    # The perception node consolidates all monitor location outputs into
    # state["object_locations"] (a generic {obj_id: region} dict).
    # That is the sole source written here — task-specific key names
    # never appear in this file.
    #
    # Perception is the sole authority for task_state location variables
    # (source: "perception" in state_schema).  Skill effects do NOT write
    # here — a mismatch between what the robot intended and what the camera
    # sees is meaningful signal (skill failure, object still in motion).
    # Monitors should emit "unknown" for any object they are responsible
    # for but did not detect this frame, so task_state never retains a
    # stale confident-but-outdated value.
    object_locations = state.get("object_locations") or {}
    for obj_id, region in object_locations.items():
        # Only known SSG nodes get mirrored to state-schema vars.
        # Generic fallback ids (emitted before identity is resolved) are
        # present in the raw monitor output but not registered as nodes.
        if obj_id not in ssg._nodes:
            continue
        ssg.set_task_state(f"{obj_id}_location", region)
        # SSG location edges only updated when detection is confident.
        # "unknown" leaves the edge at its last known value so graph
        # reasoner queries remain usable during momentary occlusions.
        if region != "unknown":
            try:
                ssg.set_location(obj_id, region)
            except Exception as exc:
                logger.debug(
                    "Could not set SSG location for %s → %s: %s",
                    obj_id, region, exc,
                )

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

    # Only add slots this node produced — the monitor_outputs reducer will
    # merge them on top of whatever the parallel sensing branches wrote.
    new_monitor_slots: Dict[str, Any] = {}
    if sound_dict:
        new_monitor_slots["sound"] = sound_dict
    # If a fresh intent result came in via the slot, mirror it into
    # monitor_outputs so the dashboard sees the latest prediction.
    if bg_intent is not None:
        new_monitor_slots["intent"] = bg_intent

    updates: Dict[str, Any] = {
        "ssg": ssg.to_dict(),
        "task_state": dict(ssg.task_state),
        "completed_steps": completed_steps,
        "object_locations": obj_locs,
    }
    if new_monitor_slots:
        updates["monitor_outputs"] = new_monitor_slots
    # Propagate freshly drained intent into flat state so decide_action
    # reads it on the same cycle.
    if bg_intent is not None:
        updates["intent_result"] = bg_intent
        print(f"  [update_ssg_node] Returning intent_result: current_action={bg_intent.get('current_action')}, state keys={list(bg_intent.keys())}")
    else:
        print(f"  [update_ssg_node] No bg_intent drained, returning: {list(updates.keys())}")
    return updates


def _ssg_change_digest(state: AuraGraphState) -> str:
    """Return a stable hash over the fields that should trigger a decision.

    Hashing the full SSG is noisy (timestamps, snapshot counters). Instead
    we digest a curated slice: object locations, robot state, completed
    steps, and the current intent phase/action. That keeps change detection
    meaningful without over-firing on cosmetic updates.
    """
    import hashlib
    ssg_dict = state.get("ssg") or {}
    ts = ssg_dict.get("task_state") or {}
    intent = state.get("intent_result") or {}

    payload = {
        "object_locations": dict(state.get("object_locations") or {}),
        "completed_steps": sorted(state.get("completed_steps") or []),
        "robot_state": ts.get("robot_state"),
        "robot_active_program": ts.get("robot_active_program"),
        # intent — include every field so any meaningful change triggers a decision.
        # List fields are sorted for stable hashing; everything else is kept as-is.
        **{
            k: (sorted(v) if isinstance(v, list) else v)
            for k, v in intent.items()
        },
        "human_requesting_help": bool(state.get("human_requesting_help")),
        "activity_detected": bool(state.get("activity_detected")),
    }
    # DEBUG PRINT: Print the payload being hashed so the user can inspect it
    # print(f"  [_ssg_change_digest] Payload generating hash:\n    {payload}")
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def check_ssg_change_node(state: AuraGraphState) -> dict:
    """Compute whether the SSG has changed since the last cycle.

    Writes ``ssg_changed`` + refreshed ``last_ssg_hash`` so the router can
    decide whether to run ``decide_action`` this tick. The first cycle is
    always treated as a change so the initial decision runs.
    """
    new_hash = _ssg_change_digest(state)
    prev_hash = state.get("last_ssg_hash") or ""
    changed = (new_hash != prev_hash)
    
    # print(f"  [check_ssg_change_node] prev_hash={prev_hash}, new_hash={new_hash}, changed={changed}") # DEBUG PRINT: Show the previous and new hash values and whether a change was detected
    # if not changed:
    #     print(f"  [check_ssg_change_node] No SSG change detected. Skipping decide_action.")
        
    return {
        "ssg_changed": changed,
        "last_ssg_hash": new_hash,
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
    config = state.get("config") or {}
    use_intent = "intent" in config.get("active_monitors", ["intent", "gesture"])

    print(f"  [decide_action_node] Called. intent_result present: {intent_result is not None}, use_intent: {use_intent}")
    if use_intent and not intent_result:
        print(f"  [decide_action_node] No intent_result (and use_intent is True); returning empty dict")
        return {}

    intent_result = intent_result or {}

    timestamp = state.get("current_timestamp_sec", 0.0)
    help_requested = state.get("human_requesting_help", False)

    actions: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []

    if help_requested:
        reasoning_parts.append("Human requesting help (gesture detected)")

    # ── Get the current frame if available ──────────────────────────
    current_frame = None
    frames = state.get("frames_buffer", [])
    if frames:
        current_frame = frames[-1]

    # ── Brain decision engine ───────────────────────────────────────
    speech_message: Optional[str] = None
    engine = None
    try:
        engine = _get_decision_engine(state)
        prediction = await engine.decide_action(current_time_sec=timestamp, current_frame=current_frame)
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
                    "parameters": prediction.parameters,
                })
                reasoning_parts.append(
                    f"Brain: {prediction.action_id}"
                )
    except Exception as e:
        logger.warning("Brain engine error: %s", e, exc_info=True)

    # ── Push speech to human via audio bridge ─────────────────────
    audio_bridge = get_audio_bridge()
    if speech_message and audio_bridge is not None and audio_bridge.is_active:
        audio_bridge.push_speech(speech_message)
        logger.info("Decision engine → human: %s", speech_message[:100])

    # Surface Behaviour-Tree tick state so the dashboard can show it.
    bt_info: Dict[str, Any] = {}
    if engine is not None:
        bt_info = {
            "decision_mode": getattr(engine.config, "decision_mode", "hybrid"),
            "bt_trail": getattr(engine, "_last_bt_reasoning", "") or "",
            "bt_branch": getattr(engine, "_last_bt_branch", "") or "",
            "bt_llm_invoked": bool(getattr(engine, "_last_bt_llm_invoked", False)),
        }

    decision_record = {
        "timestamp_sec": timestamp,
        "frame_num": state.get("current_frame_num", 0),
        "actions": actions,
        "speech_message": speech_message,
        "reasoning": " | ".join(reasoning_parts) or "No action needed",
        "decided_at": datetime.now().isoformat(),
        **bt_info,
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

        # ── Speech action type — vocalize via Kokoro TTS, no robot call ──
        # ``announce`` is a first-class action the decision engine may emit
        # to speak to the human. It carries the text in parameters["text"]
        # (falling back to the action reason).
        if action_type == "announce":
            from aura.interfaces.tts import speak as _tts_speak
            text = (action.get("parameters") or {}).get("text") or action.get("reason") or ""
            _tts_speak(text)
            result["success"] = True
            result["mode"] = "tts"
            logger.info("[TTS] announce: %s", text[:120])
            executed.append(result)
            continue

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
                if api_call and "program_file" in api_call:
                    # Program-executor shape: translate to /api/program/execute.
                    # api_call: {"program_file": "<file>.prog",
                    #            "program_args": "k1=v1 k2=v2"} (template)
                    # Args may contain {param} placeholders filled from params.
                    program = api_call["program_file"]
                    args_template = api_call.get("program_args", "") or ""
                    try:
                        args_str = args_template.format(**params) if params else args_template
                    except KeyError as exc:
                        raise RuntimeError(
                            f"Missing parameter {exc} for skill '{action_type}' args '{args_template}'"
                        ) from exc
                    args_dict: Dict[str, str] = {}
                    for token in args_str.split():
                        if "=" in token:
                            k, v = token.split("=", 1)
                            args_dict[k] = v
                    body = {"program": program, "args": args_dict} if args_dict else {"program": program}
                    resp = robot._post("/api/program/execute", body)
                elif api_call and "service" in api_call:
                    # Bare ROS service trigger (e.g. stop). Map known ones to
                    # dedicated REST endpoints; fall back to a generic POST.
                    service = api_call["service"]
                    if service.endswith("/stop"):
                        resp = robot._post("/api/program/stop", {})
                    elif service.endswith("/pause"):
                        resp = robot._post("/api/program/pause", {})
                    else:
                        resp = {"success": False, "error": f"unmapped service {service}"}
                elif api_call:
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

            # Live mode: vocalize the dispatched command via Kokoro TTS.
            if result.get("success"):
                try:
                    from aura.interfaces.tts import speak as _tts_speak
                    announcement = _command_announcement(
                        skill, action_type, obj_name, params,
                    )
                    _tts_speak(announcement)
                    logger.info("[TTS] command: %s", announcement)
                except Exception as exc:
                    logger.debug("TTS announcement skipped: %s", exc)

        # Live mode: defer skill effects until the robot program
        # actually completes. Record them as pending on the SSG; the
        # next update_ssg_node cycle applies them once the poll
        # observes the executor returning to idle.
        # Dry-run: do NOT apply effects at all — perception on the
        # input video is the sole authority for object locations.
        if not dry_run and result.get("success") and skill and skill.effects:
            pending = list(ssg.task_state.get("pending_skill_effects") or [])
            pending.append({
                "action_type": action_type,
                "effects": dict(skill.effects),
            })
            ssg.set_task_state("pending_skill_effects", pending)

        executed.append(result)

    # In live mode, leave robot BUSY so the next _robot_status_from_api
    # poll is authoritative (avoids a stale-IDLE window before the
    # motion has physically finished). In dry-run, flip back to IDLE
    # since there is no poll to correct it.
    if dry_run:
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
    max_cycles = state.get("config", {}).get("max_cycles", 1500)

    # Termination is driven only by video EOF (set by capture_frame_node),
    # explicit error, or max_cycles. The LLM's ``task_complete`` prediction
    # is intentionally ignored — for evaluation runs we want the loop to
    # drain the entire video regardless of when the model thinks the task
    # has ended.
    is_complete = state.get("is_complete", False)

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

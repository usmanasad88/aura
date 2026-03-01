"""Config-driven LangGraph builder for AURA workflows.

Given a task config directory, :func:`build_task_graph` returns a
compiled ``StateGraph`` wired according to:

* ``task_profile.json`` → ``workflow_config.graph_topology``
* Available monitor nodes (intent, gesture, perception, …)
* Conditional edges for action execution routing

The default topology ``sense_decide_act`` produces::

    capture_frame → run_gesture → run_intent → update_ssg
          ↑                                        ↓
          │                                   decide_action
          │                                    │         │
          │                              (has action)  (no)
          │                                    ↓         │
          │                             execute_action   │
          │                                    │         │
          └────── check_complete ←─────────────┘─────────┘
                   │          │
              (done)       (loop)
                END      capture_frame

The system runs continuously — gesture detection runs every cycle and
sets ``human_requesting_help`` to signal the decision engine.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy LangGraph imports for environments without it
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    END = None
    MemorySaver = None


def build_task_graph(
    config_dir: str | Path,
    *,
    dry_run: bool = True,
    video_path: str | None = None,
    webcam_device: int | str | None = None,
    robot_url: str = "http://localhost:5050",
    speed: float = 1.0,
    model: str = "gemini-2.5-flash",
    enable_voice: bool = False,
    headless: bool = True,
    extra_config: Dict[str, Any] | None = None,
) -> tuple:
    """Build a compiled LangGraph and initial state from task config.

    Args:
        config_dir: Path to the task's ``config/`` directory.
        dry_run: If True, robot actions are logged, not executed.
        video_path: Path to source video (mutually exclusive with webcam).
        webcam_device: Webcam device index or path.
        robot_url: Base URL for the robot control HTTP API.
        speed: Playback speed (for video sources).
        model: Gemini model name for intent prediction.
        enable_voice: Enable TTS voice announcements.
        headless: Suppress visualisation windows.
        extra_config: Additional config keys to merge in.

    Returns:
        ``(compiled_graph, initial_state)`` ready for ``graph.invoke()``
        or ``graph.astream()``.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is required.  Install with:  uv add langgraph"
        )

    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # ── Load task artefacts ──────────────────────────────────────────
    dag = _load_json(config_dir / "dag.json")
    task_profile = _load_json(config_dir / "task_profile.json")
    state_schema = _load_json(config_dir / "state_schema.json")

    wf_cfg = task_profile.get("workflow_config", {})
    topology = wf_cfg.get("graph_topology", "sense_decide_act")
    decision_mode = wf_cfg.get("decision_mode", "hybrid")
    active_monitors = wf_cfg.get("active_monitors", ["intent", "gesture"])

    # ── Build initial SSG snapshot ───────────────────────────────────
    initial_ssg = _build_initial_ssg(config_dir, state_schema)

    # ── Build initial flat task_state ────────────────────────────────
    task_state: Dict[str, Any] = {}
    for var, defn in state_schema.get("state_variables", {}).items():
        task_state[var] = defn.get("default") if isinstance(defn, dict) else defn

    # ── Build initial object locations ───────────────────────────────
    env = task_profile.get("environment", {})
    movable = env.get("movable_objects", [])
    initial_delivery = set(env.get("initial_delivery_objects", []))
    obj_locs = {
        obj: ("storage" if obj in initial_delivery else "workplace")
        for obj in movable
    }

    # ── Merge runtime config ─────────────────────────────────────────
    runtime_config = {
        "task_name": task_profile.get("task_name", config_dir.parent.name),
        "config_dir": str(config_dir),
        "video_path": video_path,
        "webcam_device": int(webcam_device) if webcam_device is not None else None,
        "robot_url": robot_url,
        "dry_run": dry_run,
        "speed": speed,
        "model": model,
        "enable_voice": enable_voice,
        "decision_mode": decision_mode,
        "active_monitors": active_monitors,
        "headless": headless,
        "predict_interval": wf_cfg.get("predict_interval_sec", 3.0),
        "resume_gestures": wf_cfg.get("resume_gestures", ["Thumb_Up"]),
        **(extra_config or {}),
    }

    # ── Assemble initial state ───────────────────────────────────────
    from .state import AuraGraphState

    initial_state: AuraGraphState = {
        "dag": dag,
        "task_profile": task_profile,
        "state_schema": state_schema,
        "ssg": initial_ssg,
        "task_state": task_state,
        "monitor_outputs": {},
        "frames_buffer": [],
        "current_frame_num": 0,
        "current_timestamp_sec": 0.0,
        "intent_result": None,
        "pending_actions": [],
        "last_decision": None,
        "decision_history": [],
        "completed_steps": [],
        "object_locations": obj_locs,
        "is_complete": False,
        "human_requesting_help": False,
        "last_predict_time": 0.0,
        "error": None,
        "cycle_count": 0,
        "config": runtime_config,
    }

    # ── Build graph topology ─────────────────────────────────────────
    compiled = _build_sense_decide_act(active_monitors)

    return compiled, initial_state


# ═══════════════════════════════════════════════════════════════════════════
#  Graph topology builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_sense_decide_act(
    active_monitors: list,
) -> Any:
    """Build the standard ``sense → decide → act`` graph.

    The system runs continuously:
    * Gesture detection runs every cycle (cheap MediaPipe call)
    * Intent prediction runs on a time-based throttle
    * The decision engine always evaluates, using gesture state as a signal

    Returns a compiled ``StateGraph`` with memory checkpointer.
    """
    from .state import AuraGraphState
    from .nodes import (
        capture_frame_node,
        run_gesture_node,
        run_intent_node,
        update_ssg_node,
        decide_action_node,
        execute_action_node,
        check_complete_node,
    )

    workflow = StateGraph(AuraGraphState)

    # ── Add nodes ────────────────────────────────────────────────────
    workflow.add_node("capture_frame", capture_frame_node)

    use_gesture = "gesture" in active_monitors
    if use_gesture:
        workflow.add_node("run_gesture", run_gesture_node)

    workflow.add_node("run_intent", run_intent_node)
    workflow.add_node("update_ssg", update_ssg_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("check_complete", check_complete_node)

    # ── Wire edges ───────────────────────────────────────────────────
    workflow.set_entry_point("capture_frame")

    if use_gesture:
        # Capture → Gesture → Intent (always linear, no gating)
        def _capture_router(state: dict) -> str:
            if state.get("is_complete"):
                return "check_complete"
            return "run_gesture"

        workflow.add_conditional_edges(
            "capture_frame",
            _capture_router,
            {
                "run_gesture": "run_gesture",
                "check_complete": "check_complete",
            },
        )
        workflow.add_edge("run_gesture", "run_intent")
    else:
        # No gesture monitor — capture straight to intent
        def _capture_router_no_gesture(state: dict) -> str:
            if state.get("is_complete"):
                return "check_complete"
            return "run_intent"

        workflow.add_conditional_edges(
            "capture_frame",
            _capture_router_no_gesture,
            {
                "run_intent": "run_intent",
                "check_complete": "check_complete",
            },
        )

    # intent → ssg → decide
    workflow.add_edge("run_intent", "update_ssg")
    workflow.add_edge("update_ssg", "decide_action")

    # Conditional: execute or skip
    def _action_router(state: dict) -> str:
        actions = state.get("pending_actions") or []
        if actions:
            return "execute_action"
        return "check_complete"

    workflow.add_conditional_edges(
        "decide_action",
        _action_router,
        {
            "execute_action": "execute_action",
            "check_complete": "check_complete",
        },
    )

    workflow.add_edge("execute_action", "check_complete")

    # Loop or end
    def _loop_or_end(state: dict) -> str:
        if state.get("is_complete") or state.get("error"):
            return END
        return "capture_frame"

    workflow.add_conditional_edges(
        "check_complete",
        _loop_or_end,
        {
            END: END,
            "capture_frame": "capture_frame",
        },
    )

    # ── Compile ──────────────────────────────────────────────────────
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("Built 'sense_decide_act' graph (monitors=%s)",
                active_monitors)
    return compiled


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _build_initial_ssg(
    config_dir: Path,
    state_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct an initial SSG snapshot from config files.

    If ``initial_scene.json`` exists, uses the Brain's
    ``DecisionEngine._initialize_scene`` pattern — creating nodes,
    edges, and populating task_state defaults.

    Returns the ``graph.to_dict()`` representation.
    """
    from aura.core.scene_graph import SemanticSceneGraph
    from aura.core.scene_graph.nodes import (
        ObjectNode, AgentNode, RegionNode,
        NodeType, Affordance,
    )
    from aura.core.scene_graph.edges import SpatialRelation

    ssg = SemanticSceneGraph(name="aura_ssg")

    # Initialise task state from schema defaults
    ssg.initialize_task_state(state_schema)

    scene_path = config_dir / "initial_scene.json"
    if not scene_path.exists():
        return ssg.to_dict()

    with open(scene_path) as f:
        scene = json.load(f)

    # Regions
    for r in scene.get("regions", []):
        node = RegionNode.from_dict({
            "id": r["id"], "name": r["name"], "node_type": "REGION", **r
        })
        ssg.add_node(node)

    # Objects
    for o in scene.get("objects", []):
        node = ObjectNode.from_dict({
            "id": o["id"], "name": o["name"], "node_type": "OBJECT", **o
        })
        for aff in o.get("affordances", []):
            node.add_affordance(Affordance(**aff))
        ssg.add_node(node)
        if "initial_location" in o:
            ssg.set_location(o["id"], o["initial_location"])

    # Agents
    for a in scene.get("agents", []):
        node = AgentNode.from_dict({
            "id": a["id"], "name": a["name"], "node_type": "AGENT", **a
        })
        ssg.add_node(node)

    # Explicit initial edges
    for e in scene.get("initial_edges", []):
        relation_str = e.get("relation", "at").upper()
        try:
            relation = SpatialRelation[relation_str]
        except KeyError:
            relation = SpatialRelation.AT
        from aura.core.scene_graph.edges import SSGEdge
        edge = SSGEdge.spatial(e["source_id"], e["target_id"], relation)
        ssg.add_edge(edge)

    logger.info("Built initial SSG with %d nodes from %s",
                ssg.node_count, scene_path.name)
    return ssg.to_dict()

"""Config-driven LangGraph builder for AURA workflows.

Given a task config directory, :func:`build_task_graph` returns a
compiled ``StateGraph`` wired according to:

* ``task_profile.json`` → ``workflow_config.graph_topology``
* Available monitor nodes (intent, gesture, perception, pose, …)
* Conditional edges for action execution routing

The default topology ``sense_decide_act`` is a *fast perception loop*.
Intent prediction (the slow VLM call) runs in a parallel asyncio task
— see :class:`aura.workflow.background_intent.BackgroundIntentRunner`
— and injects results into the SSG via a shared slot read by
``update_ssg_node``. The in-graph edges are::

    capture_frame → run_gesture → run_perception → run_pose → update_ssg
                                                                  ↓
                                                          check_ssg_change
                                                          │              │
                                                     (changed)       (same)
                                                          ↓              │
                                                    decide_action        │
                                                     │        │          │
                                               (has action) (none)       │
                                                     ↓        │          │
                                              execute_action  │          │
                                                     │        │          │
                                                     └────────┴──────────┘
                                                              ↓
                                                        check_complete
                                                         │        │
                                                     (done)    (loop)
                                                       END   capture_frame

The ``run_gesture`` / ``run_perception`` / ``run_pose`` nodes are each
optional, driven by ``active_monitors`` in the task profile or a
runtime override.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

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
    model: str = "gemini-3.1-pro-preview",
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

    # Allow runtime override (launcher toggles perception / gesture on/off).
    override = (extra_config or {}).get("active_monitors_override")
    if override is not None:
        active_monitors = list(override)

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
    # Load per-backend defaults (intent_max_tokens, decision_max_tokens, …)
    # from config/default.yaml so callers don't have to pass them manually.
    llm_backend = (extra_config or {}).get("llm_backend", wf_cfg.get("llm_backend", "gemini"))
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    _default_yaml = _repo_root / "config" / "default.yaml"
    _default_cfg = yaml.safe_load(_default_yaml.read_text()) if _default_yaml.exists() else {}
    _backend_defaults: Dict[str, Any] = _default_cfg.get("backend_defaults", {}).get(llm_backend, {})

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
        "max_cycles": wf_cfg.get("max_cycles_sec", 500),
        **_backend_defaults,
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
        "frames_buffer_timestamps": [],
        "frames_buffer_frame_nums": [],
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
        "activity_detected": False,
        "last_predict_time": 0.0,
        "last_ssg_hash": "",
        "ssg_changed": False,
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
    """Build the fast-perception ``sense → decide → act`` graph with
    parallel sensing branches.

    Topology per iteration::

        capture_frame ──┬── run_pose → run_activity → run_intent ─┐
                        ├── run_perception ────────────────────────┤
                        └── run_gesture ───────────────────────────┤
                                                                    ↓
                                                              update_ssg
                                                                    ↓
                                                          check_ssg_change
                                                            │           │
                                                       (changed)     (same)
                                                            ↓           │
                                                       decide_action    │
                                                         │      │       │
                                                  (action) (none)       │
                                                         ↓      └───────┤
                                                  execute_action        │
                                                         └──────────────┤
                                                                        ↓
                                                                check_complete
                                                                  │       │
                                                              (done)  (loop)
                                                                END   capture_frame

    Branches run **in parallel**; LangGraph waits on all of them at
    ``update_ssg``. The intent branch is the longest pole — but the intent
    node itself dispatches the slow VLM call to a worker thread and returns
    immediately in realtime mode (``run_intent_node``), so the join doesn't
    stall on the LLM. In offline / eval mode the intent node blocks, giving
    deterministic predictions per cycle.

    Monitor toggles (drawn from ``active_monitors``):
      * ``"gesture"``    — MediaPipe gesture (cheap, ~30 ms)
      * ``"perception"`` — Task-specific perception (tens of ms)
      * ``"pose"``       — SAM-3D-Body pose (100 ms – 1 s)
      * ``"activity"``   — Cheap classifier on top of pose; gates intent
      * ``"intent"``     — Slow RCWPS VLM call (multi-second; async-dispatched)
    """
    from .state import AuraGraphState
    from .nodes import (
        capture_frame_node,
        run_gesture_node,
        run_perception_node,
        run_pose_node,
        run_activity_node,
        run_intent_node,
        update_ssg_node,
        check_ssg_change_node,
        decide_action_node,
        execute_action_node,
        check_complete_node,
    )

    if StateGraph is None or END is None or MemorySaver is None:
        raise ImportError("LangGraph not available")

    workflow = StateGraph(AuraGraphState)

    use_gesture = "gesture" in active_monitors
    use_perception = "perception" in active_monitors
    use_pose = "pose" in active_monitors
    use_activity = "activity" in active_monitors
    use_intent = "intent" in active_monitors

    # ── Build the per-branch chains (each is a list of node names; items
    #    after the first run sequentially within the branch). ──
    branches: list[list[str]] = []

    intent_chain: list[str] = []
    if use_pose:
        intent_chain.append("run_pose")
    if use_activity:
        intent_chain.append("run_activity")
    if use_intent:
        intent_chain.append("run_intent")
    if intent_chain:
        branches.append(intent_chain)

    if use_perception:
        branches.append(["run_perception"])
    if use_gesture:
        branches.append(["run_gesture"])

    # ── Register every node referenced by at least one branch ──
    NODE_FNS: dict[str, Any] = {
        "run_pose": run_pose_node,
        "run_activity": run_activity_node,
        "run_intent": run_intent_node,
        "run_perception": run_perception_node,
        "run_gesture": run_gesture_node,
    }
    workflow.add_node("capture_frame", capture_frame_node)
    branch_node_names: set[str] = set()
    for chain in branches:
        for n in chain:
            if n not in branch_node_names:
                workflow.add_node(n, NODE_FNS[n])
                branch_node_names.add(n)

    workflow.add_node("update_ssg", update_ssg_node)
    workflow.add_node("check_ssg_change", check_ssg_change_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("check_complete", check_complete_node)

    workflow.set_entry_point("capture_frame")

    # ── capture_frame fan-out → all branch heads in parallel ──
    branch_heads = [chain[0] for chain in branches]

    if branch_heads:
        # Conditional router: short-circuit to check_complete on EOF, else
        # return the list of branch heads so LangGraph runs them in parallel.
        def _capture_router(state: dict) -> Any:
            if state.get("is_complete"):
                return "check_complete"
            return branch_heads  # list → parallel fan-out

        path_map: dict[str, str] = {h: h for h in branch_heads}
        path_map["check_complete"] = "check_complete"
        workflow.add_conditional_edges(
            "capture_frame",
            _capture_router,
            path_map,
        )
    else:
        # No sensing monitors at all — just go straight to update_ssg.
        def _capture_passthrough(state: dict) -> str:
            return "check_complete" if state.get("is_complete") else "update_ssg"

        workflow.add_conditional_edges(
            "capture_frame",
            _capture_passthrough,
            {"update_ssg": "update_ssg", "check_complete": "check_complete"},
        )

    # ── Within-branch sequential edges + join at update_ssg ──
    for chain in branches:
        for prev, nxt in zip(chain, chain[1:]):
            workflow.add_edge(prev, nxt)
        # Last node of the branch joins update_ssg.
        workflow.add_edge(chain[-1], "update_ssg")

    # ── update_ssg → check_ssg_change → (decide_action | check_complete) ──
    workflow.add_edge("update_ssg", "check_ssg_change")

    def _change_router(state: dict) -> str:
        if state.get("ssg_changed"):
            return "decide_action"
        return "check_complete"

    workflow.add_conditional_edges(
        "check_ssg_change",
        _change_router,
        {
            "decide_action": "decide_action",
            "check_complete": "check_complete",
        },
    )

    # ── decide_action → (execute_action | check_complete) ───────────
    def _action_router(state: dict) -> str:
        actions = state.get("pending_actions") or []
        return "execute_action" if actions else "check_complete"

    workflow.add_conditional_edges(
        "decide_action",
        _action_router,
        {
            "execute_action": "execute_action",
            "check_complete": "check_complete",
        },
    )

    workflow.add_edge("execute_action", "check_complete")

    # ── Loop or end ─────────────────────────────────────────────────
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
    # Do NOT use MemorySaver: it deep-copies the full state (including the
    # frames_buffer of raw numpy arrays) on every cycle, causing unbounded
    # memory growth that hangs the system in realtime mode.
    compiled = workflow.compile()
    logger.info(
        "Built 'sense_decide_act' graph (monitors=%s, branches=%s)",
        active_monitors, branches,
    )
    return compiled


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Any:
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
        NodeType,
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

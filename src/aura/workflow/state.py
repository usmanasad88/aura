"""Unified LangGraph state definition for AURA workflows.

Defines :class:`AuraGraphState` — a single ``TypedDict`` that carries
ALL runtime context through the LangGraph execution cycle.  The state
combines:

* **SSG snapshot** — serialised ``SemanticSceneGraph`` (nodes, edges,
  task_state) so that every node can read/write the shared truth.
* **Task artefacts** — DAG, task_profile, state_schema loaded at init.
* **Monitor outputs** — latest raw results from each active monitor.
* **Flat task_state** — mirrored from SSG for backward compat with the
  RCWPS intent monitor and rule-based decision engine.
* **Decision history** — append-only log using LangGraph's ``operator.add``
  reducer so concurrent branches merge cleanly.
* **Workflow control** — flags for completion, errors, gating, etc.
"""

from __future__ import annotations

import operator
from typing import Any, Annotated, Dict, List, Optional, TypedDict


# ─── Nested sub-state dicts ─────────────────────────────────────────────────

class SSGSnapshot(TypedDict, total=False):
    """Serialised Semantic Scene Graph (``graph.to_dict()`` output)."""
    name: str
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]
    task_state: Dict[str, Any]
    last_updated: str


class MonitorOutputs(TypedDict, total=False):
    """Latest raw outputs keyed by monitor name."""
    intent: Dict[str, Any]
    gesture: Dict[str, Any]
    perception: Dict[str, Any]
    motion: Dict[str, Any]
    sound: Dict[str, Any]
    affordance: Dict[str, Any]
    performance: Dict[str, Any]
    pose_tracking: Dict[str, Any]


class WorkflowConfig(TypedDict, total=False):
    """Runtime parameters injected by the entry-point script."""
    task_name: str
    config_dir: str
    video_path: Optional[str]
    webcam_device: Optional[int]
    robot_url: str
    dry_run: bool
    speed: float
    predict_interval: float
    model: str
    enable_voice: bool
    decision_mode: str         # "hybrid" | "llm" | "rules"
    active_monitors: List[str] # ["intent","gesture","perception", …]
    headless: bool
    max_frames: Optional[int]
    frame_skip: int


# ─── Main state TypedDict ───────────────────────────────────────────────────

class AuraGraphState(TypedDict, total=False):
    """Complete LangGraph state for a single AURA workflow cycle.

    Every node function ``f(state) -> partial_state`` receives this type
    and returns a *partial* dict with only the keys it wants to update.

    The ``decision_history`` field uses ``Annotated[…, operator.add]`` so
    that multiple nodes (or parallel branches) can append entries without
    overwriting each other.
    """

    # ── task artefacts (immutable after init) ──────────────────────────
    dag: Dict[str, Any]
    task_profile: Dict[str, Any]
    state_schema: Dict[str, Any]

    # ── Semantic Scene Graph ───────────────────────────────────────────
    ssg: SSGSnapshot

    # ── flat task state (synced with ssg.task_state each cycle) ────────
    task_state: Dict[str, Any]

    # ── monitor outputs (latest per cycle) ─────────────────────────────
    monitor_outputs: MonitorOutputs

    # ── frame / video state ────────────────────────────────────────────
    frames_buffer: List[Any]           # recent cv2 images (numpy arrays)
    current_frame_num: int
    current_timestamp_sec: float

    # ── intent result (from RCWPS intent monitor) ──────────────────────
    intent_result: Optional[Dict[str, Any]]

    # ── decision & action ──────────────────────────────────────────────
    pending_actions: List[Dict[str, Any]]
    last_decision: Optional[Dict[str, Any]]
    decision_history: Annotated[List[Dict[str, Any]], operator.add]

    # ── completed step tracking ────────────────────────────────────────
    completed_steps: List[str]
    object_locations: Dict[str, str]

    # ── workflow control ───────────────────────────────────────────────
    is_complete: bool
    human_requesting_help: bool  # True when gesture (e.g. Thumb_Up) detected
    last_predict_time: float     # epoch time of last intent prediction
    error: Optional[str]
    cycle_count: int

    # ── runtime config ─────────────────────────────────────────────────
    config: WorkflowConfig

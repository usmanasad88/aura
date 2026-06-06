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
    pose: Dict[str, Any]
    activity: Dict[str, Any]


def _merge_dicts(
    a: Optional[Dict[str, Any]],
    b: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Shallow dict-merge reducer for keys that parallel branches may each
    contribute to without overwriting each other. Inner-key collisions
    resolve to the latest writer; top-level slots survive.
    """
    return {**(a or {}), **(b or {})}


_merge_monitor_outputs = _merge_dicts


def _take_latest(a: Any, b: Any) -> Any:
    """Reducer: prefer the later write; fall back to the earlier if later is None."""
    return b if b is not None else a


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
    use_ground_truth_robot_status: bool
    offline_realtime: bool     # video file: paced playback vs as-fast-as-possible
    frame_buffer_size: int
    intent_num_frames: int
    pose_server_endpoint: str
    intent_blocking: bool        # True for offline eval — wait for intent each cycle


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
    dag: Any  # list[dict] — flat array of {id, description, dependencies}
    task_profile: Dict[str, Any]
    state_schema: Dict[str, Any]

    # ── Semantic Scene Graph ───────────────────────────────────────────
    ssg: Annotated[SSGSnapshot, _take_latest]

    # ── flat task state (synced with ssg.task_state each cycle) ────────
    task_state: Annotated[Dict[str, Any], _take_latest]

    # ── monitor outputs (latest per cycle) ─────────────────────────────
    # Reducer merges per-monitor slots so parallel branches each contribute
    # without overwriting each other.
    monitor_outputs: Annotated[MonitorOutputs, _merge_monitor_outputs]

    # ── frame / video state ────────────────────────────────────────────
    frames_buffer: List[Any]           # recent cv2 images (numpy arrays)
    frames_buffer_timestamps: List[float]  # video timestamps for each frame in buffer
    frames_buffer_frame_nums: List[int]    # frame numbers for each frame in buffer
    current_frame_num: int
    current_timestamp_sec: float

    # ── intent result (from RCWPS intent monitor) ──────────────────────
    intent_result: Annotated[Optional[Dict[str, Any]], _take_latest]

    # ── decision & action ──────────────────────────────────────────────
    pending_actions: Annotated[List[Dict[str, Any]], _take_latest]
    last_decision: Annotated[Optional[Dict[str, Any]], _take_latest]
    decision_history: Annotated[List[Dict[str, Any]], operator.add]

    # ── completed step tracking ────────────────────────────────────────
    completed_steps: Annotated[List[str], _take_latest]
    # Written by run_perception inside a parallel branch; a merge reducer
    # avoids LangGraph's concurrent-update error when other branches also
    # produce state deltas in the same fan-in superstep.
    object_locations: Annotated[Dict[str, str], _merge_dicts]
    # Annotated frame rendered by run_perception (monitor.visualize) for the
    # dashboard. Numpy BGR image; consumed by the runner's set_frame and popped
    # before publishing so it never hits the SSE JSON.
    perception_vis: Annotated[Optional[Any], _take_latest]

    # ── workflow control ───────────────────────────────────────────────
    is_complete: Annotated[bool, _take_latest]
    # Scalars written by single parallel branches still need reducers so
    # LangGraph's concurrent-update check accepts the fan-in superstep.
    human_requesting_help: Annotated[bool, _take_latest]  # set by run_gesture
    activity_detected: Annotated[bool, _take_latest]      # set by run_activity
    last_predict_time: Annotated[float, _take_latest]     # set by run_intent dispatch
    last_ssg_hash: str           # hash of the previous SSG snapshot (change detection)
    ssg_changed: bool            # set by check_ssg_change_node each cycle
    error: Optional[str]
    cycle_count: int

    # ── runtime config ─────────────────────────────────────────────────
    config: WorkflowConfig

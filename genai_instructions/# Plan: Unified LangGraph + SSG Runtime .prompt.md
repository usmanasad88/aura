# Plan: Unified LangGraph + SSG Runtime for AURA

**TL;DR** — Replace the two disconnected pipelines (assistant flat-state + brain SSG) with a single **LangGraph `StateGraph`** orchestrator, generated from task config files. The SSG tracks spatial/semantic reality (object locations, agent targets, relations); a flat `task_state` dict tracks SOP progress (booleans, phase, counters). A config-driven **graph builder** reads each task's `dag.json`, `task_profile.json`, `state_schema.json`, and optional `initial_scene.json` / `robot_skills.json` to construct the appropriate graph topology and node set. The existing `GraphReasoner`, `SkillRegistry`, and `DecisionExplainer` from `src/aura/brain/` are finally wired into the active runtime. The hand_layup task is the first validation target.

---

## Steps

### 1. Define generic LangGraph state: `AuraGraphState`

Create `src/aura/workflow/state.py` with a `TypedDict` that holds both SSG and flat state side-by-side:

- `ssg_snapshot: Dict` — serialized `SemanticSceneGraph` (via `to_dict()`)
- `task_state: Dict[str, Any]` — flat key-value variables from `state_schema.json`
- `dag: Dict` — loaded `dag.json` (read-only reference)
- `task_profile: Dict` — loaded `task_profile.json`
- `monitor_outputs: Dict[str, Any]` — keyed by `MonitorType` name, latest output from each active monitor
- `frames_buffer: List[bytes]` — recent frames (base64 or raw, configurable)
- `current_frame_num: int`, `current_timestamp_sec: float`
- `decision_history: Annotated[List[Dict], operator.add]` — append-only via LangGraph reducer
- `pending_actions: List[Dict]` — robot action queue
- `completed_steps: Set[str]`
- `is_complete: bool`, `error: Optional[str]`
- `config: Dict` — runtime config (model, dry_run, robot_url, etc.)

This replaces both `WeighBottlesState` (task-specific) and `IntentResult` + flat dicts in `AURADecisionEngine`. The existing `AuraState` dataclass in `src/aura/core/types.py` can be kept as documentation/reference but the `TypedDict` is what LangGraph needs.

### 2. Implement reusable LangGraph node functions

Create `src/aura/workflow/nodes.py` with these node functions, each taking `AuraGraphState → Dict` (partial state update):

- **`capture_frame_node`** — reads from video source or webcam; appends to `frames_buffer`, updates `current_frame_num`/`current_timestamp_sec`. Sources are injected via closure at graph construction time (same pattern as `tasks/weigh_bottles/workflow/nodes.py`).
- **`gesture_gate_node`** — runs `GestureMonitor.update(frame)`. If the configured trigger gesture (default `Thumb_Up`) is detected, sets a `should_predict: bool` key. Replaces the inline gesture polling in `run_aura_assistant.py`.
- **`run_intent_node`** — calls `AURAIntentMonitor.predict(frames, timestamp, frame_num)`. Writes the parsed `IntentResult` fields into `monitor_outputs["intent"]` and merges returned state variables into `task_state`.
- **`run_perception_node`** — calls `PerceptionModule.update(frame)` if configured. Writes `PerceptionOutput` into `monitor_outputs["perception"]`.
- **`run_motion_node`** — calls `MotionPredictor.update(frame)` if configured. Writes `MotionOutput` to `monitor_outputs["motion"]`.
- **`update_ssg_node`** — **the key bridge**: constructs/updates a `SemanticSceneGraph` from monitor outputs. Uses the pattern from `tasks/weigh_bottles/workflow/nodes.py` `update_ssg_node` but generalized:
  - Reads `initial_scene.json` to bootstrap nodes/regions on first call
  - Maps detected objects → `ObjectNode`s with location edges
  - Maps intent output → `AgentNode` targets, human state
  - Maps `task_state` variables → `ssg.update_task_state()`
  - Maps object location variables (from `state_schema.json`, e.g., `roller_location`) → `ssg.set_location()` calls
  - Persists SSG across iterations via `ssg_snapshot` (serialize → deserialize each cycle)
- **`decide_action_node`** — **the unified decision maker**: instantiates the Brain's `DecisionEngine` (or reuses a cached instance) with the live SSG + `SkillRegistry`. Calls `decide_action()` which uses `GraphReasoner.get_available_actions()` + `get_proactive_opportunities()`, then LLM reasoning (Gemini) or rule-based fallback. Also evaluates safety rules and timers from `task_profile`. Writes decision to `pending_actions` and `decision_history`. The `DecisionExplainer.explain_action()` output goes into the decision record.
- **`execute_action_node`** — pops from `pending_actions`, resolves program via `program_map` or `SkillRegistry`, calls `RobotControlClient` (or dry-run logs), updates `task_state` and SSG (`ssg.set_location` for moved objects).
- **`check_complete_node`** — checks if any DAG `end_nodes` are in `completed_steps`, or if video source is exhausted, or error. Sets `is_complete`.

### 3. Build a config-driven graph builder

Create `src/aura/workflow/builder.py` with a function `build_task_graph(config_dir: Path, **runtime_kwargs) → CompiledGraph`:

- Loads `dag.json`, `task_profile.json`, `state_schema.json` from `config_dir`
- Optionally loads `initial_scene.json`, `robot_skills.json` if present
- Reads a new `workflow_config` section from `task_profile.json` (or a separate `workflow.json`) that declares:
  - `active_monitors: ["intent", "gesture"]` (or `["intent", "gesture", "perception", "motion"]`)
  - `trigger_mode: "gesture"` or `"continuous"` or `"interval"`
  - `decision_mode: "llm"` or `"rules"` or `"hybrid"`
- Constructs a `StateGraph(AuraGraphState)` by:
  1. Always adding `capture_frame_node`
  2. Adding `gesture_gate_node` if `trigger_mode == "gesture"`
  3. Adding monitor nodes for each active monitor
  4. Always adding `update_ssg_node`, `decide_action_node`, `check_complete_node`
  5. Adding conditional edges (gesture gate → intent only when triggered; decision → execute only when `pending_actions` non-empty)
  6. Setting `START → capture_frame_node`, loop from `check_complete_node → capture_frame_node | END`
- Returns `graph.compile(checkpointer=MemorySaver())` — enabling short-term memory for free via LangGraph's built-in checkpointing

### 4. Create the new entry point

Create `scripts/run_aura.py` — a streamlined replacement for `run_aura_assistant.py`:

- Same CLI args (task, video, webcam, robot-url, speed, predict-interval, model, dry-run, voice, etc.)
- Calls `build_task_graph(config_dir, ...)` to get a compiled graph
- Constructs initial `AuraGraphState` with loaded configs, empty SSG, default task_state from schema
- Runs the graph in a loop (`graph.invoke(state)` per iteration, or `graph.astream()` for async)
- Prints decisions using existing `_print_intent`/`_print_actions` formatting
- On exit, calls `DecisionExplainer.generate_decision_report()` and `engine.save_summary()`

### 5. Migrate hand_layup config

Extend `tasks/hand_layup/config/` with:

- **`initial_scene.json`** — define regions (`storage`, `workplace`, `scale_area`), objects (`resin_bottle`, `hardener_bottle`, `roller`, `mixing_cup`, etc.) with initial locations and affordances, agents (`human`, `robot`). Follow the pattern from `tasks/tea_making/config/initial_scene.json` which already demonstrates this structure.
- **`robot_skills.json`** — convert the existing `program_map` entries into proper skill definitions with preconditions/effects. E.g., `deliver_to_workplace` for roller has precondition `roller.location == "storage"` and effect `roller.location == "workplace"`. Follow `tasks/tea_making/config/robot_skills.json` pattern.
- **`workflow_config`** section in `task_profile.json` — add `active_monitors: ["intent", "gesture"]`, `trigger_mode: "gesture"`, `decision_mode: "hybrid"`.

### 6. Unify monitor output flow through SSG

Modify `src/aura/core/scene_graph/graph.py` — add a convenience method `update_from_monitor_outputs(outputs: Dict[str, MonitorOutput])` that:

- Dispatches to existing Brain `update_from_perception()`, `update_from_intent()`, `update_from_motion()` patterns, but as SSG instance methods rather than tied to the Brain class. This avoids duplicating the logic that's currently spread between `brain/decision_engine.py` `update_from_*` methods and `weigh_bottles/workflow/nodes.py` `update_ssg_node`.

### 7. Connect `DecisionExplainer` to every decision

In the `decide_action_node`, after every decision (act or wait):

- Call `explainer.record_decision(DecisionRecord(...))` with evidence gathered from SSG edges (via `_collect_evidence()`)
- Include the SSG-sourced explanation in `decision_history` entries
- This directly implements the paper's claim of "every decision includes explicit reasoning that can be inspected"

### 8. Consolidate duplicate code

- **Deprecate** `src/aura/assistant/decision_engine.py` (`AURADecisionEngine`) — its delivery/return logic migrates into `decide_action_node` rules combined with `GraphReasoner.get_proactive_opportunities()` + `SkillRegistry` precondition checking
- **Keep** `src/aura/assistant/intent_monitor.py` (`AURAIntentMonitor`) as the Gemini VLM caller — it works well and the RCWPS prompting strategy is what the paper describes. It becomes a LangGraph node's internal implementation.
- **Retire** the duplicate `IntentMonitor` in `src/aura/monitors/intent_monitor.py` — or merge the two into one class

### 9. Wire voice as optional node

If `--voice` is passed, the graph builder adds a `voice_output_node` after `decide_action_node` that forwards voice messages through `SoundMonitor` / `VoiceActionBridge`. This replaces the inline `voice_callback` closure in the current `run_aura_assistant.py`.

---

## Verification

1. **Dry-run regression**: `uv run python scripts/run_aura.py --task hand_layup --video demo_data/layup_demo/layup_gesture_demo.mp4 --dry-run` — should produce the same sequence of decisions and voice messages as the current `run_aura_assistant.py`
2. **SSG snapshot inspection**: Each decision log in `logs/intent_monitor/session_*/call_*/` should now include a `ssg_snapshot.json` alongside the existing `decision.json`, showing full node/edge state at decision time
3. **Explainability check**: Each decision record should contain an `evidence` field citing SSG edges (e.g., "roller AT storage → deliver to workplace because consolidate_with_roller predicted next")
4. **Config-only new task**: Create a new task by copying the hand_layup config dir, changing node names and objects — the runtime should work without code changes, validating the config-driven graph builder

---

## Design Decisions

- **SSG + flat state coexistence**: SSG stores spatial/semantic relations; `task_state` dict stores SOP progress booleans. The `update_ssg_node` syncs between them each cycle (object locations from `task_state` → SSG edges; SSG `task_state` dict ← `task_state`). This avoids forcing all state into graph nodes while giving the LLM both structured + relational context.
- **Task-specific graphs from config, not code**: Graph topology is built dynamically by `builder.py` reading config files. Task authors declare monitors and routing via JSON, not by writing Python `StateGraph` code. The `weigh_bottles` hand-coded graph becomes a reference/example but the generic builder is the standard path.
- **Keep `AURAIntentMonitor`**: The RCWPS Gemini prompting (rolling context window + previous state + DAG + state schema) is the paper's core innovation. It stays as the `run_intent_node` implementation. The Brain's `_llm_decide_action()` is a separate LLM call for action selection, not intent prediction — both are needed.
- **Two LLM calls per cycle**: (1) Intent Monitor → "what is human doing / what's next?" (Gemini Flash), (2) Decision Engine → "should the robot act, and how?" (Gemini Pro). This matches the paper's separation of perception-layer vs brain-layer reasoning. The gesture gate makes this cost-manageable since both only fire on trigger.
- **Chose LangGraph `MemorySaver` over custom history**: LangGraph's built-in checkpointer gives short-term memory (state persistence across iterations) for free. Long-term memory (ChromaDB) is deferred — the plan focuses on getting the core SSG + LangGraph loop working first.

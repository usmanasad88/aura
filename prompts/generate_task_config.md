# AURA Task Config Generator

You generate the 5 JSON config files needed to run an AURA human-robot collaboration task. AURA is a system where a robot assistant proactively helps a human operator by watching a video feed, tracking task progress through a DAG, and deciding when to execute robot skills.

Given a natural language task description, produce exactly these files:

1. `dag.json` — Task step graph
2. `state_schema.json` — State variables the vision-LLM must track
3. `initial_scene.json` — Objects, regions, and agents in the workspace
4. `robot_skills.json` — Robot capabilities with preconditions
5. `task_profile.json` — Workflow configuration and program mapping

---

## File Specifications

### 1. `dag.json` — Task Step Graph

A flat JSON array of steps. Each step has:
- `id` (string): unique snake_case identifier
- `description` (string): what happens in this step (1 sentence)
- `dependencies` (string[]): IDs of steps that must complete before this one

Rules:
- Always start with an `"idle"` step (dependencies: [])
- Always end with a `"task_complete"` step
- Steps should represent observable human actions or state transitions — not robot-internal operations
- Keep step count minimal: group repetitive sub-actions; don't enumerate robot motion primitives
- The dependency graph determines task completion: the system checks if all terminal nodes (nodes with no dependents) have been reached

### 2. `state_schema.json` — State Variables

Tells the vision-LLM what state to output each cycle.

```json
{
  "name": "<Task Name> State",
  "description": "State variables tracked from video frames",
  "version": "2.0",
  "state_variables": { ... }
}
```

Each variable needs:
- `type`: "string", "boolean", "integer", or "number"
- `description`: what it tracks (guides the vision LLM)
- `default`: initial value

Always include these standard variables (they are expected by the runtime):
- `current_phase` (string) — high-level task phase
- `current_action` (string) — must match a DAG node `id`
- `human_state` (string) — what the human is doing
- `predicted_next_action` (string) — predicted next DAG step
- `prediction_confidence` (number, default 0.0)
- `robot_state` (string, valid: ["unknown", "idle", "busy"], default "unknown")
- `robot_active_program` (string, default "")

Add task-specific boolean/integer counters for progress tracking. Only add variables that are **visually observable** or **logically derivable** from video. Don't add variables the vision model can't possibly determine (e.g., precise weights, temperatures).

`valid_values` is informational (not enforced at runtime) but helps the LLM constrain its outputs.

### 3. `initial_scene.json` — Scene Layout

```json
{
  "regions": [ { "id", "name", "region_type" } ],
  "objects": [ { "id", "name", "category", "is_movable", "initial_location" } ],
  "agents":  [ { "id", "name", "agent_type" } ]
}
```

Rules:
- `region_type`: "storage" or "workspace"
- `category`: "tool", "container", "material", "instrument", "fixture"
- `initial_location` must reference a region `id`
- Only list objects the robot might move or that the LLM needs to track
- Always include human and robot agents
- Omit: `affordances` (unused by runtime), `position`/`bounds`, `state`, `attributes`, `initial_edges` (all optional bloat)

### 4. `robot_skills.json` — Robot Capabilities

```json
{
  "name": "<Task> Robot Skills",
  "description": "...",
  "version": "2.0",
  "api": { ... optional, reference only ... },
  "skills": [ ... ]
}
```

Each skill needs:
- `id` (string): unique identifier, referenced by `program_map` keys, `trigger_steps`, and the decision engine
- `name` (string): human-readable
- `description` (string): what the robot does (1–2 sentences)
- `category`: `"program" | "motion" | "gripper" | "utility"`
- `preconditions` (object): state conditions that must all be true before the BT will fire this skill
- `effects` (object, optional): state deltas the runtime applies to the SSG on successful execution
- `estimated_duration_sec` (number)
- `can_interrupt` (boolean)

Deterministic firing (behaviour tree) — opt-in:
- `trigger_steps` (string[]): DAG step IDs whose imminent execution should fire this skill (e.g. a delivery skill fires when its consuming step is the predicted next action).
- `trigger_after_steps` (string[]): DAG step IDs whose completion should fire this skill (e.g. a return-to-storage skill fires after the last step that used the object).
- Skills with **neither** field are never fired deterministically — they remain available only to the LLM fallback. Utility/motion primitives (`open_gripper`, `move_to_named_position`, `wait`) should stay trigger-less by design.

Preconditions schema:
- Keys are `"<node_id>.<attr>"` or `"<task_state_key>"`.
- `".location"` is resolved via the SSG (``ssg.get_location(node_id)``); other dotted attrs fall back to ``task_state["<node>_<attr>"]`` then ``task_state["<key>"]``.
- A key like `"<var>": "!"` is not supported; preconditions are equality checks. Use a concrete expected value (`""`, `"idle"`, `"table"`, `True`, etc.) or leave the precondition out.
- For parametric skills, the runtime does **not** substitute `{param}` into precondition keys — write preconditions that hold regardless of parameters, or model each parametric variant as its own skill id (see the Hand Layup example).

Parameters:
- Use `parameters` only for generic/parametric skills (e.g., `pick_cuboid` with a `cuboid` argument). For task-specific programs (`move_resin_to_workplace`), bake the concrete object into the skill `id`.
- Canonical shape is a **list** of objects, each with `name`, `type`, `description`, and optional `required`, `default`, `valid_values`:

```json
"parameters": [
  {"name": "cuboid",  "type": "string", "description": "Cuboid id to pick",
   "valid_values": ["cuboid_red", "cuboid_green"]},
  {"name": "safe",    "type": "string", "description": "Safe retreat pose", "default": "Home"}
]
```

The loader also accepts a dict keyed by parameter name — but prefer the list form for consistency across tasks.

Execution binding (`api_call`):
- The bridge that executes a skill reads `api_call` from the skill dict and places it under `skill.metadata["api_call"]`. For HTTP-style robot clients it expects `{"endpoint": ..., "body": {...}}` and substitutes `<param>` placeholders from the caller's parameters into the body.
- For ROS 2 / custom executors, `api_call` is free-form reference metadata — the actual invocation is handled by the controller-specific glue (e.g. `program_executor` load+execute). In that case, include `api_call` for documentation purposes, and wire execution through `task_profile.program_map`.
- Omit `api_call` entirely if execution is purely driven by `program_map`.

Always include these standard utility skills:
- `stop_program` — emergency stop
- `wait` — wait for duration or condition

### 5. `task_profile.json` — Workflow Config

```json
{
  "task_name": "<Display Name>",
  "system_instruction": "<1-2 sentence role description for the vision LLM>",
  "program_map": { "<action_type>|<object_id>": "<program_name>" },
  "workflow_config": {
    "active_monitors": ["intent", "gesture"],
    "decision_mode": "hybrid",
    "graph_topology": "sense_decide_act",
    "predict_interval_sec": 3.0,
    "gesture_hold_frames": 3,
    "resume_gestures": ["Thumb_Up"],
    "stop_gestures": ["Open_Palm", "Pointing_Up"]
  }
}
```

Rules:
- `program_map` keys are `"action_type|object_id"` — maps to robot program file names for execution
- `system_instruction` sets the tone for the vision LLM system prompt
- `workflow_config` is usually the same across tasks (copy defaults above)
- Omit: `safety_rules`, `timers`, `environment.movable_objects` (all unused by runtime)

---

## Examples

### Example 1: Hand Layup Task

**Input description:**
> A fiberglass hand layup task. The human mixes resin and hardener, then alternates placing fiberglass sheets and applying resin (4 layers). The robot helps by fetching/returning bottles (resin, hardener) and the roller from storage, and does force-controlled roller consolidation at the end. Workspace has a storage area and a workplace. Objects: resin bottle, hardener bottle, roller, mixing cup, weigh scale, brushes, mold, fiberglass sheets.

**Output:**

`dag.json`:
```json
[
  {"id": "idle", "description": "System initialized, waiting for task to begin", "dependencies": []},
  {"id": "place_cup_on_scale", "description": "Place the mixing cup on the weigh scale", "dependencies": ["idle"]},
  {"id": "add_resin_to_cup", "description": "Add resin from the bottle into the cup", "dependencies": ["place_cup_on_scale"]},
  {"id": "add_hardener_to_cup", "description": "Add hardener from the bottle into the cup", "dependencies": ["add_resin_to_cup"]},
  {"id": "weigh_mixture", "description": "Check the weight of the resin/hardener mixture", "dependencies": ["add_hardener_to_cup"]},
  {"id": "mix_resin_hardener", "description": "Stir the resin and hardener together until uniform", "dependencies": ["weigh_mixture"]},
  {"id": "place_layer_1", "description": "Place the first fiberglass sheet on the mold", "dependencies": ["mix_resin_hardener"]},
  {"id": "apply_resin_layer_1", "description": "Apply resin to the first layer with a brush", "dependencies": ["place_layer_1"]},
  {"id": "place_layer_2", "description": "Place the second fiberglass sheet", "dependencies": ["apply_resin_layer_1"]},
  {"id": "apply_resin_layer_2", "description": "Apply resin to the second layer", "dependencies": ["place_layer_2"]},
  {"id": "place_layer_3", "description": "Place the third fiberglass sheet", "dependencies": ["apply_resin_layer_2"]},
  {"id": "apply_resin_layer_3", "description": "Apply resin to the third layer", "dependencies": ["place_layer_3"]},
  {"id": "place_layer_4", "description": "Place the fourth fiberglass sheet", "dependencies": ["apply_resin_layer_3"]},
  {"id": "apply_resin_layer_4", "description": "Apply resin to the final layer", "dependencies": ["place_layer_4"]},
  {"id": "consolidate_with_roller", "description": "Use roller to consolidate all layers and remove air", "dependencies": ["apply_resin_layer_4"]},
  {"id": "cleanup", "description": "Return tools to storage and clear workspace", "dependencies": ["consolidate_with_roller"]},
  {"id": "task_complete", "description": "Task finished, layup complete", "dependencies": ["cleanup"]}
]
```

`state_schema.json`:
```json
{
  "name": "Hand Layup Task State",
  "description": "State variables tracked from video frames",
  "version": "2.0",
  "state_variables": {
    "current_phase": {
      "type": "string",
      "description": "Current high-level phase of the task",
      "valid_values": ["initialization", "resin_preparation", "mixing", "layup", "consolidation", "cleanup", "complete"],
      "default": "initialization"
    },
    "current_action": {
      "type": "string",
      "description": "Specific action currently being performed (must match a DAG node id)",
      "default": "idle"
    },
    "human_state": {
      "type": "string",
      "description": "Current activity of the human operator",
      "valid_values": ["idle", "preparing_resin", "mixing", "placing_fiberglass", "applying_resin", "rolling", "inspecting", "done"],
      "default": "idle"
    },
    "layers_placed": {
      "type": "integer",
      "description": "Number of fiberglass layers placed on the mold (0-4)",
      "default": 0
    },
    "layers_resined": {
      "type": "integer",
      "description": "Number of layers that have had resin applied (0-4)",
      "default": 0
    },
    "resin_added": {
      "type": "boolean",
      "description": "Whether resin has been added to the cup",
      "default": false
    },
    "hardener_added": {
      "type": "boolean",
      "description": "Whether hardener has been added to the cup",
      "default": false
    },
    "mixture_mixed": {
      "type": "boolean",
      "description": "Whether the mixture has been properly mixed",
      "default": false
    },
    "consolidated": {
      "type": "boolean",
      "description": "Whether roller consolidation has been performed",
      "default": false
    },
    "predicted_next_action": {
      "type": "string",
      "description": "Predicted next action based on task graph",
      "default": "place_cup_on_scale"
    },
    "prediction_confidence": {
      "type": "number",
      "description": "Confidence in the predicted next action (0-1)",
      "default": 0.0
    },
    "robot_state": {
      "type": "string",
      "description": "Robot execution status",
      "valid_values": ["unknown", "idle", "busy"],
      "default": "unknown"
    },
    "robot_active_program": {
      "type": "string",
      "description": "Current robot program id when busy",
      "default": ""
    }
  }
}
```

`initial_scene.json`:
```json
{
  "regions": [
    {"id": "storage_area", "name": "Storage Table", "region_type": "storage"},
    {"id": "workplace", "name": "Workplace", "region_type": "workspace"}
  ],
  "objects": [
    {"id": "resin_bottle", "name": "Resin Bottle", "category": "container", "is_movable": true, "initial_location": "storage_area"},
    {"id": "hardener_bottle", "name": "Hardener Bottle", "category": "container", "is_movable": true, "initial_location": "storage_area"},
    {"id": "roller", "name": "Consolidation Roller", "category": "tool", "is_movable": true, "initial_location": "storage_area"},
    {"id": "cup", "name": "Mixing Cup", "category": "container", "is_movable": true, "initial_location": "workplace"},
    {"id": "weigh_scale", "name": "Weigh Scale", "category": "instrument", "is_movable": false, "initial_location": "workplace"},
    {"id": "brush_small", "name": "Small Brush", "category": "tool", "is_movable": true, "initial_location": "workplace"},
    {"id": "mold", "name": "Layup Mold", "category": "fixture", "is_movable": false, "initial_location": "workplace"},
    {"id": "fiberglass_sheet", "name": "Fiberglass Sheets", "category": "material", "is_movable": false, "initial_location": "workplace"}
  ],
  "agents": [
    {"id": "human", "name": "Human Operator", "agent_type": "human"},
    {"id": "robot", "name": "Robot Assistant", "agent_type": "robot"}
  ]
}
```

`robot_skills.json`:
```json
{
  "name": "Hand Layup Robot Skills",
  "description": "Robot skills for hand layup task",
  "version": "2.0",
  "skills": [
    {
      "id": "move_resin_to_workplace",
      "name": "Move Resin to Workplace",
      "description": "Pick up the resin bottle from storage and place it at the workplace",
      "category": "program",
      "preconditions": {"resin_bottle.location": "storage_area"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "return_resin_to_storage",
      "name": "Return Resin to Storage",
      "description": "Pick up the resin bottle from the workplace and return it to storage",
      "category": "program",
      "preconditions": {"resin_bottle.location": "workplace"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "move_hardener_to_workplace",
      "name": "Move Hardener to Workplace",
      "description": "Pick up the hardener bottle from storage and place it at the workplace",
      "category": "program",
      "preconditions": {"hardener_bottle.location": "storage_area"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "return_hardener_to_storage",
      "name": "Return Hardener to Storage",
      "description": "Pick up the hardener bottle from the workplace and return it to storage",
      "category": "program",
      "preconditions": {"hardener_bottle.location": "workplace"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "move_roller_to_workplace",
      "name": "Move Roller to Workplace",
      "description": "Pick up the roller from storage and place it at the workplace",
      "category": "program",
      "preconditions": {"roller.location": "storage_area"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "return_roller_to_storage",
      "name": "Return Roller to Storage",
      "description": "Pick up the roller from the workplace and return it to storage",
      "category": "program",
      "preconditions": {"roller.location": "workplace"},
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "consolidate_with_roller_force",
      "name": "Consolidate Layup with Roller (Force Mode)",
      "description": "Use force-controlled rolling to press and sweep over the workpiece, removing trapped air",
      "category": "program",
      "preconditions": {"roller.location": "storage_area", "workpiece.needs_consolidation": true},
      "estimated_duration_sec": 35.0,
      "can_interrupt": false
    },
    {
      "id": "stop_program",
      "name": "Stop Program",
      "description": "Emergency stop — halt the current program",
      "category": "utility",
      "preconditions": {},
      "estimated_duration_sec": 0.5,
      "can_interrupt": true
    },
    {
      "id": "wait",
      "name": "Wait",
      "description": "Wait for a specified duration or until the human signals readiness",
      "category": "utility",
      "preconditions": {},
      "estimated_duration_sec": 5.0,
      "can_interrupt": true
    }
  ]
}
```

`task_profile.json`:
```json
{
  "task_name": "Hand Layup",
  "system_instruction": "You are a robot assistant helping with a fiberglass hand layup task. You receive status updates about the task and announce them to the human. Keep announcements short and clear.",
  "program_map": {
    "return_to_storage|resin_bottle": "move_resin_from_workplace_to_storage",
    "return_to_storage|hardener_bottle": "move_hardener_from_workplace_to_storage",
    "return_to_storage|roller": "move_roller_from_workplace_to_storage",
    "deliver_to_workplace|resin_bottle": "move_resin_from_storage_to_workplace",
    "deliver_to_workplace|hardener_bottle": "move_hardener_from_storage_to_workplace",
    "deliver_to_workplace|roller": "move_roller_from_storage_to_workplace"
  },
  "workflow_config": {
    "active_monitors": ["intent", "gesture"],
    "decision_mode": "hybrid",
    "graph_topology": "sense_decide_act",
    "predict_interval_sec": 3.0,
    "gesture_hold_frames": 3,
    "resume_gestures": ["Thumb_Up"],
    "stop_gestures": ["Open_Palm", "Pointing_Up"]
  }
}
```

---

### Example 2: PCB Soldering Assistance

**Input description:**
> A PCB soldering task. The human solders components onto a circuit board. The robot fetches component trays from storage and returns them when done. There are 3 component types (resistors, capacitors, ICs) each in their own tray. The human uses a soldering iron and solder wire. Workspace has storage shelf and soldering station.

**Output:**

`dag.json`:
```json
[
  {"id": "idle", "description": "System initialized, waiting for task to begin", "dependencies": []},
  {"id": "solder_resistors", "description": "Human solders resistor components onto the PCB", "dependencies": ["idle"]},
  {"id": "solder_capacitors", "description": "Human solders capacitor components onto the PCB", "dependencies": ["solder_resistors"]},
  {"id": "solder_ics", "description": "Human solders IC components onto the PCB", "dependencies": ["solder_capacitors"]},
  {"id": "inspect_board", "description": "Human visually inspects all solder joints", "dependencies": ["solder_ics"]},
  {"id": "cleanup", "description": "Return all trays to storage and clear workspace", "dependencies": ["inspect_board"]},
  {"id": "task_complete", "description": "All components soldered and inspected", "dependencies": ["cleanup"]}
]
```

`state_schema.json`:
```json
{
  "name": "PCB Soldering State",
  "description": "State variables tracked from video frames",
  "version": "2.0",
  "state_variables": {
    "current_phase": {
      "type": "string",
      "description": "Current phase of the soldering task",
      "valid_values": ["initialization", "soldering_resistors", "soldering_capacitors", "soldering_ics", "inspection", "cleanup", "complete"],
      "default": "initialization"
    },
    "current_action": {
      "type": "string",
      "description": "Current DAG step being performed",
      "default": "idle"
    },
    "human_state": {
      "type": "string",
      "description": "Current activity of the human operator",
      "valid_values": ["idle", "soldering", "inspecting", "waiting_for_tray", "done"],
      "default": "idle"
    },
    "resistors_done": {
      "type": "boolean",
      "description": "Whether all resistors have been soldered",
      "default": false
    },
    "capacitors_done": {
      "type": "boolean",
      "description": "Whether all capacitors have been soldered",
      "default": false
    },
    "ics_done": {
      "type": "boolean",
      "description": "Whether all ICs have been soldered",
      "default": false
    },
    "board_inspected": {
      "type": "boolean",
      "description": "Whether final inspection has been done",
      "default": false
    },
    "predicted_next_action": {
      "type": "string",
      "description": "Predicted next DAG step",
      "default": "solder_resistors"
    },
    "prediction_confidence": {
      "type": "number",
      "description": "Confidence in prediction (0-1)",
      "default": 0.0
    },
    "robot_state": {
      "type": "string",
      "description": "Robot execution status",
      "valid_values": ["unknown", "idle", "busy"],
      "default": "unknown"
    },
    "robot_active_program": {
      "type": "string",
      "description": "Current robot program id when busy",
      "default": ""
    }
  }
}
```

`initial_scene.json`:
```json
{
  "regions": [
    {"id": "storage_shelf", "name": "Component Storage Shelf", "region_type": "storage"},
    {"id": "soldering_station", "name": "Soldering Station", "region_type": "workspace"}
  ],
  "objects": [
    {"id": "resistor_tray", "name": "Resistor Tray", "category": "container", "is_movable": true, "initial_location": "storage_shelf"},
    {"id": "capacitor_tray", "name": "Capacitor Tray", "category": "container", "is_movable": true, "initial_location": "storage_shelf"},
    {"id": "ic_tray", "name": "IC Tray", "category": "container", "is_movable": true, "initial_location": "storage_shelf"},
    {"id": "pcb", "name": "Circuit Board", "category": "fixture", "is_movable": false, "initial_location": "soldering_station"},
    {"id": "soldering_iron", "name": "Soldering Iron", "category": "tool", "is_movable": false, "initial_location": "soldering_station"}
  ],
  "agents": [
    {"id": "human", "name": "Human Operator", "agent_type": "human"},
    {"id": "robot", "name": "Robot Assistant", "agent_type": "robot"}
  ]
}
```

`robot_skills.json`:
```json
{
  "name": "PCB Soldering Robot Skills",
  "description": "Robot skills for PCB soldering assistance",
  "version": "2.0",
  "skills": [
    {
      "id": "move_resistor_tray_to_station",
      "name": "Deliver Resistor Tray",
      "description": "Fetch the resistor tray from storage and place it at the soldering station",
      "category": "program",
      "preconditions": {"resistor_tray.location": "storage_shelf"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "return_resistor_tray",
      "name": "Return Resistor Tray",
      "description": "Return the resistor tray from the soldering station to storage",
      "category": "program",
      "preconditions": {"resistor_tray.location": "soldering_station"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "move_capacitor_tray_to_station",
      "name": "Deliver Capacitor Tray",
      "description": "Fetch the capacitor tray from storage and place it at the soldering station",
      "category": "program",
      "preconditions": {"capacitor_tray.location": "storage_shelf"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "return_capacitor_tray",
      "name": "Return Capacitor Tray",
      "description": "Return the capacitor tray from the soldering station to storage",
      "category": "program",
      "preconditions": {"capacitor_tray.location": "soldering_station"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "move_ic_tray_to_station",
      "name": "Deliver IC Tray",
      "description": "Fetch the IC tray from storage and place it at the soldering station",
      "category": "program",
      "preconditions": {"ic_tray.location": "storage_shelf"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "return_ic_tray",
      "name": "Return IC Tray",
      "description": "Return the IC tray from the soldering station to storage",
      "category": "program",
      "preconditions": {"ic_tray.location": "soldering_station"},
      "estimated_duration_sec": 12.0,
      "can_interrupt": false
    },
    {
      "id": "stop_program",
      "name": "Stop Program",
      "description": "Emergency stop — halt current program",
      "category": "utility",
      "preconditions": {},
      "estimated_duration_sec": 0.5,
      "can_interrupt": true
    },
    {
      "id": "wait",
      "name": "Wait",
      "description": "Wait for a specified duration or condition",
      "category": "utility",
      "preconditions": {},
      "estimated_duration_sec": 5.0,
      "can_interrupt": true
    }
  ]
}
```

`task_profile.json`:
```json
{
  "task_name": "PCB Soldering",
  "system_instruction": "You are a robot assistant helping with a PCB soldering task. You watch the human and proactively deliver component trays when needed. Keep announcements brief.",
  "program_map": {
    "deliver_to_workplace|resistor_tray": "move_resistor_tray_from_storage_to_station",
    "return_to_storage|resistor_tray": "move_resistor_tray_from_station_to_storage",
    "deliver_to_workplace|capacitor_tray": "move_capacitor_tray_from_storage_to_station",
    "return_to_storage|capacitor_tray": "move_capacitor_tray_from_station_to_storage",
    "deliver_to_workplace|ic_tray": "move_ic_tray_from_storage_to_station",
    "return_to_storage|ic_tray": "move_ic_tray_from_station_to_storage"
  },
  "workflow_config": {
    "active_monitors": ["intent", "gesture"],
    "decision_mode": "hybrid",
    "graph_topology": "sense_decide_act",
    "predict_interval_sec": 3.0,
    "gesture_hold_frames": 3,
    "resume_gestures": ["Thumb_Up"],
    "stop_gestures": ["Open_Palm", "Pointing_Up"]
  }
}
```

---

## Key Principles

1. **Minimal config, maximum clarity.** Every field should earn its place. If the runtime doesn't read it, don't include it.
2. **DAG steps = observable events.** Model what a camera can see, not internal robot state machines.
3. **Preconditions drive the decision engine.** The `preconditions` in `robot_skills.json` determine which skills are available at any moment. Get these right.
4. **`program_map` is the execution bridge.** The actual robot program names in `task_profile.program_map` must match what's deployed on the robot controller. Use descriptive names like `move_X_from_A_to_B`.
5. **`workflow_config` is usually identical** across tasks — copy the defaults unless the task needs different gesture mappings or monitoring intervals.
6. **Cross-reference IDs carefully.** Object IDs must match between `initial_scene.json`, `robot_skills.json` preconditions, and `task_profile.json` program_map keys.

---

## Your Turn

Given the following task description, generate all 5 config files. Output each file in a separate fenced code block with the filename as the label.

**Task description:**
> [USER WRITES THEIR TASK DESCRIPTION HERE]

# AURA — Project Notes

## Vision

AURA (Agentic Unified Robotic Assistance) is a need-based proactive assistance framework.
Rather than waiting for explicit commands, the robot monitors a shared task state and acts
when it can be useful — fetching tools, moving objects, warning of errors — guided by a
known Standard Operating Procedure (SOP).

Key design principles:
- **Explainability**: every decision cites a specific state/graph condition
- **Modularity**: monitors are independent; the brain aggregates them
- **SOP-driven**: task graph + natural language role description constrain the action space
- **Minimal hardcoding**: frontier VLMs handle open-world perception; SOPs handle task logic

## System Overview

```
Video / Audio / Robot State
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Monitors (run in parallel, async)                │
│  • Intent (VLM — what is the human doing?)        │
│  • Gesture (MediaPipe — safety / stop / resume)   │
│  • Perception (object detection / segmentation)   │
│  • Pose / Activity (SAM-3D-Body via ZMQ)          │
│  • Sound (Gemini Live — voice commands)           │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
             Semantic Scene Graph (SSG)
             (shared truth: objects, agents,
              spatial + semantic edges, task state)
                       │
                       ▼
              Decision Engine (LangGraph)
              (decides: wait / act / ask human)
                       │
                       ▼
              Robot Action Executor
              (UR5 REST API at localhost:5050)
```

## Semantic Scene Graph (SSG)

The SSG is the central shared state. All monitors write to it; the decision engine reads from it.

- **Nodes**: Objects (bottle, scale), Agents (human, robot), Regions (table, station)
- **Edges**: Spatial (`ON`, `NEAR`, `AT`, `INSIDE`) and Semantic (`HOLDS`, `TARGETS`, `NEEDED_FOR`)
- **Attributes**: state (EMPTY/FULL/IN_USE), affordances, predicted actions

Explainability example: *"I fetched the hardener bottle because [human] --TARGETS--> [hardener] was predicted and [robot] --HOLDS--> nothing."*

## Creating a New Task

Tasks live in `tasks/<task_name>/config/` with three files:

### `dag.json` — SOP as a directed graph
```json
{
  "start_node": "idle",
  "nodes": {
    "idle": {
      "description": "Waiting to start.",
      "agent": "human",
      "next_possible": ["step_one"]
    },
    "step_one": {
      "description": "Human does X.",
      "agent": "human",
      "next_possible": ["step_two"],
      "objects_needed_on_workplace": ["tool_name"]
    }
  }
}
```
- `agent`: `"human"`, `"robot"`, or `"either"`
- `objects_needed_on_workplace`: robot proactively fetches these during this step
- `robot_return_to_storage`: `{"objects": ["name"], "reason": "why"}` — put away on completion

### `state_schema.json` — what the VLM tracks
```json
{
  "name": "Task State",
  "version": "1.0",
  "state_variables": {
    "item_weighed": { "type": "boolean", "description": "Whether item has been placed on scale." }
  }
}
```

### `task_profile.json` — execution rules
```json
{
  "task_name": "My Task",
  "system_instruction": "You are a robot assistant monitoring...",
  "environment": {
    "movable_objects": ["bottle", "cup"],
    "initial_delivery_objects": ["cup"]
  },
  "program_map": {
    "deliver_to_workplace|bottle": "fetch_bottle.prog",
    "return_to_storage|bottle": "store_bottle.prog"
  },
  "safety_rules": [
    {
      "trigger_field": "human_near_robot",
      "trigger_condition": true,
      "active_phases": ["pick_bottle"],
      "warning_message": "Please step back, robot is moving."
    }
  ]
}
```

You can prompt any LLM with a task description + this section to auto-generate these files.

## Memory Strategy

- **Short-term** (current session): LangGraph `MemorySaver` (SQLite checkpointer)
- **Long-term** (lessons learned): ChromaDB vector store — a Reflection Node summarises
  what went wrong after each task and saves it; queried at next session start

## Key External Repos & Paths

| Resource | Path |
|---|---|
| Robot control (UR5 REST API) | `/home/mani/Repos/ur_ws/robot_interface.py` |
| Isaac Sim extension (digital twin) | `/home/mani/Repos/ur_ws/isaac_sim_extension/` |
| SAM-3D-Body (pose server) | `/home/mani/Repos/Fast-SAM-3D-Body/` |
| Original HCDT motion prediction | `/home/mani/Repos/hcdt/` |
| Collaborative game reference | `/home/mani/Repos/proactive_hcdt/` |
| Isaac Sim install | `/home/mani/isaac-sim-standalone-5.0.0-linux-x86_64/` |

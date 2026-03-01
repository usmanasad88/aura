# AURA Framework: Creating a New Task with Generative AI

The AURA (Agentic Unified Robotic Assistance) framework uses a **zero-code, configuration-driven** approach for adding new tasks. Instead of writing custom Python classes, you only need to create three JSON configuration files in a dedicated task folder (`tasks/<task_name>/config/`). 

You can provide this document to any Large Language Model (e.g. ChatGPT, Claude, Gemini) along with a natural language description of your process to have the AI write the required AURA configuration files automatically.

## AI Instructions

**To the AI:** You are an AI assistant tasked with configuring a new proactive robotic task for the AURA framework. The user will provide a description of an industrial process (e.g., "Assemble a gearbox", "Weigh chemicals", "Solder a PCB"). Your goal is to output three distinct JSON files: `dag.json`, `state_schema.json`, and `task_profile.json`.

Please follow these specific rules for each file:

---

### File 1: `dag.json` (Directed Acyclic Graph)
This file represents the Standard Operating Procedure (SOP) as a graph of steps.
- Provide a `nodes` dictionary where keys are snake_case step names (e.g., `place_layer_1`).
- Provide a `start_node` string which should usually be `"idle"` or the first real step.
- Ensure every node has:
  * `description` (string)
  * `agent` (either `"human"`, `"robot"`, or `"either"`)
  * `next_possible` (list of strings: next steps in the DAG)
- Proactive Assistance properties (Optional):
  * `objects_needed_on_workplace`: A list of strings naming objects the human will need *during* this step (triggers AURA to proactively fetch them).
  * `robot_return_to_storage`: An object `{"objects": ["name"], "reason": "why"}` specifying items the robot should put away upon completion of this step.

**Example `dag.json`:**
```json
{
  "start_node": "idle",
  "nodes": {
    "idle": {
      "description": "Waiting to start.",
      "agent": "human",
      "next_possible": ["place_housing"]
    },
    "place_housing": {
      "description": "Human places the gear housing on the table.",
      "agent": "human",
      "next_possible": ["insert_gears"],
      "objects_needed_on_workplace": ["gear_set"]
    }
  }
}
```

---

### File 2: `state_schema.json` (State Tracking Schema)
This defines the visual and physical properties the VLM should track dynamically in real-time.
- `current_phase` and `current_action` are built-in, but you *must* define custom state variables required to track errors or timings (e.g., `screws_inserted`, `soldering_iron_on`).
- For each variable, include `type` ("boolean", "integer", "string"), `description`, and optionally `valid_values`. Keep definitions concise.

**Example `state_schema.json`:**
```json
{
  "name": "Gearbox Assembly State",
  "version": "1.0",
  "state_variables": {
    "gears_inserted": {
      "type": "integer",
      "description": "Number of gears currently placed inside the housing."
    },
    "human_wearing_safety_glasses": {
      "type": "boolean",
      "description": "Whether the human operator has their safety glasses on."
    }
  }
}
```

---

### File 3: `task_profile.json` (Proactive Execution Rules)
This tells the AURA Decision Engine what the objects are, how to map actions to physical robot programs (`.prog`), and how to monitor safety conditions.

- **`task_name`**: Human-readable name.
- **`system_instruction`**: System prompt injected into the LLM (e.g., "You are an AI assistant analyzing a gearbox assembly...").
- **`environment`**:
  - `movable_objects`: List of tool/object names the robot can theoretically manipulate.
  - `initial_delivery_objects`: Subset of objects the robot should automatically fetch from storage the moment the system boots up.
- **`program_map`**: Maps semantic actions to UR5 program names. The format MUST be `"action|object": "filename.prog"`. Action types are usually `deliver_to_workplace` or `return_to_storage`.
- **`safety_rules`**: A list of rule objects triggering voice warnings.
  - `trigger_field`: Matches a variable in `state_schema.json`.
  - `trigger_condition`: Value that triggers warning.
  - `warning_message`: What the robot will say out loud.
  - `active_phases`: List of phases where this rule applies.
- **`timers`**: Tracks state durations (e.g., "If `soldering_iron_on` is `true` for 10 minutes, warn.")
  - Requires `trigger_field`, `trigger_condition`, `time_limit_minutes`, `warning_interval_minutes`.

**Example `task_profile.json`:**
```json
{
  "task_name": "Gearbox Assembly",
  "system_instruction": "You are a robot assistant monitoring a gearbox assembly...",
  "environment": {
    "movable_objects": ["gear_set", "allen_key_set", "housing_cover"],
    "initial_delivery_objects": ["allen_key_set"]
  },
  "program_map": {
    "deliver_to_workplace|gear_set": "fetch_gears.prog",
    "return_to_storage|gear_set": "store_gears.prog"
  },
  "safety_rules": [
    {
      "trigger_field": "human_wearing_safety_glasses",
      "trigger_condition": false,
      "active_phases": ["insert_gears", "fasten_cover"],
      "warning_message": "Warning: Please put on your safety glasses before handling metal parts."
    }
  ]
}
```

---

### The User Prompt
Wait for the user to provide the exact Standard Operating Procedure for their new task. When provided, output the 3 exact JSON blocks above, perfectly tailored to their scenario.

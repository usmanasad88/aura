# Kettle Tea Making Task for AURA

Task-specific configuration for a **collaborative electric-kettle tea making**
proactive-assistance demo. Tea is brewed in a **cup** with a **tea bag** and
**powdered milk**, using water boiled in an **electric kettle**.

**This folder can be deleted without affecting the base AURA framework.**

## Why a new task (vs. `tea_making` / `milk_tea_making`)

- `tea_making` boils **water in a pot on an induction cooker**, adds chai powder,
  then powdered milk in the cup.
- `milk_tea_making` boils **milk in a saucepan** and strains the brew through a sieve.
- **This task** uses an **electric kettle**: the human opens the lid to signal
  intent, water is poured in, the lid is closed and the kettle switched on
  (blue LED), and the tea is assembled in the cup once the water boils.

## Process

1. Key items start in **storage**.
2. The human signals intent to make tea by **opening the kettle lid**
   (`lid_open = true`). **The robot cannot open the lid.**
3. If the bottle is in storage → the robot **brings the bottle** to the human.
   The **human always pours** the water into the kettle.
4. Once water is poured, the **lid is closed** and the **kettle is turned on**.
5. A kettle that is on shows a **blue LED**; when the water boils the **LED turns off**
   (`kettle_on = true → water_boiled = true`).
6. The robot brings the **powdered milk container, cup, spoon, tea bag
   and biscuits** to the working area.
7. The human assembles their tea (`prepare_tea`): adds the tea bag, pours the
   boiled water, adds milk and stirs.

> The human can carry out any of these steps themselves; the robot assists
> proactively.

## Task flow (DAG)

```
idle → open_kettle_lid
open_kettle_lid → bring_water_bottle, setup_workspace
bring_water_bottle → pour_water_into_kettle → close_lid_and_turn_on → boil_water
boil_water + setup_workspace → prepare_tea → cleanup → task_complete
```

## Robot role

The robot proactively:
1. **brings** the water bottle to the human (the human pours it);
2. **closes** the lid and **turns the kettle on** (it cannot open the lid);
3. **retrieves** the powdered milk, cup, spoon, tea bag and biscuits to
   the working area;
4. **returns** unused items to storage during cleanup.

## Contents

- `config/dag.json` — task DAG (action sequence + dependencies)
- `config/state_schema.json` — state variables the intent monitor tracks from video
  (includes the `lid_open`, `kettle_on`, `water_boiled` booleans)
- `config/task_profile.json` — system instruction, keyword program map, safety rules, timers, workflow config
- `config/robot_skills.json` — parametric robot skills (ROS 2 `program_executor`),
  including a keyword `pick_and_place_items.prog` and dedicated kettle programs
- `config/initial_scene.json` — regions, objects (all start in `storage_area`), agents

## Run

```bash
uv run python scripts/run_aura.py --task kettle_tea_making --dry-run
```

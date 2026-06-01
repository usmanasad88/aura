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
3. Water handling depends on the bottle cap:
   - cap **closed** + bottle in storage → the robot **brings the bottle** to the human;
   - cap **open** → water can be **poured into the kettle** (robot or human).
4. Once water is poured, the **lid is closed** and the **kettle is turned on**.
5. A kettle that is on shows a **blue LED**; when the water boils the **LED turns off**
   (`kettle_on = true → water_boiled = true`).
6. The robot brings the **powdered milk container, cup, spoon, tea bag, sugar
   container and biscuits** to the working area (the tea bag and spoon may be
   dropped into the cup).
7. The human assembles their tea (`prepare_tea`): adds milk and sugar and stirs.

> The human can carry out any of these steps themselves; the robot assists
> proactively.

## Task flow (DAG)

```
idle → open_kettle_lid
open_kettle_lid → bring_water_bottle, setup_workspace
bring_water_bottle → pour_water_into_kettle → close_lid_and_turn_on → boil_water
setup_workspace → add_tea_bag
boil_water + add_tea_bag → pour_boiled_water_into_cup → prepare_tea → cleanup → task_complete
```

## Robot role

The robot proactively:
1. **brings** the water bottle to the human when its cap is closed;
2. **pours** water into the kettle, **closes** the lid, and **turns it on**
   (it cannot open the lid);
3. **retrieves** the powdered milk, cup, spoon, tea bag, sugar and biscuits to
   the working area (tea bag / spoon may go into the cup);
4. **returns** unused items to storage during cleanup.

## Contents

- `config/dag.json` — task DAG (action sequence + dependencies)
- `config/state_schema.json` — state variables the intent monitor tracks from video
  (includes the `lid_open`, `water_bottle_cap_open`, `kettle_on`, `water_boiled` booleans)
- `config/task_profile.json` — system instruction, keyword program map, safety rules, timers, workflow config
- `config/robot_skills.json` — parametric robot skills (ROS 2 `program_executor`),
  including a keyword `pick_and_place_items.prog` and dedicated kettle programs
- `config/initial_scene.json` — regions, objects (all start in `storage_area`), agents

## Run

```bash
uv run python scripts/run_aura.py --task kettle_tea_making --dry-run
```

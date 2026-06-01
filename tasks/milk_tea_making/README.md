# Milk Tea Making Task for AURA

Task-specific configuration for a **collaborative milk tea making** proactive-assistance
demo. The recipe is the *boiled-milk* ("doodh-patti") style observed in the
[Ego-Exo4D](https://ego-exo4d-data.org/) **"Making Milk Tea"** cooking takes
(e.g. `iiith_cooking_05_1`): milk is the boiling base, tea and sugar go into the
pot, the brew is simmered, then **strained through a fine-mesh sieve** into the mug.

**This folder can be deleted without affecting the base AURA framework.**

## Why a new task (vs. `tea_making`)

The existing `tea_making` task is a synthetic *chai* recipe: boil **water**, add
chai powder, pour into cups, then add **powdered milk** to the cup. The Ego-Exo4D
milk-tea takes are different in three ways, all reflected here:

1. **Milk is the boiling base** — poured into the saucepan and heated directly
   (`pour_milk_into_pot`, `heat_milk`), not added as powder at the end.
2. **Tea and sugar go into the pot** while it brews (`add_tea`, `add_sugar`).
3. **Straining is the signature step** — the brewed tea is poured through a
   fine-mesh **sieve** into the mug (`position_sieve` → `strain_tea_into_cup`).

> A single DAG/state schema can't capture every variation across the 73 takes
> (some measure milk into the mug first, some add water, some use tea bags vs.
> loose leaves, sugar timing varies). This models the **dominant flow**; see
> *Known variations* below.

## Contents

- `config/dag.json` — task DAG (action sequence + dependencies)
- `config/state_schema.json` — state variables the intent monitor tracks from video
- `config/task_profile.json` — system instruction, program map, safety rules, timers, workflow config
- `config/robot_skills.json` — available robot skills (UR5 External Control API)
- `config/initial_scene.json` — regions, objects (all start at `unknown` location), agents
- `ground_truth/` — per-video intent ground-truth files, e.g.
  `milk_tea_making.<take>.intent_gt.json` (created per annotated video, like
  `tasks/tea_making/ground_truth/tea_making.intent_gt.json`)

## Task flow (DAG)

```
idle → setup_workspace
setup_workspace → pour_milk_into_pot, add_water (optional), position_sieve
pour_milk_into_pot → place_pot_on_stove → turn_on_stove → heat_milk → add_tea
add_tea → ask_sugar_preference → (add_sugar | skip_sugar) → simmer_tea
simmer_tea → stir_pot, adjust_heat → turn_off_stove
turn_off_stove + position_sieve → strain_tea_into_cup → cleanup → task_complete
```

## Robot role

The robot proactively:
1. **retrieves** the saucepan, mug, milk/tea/sugar containers, sieve and spoon to the working area;
2. **holds and positions the sieve** over the mug so the human can strain the tea (`hold_sieve_over_mug`);
3. **assists** with sugar dispensing into the pot and stirring;
4. **returns** unused items to storage during cleanup.

## Known variations (not all modeled in the single DAG)

- **Milk measuring**: some takes pour milk into the mug first as a measure, then into the pot.
- **Water**: some add a splash of water (`add_water`, optional branch); pure-milk takes skip it.
- **Tea form**: loose tea leaves vs. tea bag (both map to `add_tea`).
- **Sugar timing**: usually added to the pot mid-brew; a few add it to the cup at the end.
- **Spices**: a few takes add cinnamon / ginger / nutmeg — treated as `Other`.

## Provenance

Step inventory, object set, and timings were derived from the 31 richest
Ego-Exo4D milk-tea takes (keystep + expert commentary + relations), aggregated in
`/home/mani/Repos/annotations/milk_tea_rich/`.

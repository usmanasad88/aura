# Tea Making Task for AURA

This folder contains the task-specific configuration for the tea-making proactive assistance demo.

**This folder can be deleted without affecting the base AURA framework.**

## Contents

- `config/tea_making.yaml` - Main task configuration
- `config/tea_making_dag.json` - Task DAG (action sequence)
- `config/tea_making_state.json` - Task state schema
- `config/robot_skills.json` - Available robot skills
- `config/initial_scene.json` - Initial scene graph setup
- `demo/run_tea_demo.py` - Main demo script
- `demo/test_tea_making.py` - Unit tests

## Robot Capabilities

For this task, the robot can:
1. **retrieve_object** - Pick from storage, place in working area
2. **return_object** - Return object to storage
3. **add_sugar** - Add sugar to cup (learned skill)
4. **stir** - Stir contents of cup
5. **ask_preference** - Ask human about preferences (e.g., sugar amount)

## Task Flow

1. Human starts heating water on induction cooker
2. Robot retrieves cups, tea, milk, sugar to working area
3. When water boils, human adds chai powder
4. Human pours tea into cups
5. Robot adds powdered milk if requested
6. Robot asks about sugar preference
7. Robot adds sugar based on preference
8. Robot stirs the tea
9. Robot returns unused items to storage

## Running the Demo

```bash
cd /home/mani/Repos/aura
python -m tasks.tea_making.demo.run_tea_demo --video demo_data/002.360
```

## Wizard-of-Oz Evaluation

The ground truth timing file should contain timestamps of when a human operator
would have triggered each robot action. This enables:
- Measuring prediction accuracy
- Timing error analysis
- Comparing algorithmic vs human decisions

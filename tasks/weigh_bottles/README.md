# Weigh Bottles Task for AURA

This folder contains the task-specific configuration for the bottle weighing proactive assistance demo.

**This folder can be deleted without affecting the base AURA framework.**

## Contents

- `config/weigh_bottles.yaml` - Main task configuration
- `config/weigh_bottles_dag.json` - Task DAG (action sequence)
- `config/weigh_bottles_state.json` - Task state schema
- `config/ground_truth.json` - Ground truth from Wizard-of-Oz experiment
- `monitors/` - Task-specific monitors:
  - `affordance_monitor.py` - Robot program execution management
  - `performance_monitor.py` - Gemini-based failure detection
- `demo/test_perception.py` - Perception testing with SAM3
- `demo/test_intent.py` - Intent prediction testing
- `demo/test_performance.py` - Affordance and performance monitor testing

## Robot Capabilities

For this task, the robot can:
1. **pick_hardener** - Pick hardener bottle from storage, deliver to human
2. **pick_resin** - Pick resin bottle from storage, deliver to human
3. **return_hardener** - Return hardener bottle to storage
4. **return_resin** - Return resin bottle to storage
5. **go_home** - Return to home position

## Task Flow

1. Robot starts at home position
2. Robot picks hardener bottle and delivers to human
3. Human weighs hardener bottle
4. Robot returns to home
5. Robot picks resin bottle and delivers to human
6. Human weighs resin bottle
7. Robot returns to home
8. Robot collects hardener and returns to storage
9. Robot collects resin and returns to storage
10. Robot returns to home - task complete

## Running the Demo

```bash
cd /home/mani/Repos/aura
python -m tasks.weigh_bottles.demo.run_weigh_bottles_demo --video demo_data/weigh_bottles/video.mp4

# With 360 gripper video
python -m tasks.weigh_bottles.demo.run_weigh_bottles_demo \
    --video demo_data/weigh_bottles/video.mp4 \
    --gripper-video demo_data/weigh_bottles/exp.mp4

# Headless mode with results
python -m tasks.weigh_bottles.demo.run_weigh_bottles_demo \
    --video demo_data/weigh_bottles/video.mp4 \
    --headless --output output.mp4 --save-results results.json
```

## Data Files

Located in `demo_data/weigh_bottles/`:
- `video.mp4` - Third-person workspace view
- `exp.mp4` - 360° GoPro Max on robot gripper
- `joint_states.h5` - Robot joint state recording
- `program_events.json` - Ground truth timing from Wizard-of-Oz

## Wizard-of-Oz Evaluation

The ground truth timing file contains timestamps of when the human operator
triggered each robot program during the data collection:
- `pick_hardener_bottle.prog` - Started at 3.50s
- `pick_resin_bottle.prog` - Started at 46.70s
- `return_hardener_bottle.prog` - Started at 98.82s
- `return_resin_bottle.prog` - Started at 144.87s

## Testing Monitors

### Test Affordance and Performance Monitors

```bash
cd /home/mani/Repos/aura
python -m tasks.weigh_bottles.demo.test_performance \
    --video demo_data/weigh_bottles/video.mp4

# Headless mode (no display)
python -m tasks.weigh_bottles.demo.test_performance \
    --video demo_data/weigh_bottles/video.mp4 --headless

# Skip Gemini calls (affordance only)
python -m tasks.weigh_bottles.demo.test_performance \
    --video demo_data/weigh_bottles/video.mp4 --no-performance
```

### Test Intent Prediction

```bash
python -m tasks.weigh_bottles.demo.test_intent \
    --video demo_data/weigh_bottles/video.mp4
```

### Test Perception

```bash
python -m tasks.weigh_bottles.demo.test_perception \
    --video demo_data/weigh_bottles/video.mp4
```

## Using the Monitors in Code

```python
from tasks.weigh_bottles import (
    WeighBottlesAffordanceMonitor,
    WeighBottlesPerformanceMonitor,
    PerformanceStatus,
    FailureType,
)

# Affordance monitor - manages program execution
affordance = WeighBottlesAffordanceMonitor()
await affordance.start_program("pick_hardener_bottle.prog")
affordance.mark_program_complete("pick_hardener_bottle.prog")

# Performance monitor - detects failures via Gemini
from tasks.weigh_bottles.monitors.performance_monitor import PerformanceMonitorConfig
perf_config = PerformanceMonitorConfig(model="gemini-2.0-flash")
performance = WeighBottlesPerformanceMonitor(perf_config)
performance.set_current_instruction("Pick hardener bottle")
performance.add_frame(frame)
result = await performance.check_performance()
if not result.is_ok:
    print(f"Failure: {result.failure_type.name} - {result.reasoning}")
```

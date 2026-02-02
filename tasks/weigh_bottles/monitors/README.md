# Weigh Bottles Task-Specific Monitors

This folder contains task-specific monitors for the bottle weighing task. These monitors can be deleted without affecting the base AURA framework.

## Monitors

### AffordanceMonitor (`affordance_monitor.py`)

Manages robot program execution for the bottle weighing task. Features:

- **Sequential execution**: Programs must run one after another
- **Prerequisites**: Each program has prerequisites (completed programs)
- **Robot state tracking**: Tracks if robot is moving or at home position
- **Auto-completion**: Marks programs complete when robot stops moving

Available programs:
1. `pick_hardener_bottle.prog` - Pick hardener from storage, deliver to human
2. `pick_resin_bottle.prog` - Pick resin from storage, deliver to human  
3. `return_hardener_bottle.prog` - Return hardener to storage
4. `return_resin_bottle.prog` - Return resin to storage

Usage:
```python
from tasks.weigh_bottles.monitors import WeighBottlesAffordanceMonitor

monitor = WeighBottlesAffordanceMonitor()

# Get available programs
available = monitor.get_available_programs()

# Start a program
await monitor.start_program("pick_hardener_bottle.prog")

# Mark complete when robot stops
monitor.mark_program_complete("pick_hardener_bottle.prog")
```

### PerformanceMonitor (`performance_monitor.py`)

Uses Gemini to analyze video frames and detect task execution failures. Features:

- **Visual analysis**: Analyzes 2-3 seconds of video frames
- **Failure detection**: Detects dropped bottles, failed grasps, collisions, etc.
- **Confidence scoring**: Reports confidence level for detections
- **Status levels**: OK, WARNING, ERROR, CRITICAL

Detected failure types:
- `DROPPED_OBJECT` - Bottle falling from gripper
- `FAILED_GRASP` - Gripper missed the bottle
- `COLLISION` - Unexpected contact with objects
- `WRONG_TRAJECTORY` - Moving in wrong direction
- `ROBOT_STUCK` - No motion when expected
- `HUMAN_INTERVENTION` - Human blocking or intervening

Usage:
```python
from tasks.weigh_bottles.monitors import (
    WeighBottlesPerformanceMonitor,
    PerformanceMonitorConfig,
)

config = PerformanceMonitorConfig(
    fps=2.0,
    model="gemini-2.0-flash",
)
monitor = WeighBottlesPerformanceMonitor(config)

# Set current instruction
monitor.set_current_instruction("Pick hardener bottle from storage")

# Add frames
monitor.add_frame(frame)

# Check performance
result = await monitor.check_performance()
if not result.is_ok:
    print(f"Failure: {result.failure_type.name}")
```

## Testing

Run the test script:
```bash
cd /home/mani/Repos/aura
python -m tasks.weigh_bottles.demo.test_performance \
    --video demo_data/weigh_bottles/video.mp4
```

Options:
- `--headless` - Run without display window
- `--no-performance` - Skip Gemini performance checks
- `--save-video` - Save output video with overlays
- `--frame-skip N` - Process every Nth frame (default: 30)

## Environment Variables

- `GEMINI_API_KEY` - Required for performance monitoring with Gemini

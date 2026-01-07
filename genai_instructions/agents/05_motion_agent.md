# Agent 05: Motion & Intent Prediction Agents

## Task: Implement Real-time Intent and Motion Prediction Monitors

### Objective
Create two complementary monitors for understanding human intention and predicting future actions:

1. **IntentMonitor**: Uses Gemini to analyze video frames and predict current/next actions using configurable task graphs
2. **MotionPredictor**: Uses MediaPipe for fast hand tracking and trajectory prediction

### Architecture Overview

```
                    Webcam Stream
                         │
           ┌─────────────┴─────────────┐
           │                           │
     IntentMonitor               MotionPredictor
     (2fps, Gemini)              (15fps, MediaPipe)
           │                           │
           ▼                           ▼
    IntentOutput                  MotionOutput
    - Current action              - Hand positions
    - Predicted next              - Predicted trajectory
    - Confidence                  - Velocity
    - Reasoning                   - Collision risk
           │                           │
           └─────────────┬─────────────┘
                         │
                    Event Bus
```

### Prerequisites
- Sprint 1 complete (core types and monitor interface)
- GEMINI_API_KEY environment variable set
- Webcam available for testing
- Optional: MediaPipe for hand landmark tracking

---

## Part 1: IntentMonitor (Gemini-based Action Recognition)

### Reference Code
- `/home/mani/Repos/hcdt/realtime_test/realtime_action_node.py` - Gemini-based action prediction with frame buffering
- `/home/mani/Repos/hcdt/realtime_test/config.yaml` - Task graph configuration patterns

### Key Concepts

The IntentMonitor:
1. **Captures frames over time window**: Buffer last N frames (e.g., 2fps × 2 seconds = 4 frames)
2. **Configurable task graphs**: Define possible actions and their dependencies via JSON files
3. **State tracking**: Track scene state variables (holding object, gaze target, etc.)
4. **Action prediction**: Predict current action and next likely action from task graph
5. **Real-time updates**: Query Gemini every ~3 seconds with frame buffer

### Files

#### 1. `src/aura/monitors/intent_monitor.py` ✓ IMPLEMENTED

Key components:
- `IntentPrediction` dataclass - Result structure
- `IntentMonitor` class - Main monitor with:
  - Configurable DAG/state schema loading
  - Frame buffering at 2fps
  - Gemini queries with JSON response schema
  - Action-to-IntentType mapping
- `visualize_intent()` - Visualization helper

#### 2. `src/aura/utils/config.py` - IntentMonitorConfig ✓ IMPLEMENTED

```python
class IntentMonitorConfig(MonitorConfig):
    fps: float = 2.0  # Frame capture rate
    capture_duration: float = 2.0  # Buffer window in seconds
    prediction_interval: float = 3.0  # Gemini query interval
    max_image_dimension: int = 512  # Image resize for Gemini
    model: str = "gemini-2.0-flash-exp"
    # Task graph configuration
    dag_file: Optional[str] = None  # Path to DAG JSON
    state_file: Optional[str] = None  # Path to state schema JSON
    task_name: str = "activity monitoring"
    # Prompt customization
    system_prompt: Optional[str] = None
    task_context: Optional[str] = None
    analysis_instructions: Optional[str] = None
    output_format: Optional[str] = None
```

#### 3. Default Task Graphs ✓ CREATED

Located in `config/task_graphs/`:
- `generic_activity_dag.json` - General-purpose action DAG
- `generic_activity_state.json` - State variable schema

### Usage Example

```python
from aura.monitors import IntentMonitor
from aura.utils.config import IntentMonitorConfig

# Basic usage with defaults
config = IntentMonitorConfig(enabled=True)
monitor = IntentMonitor(config)

# With custom task graph
config = IntentMonitorConfig(
    enabled=True,
    dag_file="config/task_graphs/assembly_task_dag.json",
    state_file="config/task_graphs/assembly_state.json",
    task_name="Assembly Task Monitoring"
)
monitor = IntentMonitor(config)

# Process frames
async def run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        output = await monitor.update(frame=frame)
        if output and output.intent:
            print(f"Current: {output.intent.type.name}")
```

---

## Part 2: MotionPredictor (MediaPipe Hand Tracking)

### Key Concepts

The MotionPredictor:
1. **Fast hand tracking**: Uses MediaPipe Hands at 15fps
2. **Trajectory prediction**: Linear extrapolation with damping
3. **Velocity estimation**: Tracks hand velocity for collision prediction
4. **Lightweight**: No API calls, runs locally

### Files

#### 1. `src/aura/monitors/motion_predictor.py` ✓ IMPLEMENTED

Key components:
- `HandTrackingResult` dataclass - Per-frame hand tracking result
- `MotionPredictor` class - Main monitor with:
  - MediaPipe Hands integration
  - Trajectory prediction (linear extrapolation)
  - Velocity estimation
- `visualize_motion_prediction()` - Visualization helper

#### 2. `src/aura/utils/config.py` - MotionPredictorConfig ✓ UPDATED

```python
class MotionPredictorConfig(MonitorConfig):
    fps: float = 15.0  # Frame capture rate (fast for tracking)
    window_duration: float = 1.0  # Tracking history window
    prediction_horizon: float = 0.5  # Predict 0.5s ahead
    use_hand_tracking: bool = True
    smooth_trajectory: bool = True
    damping_factor: float = 0.95
```

### Usage Example

```python
from aura.monitors import MotionPredictor
from aura.utils.config import MotionPredictorConfig

config = MotionPredictorConfig(
    enabled=True,
    fps=15.0,
    prediction_horizon=0.5
)
predictor = MotionPredictor(config)

# Process frames
async def run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        output = await predictor.update(frame=frame)
        if output and output.predictions:
            trajectory = output.predictions[0].predicted_trajectory
            print(f"Predicted positions: {len(trajectory.poses)}")
```

---

## Combining Both Monitors

For complete human intention understanding, use both monitors together:

```python
from aura.monitors import IntentMonitor, MotionPredictor, MonitorEventBus

# Create monitors
intent_monitor = IntentMonitor(IntentMonitorConfig(enabled=True))
motion_predictor = MotionPredictor(MotionPredictorConfig(enabled=True))

# Create shared event bus
bus = MonitorEventBus()
intent_monitor.set_event_bus(bus)
motion_predictor.set_event_bus(bus)

# Process frames through both
async def run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        # Fast motion tracking (every frame)
        motion_output = await motion_predictor.update(frame=frame)
        
        # Slower intent recognition (periodic)
        intent_output = await intent_monitor.update(frame=frame)
        
        # Combine insights
        if motion_output and intent_output:
            # Motion gives immediate hand position
            # Intent gives semantic action understanding
            pass
```

---

## Testing

### Run Unit Tests

```bash
cd /home/mani/Repos/aura
source .venv/bin/activate

# Test IntentMonitor (no webcam required)
PYTHONPATH="" .venv/bin/python tests/test_intent.py --unit

# Test MotionPredictor (no webcam required)
PYTHONPATH="" .venv/bin/python tests/test_motion.py --unit
```

### Run Live Tests (Webcam Required)

```bash
# Live intent recognition (10 seconds)
PYTHONPATH="" .venv/bin/python tests/test_intent.py --live --duration 30

# Live motion tracking (10 seconds)
PYTHONPATH="" .venv/bin/python tests/test_motion.py --live --duration 30
```

---

## Custom Task Graphs

### Creating a Custom DAG

Create a JSON file like `config/task_graphs/your_task_dag.json`:

```json
{
  "name": "Your Task Name",
  "description": "Description of the task",
  "nodes": {
    "start": {
      "description": "Initial state",
      "visual_cues": ["waiting", "hands at rest"],
      "next_possible": ["action1", "action2"]
    },
    "action1": {
      "description": "First action",
      "visual_cues": ["reaching", "arm extended"],
      "next_possible": ["action3", "end"]
    }
  },
  "start_node": "start",
  "end_nodes": ["end"]
}
```

### Creating a State Schema

Create `config/task_graphs/your_task_state.json`:

```json
{
  "name": "Your Task State",
  "state_variables": {
    "current_step": {
      "type": "number",
      "description": "Current step in the task",
      "default": 0
    },
    "holding_tool": {
      "type": "string",
      "description": "Tool currently being held",
      "default": null
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "current_action": {"type": "string"},
      "confidence": {"type": "number"}
    },
    "required": ["current_action", "confidence"]
  }
}
```

### Using Custom Task Graphs

```python
config = IntentMonitorConfig(
    enabled=True,
    dag_file="config/task_graphs/your_task_dag.json",
    state_file="config/task_graphs/your_task_state.json",
    task_name="Your Custom Task"
)
monitor = IntentMonitor(config)
```

---

## Implementation Notes

### Intent Monitor Design Decisions

1. **Frame buffering at 2fps**: Balances detail vs. context. 4 frames over 2 seconds provides enough motion information.

2. **Prediction interval of 3 seconds**: Gemini API calls have latency. 3-second intervals allow complete processing while maintaining responsiveness.

3. **JSON response schema**: Using Gemini's structured output ensures consistent, parseable responses.

4. **Configurable prompts**: All prompt parts can be customized via config for different tasks.

### Motion Predictor Design Decisions

1. **15fps tracking**: MediaPipe is fast enough for real-time tracking at 15fps without GPU.

2. **Linear extrapolation with damping**: Simple but effective for short-term predictions. Damping factor (0.95) prevents trajectory from extending too far.

3. **Separate from intent**: Keeps the module lightweight and focused. Intent recognition requires more context (Gemini), motion tracking needs speed (MediaPipe).

---

## Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `src/aura/monitors/intent_monitor.py` | NEW | Gemini-based intent recognition |
| `src/aura/monitors/motion_predictor.py` | REPLACED | MediaPipe hand tracking |
| `src/aura/monitors/__init__.py` | UPDATED | Exports both monitors |
| `src/aura/utils/config.py` | UPDATED | IntentMonitorConfig added, MotionPredictorConfig updated |
| `config/task_graphs/generic_activity_dag.json` | NEW | Default task DAG |
| `config/task_graphs/generic_activity_state.json` | NEW | Default state schema |
| `tests/test_intent.py` | NEW | IntentMonitor tests |
| `tests/test_motion.py` | NEW | MotionPredictor tests |

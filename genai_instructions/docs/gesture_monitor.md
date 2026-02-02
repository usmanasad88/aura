# Gesture Recognition Monitor

## Overview

The Gesture Recognition Monitor (`GestureMonitor`) is a core AURA component that uses MediaPipe's Gesture Recognition API to detect hand gestures in real-time. It can be used for:

- **Safety Control**: Stop/resume robot operations based on predefined gestures
- **Intent Recognition**: Map gestures to human intent types (reaching, grasping, etc.)
- **Human-Robot Interaction**: Non-verbal commands and communication

## Features

- Real-time gesture detection using MediaPipe
- Multi-hand tracking (up to 2 hands by default)
- Debouncing for stable gesture recognition
- Safety trigger system (stop/resume gestures)
- Intent mapping from gestures
- Visualization support
- Gesture statistics and history tracking

## Recognized Gestures

MediaPipe's gesture recognizer can detect the following gestures:

- `Closed_Fist`
- `Open_Palm`
- `Pointing_Up`
- `Thumb_Down`
- `Thumb_Up`
- `Victory` (peace sign)
- `ILoveYou` (🤟)

## Installation

MediaPipe is already included in AURA's dependencies. If you need to install it manually:

```bash
# Using uv (recommended for AURA project)
uv add mediapipe

# Using pip
pip install mediapipe
```

## Configuration

The `GestureMonitorConfig` class controls behavior:

```python
from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

config = GestureMonitorConfig(
    enabled=True,
    update_rate_hz=10.0,              # Processing rate
    timeout_sec=5.0,                  # Timeout for processing
    model_path="",                    # Auto-downloads if empty
    min_detection_confidence=0.5,     # Hand detection threshold
    min_tracking_confidence=0.5,      # Hand tracking threshold
    num_hands=2,                      # Max hands to track
    gesture_hold_frames=3,            # Debounce: frames before trigger
    
    # Safety gestures
    stop_gestures={'Open_Palm', 'Pointing_Up'},
    resume_gestures={'Thumb_Up', 'Victory'},
    
    # Intent mapping
    enable_intent_mapping=True,
)

monitor = GestureMonitor(config)
```

## Usage

### Basic Usage

```python
import cv2
import asyncio
from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

async def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Create monitor
    monitor = GestureMonitor()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        output = await monitor.update(frame=frame)
        
        # Check results
        if output.is_valid:
            print(f"Gestures: {[g.gesture_name for g in output.gestures]}")
            print(f"Safety triggered: {output.safety_triggered}")
            
            if output.intent:
                print(f"Detected intent: {output.intent.type.name}")
        
        # Visualize (optional)
        viz_frame = monitor.get_visualization_frame(frame, output)
        cv2.imshow("Gesture Detection", viz_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())
```

### Safety Control Integration

```python
# Example: Robot safety system
monitor = GestureMonitor(GestureMonitorConfig(
    stop_gestures={'Open_Palm', 'Pointing_Up'},
    resume_gestures={'Thumb_Up'},
))

while robot_running:
    frame = get_camera_frame()
    output = await monitor.update(frame=frame)
    
    if output.safety_triggered:
        robot.emergency_stop()
        print("🛑 SAFETY STOP - Open Palm detected!")
    else:
        robot.resume()
```

### Intent Recognition

```python
from aura.core import IntentType

config = GestureMonitorConfig(
    enable_intent_mapping=True,
)
monitor = GestureMonitor(config)

output = await monitor.update(frame=frame)

if output.intent:
    if output.intent.type == IntentType.GRASPING:
        print("Human is about to grasp something")
    elif output.intent.type == IntentType.GESTURING:
        print(f"Human is gesturing: {output.dominant_gesture}")
```

## Integration with AURA Brain

```python
from aura.brain import Brain
from aura.monitors.gesture_monitor import GestureMonitor

# Register monitor with brain
brain = Brain()
gesture_monitor = GestureMonitor()
brain.register_monitor(gesture_monitor)

# Monitor will automatically publish to event bus
# Brain will receive gesture events and safety signals
```

## Test Script

Run the included test script to verify gesture recognition:

```bash
# Basic test with webcam
python scripts/test_gesture_monitor.py --show_viz

# With RealSense camera
python scripts/test_gesture_monitor.py --use_realsense --show_viz

# Custom camera and FPS
python scripts/test_gesture_monitor.py --camera_id 1 --fps 15 --show_viz

# Specify model path
python scripts/test_gesture_monitor.py --model_path ~/.mediapipe/gesture_recognizer.task
```

Test script controls:
- Press `q` to quit
- Press `r` to reset safety state

## Model Download

The gesture recognizer model is automatically downloaded on first use to:
- `~/.cache/mediapipe/gesture_recognizer.task` (preferred)
- `~/.mediapipe/gesture_recognizer.task`

If you want to manually download:
```bash
mkdir -p ~/.cache/mediapipe
curl -o ~/.cache/mediapipe/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## API Reference

### GestureMonitor

#### Methods

- `__init__(config: GestureMonitorConfig)`: Initialize monitor
- `async update(frame: np.ndarray) -> GestureOutput`: Process frame and detect gestures
- `get_visualization_frame(frame, output) -> np.ndarray`: Draw gesture visualization
- `reset_safety()`: Reset safety trigger state
- `get_gesture_statistics() -> Dict`: Get gesture detection statistics

#### Properties

- `monitor_type`: Returns `MonitorType.INTENT`
- `safety_triggered`: Current safety state (bool)
- `last_gesture`: Most recent detected gesture (str)

### GestureOutput

Output data class containing:

```python
@dataclass
class GestureOutput(MonitorOutput):
    gestures: List[GestureRecognitionResult]  # All detected gestures
    dominant_gesture: Optional[str]            # Most stable gesture
    safety_triggered: bool                     # Safety stop signal
    intent: Optional[Intent]                   # Mapped intent
```

### GestureRecognitionResult

Individual gesture result:

```python
@dataclass
class GestureRecognitionResult:
    gesture_name: str          # Name of gesture
    confidence: float          # Score 0-1
    handedness: str           # "Left" or "Right"
    hand_landmarks: List      # MediaPipe landmarks
    timestamp: datetime       # Detection time
```

## Customization

### Custom Gesture-to-Intent Mapping

Override the default mapping:

```python
from aura.core import IntentType

monitor = GestureMonitor()

# Custom mapping
monitor.GESTURE_TO_INTENT = {
    'Pointing_Up': IntentType.REACHING,
    'Open_Palm': IntentType.IDLE,
    'Closed_Fist': IntentType.GRASPING,
    'Thumb_Up': IntentType.IDLE,
    'Victory': IntentType.GESTURING,
}
```

### Custom Safety Gestures

```python
config = GestureMonitorConfig(
    stop_gestures={'Closed_Fist', 'Thumb_Down'},  # Custom stop
    resume_gestures={'Thumb_Up', 'ILoveYou'},     # Custom resume
)
```

## Performance Considerations

- **Processing Rate**: Default 10 Hz is good balance. Higher rates (15-30 Hz) give better responsiveness but use more CPU.
- **Debouncing**: `gesture_hold_frames=3` requires gesture to be stable for 3 frames before triggering. Increase for more stability, decrease for faster response.
- **Number of Hands**: Tracking 2 hands is more expensive than 1. Set `num_hands=1` if only one hand is relevant.
- **Confidence Thresholds**: Lower thresholds (`0.3-0.4`) detect more gestures but with false positives. Higher (`0.6-0.7`) are more accurate but may miss gestures.

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Gesture recognizer model not found
```
**Solution**: Model will auto-download. If download fails, manually download from the URL above.

### MediaPipe Not Installed
```
RuntimeError: MediaPipe is required for gesture recognition
```
**Solution**: `uv add mediapipe` or `pip install mediapipe`

### Low Frame Rate
**Symptoms**: Laggy visualization, delayed gesture detection
**Solutions**:
- Reduce `update_rate_hz` to 5-8
- Set `num_hands=1` if detecting single hand
- Use lower resolution camera (640x480)

### False Gesture Detections
**Symptoms**: Random gestures detected, unstable recognition
**Solutions**:
- Increase `gesture_hold_frames` to 5-10
- Increase `min_detection_confidence` to 0.6-0.7
- Ensure good lighting
- Keep hands in camera view

### Safety State Stuck
**Symptoms**: Safety remains triggered even after gesture removed
**Solutions**:
- Call `monitor.reset_safety()`
- Ensure resume gesture is in configured set
- Check gesture history: `monitor.get_gesture_statistics()`

## Example Applications

### 1. Collaborative Assembly

Human shows specific gestures to request tools or pause robot:
```python
config = GestureMonitorConfig(
    stop_gestures={'Open_Palm'},           # Pause
    resume_gestures={'Thumb_Up'},          # Continue
)
# Add custom gestures for tool requests
```

### 2. Composite Layup Demo

Operator uses gestures during fiberglass layup:
```python
# Stop robot when working in danger zone
# Point to indicate next ply location
# Thumbs up when ply is correctly placed
```

### 3. Teleoperation Assist

Combine gesture control with robot arm movements:
```python
# Victory = "I'm done with this area"
# Pointing = "Move robot to this location"
# Closed Fist = "Activate gripper"
```

## Related Components

- [Intent Monitor](./intent_monitor.md): High-level intent detection using Gemini
- [Motion Predictor](./motion_predictor.md): Hand trajectory prediction
- [Sound Monitor](./sound_monitor.md): Voice command detection
- [Monitor Event Bus](./monitor_bus.md): Event system integration

## References

- [MediaPipe Gesture Recognition](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- AURA Documentation: [Documentation.md](../Documentation.md)

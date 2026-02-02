# Gesture Recognition Integration

## Overview

MediaPipe-based gesture recognition has been integrated into the AURA framework. This provides real-time hand gesture detection for safety control, intent recognition, and human-robot interaction.

## Installation

The project uses `uv` for dependency management. MediaPipe is already included in the dependencies.

### Setup with uv

```bash
# Install dependencies
cd /home/mani/Repos/aura
uv sync

# Or add mediapipe explicitly if needed
uv add mediapipe
```

### Manual Installation (if not using uv)

```bash
pip install mediapipe
```

## Quick Start

### 1. Test Gesture Recognition

Run the test script to verify the installation:

```bash
# Basic test with webcam visualization
python scripts/test_gesture_monitor.py --show_viz

# With higher FPS for responsive tracking
python scripts/test_gesture_monitor.py --show_viz --fps 15

# With RealSense camera
python scripts/test_gesture_monitor.py --use_realsense --show_viz
```

### 2. Use in Your Code

```python
import asyncio
import cv2
from aura.monitors import GestureMonitor, GestureMonitorConfig

async def main():
    # Configure gesture monitor
    config = GestureMonitorConfig(
        stop_gestures={'Open_Palm', 'Pointing_Up'},
        resume_gestures={'Thumb_Up', 'Victory'},
        num_hands=2,
        gesture_hold_frames=3,  # Debounce
    )
    
    # Create monitor
    monitor = GestureMonitor(config)
    
    # Capture and process
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        output = await monitor.update(frame=frame)
        
        # Check results
        if output.is_valid:
            print(f"Safety: {'STOP' if output.safety_triggered else 'SAFE'}")
            print(f"Gestures: {[g.gesture_name for g in output.gestures]}")
            
            if output.intent:
                print(f"Intent: {output.intent.type.name}")
        
        # Visualize
        viz = monitor.get_visualization_frame(frame, output)
        cv2.imshow("Gesture Monitor", viz)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())
```

## Recognized Gestures

MediaPipe can detect these gestures:
- `Closed_Fist` - Closed hand
- `Open_Palm` - Open hand (default STOP gesture)
- `Pointing_Up` - Index finger pointing up (default STOP gesture)
- `Thumb_Down` - Thumbs down
- `Thumb_Up` - Thumbs up (default RESUME gesture)
- `Victory` - Peace sign / V sign (default RESUME gesture)
- `ILoveYou` - 🤟 sign

## Configuration Options

```python
GestureMonitorConfig(
    enabled=True,                     # Enable/disable monitor
    update_rate_hz=10.0,             # Processing rate
    timeout_sec=5.0,                  # Timeout for processing
    model_path="",                    # Model path (auto-downloads if empty)
    min_detection_confidence=0.5,     # Hand detection threshold
    min_tracking_confidence=0.5,      # Hand tracking threshold
    num_hands=2,                      # Max hands to track (1-2)
    gesture_hold_frames=3,            # Debounce frames
    stop_gestures={'Open_Palm', 'Pointing_Up'},  # Safety stop gestures
    resume_gestures={'Thumb_Up', 'Victory'},     # Safety resume gestures
    enable_intent_mapping=True,       # Map gestures to intents
)
```

## Integration with AURA System

### With Brain

```python
from aura.brain import Brain
from aura.monitors import GestureMonitor

brain = Brain()
gesture_monitor = GestureMonitor()

# Register monitor with brain
brain.register_monitor(gesture_monitor)

# Monitor will automatically publish to event bus
```

### Safety Control

```python
# Monitor gesture for safety control
while robot_running:
    frame = camera.capture()
    output = await gesture_monitor.update(frame=frame)
    
    if output.safety_triggered:
        robot.emergency_stop()
        print("🛑 STOP gesture detected!")
    else:
        robot.resume()
```

### Intent Recognition

```python
from aura.core import IntentType

output = await gesture_monitor.update(frame=frame)

if output.intent:
    if output.intent.type == IntentType.GRASPING:
        print("Human is grasping")
    elif output.intent.type == IntentType.GESTURING:
        print(f"Gesture: {output.dominant_gesture}")
```

## Files Added

### Core Implementation
- `src/aura/monitors/gesture_monitor.py` - Main gesture monitor class
  - `GestureMonitor` - Monitor implementation
  - `GestureMonitorConfig` - Configuration class
  - `GestureOutput` - Output data class
  - `GestureRecognitionResult` - Individual gesture result

### Test & Scripts
- `scripts/test_gesture_monitor.py` - Test script with visualization

### Documentation
- `genai_instructions/docs/gesture_monitor.md` - Complete documentation
- `GESTURE_INTEGRATION.md` - This file

## Model Download

The MediaPipe gesture recognizer model is automatically downloaded on first use to:
```
~/.cache/mediapipe/gesture_recognizer.task
```

Manual download (if needed):
```bash
mkdir -p ~/.cache/mediapipe
curl -o ~/.cache/mediapipe/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Troubleshooting

### MediaPipe Import Error
```bash
# Using uv
uv add mediapipe

# Using pip
pip install mediapipe
```

### Camera Not Found
```python
# Try different camera IDs
python scripts/test_gesture_monitor.py --camera_id 1 --show_viz

# Or use RealSense
python scripts/test_gesture_monitor.py --use_realsense --show_viz
```

### Low Performance
- Reduce FPS: `--fps 5`
- Track single hand: `--num_hands 1`
- Use lower resolution camera

### False Detections
- Increase debounce: set `gesture_hold_frames=5` or higher
- Increase confidence: set `min_detection_confidence=0.6`
- Ensure good lighting

## Performance Tips

1. **Frame Rate**: 10 Hz is a good balance. Use 5-8 Hz for lower-end hardware, 15-30 Hz for responsive systems.

2. **Debouncing**: `gesture_hold_frames=3` requires gesture to be stable for 3 frames. Increase for stability, decrease for responsiveness.

3. **Number of Hands**: Tracking 2 hands costs ~2x CPU. Use `num_hands=1` when only one hand is relevant.

4. **Camera Resolution**: 640x480 is sufficient and faster than 1080p.

## Example Applications

### 1. Composite Layup Demo
```python
# Operator gestures during fiberglass layup
# - Open palm to pause robot
# - Thumbs up when ply is placed correctly
# - Point to indicate next location
```

### 2. Collaborative Sorting Game
```python
# Players use gestures for commands
# - Victory sign when task complete
# - Open palm to stop robot
# - Point to direct robot attention
```

### 3. Robot Teleoperation
```python
# Gesture-based robot control
# - Point to move robot
# - Closed fist to activate gripper
# - Thumbs up to confirm action
```

## Related Documentation

- [Main Documentation](genai_instructions/Documentation.md)
- [Gesture Monitor Details](genai_instructions/docs/gesture_monitor.md)
- [Motion Predictor](src/aura/monitors/motion_predictor.py) - Hand tracking
- [Intent Monitor](src/aura/monitors/intent_monitor.py) - High-level intent

## Dependencies

The gesture monitor requires:
- `mediapipe>=0.10.31` (already in pyproject.toml)
- `opencv-python>=4.11.0` (already in pyproject.toml)
- `numpy` (standard dependency)

All dependencies are managed through `uv` and specified in `pyproject.toml`.

## Testing

```bash
# Run unit tests (when implemented)
pytest tests/test_monitors/test_gesture_monitor.py

# Run integration test
python scripts/test_gesture_monitor.py --show_viz --fps 10

# Test with different gestures
# Hold each gesture for 3+ seconds to verify recognition:
# - Open Palm (should trigger STOP)
# - Pointing Up (should trigger STOP)
# - Thumbs Up (should trigger RESUME)
# - Victory sign (should trigger RESUME)
```

## Future Enhancements

Potential improvements:
1. Custom gesture training
2. Gesture sequences (e.g., "wave twice to activate")
3. Distance/depth estimation for gestures
4. Multi-person gesture tracking
5. Gesture velocity/dynamics analysis

## Support

For issues or questions:
1. Check the detailed documentation: `genai_instructions/docs/gesture_monitor.md`
2. Review the example: `scripts/test_gesture_monitor.py`
3. Verify MediaPipe installation: `python -c "import mediapipe; print(mediapipe.__version__)"`

## License

Part of the AURA framework. See main project LICENSE.

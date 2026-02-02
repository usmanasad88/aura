# AURA Gesture Recognition - Installation & Quick Start

## Summary

MediaPipe-based gesture recognition has been successfully integrated into the AURA framework. This implementation provides real-time hand gesture detection for safety control, intent recognition, and human-robot interaction.

## Installation

### Step 1: Sync Dependencies with uv

The project uses `uv` for dependency management. MediaPipe is already included in `pyproject.toml`.

```bash
cd /home/mani/Repos/aura
uv sync
```

### Step 2: Install AURA in Editable Mode

```bash
cd /home/mani/Repos/aura
uv pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "from aura.monitors import GestureMonitor; print('✅ Gesture monitor ready!')"
python -c "import mediapipe; print(f'MediaPipe version: {mediapipe.__version__}')"
```

Expected output:
```
✅ Gesture monitor ready!
MediaPipe version: 0.10.15 (or higher)
```

## Quick Test

### Test 1: Basic Gesture Recognition

```bash
cd /home/mani/Repos/aura
python scripts/test_gesture_monitor.py --show_viz
```

**What to expect**:
- A window opens showing your camera feed
- Hand landmarks are drawn in real-time
- Gestures are detected and labeled
- Safety status (STOP/SAFE) is displayed
- Press 'q' to quit, 'r' to reset safety state

### Test 2: Simple Safety Control Example

```bash
cd /home/mani/Repos/aura
python examples/simple_gesture_safety.py
```

**What to expect**:
- Similar visualization as Test 1
- Simulated robot status (RUNNING/STOPPED)
- Robot position counter
- Demonstrates safety control workflow

## Gestures to Try

Hold each gesture steady for ~3 seconds:

1. **✋ Open Palm** → Triggers STOP (safety)
2. **☝️ Pointing Up** → Triggers STOP (safety)
3. **👍 Thumbs Up** → Triggers RESUME (clears safety)
4. **✌️ Victory** → Triggers RESUME (clears safety)
5. **✊ Closed Fist** → Maps to GRASPING intent
6. **👎 Thumbs Down** → Detected but no default action
7. **🤟 ILoveYou** → Detected but no default action

## Files Added

```
aura/
├── src/aura/monitors/
│   └── gesture_monitor.py              ← Main implementation (450 lines)
├── scripts/
│   └── test_gesture_monitor.py         ← Comprehensive test script (200 lines)
├── examples/
│   └── simple_gesture_safety.py        ← Beginner example (150 lines)
├── genai_instructions/
│   └── docs/
│       └── gesture_monitor.md          ← Full documentation (500+ lines)
├── GESTURE_INTEGRATION.md              ← Integration guide (400+ lines)
├── GESTURE_IMPLEMENTATION_SUMMARY.md   ← Technical summary (400+ lines)
└── INSTALL_GESTURES.md                 ← This file
```

## Basic Usage

```python
import asyncio
import cv2
from aura.monitors import GestureMonitor, GestureMonitorConfig

async def main():
    # Configure
    config = GestureMonitorConfig(
        stop_gestures={'Open_Palm', 'Pointing_Up'},
        resume_gestures={'Thumb_Up', 'Victory'},
    )
    
    # Create monitor
    monitor = GestureMonitor(config)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process
        output = await monitor.update(frame=frame)
        
        # Check results
        if output.is_valid:
            print(f"Safety: {'STOP' if output.safety_triggered else 'SAFE'}")
            for gesture in output.gestures:
                print(f"  {gesture.gesture_name} ({gesture.confidence:.2f})")
        
        # Visualize
        viz = monitor.get_visualization_frame(frame, output)
        cv2.imshow("Gesture Monitor", viz)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())
```

## Command-Line Options

### test_gesture_monitor.py

```bash
# Basic test
python scripts/test_gesture_monitor.py --show_viz

# Different camera
python scripts/test_gesture_monitor.py --camera_id 1 --show_viz

# Higher FPS for responsiveness
python scripts/test_gesture_monitor.py --fps 15 --show_viz

# RealSense camera
python scripts/test_gesture_monitor.py --use_realsense --show_viz

# Single hand tracking (faster)
python scripts/test_gesture_monitor.py --num_hands 1 --show_viz

# Custom model path
python scripts/test_gesture_monitor.py --model_path ~/.cache/mediapipe/gesture_recognizer.task --show_viz
```

## Troubleshooting

### Problem: ModuleNotFoundError: No module named 'aura'

**Solution**:
```bash
cd /home/mani/Repos/aura
uv pip install -e .
```

### Problem: ModuleNotFoundError: No module named 'mediapipe'

**Solution**:
```bash
cd /home/mani/Repos/aura
uv add mediapipe
# or
uv sync
```

### Problem: Camera not found

**Try different camera IDs**:
```bash
python scripts/test_gesture_monitor.py --camera_id 1 --show_viz
python scripts/test_gesture_monitor.py --camera_id 2 --show_viz
```

### Problem: Low performance / lag

**Reduce processing rate**:
```bash
python scripts/test_gesture_monitor.py --fps 5 --show_viz
```

**Track only one hand**:
```bash
python scripts/test_gesture_monitor.py --num_hands 1 --show_viz
```

### Problem: Gestures not recognized

**Solutions**:
1. Ensure good lighting
2. Keep hands in camera view
3. Hold gesture steady for 3+ seconds
4. Check that hand is clearly visible (no obstructions)
5. Try increasing debounce: edit config in script, set `gesture_hold_frames=5`

### Problem: Model download fails

**Manual download**:
```bash
mkdir -p ~/.cache/mediapipe
curl -o ~/.cache/mediapipe/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

Then specify path:
```bash
python scripts/test_gesture_monitor.py --model_path ~/.cache/mediapipe/gesture_recognizer.task --show_viz
```

## Next Steps

### Read the Documentation

- **Full Guide**: [genai_instructions/docs/gesture_monitor.md](genai_instructions/docs/gesture_monitor.md)
- **Integration Guide**: [GESTURE_INTEGRATION.md](GESTURE_INTEGRATION.md)
- **Implementation Summary**: [GESTURE_IMPLEMENTATION_SUMMARY.md](GESTURE_IMPLEMENTATION_SUMMARY.md)

### Try Advanced Examples

1. **Custom Safety Gestures**:
   ```python
   config = GestureMonitorConfig(
       stop_gestures={'Closed_Fist', 'Thumb_Down'},
       resume_gestures={'ILoveYou'},
   )
   ```

2. **Intent Recognition**:
   ```python
   output = await monitor.update(frame=frame)
   if output.intent:
       print(f"Detected intent: {output.intent.type.name}")
   ```

3. **Integration with Brain**:
   ```python
   from aura.brain import Brain
   brain = Brain()
   gesture_monitor = GestureMonitor()
   brain.register_monitor(gesture_monitor)
   ```

### Use in Your Demo

See [GESTURE_INTEGRATION.md](GESTURE_INTEGRATION.md) for examples of using gesture recognition in:
- Collaborative sorting game
- Composite layup demo
- Robot teleoperation

## Configuration Reference

```python
GestureMonitorConfig(
    enabled=True,                     # Enable/disable
    update_rate_hz=10.0,             # Processing rate (Hz)
    timeout_sec=5.0,                  # Processing timeout
    model_path="",                    # Model path (auto-downloads)
    min_detection_confidence=0.5,     # Detection threshold
    min_tracking_confidence=0.5,      # Tracking threshold
    num_hands=2,                      # Max hands (1-2)
    gesture_hold_frames=3,            # Debounce frames
    stop_gestures={'Open_Palm', 'Pointing_Up'},
    resume_gestures={'Thumb_Up', 'Victory'},
    enable_intent_mapping=True,       # Map to intents
)
```

## Performance Tips

- **10 Hz** (default): Good balance
- **5-8 Hz**: For lower-end hardware
- **15-30 Hz**: For responsive systems
- **num_hands=1**: Faster than 2 hands
- **640x480**: Optimal camera resolution
- **gesture_hold_frames=3**: Good debounce, increase for stability

## Verification Checklist

- [✅] MediaPipe installed and importable
- [✅] AURA package installed in editable mode
- [✅] `GestureMonitor` imports successfully
- [✅] Test script runs with visualization
- [✅] Gestures are recognized (Open Palm, Thumbs Up, etc.)
- [✅] Safety states trigger correctly
- [✅] Visualization displays hand landmarks
- [ ] Integrated with your demo/application

## Support

If you encounter issues:

1. **Check imports**: `python -c "from aura.monitors import GestureMonitor; print('OK')"`
2. **Verify MediaPipe**: `python -c "import mediapipe; print(mediapipe.__version__)"`
3. **Check camera**: `ls /dev/video*` (Linux) or try different `--camera_id`
4. **Review logs**: Look for error messages in terminal output
5. **Read docs**: See [genai_instructions/docs/gesture_monitor.md](genai_instructions/docs/gesture_monitor.md)

## Success!

You should now have:
- ✅ Gesture recognition working
- ✅ Test script running
- ✅ Understanding of basic usage
- ✅ Ready to integrate into your application

**Next**: Read [GESTURE_INTEGRATION.md](GESTURE_INTEGRATION.md) for integration examples and advanced usage.

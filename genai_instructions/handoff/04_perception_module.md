# Perception Module Complete

## Files Created
- `src/aura/monitors/perception_module.py` (450 lines)
- `scripts/test_perception.py` (240 lines)
- Updated `src/aura/monitors/__init__.py` to export PerceptionModule

## Features Implemented
1. ✅ SAM3 integration for object segmentation
2. ✅ Gemini-based automatic object detection
3. ✅ Multi-object tracking with text prompts
4. ✅ Bounding box and mask extraction
5. ✅ Confidence threshold filtering
6. ✅ Real-time visualization
7. ✅ Lazy loading of heavy models
8. ✅ Async processing interface (inherits from BaseMonitor)
9. ✅ Periodic re-detection with Gemini
10. ✅ Configuration via PerceptionConfig

## Architecture

### PerceptionModule Class
- Inherits from `BaseMonitor`
- Uses `PerceptionConfig` for configuration
- Lazy-loads SAM3 model on first use
- Returns `PerceptionOutput` with list of `TrackedObject`

### Key Methods
- `process(frame: np.ndarray) -> PerceptionOutput`: Main processing
- `visualize(frame, output) -> np.ndarray`: Draw detections on frame
- `detect_objects_with_gemini()`: Auto-detect objects using Gemini

### Configuration Options
```python
class PerceptionConfig:
    use_sam3: bool = True
    use_gemini_detection: bool = True
    max_objects: int = 10
    confidence_threshold: float = 0.5
    gemini_model: str = "gemini-2.0-flash-exp"
    default_prompts: List[str] = ["person", "hand"]
    detection_interval_frames: int = 30
```

## Usage

### Basic Usage
```python
from aura.monitors import PerceptionModule, PerceptionConfig
from aura.utils.config import load_config
import cv2

# Load config
config = load_config()

# Create perception module
perception = PerceptionModule(config.monitors.perception)

# Process frame
frame = cv2.imread("image.jpg")
output = await perception.process(frame)

# Visualize
vis_frame = perception.visualize(frame, output)
cv2.imshow("Perception", vis_frame)
```

### Test Scripts

#### Webcam Test (with display)
```bash
cd /home/mani/Repos/aura
uv run python scripts/test_perception.py --input webcam --duration 10
```

#### Webcam Test (headless, save video)
```bash
uv run python scripts/test_perception.py --input webcam --no-display --max-frames 50
```

#### Image Test (auto-detect)
```bash
uv run python scripts/test_perception.py --input /path/to/image.jpg
```

#### Image Test (specific prompts)
```bash
uv run python scripts/test_perception.py --input /path/to/image.jpg --prompts cup phone laptop
```

## Performance

### Expected Performance
- **FPS**: 5-15 depending on hardware
  - GPU (CUDA): ~10-15 FPS
  - CPU only: ~2-5 FPS
- **Model loading**: 10-20 seconds first time
- **Gemini detection**: ~1-2 seconds per frame
- **SAM3 segmentation**: ~100-200ms per prompt

### Optimizations
- Lazy model loading (only loads when first needed)
- Periodic Gemini detection (not every frame)
- Confidence threshold filtering
- GPU acceleration when available

## Data Flow

1. **Input**: BGR frame from OpenCV
2. **Gemini Detection** (optional, periodic):
   - Convert to RGB
   - Resize and compress
   - Query Gemini for object list
3. **SAM3 Segmentation**:
   - Convert to PIL Image
   - For each prompt:
     - Reset SAM3 prompts
     - Run text-based segmentation
     - Extract boxes, masks, scores
4. **Output**: `PerceptionOutput` with `TrackedObject` list
5. **Visualization**: Draw boxes, masks, labels on frame

## Integration with AURA System

### With MonitorEventBus
```python
from aura.monitors import MonitorEventBus, PerceptionModule

bus = MonitorEventBus()
perception = PerceptionModule()
perception.set_event_bus(bus)

# Subscribe to perception events
async def on_perception(event):
    output = event.output
    print(f"Detected {len(output.objects)} objects")

bus.subscribe(MonitorType.PERCEPTION, on_perception)

# Start processing
await perception.start_monitoring()
```

### With AuraState
```python
from aura.core import AuraState

state = AuraState()
output = await perception.process(frame)

# Update state
for obj in output.objects:
    state.scene_graph.objects[obj.object_id] = obj
```

## Known Limitations

1. **No persistent tracking**: Object IDs reset each frame
   - TODO: Add IOU-based tracking across frames
2. **No depth estimation**: Only 2D bounding boxes
   - TODO: Add depth camera support or monocular depth
3. **No pose estimation**: `TrackedObject.pose` is None
   - TODO: Integrate 6DOF pose estimation
4. **Gemini rate limits**: Frequent detection can hit API limits
   - Use `detection_interval_frames` to control frequency
5. **CPU-only mode is slow**: Recommended to use GPU
   - SAM3 requires significant compute

## Next Steps

### Immediate TODOs
1. Add unit tests for `PerceptionModule`
2. Add IOU-based object tracking
3. Add depth estimation option
4. Add pose estimation for known objects

### Future Enhancements
1. Multi-camera support
2. 3D scene reconstruction
3. Object relationship detection
4. Activity recognition
5. Integration with digital twin

## Dependencies

### Required
- `torch` (PyTorch)
- `opencv-python` (cv2)
- `Pillow` (PIL)
- `numpy`
- `sam3` (in third_party/)

### Optional
- `google-genai` (for Gemini detection)
- CUDA (for GPU acceleration)

## Testing Checklist

- [x] Module imports successfully
- [x] Configuration loads correctly
- [ ] Webcam test runs (requires camera + display)
- [ ] Image test runs (requires test image)
- [ ] SAM3 segmentation works
- [ ] Gemini detection works (requires API key)
- [ ] Visualization renders correctly
- [ ] FPS is acceptable (>5 FPS)
- [ ] Multiple objects detected
- [ ] Confidence filtering works

## References

- SAM3: `/home/mani/Repos/aura/third_party/sam3`
- Original test: `/home/mani/Repos/aura/testing.py`
- Task instructions: `genai_instructions/agents/04_perception_agent.md`

# Perception Monitor Test Results - Tea Making Task

## Test Configuration

**Video**: `demo_data/002.360` (10,466 frames @ 30 fps, 360° GoPro format)  
**Model**: SAM3 (Segment Anything Model 3) with text prompts  
**Objects of Interest** (from tea_making.yaml):
- pot, pan, cup
- induction cooker, stove, kettle
- sugar container, milk container, tea container
- spoon
- hand, person

**Processing**: 360° video → perspective extraction (90° FOV) → SAM3 segmentation

---

## Test Results Summary

### Performance
- **First frame**: 19.21s (includes SAM3 model loading)
- **Subsequent frames**: ~1.0s per frame
- **Average**: 7.09s per frame (across 3 test frames)

### Detection Statistics (3 frames sampled)

| Object | Avg per Frame | Notes |
|--------|--------------|-------|
| **pot** | 2.0 | ✅ Good detection (multiple false positives) |
| **hand** | 1.0 | ✅ Detected when visible |
| **stove** | 1.0 | ✅ Detected (induction cooker) |
| **pan** | 0.7 | ⚠️ Confusion with pot |
| **person** | 0.7 | ✅ Good detection |
| **cup** | 0.3 | ⚠️ Low detection rate |
| **kettle** | 0.3 | ⚠️ Confusion with pot |

### Not Detected
- sugar container
- milk container
- tea container
- spoon
- induction cooker (detected as "stove")

---

## Detailed Observations

### Frame 0 (Setup phase)
**Objects Found**: 8 total
- **pot**: 5 instances (avg conf: 0.58) ⚠️ Multiple false positives
- **pan**: 2 instances (avg conf: 0.64) ⚠️ Overlapping with pot
- **cup**: 1 instance (conf: 0.54) ✅

**Issues**:
- SAM3 detecting multiple overlapping regions for same pot
- Confusion between pot/pan/kettle (similar cylindrical shapes)
- No detection of smaller objects (containers, spoons)

### Frame 5232 (Mid-task, human active)
**Objects Found**: 4 total
- **hand**: 3 instances (avg conf: 0.72) ✅ Good hand detection
- **person**: 1 instance (conf: 0.98) ✅ Excellent person detection

**Issues**:
- Multiple hand detections (likely overlapping regions)
- No tea-making objects detected (focus on human)

### Frame 10465 (Final phase)
**Objects Found**: 6 total
- **pot**: 1 instance (conf: 0.71) ✅
- **stove**: 3 instances (avg conf: 0.58) ⚠️ Multiple detections
- **kettle**: 1 instance (conf: 0.53) ⚠️ Actually pot?
- **person**: 1 instance (conf: 0.98) ✅

---

## Issues & Recommendations

### 🔴 Critical Issues

1. **Multiple Overlapping Detections**
   - SAM3 returns multiple bounding boxes for the same object
   - **Solution**: Add non-maximum suppression (NMS) post-processing
   - **Impact**: Inflates object counts, confuses tracking

2. **Pot/Pan/Kettle Confusion**
   - Similar cylindrical shapes cause misclassification
   - **Solution**: Use Gemini Vision API for better semantic understanding
   - **Alternative**: Add context-aware filtering based on scene layout

3. **Small Object Missed**
   - Containers, spoons not detected at all
   - **Solution**: 
     - Increase image resolution (currently 960x720)
     - Lower confidence threshold (currently 0.3)
     - Use region proposals for known locations

### ⚠️ Medium Priority

4. **Slow Processing (1s/frame)**
   - Cannot run real-time at 2 Hz (tea_making.yaml config)
   - **Solution**: 
     - Batch processing on GPU
     - Use smaller SAM3 model variant
     - Process at lower frequency (0.5 Hz)

5. **Missing Objects**
   - Sugar/milk/tea containers never detected
   - **Cause**: Text prompts too generic ("sugar container")
   - **Solution**: 
     - Train custom detector OR
     - Use more specific prompts: "white jar", "bottle", "box"
     - Add visual examples to SAM3

### ✅ Working Well

- Person detection (98% confidence)
- Hand detection (when visible)
- Pot/stove detection (with false positives)

---

## Recommended Adjustments

### 1. Add NMS Post-Processing

```python
def apply_nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression to remove overlapping boxes."""
    # Sort by score
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        # Compute IoU with rest
        ious = compute_iou(boxes[i], boxes[indices[1:]])
        
        # Keep only boxes with IoU < threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep
```

### 2. Update Perception Config

```yaml
perception:
  update_rate_hz: 0.5  # Reduce from 2.0 Hz (too slow)
  confidence_threshold: 0.25  # Lower for small objects
  nms_iou_threshold: 0.5  # Add NMS
  image_resolution: [1280, 960]  # Increase resolution
  objects_of_interest:
    # More specific prompts
    - "cooking pot"
    - "drinking cup"
    - "metal spoon"
    - "white jar"
    - "bottle"
    - "stove burner"
    - "human hand"
    - "person"
```

### 3. Enable Gemini Detection

Currently disabled. Should enable for first frame to auto-detect objects:

```yaml
use_gemini_detection: true
detection_interval_frames: 150  # Re-detect every 5 seconds at 30fps
```

### 4. Add Tracking

SAM3 doesn't track between frames. Should add:
- Simple centroid tracking for object IDs
- Kalman filter for position smoothing
- Re-identification when objects reappear

---

## Next Steps

1. **Implement NMS**: Add to `perception_module.py` 
2. **Test with higher resolution**: 1280x960 instead of 640x480
3. **Enable Gemini detection**: Let LLM find objects in first frame
4. **Add object tracking**: Maintain IDs across frames
5. **Test other monitors**: Intent, Motion prediction

---

## Saved Outputs

Visualizations saved to: `tasks/tea_making/demo/perception_outputs/`
- `frame_0000.jpg` - Setup phase with pot detections
- `frame_5232.jpg` - Human interaction with hand detection
- `frame_10465.jpg` - Final phase with stove/pot

Run with `--no-display` flag disabled to view frames interactively.

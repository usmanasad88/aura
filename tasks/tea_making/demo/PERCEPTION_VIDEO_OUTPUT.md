# Perception Video Output - Tea Making Task

## Overview

Created visualization videos showing the perception monitor's real-time object detection and segmentation overlays on the 360° GoPro tea-making video.

## Output Files

### Test Videos
- **`test_short.mp4`** (24s processing, 6 frames)
  - Duration: 3 seconds at 1 FPS
  - Frames: 0, 30, 60, 90, 120, 150 (every 1 second of source)
  - Purpose: Quick validation test

### Full Video (In Progress)
- **`tea_making_perception_full.mp4`** 
  - Source: 10,466 frames @ 30 FPS = 349 seconds = 5:49 minutes
  - Output: ~349 frames @ 1 FPS = 5:49 minutes
  - Processing: Every 30th frame (1 Hz sample rate)
  - Estimated time: ~6 minutes (1s/frame × 349 frames)
  - Status: Running in background (check: `tail -f /tmp/perception_full.log`)

## Video Features

Each frame includes:

### Object Detection Overlays
- **Bounding boxes** (colored by object class)
- **Segmentation masks** (semi-transparent overlays)
- **Confidence scores** for each detection
- **Object labels** with class name

### Color Coding
- 🟢 Green: pot
- 🔵 Blue: pan
- 🔴 Red: cup
- 🟡 Yellow: induction cooker/stove
- 🟣 Magenta: spoon
- 🔶 Cyan: hand
- Additional colors for other objects

### Information Display
- **Top-left**: Object count summary
- **Per-class counts** with color-coded labels
- **Bottom overlay**:
  - Frame number / total
  - Video timestamp
  - Processing ETA (for live creation)

## Detection Performance

### Objects Successfully Detected
| Object | Detection Rate | Avg Confidence |
|--------|----------------|----------------|
| person | ~70% | 98% |
| hand | ~50% | 90-97% |
| pot | ~100% | 58-82% |
| pan | ~50% | 51-89% |
| cup | ~20% | 54% |
| stove | ~30% | 58% |

### Objects Not Detected
- sugar container
- milk container  
- tea container
- spoon

## Technical Details

### Processing Pipeline
1. **Input**: 360° equirectangular frame (2160×1080)
2. **Perspective extraction**: 90° FOV front view (960×720)
3. **SAM3 segmentation**: Text-prompted detection
4. **Visualization**: Bounding boxes + masks + labels
5. **Output**: MP4 video (MPEG-4 codec)

### Performance
- **First frame**: 12-19s (includes model loading)
- **Subsequent frames**: 1.0-1.1s per frame
- **GPU**: CUDA-accelerated SAM3 model

### Quality Settings
- Resolution: 960×720 pixels
- Codec: MPEG-4 (mp4v)
- Bitrate: ~1.4 Mbps
- Frame rate: 1 FPS (from 30 FPS source, skip=30)

## Usage

### Create Custom Video Segment

```bash
# Process specific time range
python tasks/tea_making/demo/test_perception.py \
    --create-video \
    --start 0 \
    --end 900 \
    --skip 15 \
    --output my_output.mp4
```

### Parameters
- `--start`: First frame (0-based index)
- `--end`: Last frame to process
- `--skip`: Process every Nth frame (15 = 2fps from 30fps source)
- `--preview`: Show live preview during creation
- `--output`: Output video file path

### Frame Rate Conversion
- `--skip 30` → 1 FPS output (recommended for review)
- `--skip 15` → 2 FPS output (smoother but longer processing)
- `--skip 5` → 6 FPS output (very long processing)

## Known Issues

See [PERCEPTION_TEST_RESULTS.md](PERCEPTION_TEST_RESULTS.md) for detailed analysis:

1. **Multiple overlapping detections** - Same object detected multiple times
2. **Pot/pan/kettle confusion** - Similar shapes misclassified
3. **Small objects missed** - Containers and utensils not detected
4. **Processing speed** - 1s/frame too slow for real-time

## Next Steps

1. ✅ Video output created
2. 🔄 Review full video when complete
3. ⏳ Implement NMS for duplicate removal
4. ⏳ Test with Gemini Vision for better detection
5. ⏳ Add object tracking across frames

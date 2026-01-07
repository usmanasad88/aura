# Agent 04: Perception Module Agent

## Task: Implement Perception Module with SAM3 Integration

### Objective
Create the perception module that detects, segments, and tracks objects using SAM3 and optionally Gemini for scene understanding.

### Prerequisites
- Sprint 1 complete (core types and monitor interface)
- SAM3 installed in third_party/
- Webcam or video source available for testing

### Reference Code
- `/home/mani/Repos/aura/testing.py` - Working SAM3 real-time example

### Files to Create

#### 1. `src/aura/monitors/perception_module.py`

```python
"""Perception module for object detection and segmentation."""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# Lazy imports for heavy dependencies
_sam3_model = None
_sam3_processor = None

from aura.core import (
    MonitorType, PerceptionOutput, 
    TrackedObject, BoundingBox, Pose3D
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig


logger = logging.getLogger(__name__)


@dataclass
class PerceptionConfig(MonitorConfig):
    """Configuration for perception module."""
    use_sam3: bool = True
    use_gemini_detection: bool = True
    max_objects: int = 10
    confidence_threshold: float = 0.5
    gemini_model: str = "gemini-2.0-flash"
    default_prompts: List[str] = field(default_factory=lambda: ["person", "hand"])
    detection_interval_frames: int = 30  # Re-run detection every N frames
    

def _load_sam3():
    """Lazy-load SAM3 model."""
    global _sam3_model, _sam3_processor
    if _sam3_model is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        logger.info("Loading SAM3 model...")
        _sam3_model = build_sam3_image_model()
        _sam3_processor = Sam3Processor(_sam3_model)
        logger.info("SAM3 model loaded")
    return _sam3_model, _sam3_processor


def detect_objects_with_gemini(
    frame_rgb: np.ndarray, 
    max_objects: int = 5,
    model: str = "gemini-2.0-flash"
) -> List[str]:
    """Use Gemini to detect important objects in a frame.
    
    Args:
        frame_rgb: RGB numpy array of the frame
        max_objects: Maximum number of objects to detect
        model: Gemini model to use
        
    Returns:
        List of object names to track
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("Gemini not available, using default objects")
        return ["person", "hand"]
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, using default objects")
        return ["person", "hand"]
    
    # Convert frame to base64 JPEG
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize if too large
    max_dim = 640
    if max(pil_image.size) > max_dim:
        ratio = max_dim / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    image_bytes = buffer.getvalue()
    
    # Query Gemini
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Analyze this image and list the {max_objects} most important/prominent objects.

Focus on:
- Objects that are clearly visible
- Objects that might move or be interacted with
- Rigid objects which can be tracked

Respond with ONLY a JSON array of object names, like:
["laptop", "phone", "coffee mug", "chair"]

Keep object names simple and singular."""
    
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)
    ]
    
    contents = [types.Content(role="user", parts=parts)]
    
    generate_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.3,
    )
    
    logger.info("Querying Gemini for object detection...")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_config,
        )
        
        response_text = response.text
        logger.debug(f"Gemini response: {response_text}")
        
        # Parse JSON array from response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            import json
            objects = json.loads(json_match.group())
            return objects[:max_objects]
        else:
            logger.warning("Could not parse Gemini response")
            return ["person", "hand"]
            
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        return ["person", "hand"]


class PerceptionModule(BaseMonitor):
    """Perception module for object detection and segmentation.
    
    Uses SAM3 for segmentation and optionally Gemini for 
    scene-based object detection.
    """
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        super().__init__(config or PerceptionConfig())
        self.config: PerceptionConfig = self.config
        
        self._current_prompts: List[str] = list(self.config.default_prompts)
        self._frame_count = 0
        self._object_id_counter = 0
        self._tracked_objects: Dict[str, TrackedObject] = {}
        
        # Load SAM3 if enabled
        if self.config.use_sam3:
            _load_sam3()
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERCEPTION
    
    @property
    def current_prompts(self) -> List[str]:
        """Get current detection prompts."""
        return self._current_prompts
    
    def set_prompts(self, prompts: List[str]):
        """Set object prompts to detect."""
        self._current_prompts = prompts
        logger.info(f"Updated prompts: {prompts}")
    
    def add_prompt(self, prompt: str):
        """Add a prompt to detect."""
        if prompt not in self._current_prompts:
            self._current_prompts.append(prompt)
    
    async def _process(
        self, 
        frame: Optional[np.ndarray] = None,
        auto_detect: bool = False
    ) -> PerceptionOutput:
        """Process a frame and detect objects.
        
        Args:
            frame: BGR numpy array from OpenCV (or RGB)
            auto_detect: If True, use Gemini to detect objects in first frame
            
        Returns:
            PerceptionOutput with detected objects
        """
        if frame is None:
            return PerceptionOutput(is_valid=False, error="No frame provided")
        
        self._frame_count += 1
        
        # Convert to RGB if needed (assume BGR from OpenCV)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Auto-detect objects with Gemini on first frame or periodically
        if auto_detect and self.config.use_gemini_detection:
            if (self._frame_count == 1 or 
                self._frame_count % self.config.detection_interval_frames == 0):
                detected = detect_objects_with_gemini(
                    frame_rgb, 
                    self.config.max_objects,
                    self.config.gemini_model
                )
                self._current_prompts = detected
                logger.info(f"Auto-detected objects: {detected}")
        
        # Run SAM3 segmentation
        objects = []
        
        if self.config.use_sam3:
            model, processor = _load_sam3()
            image = Image.fromarray(frame_rgb)
            height, width = frame.shape[:2]
            
            for prompt in self._current_prompts:
                try:
                    # Process with SAM3
                    inference_state = processor.set_image(image)
                    output = processor.set_text_prompt(
                        state=inference_state, 
                        prompt=prompt
                    )
                    
                    masks = output.get("masks")
                    boxes = output.get("boxes")
                    scores = output.get("scores")
                    
                    if masks is not None and len(masks) > 0:
                        # Process each detection
                        for i in range(min(len(masks), 3)):  # Max 3 per prompt
                            score = float(scores[i]) if scores is not None else 1.0
                            
                            if score < self.config.confidence_threshold:
                                continue
                            
                            # Create bounding box
                            bbox = None
                            if boxes is not None and i < len(boxes):
                                box = boxes[i]
                                if hasattr(box, 'cpu'):
                                    box = box.cpu().numpy()
                                bbox = BoundingBox(
                                    x_min=int(box[0]),
                                    y_min=int(box[1]),
                                    x_max=int(box[2]),
                                    y_max=int(box[3])
                                )
                            
                            # Get mask
                            mask = None
                            if masks is not None and i < len(masks):
                                mask_data = masks[i]
                                if hasattr(mask_data, 'cpu'):
                                    mask_data = mask_data.cpu().numpy()
                                mask = mask_data.astype(np.uint8)
                            
                            # Create tracked object
                            obj_id = f"{prompt}_{self._object_id_counter}"
                            self._object_id_counter += 1
                            
                            obj = TrackedObject(
                                id=obj_id,
                                name=prompt,
                                category=prompt,
                                bbox=bbox,
                                mask=mask,
                                confidence=score,
                                last_seen=datetime.now()
                            )
                            objects.append(obj)
                            
                except Exception as e:
                    logger.error(f"Error processing prompt '{prompt}': {e}")
                    continue
        
        return PerceptionOutput(
            objects=objects,
            scene_description=f"Detected {len(objects)} objects"
        )
    
    def visualize(
        self, 
        frame: np.ndarray, 
        output: PerceptionOutput,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Draw detection results on frame.
        
        Args:
            frame: BGR image to draw on
            output: PerceptionOutput with detections
            colors: Optional dict mapping category to BGR color
            
        Returns:
            Frame with visualizations drawn
        """
        result = frame.copy()
        
        default_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, obj in enumerate(output.objects):
            color = default_colors[i % len(default_colors)]
            if colors and obj.category in colors:
                color = colors[obj.category]
            
            # Draw mask
            if obj.mask is not None:
                mask = obj.mask
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                overlay = result.copy()
                overlay[mask > 0] = color
                result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # Draw bounding box
            if obj.bbox:
                cv2.rectangle(
                    result,
                    (obj.bbox.x_min, obj.bbox.y_min),
                    (obj.bbox.x_max, obj.bbox.y_max),
                    color, 2
                )
                
                # Label
                label = f"{obj.name}: {obj.confidence:.2f}"
                cv2.putText(
                    result, label,
                    (obj.bbox.x_min, obj.bbox.y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        return result
```

#### 2. `scripts/test_perception.py`

```python
#!/usr/bin/env python
"""Test script for perception module."""

import argparse
import asyncio
import time
import cv2
import sys

from aura.monitors.perception_module import PerceptionModule, PerceptionConfig


async def test_webcam(duration: float = 10.0, auto_detect: bool = True):
    """Test perception with webcam input."""
    print("Initializing perception module...")
    
    config = PerceptionConfig(
        use_sam3=True,
        use_gemini_detection=auto_detect,
        max_objects=5,
        confidence_threshold=0.3,
    )
    perception = PerceptionModule(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Running perception test for {duration} seconds...")
    print("Press 'q' to quit, 'a' to add prompt, 'd' to detect objects")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run perception
            is_first = frame_count == 0
            output = await perception.update(frame=frame, auto_detect=is_first)
            frame_count += 1
            
            if output and output.is_valid:
                # Visualize
                vis_frame = perception.visualize(frame, output)
                
                # Add info overlay
                fps = frame_count / (time.time() - start_time)
                info = f"FPS: {fps:.1f} | Objects: {len(output.objects)}"
                cv2.putText(vis_frame, info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show prompts
                prompts_text = f"Prompts: {', '.join(perception.current_prompts)}"
                cv2.putText(vis_frame, prompts_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Perception Test", vis_frame)
            else:
                cv2.imshow("Perception Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                prompt = input("Enter object to detect: ").strip()
                if prompt:
                    perception.add_prompt(prompt)
                    print(f"Added prompt: {prompt}")
            elif key == ord('d'):
                # Force re-detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from aura.monitors.perception_module import detect_objects_with_gemini
                detected = detect_objects_with_gemini(frame_rgb)
                perception.set_prompts(detected)
                print(f"Detected: {detected}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")


async def test_image(image_path: str, prompts: list):
    """Test perception on a single image."""
    print(f"Testing perception on: {image_path}")
    
    config = PerceptionConfig(
        use_sam3=True,
        use_gemini_detection=not bool(prompts),
        default_prompts=prompts if prompts else ["object"],
    )
    perception = PerceptionModule(config)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    output = await perception.update(frame=frame, auto_detect=True)
    
    if output and output.is_valid:
        print(f"Detected {len(output.objects)} objects:")
        for obj in output.objects:
            print(f"  - {obj.name} (confidence: {obj.confidence:.2f})")
        
        vis_frame = perception.visualize(frame, output)
        
        output_path = image_path.rsplit('.', 1)[0] + "_detected.jpg"
        cv2.imwrite(output_path, vis_frame)
        print(f"Saved visualization to: {output_path}")
    else:
        print("Perception failed or no objects detected")


def main():
    parser = argparse.ArgumentParser(description="Test perception module")
    parser.add_argument("--input", type=str, default="webcam",
                       help="Input source: 'webcam' or path to image")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Duration for webcam test (seconds)")
    parser.add_argument("--prompts", type=str, nargs="*", default=[],
                       help="Object prompts to detect")
    parser.add_argument("--no-auto-detect", action="store_true",
                       help="Disable Gemini auto-detection")
    
    args = parser.parse_args()
    
    if args.input == "webcam":
        asyncio.run(test_webcam(
            duration=args.duration,
            auto_detect=not args.no_auto_detect
        ))
    else:
        asyncio.run(test_image(args.input, args.prompts))


if __name__ == "__main__":
    main()
```

### Validation

```bash
cd /home/mani/Repos/aura

# Test with webcam (requires display)
uv run python scripts/test_perception.py --input webcam --duration 10

# Test with image
uv run python scripts/test_perception.py --input /path/to/image.jpg --prompts cup phone

# Run unit tests
uv run pytest tests/test_monitors/test_perception.py -v
```

### Expected Behavior

1. SAM3 model loads (may take 10-20s first time)
2. Gemini detects objects in first frame (if enabled)
3. SAM3 segments each detected object
4. Bounding boxes and masks are drawn on frame
5. Real-time FPS ~5-15 depending on hardware

### Handoff Notes

Create `genai_instructions/handoff/04_perception.md`:

```markdown
# Perception Module Complete

## Files Created
- src/aura/monitors/perception_module.py
- scripts/test_perception.py

## Dependencies
- SAM3 (third_party/sam3)
- google-genai for Gemini
- OpenCV, PIL

## Key Features
1. SAM3 segmentation with text prompts
2. Gemini auto-detection of objects
3. Real-time visualization
4. Configurable confidence threshold

## Known Limitations
- FPS limited by SAM3 inference (~5-15 FPS)
- Gemini detection adds latency on first frame
- No persistent object tracking (TODO)

## Next Steps
- Add object tracking across frames (IOU matching)
- Add depth estimation if depth camera available
- Integrate with digital twin for 3D pose
```

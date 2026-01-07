"""Perception module for object detection and segmentation."""

import os
import re
import time
import logging
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
from aura.monitors.base_monitor import BaseMonitor
from aura.utils.config import PerceptionConfig


logger = logging.getLogger(__name__)

def _load_sam3():
    """Lazy-load SAM3 model."""
    global _sam3_model, _sam3_processor
    
    if _sam3_model is None or _sam3_processor is None:
        import torch
        logger.info("Loading SAM3 model...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        _sam3_model = build_sam3_image_model()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _sam3_model = _sam3_model.to(device)
            logger.info(f"SAM3 model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU (will be slower)")
        
        _sam3_processor = Sam3Processor(_sam3_model)
        logger.info("SAM3 model loaded successfully!")
    
    return _sam3_model, _sam3_processor


def detect_objects_with_gemini(
    frame_rgb: np.ndarray, 
    max_objects: int = 5,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-exp"
) -> List[str]:
    """Use Gemini to detect important objects in a frame.
    
    Args:
        frame_rgb: RGB numpy array of the frame
        max_objects: Maximum number of objects to detect
        api_key: Gemini API key (uses env var if None)
        model: Gemini model to use
        
    Returns:
        List of object names to track
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai not installed, using default prompts")
        return ["person", "hand"]
    
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, using default prompts")
        return ["person", "hand"]
    
    # Convert frame to JPEG bytes
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
    
    logger.info("🔍 Querying Gemini to detect objects...")
    
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
            logger.info(f"✅ Detected objects: {objects}")
            return objects[:max_objects]
        else:
            logger.warning("Failed to parse Gemini response, using defaults")
            return ["person", "hand"]
            
    except Exception as e:
        logger.error(f"Gemini query failed: {e}, using defaults")
        return ["person", "hand"]


class PerceptionModule(BaseMonitor):
    """Perception module for object detection and segmentation.
    
    Uses SAM3 for segmentation and optionally Gemini for object detection.
    """
    
    @property
    def monitor_type(self) -> MonitorType:
        """Return the monitor type."""
        return MonitorType.PERCEPTION
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        """Initialize perception module.
        
        Args:
            config: Configuration for perception module
        """
        if config is None:
            config = PerceptionConfig(monitor_type=MonitorType.PERCEPTION)
        elif not isinstance(config, PerceptionConfig):
            # Convert generic config to PerceptionConfig
            config_dict = config.__dict__.copy()
            config = PerceptionConfig(**config_dict)
        
        super().__init__(config)
        
        self.config: PerceptionConfig = config
        self.prompts: List[str] = config.default_prompts.copy()
        self.frame_count = 0
        self.last_detection_frame = -1
        
        # Lazy-load SAM3 on first use
        self._model_loaded = False
        
        # Colors for visualization (BGR format)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
    
    def _ensure_model_loaded(self):
        """Ensure SAM3 model is loaded."""
        if not self._model_loaded and self.config.use_sam3:
            _load_sam3()
            self._model_loaded = True
    
    async def _process(self, **kwargs) -> Optional[PerceptionOutput]:
        """Internal processing method required by BaseMonitor.
        
        This is called by the BaseMonitor.process() method.
        For perception, we need a frame to process, so users should call
        process_frame() directly.
        """
        # This shouldn't be called directly for perception module
        # Users should call process_frame() instead
        raise NotImplementedError(
            "PerceptionModule requires a frame. Use process_frame(frame) instead."
        )
    
    async def process_frame(self, frame: np.ndarray, **kwargs) -> Optional[PerceptionOutput]:
        """Process a frame and detect/segment objects.
        
        Args:
            frame: BGR image from camera (OpenCV format)
            **kwargs: Additional arguments
            
        Returns:
            PerceptionOutput with detected objects, or None if processing fails
        """
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            # Auto-detect objects with Gemini on first frame or periodically
            if (self.config.use_gemini_detection and 
                (self.frame_count == 0 or 
                 (self.frame_count - self.last_detection_frame >= self.config.detection_interval_frames))):
                logger.info("Running Gemini detection...")
                self.prompts = detect_objects_with_gemini(
                    frame_rgb,
                    max_objects=self.config.max_objects,
                    model=self.config.gemini_model
                )
                self.last_detection_frame = self.frame_count
            
            # Run SAM3 segmentation for each prompt
            tracked_objects = []
            
            if self.config.use_sam3:
                import torch
                _, processor = _load_sam3()
                
                image_pil = Image.fromarray(frame_rgb)
                
                with torch.no_grad():
                    inference_state = processor.set_image(image_pil)
                    
                    for prompt_idx, prompt in enumerate(self.prompts):
                        # Reset prompts before each new text query
                        processor.reset_all_prompts(inference_state)
                        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                        
                        if output["boxes"] is not None and len(output["boxes"]) > 0:
                            boxes = output["boxes"].cpu().numpy()
                            scores = output["scores"].cpu().numpy()
                            masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
                            
                            # Filter by confidence threshold
                            valid_indices = scores >= self.config.confidence_threshold
                            boxes = boxes[valid_indices]
                            scores = scores[valid_indices]
                            if masks is not None:
                                masks = masks[valid_indices]
                            
                            # Create TrackedObject for each detection
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                x1, y1, x2, y2 = box.astype(int)
                                
                                # Clamp to image bounds
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(width, x2), min(height, y2)
                                
                                # Extract mask if available
                                mask = None
                                if masks is not None and i < len(masks):
                                    mask_data = masks[i]
                                    if mask_data.ndim == 3:
                                        mask_data = mask_data[0]
                                    mask = (mask_data > 0.5).astype(np.uint8)
                                
                                # Create TrackedObject
                                obj = TrackedObject(
                                    id=f"{prompt}_{i}",
                                    name=prompt,
                                    category=prompt,
                                    bbox=BoundingBox(
                                        x_min=float(x1),
                                        y_min=float(y1),
                                        x_max=float(x2),
                                        y_max=float(y2)
                                    ),
                                    confidence=float(score),
                                    mask=mask,
                                    # Pose estimation would go here (TODO)
                                    pose=None,
                                    last_seen=datetime.now()
                                )
                                
                                tracked_objects.append(obj)
            
            self.frame_count += 1
            
            # Create PerceptionOutput
            output = PerceptionOutput(
                timestamp=datetime.now(),
                objects=tracked_objects,
                scene_description=f"Frame {self.frame_count}: Tracking {', '.join(self.prompts)}"
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Perception processing failed: {e}", exc_info=True)
            return None
    
    def visualize(self, frame: np.ndarray, output: PerceptionOutput) -> np.ndarray:
        """Draw detection results on frame.
        
        Args:
            frame: BGR image to draw on
            output: PerceptionOutput to visualize
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw each detected object
        for obj in output.objects:
            # Get color for this object class
            prompt_idx = self.prompts.index(obj.name) if obj.name in self.prompts else 0
            color = self.colors[prompt_idx % len(self.colors)]
            
            # Draw bounding box
            x1, y1 = int(obj.bbox.x_min), int(obj.bbox.y_min)
            x2, y2 = int(obj.bbox.x_max), int(obj.bbox.y_max)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{obj.name}: {obj.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw mask if available
            if obj.mask is not None:
                colored_mask = np.zeros_like(vis_frame)
                colored_mask[:, :, 0] = obj.mask * (color[0] // 2)
                colored_mask[:, :, 1] = obj.mask * (color[1] // 2)
                colored_mask[:, :, 2] = obj.mask * (color[2] // 2)
                vis_frame = cv2.addWeighted(vis_frame, 1, colored_mask, 0.4, 0)
        
        # Draw summary info
        info_text = f"Objects: {len(output.objects)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw per-class counts
        y_offset = 60
        for prompt_idx, prompt in enumerate(self.prompts):
            color = self.colors[prompt_idx % len(self.colors)]
            count = sum(1 for obj in output.objects if obj.name == prompt)
            cv2.putText(vis_frame, f"{prompt}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        return vis_frame

"""YOLO-based person detector for cropping frames to person bounding boxes.

Uses YOLOv8 to detect people in frames and returns cropped regions,
suitable for feeding into downstream monitors (gesture, pose, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default model path (GVHMR checkpoint)
_DEFAULT_MODEL_PATH = Path.home() / "GVHMR/inputs/checkpoints/yolo/yolov8x.pt"


@dataclass
class PersonCrop:
    """A detected person's bounding box and cropped image."""
    bbox_xyxy: np.ndarray      # [x1, y1, x2, y2] in original frame coords
    crop: np.ndarray           # Cropped BGR image
    confidence: float
    track_id: Optional[int] = None

    @property
    def width(self) -> int:
        return int(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> int:
        return int(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> int:
        return self.width * self.height


class PersonDetector:
    """Detects people using YOLOv8 and returns cropped person regions.

    Args:
        model_path: Path to YOLOv8 weights (.pt). Falls back to default GVHMR path.
        confidence: Minimum detection confidence.
        device: "cuda", "cpu", or None for auto.
        pad_ratio: Fractional padding around bbox (0.1 = 10% on each side).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.5,
        device: str | None = None,
        pad_ratio: float = 0.1,
    ):
        from ultralytics import YOLO

        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {path}. "
                f"Provide a valid path or place yolov8x.pt at {_DEFAULT_MODEL_PATH}"
            )

        self.model = YOLO(str(path))
        self.confidence = confidence
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.pad_ratio = pad_ratio
        logger.info(f"PersonDetector loaded {path.name} on {self.device}")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect(self, frame: np.ndarray) -> list[PersonCrop]:
        """Detect people in a BGR frame and return cropped regions.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of PersonCrop sorted by area (largest first).
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.confidence,
            classes=0,  # person class only
            verbose=False,
        )

        h, w = frame.shape[:2]
        crops: list[PersonCrop] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0].cpu())
                track_id = int(box.id[0].cpu()) if box.id is not None else None

                # Apply padding
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                pad_x = bw * self.pad_ratio
                pad_y = bh * self.pad_ratio
                x1 = max(0, int(xyxy[0] - pad_x))
                y1 = max(0, int(xyxy[1] - pad_y))
                x2 = min(w, int(xyxy[2] + pad_x))
                y2 = min(h, int(xyxy[3] + pad_y))

                crop = frame[y1:y2, x1:x2].copy()
                crops.append(PersonCrop(
                    bbox_xyxy=np.array([x1, y1, x2, y2]),
                    crop=crop,
                    confidence=conf,
                    track_id=track_id,
                ))

        # Sort by area descending (largest person first)
        crops.sort(key=lambda c: c.area, reverse=True)
        return crops

    def detect_largest(self, frame: np.ndarray) -> Optional[PersonCrop]:
        """Detect the largest person in frame."""
        crops = self.detect(frame)
        return crops[0] if crops else None

"""Webcam frame source using OpenCV VideoCapture."""

import logging
import time
from typing import Optional

import cv2

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)


class WebcamSource(FrameSource):
    """Live frame source from a webcam / USB camera.

    Args:
        device: Device index (default ``0``) or a V4L2 device path
            such as ``"/dev/video0"``.
        fps: Target capture FPS.  The camera's actual FPS may differ.
        width: Requested frame width (``None`` = camera default).
        height: Requested frame height (``None`` = camera default).
    """

    def __init__(
        self,
        device: int | str = 0,
        fps: float = 30.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self._device = device
        self._target_fps = fps
        self._req_width = width
        self._req_height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._cap is not None:
            return
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Could not open webcam device: {self._device}"
            )

        # Apply requested settings
        if self._req_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        if self._req_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        self._frame_count = 0
        self._start_time = time.monotonic()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "WebcamSource opened: device=%s  %dx%d @ %.1f fps",
            self._device, actual_w, actual_h, actual_fps,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("WebcamSource closed (device=%s)", self._device)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ret, image = self._cap.read()
        if not ret:
            return None

        h, w = image.shape[:2]
        ts = time.monotonic() - self._start_time
        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"webcam:{self._device}",
            width=w,
            height=h,
        )
        self._frame_count += 1
        return frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or self._target_fps
        return self._target_fps

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self._req_width or 0, self._req_height or 0)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

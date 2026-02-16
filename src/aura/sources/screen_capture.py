"""Live screen capture frame source using mss."""

import logging
import time
from typing import Optional

import numpy as np

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)


class ScreenCaptureSource(FrameSource):
    """Live frame source from screen / monitor capture.

    Requires the ``mss`` package (``pip install mss``).

    Args:
        monitor: Monitor index (``0`` = all monitors combined,
            ``1`` = primary, ``2`` = secondary, etc.).
        region: Optional ``(left, top, width, height)`` sub-region to
            capture.  When ``None`` the full monitor area is used.
        fps: Target capture rate.  The actual rate depends on system
            performance.
    """

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[tuple[int, int, int, int]] = None,
        fps: float = 15.0,
    ):
        self._monitor_idx = monitor
        self._region = region
        self._target_fps = fps

        self._sct = None  # mss instance
        self._bbox: Optional[dict] = None
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        try:
            import mss
        except ImportError:
            raise ImportError(
                "ScreenCaptureSource requires the 'mss' package. "
                "Install it with: pip install mss"
            )

        self._sct = mss.mss()

        if self._region is not None:
            left, top, w, h = self._region
            self._bbox = {
                "left": left, "top": top, "width": w, "height": h,
            }
            self._width, self._height = w, h
        else:
            mon = self._sct.monitors[self._monitor_idx]
            self._bbox = mon
            self._width = mon["width"]
            self._height = mon["height"]

        self._frame_count = 0
        self._start_time = time.monotonic()
        self._opened = True

        logger.info(
            "ScreenCaptureSource opened: monitor=%d  %dx%d @ %.1f fps target",
            self._monitor_idx, self._width, self._height, self._target_fps,
        )

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        self._opened = False
        logger.info("ScreenCaptureSource closed")

    def read(self) -> Optional[Frame]:
        if not self._opened or self._sct is None:
            return None

        raw = self._sct.grab(self._bbox)
        # mss returns BGRA; convert to BGR for OpenCV convention
        image = np.array(raw, dtype=np.uint8)[:, :, :3].copy()

        ts = time.monotonic() - self._start_time
        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"screen:{self._monitor_idx}",
            width=self._width,
            height=self._height,
        )
        self._frame_count += 1
        return frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._target_fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def is_open(self) -> bool:
        return self._opened

"""Live screen capture frame source using mss."""

import logging
import threading
import time
from typing import Optional

import numpy as np

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)


class ScreenCaptureSource(FrameSource):
    """Live frame source from screen / monitor capture.

    Requires the ``mss`` package (``pip install mss``).

    Thread-safe: a fresh ``mss`` instance is created per-thread because
    the underlying X11 display handle is stored in thread-local storage
    and cannot be shared across threads (as LangGraph does).

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

        self._tls = threading.local()  # per-thread mss instances
        self._bbox: Optional[dict] = None
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Internal: get or create a per-thread mss instance
    # ------------------------------------------------------------------

    def _get_sct(self):
        """Return an mss instance for the current thread."""
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            import mss
            sct = mss.mss()
            self._tls.sct = sct
        return sct

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        try:
            import mss  # noqa: F401 — verify importable
        except ImportError:
            raise ImportError(
                "ScreenCaptureSource requires the 'mss' package. "
                "Install it with: pip install mss"
            )

        sct = self._get_sct()

        if self._region is not None:
            left, top, w, h = self._region
            self._bbox = {
                "left": left, "top": top, "width": w, "height": h,
            }
            self._width, self._height = w, h
        else:
            mon = sct.monitors[self._monitor_idx]
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
        # Best-effort close of the calling thread's instance
        sct = getattr(self._tls, "sct", None)
        if sct is not None:
            sct.close()
            self._tls.sct = None
        self._opened = False
        logger.info("ScreenCaptureSource closed")

    def read(self) -> Optional[Frame]:
        if not self._opened:
            return None

        sct = self._get_sct()

        # Resolve bbox on first read in this thread if it was opened on
        # another thread (region is already absolute so it's fine).
        bbox = self._bbox
        if bbox is None:
            mon = sct.monitors[self._monitor_idx]
            self._bbox = mon
            bbox = mon
            self._width = mon["width"]
            self._height = mon["height"]

        raw = sct.grab(bbox)
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

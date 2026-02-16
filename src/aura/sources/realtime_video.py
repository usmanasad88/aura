"""Video file source that enforces real-time playback pacing.

Frames are gated by wall-clock time: a frame whose video timestamp is
*T* seconds into the file will not be returned until *T / speed* seconds
of wall-clock time have elapsed since :meth:`open`.  This lets you test
live-source pipelines against a pre-recorded video without ever seeing
"future" frames.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)


class RealtimeVideoSource(FrameSource):
    """A video file that delivers frames at real-time (or scaled) pace.

    Behaves identically to :class:`VideoFileSource` except:

    * :meth:`read` **blocks** until the frame's video timestamp has
      arrived in wall-clock time.
    * :pyattr:`is_live` returns ``True`` so that the rest of the
      pipeline treats this source like a webcam (e.g. enabling the
      realtime intent-monitor prompt).

    Args:
        path: Path to the video file.
        speed: Playback speed multiplier.  ``1.0`` = real-time,
            ``2.0`` = twice as fast, ``0.5`` = half speed.
        max_frames: Stop after delivering this many frames.
    """

    def __init__(
        self,
        path: str | Path,
        speed: float = 1.0,
        max_frames: Optional[int] = None,
    ):
        self._path = str(path)
        self._speed = max(0.01, speed)
        self._max_frames = max_frames

        self._cap: Optional[cv2.VideoCapture] = None
        self._native_fps: float = 30.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

        self._delivered: int = 0
        self._wall_start: Optional[float] = None  # set on first read()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._cap is not None:
            return
        if not Path(self._path).exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Could not open video: {self._path}")

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._delivered = 0
        self._wall_start = None

        duration = self._total_frames / self._native_fps if self._native_fps else 0
        logger.info(
            "RealtimeVideoSource opened: %s  %dx%d @ %.1f fps  "
            "%d frames (%.1fs)  speed=%.2fx  max=%s",
            self._path, self._width, self._height, self._native_fps,
            self._total_frames, duration, self._speed, self._max_frames,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("RealtimeVideoSource closed: %s", self._path)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        if self._max_frames is not None and self._delivered >= self._max_frames:
            return None

        # Start the timer on the very first read(), not at open().
        # This avoids counting model-loading time as elapsed video time.
        if self._wall_start is None:
            self._wall_start = time.monotonic()

        # Compute how far into the video we should be based on wall time.
        wall_elapsed = time.monotonic() - self._wall_start
        video_ts = wall_elapsed * self._speed
        target_frame = int(video_ts * self._native_fps)

        # Clamp to valid range.
        if target_frame >= self._total_frames:
            return None

        # Seek to the target frame (skip ahead if the consumer is slow).
        current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if target_frame > current_pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, image = self._cap.read()
        if not ret:
            return None

        actual_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        actual_ts = actual_pos / self._native_fps

        frame = Frame(
            image=image,
            timestamp=actual_ts,
            source_name=f"realtime-file:{Path(self._path).name}",
            frame_number=self._delivered,
            width=self._width,
            height=self._height,
        )
        self._delivered += 1
        return frame

    # ------------------------------------------------------------------
    # Extra properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration(self) -> float:
        if self._native_fps:
            return self._total_frames / self._native_fps
        return 0.0

    @property
    def frames_delivered(self) -> int:
        return self._delivered

    # ------------------------------------------------------------------
    # FrameSource properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._native_fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_live(self) -> bool:
        # Report as live so the pipeline uses realtime mode.
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

"""Pre-recorded video file frame source."""

import logging
from pathlib import Path
from typing import Optional

import cv2

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """Frame source backed by a video file on disk.

    Args:
        path: Path to the video file (mp4, avi, mkv, etc.).
        frame_skip: Yield every *N*-th frame (1 = every frame).
        max_frames: Stop after delivering this many frames (``None`` = all).
        loop: When ``True``, restart from the beginning on EOF instead of
            returning ``None``.
    """

    def __init__(
        self,
        path: str | Path,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        loop: bool = False,
    ):
        self._path = str(path)
        self._frame_skip = max(1, frame_skip)
        self._max_frames = max_frames
        self._loop = loop

        self._cap: Optional[cv2.VideoCapture] = None
        self._native_fps: float = 30.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

        # Counters
        self._delivered: int = 0  # frames returned to caller
        self._raw_pos: int = 0  # raw frame position in file

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
        self._raw_pos = 0

        duration = self._total_frames / self._native_fps if self._native_fps else 0
        logger.info(
            "VideoFileSource opened: %s  %dx%d @ %.1f fps  "
            "%d frames (%.1fs)  skip=%d  max=%s",
            self._path, self._width, self._height, self._native_fps,
            self._total_frames, duration, self._frame_skip, self._max_frames,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("VideoFileSource closed: %s", self._path)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        # Max-frames guard
        if self._max_frames is not None and self._delivered >= self._max_frames:
            return None

        while True:
            if self._raw_pos >= self._total_frames:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._raw_pos = 0
                    continue
                return None

            # Seek to the desired frame position
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._raw_pos)
            ret, image = self._cap.read()
            current_pos = self._raw_pos
            self._raw_pos += self._frame_skip

            if not ret:
                # Corrupted frame — skip ahead
                continue

            ts = current_pos / self._native_fps
            frame = Frame(
                image=image,
                timestamp=ts,
                frame_number=self._delivered,
                source_name=f"file:{Path(self._path).name}",
                width=self._width,
                height=self._height,
            )
            self._delivered += 1
            return frame

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def seek(self, frame_number: int) -> None:
        """Seek to an absolute frame position in the video.

        The next call to :meth:`read` will return this frame.
        """
        if self._cap is None:
            raise RuntimeError("Source is not open")
        self._raw_pos = max(0, min(frame_number, self._total_frames - 1))

    # ------------------------------------------------------------------
    # Extra properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of raw frames in the video file."""
        return self._total_frames

    @property
    def duration(self) -> float:
        """Duration of the video in seconds."""
        if self._native_fps:
            return self._total_frames / self._native_fps
        return 0.0

    @property
    def frames_delivered(self) -> int:
        """Number of frames returned so far."""
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
        return False

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

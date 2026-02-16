"""Abstract base class for all AURA frame sources."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Iterator, Optional

from aura.sources.frame import Frame


class FrameSource(ABC):
    """Uniform interface for providing video frames to the AURA pipeline.

    Concrete implementations exist for webcams, pre-recorded video files,
    and live screen capture.  All sources produce :class:`Frame` objects
    with BGR uint8 images (the OpenCV convention used throughout AURA).

    Usage::

        with VideoFileSource("video.mp4") as src:
            for frame in src:
                output = await monitor.update(frame=frame.image)
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open the underlying capture device / file."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture device / file."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read the next frame.

        Returns:
            A :class:`Frame`, or ``None`` when the source is exhausted
            (EOF for files) or on a transient read error.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second (native for files, target for live sources)."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` of the frames produced by this source."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """``True`` for real-time sources (webcam, screen capture)."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """``True`` when the source has been opened and not yet closed."""

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    # ------------------------------------------------------------------
    # BaseMonitor integration
    # ------------------------------------------------------------------

    def as_input_provider(self) -> Callable[[], Dict[str, Any]]:
        """Return a callable compatible with
        :meth:`BaseMonitor.start_continuous`.

        The callable returns ``{"frame": <BGR ndarray>}`` on each
        invocation, or ``{"frame": None}`` when the source is exhausted.
        """

        def _provider() -> Dict[str, Any]:
            frame = self.read()
            if frame is not None:
                return {"frame": frame.image}
            return {"frame": None}

        return _provider

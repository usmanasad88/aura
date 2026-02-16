"""Frame dataclass for the AURA FrameSource abstraction."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single video frame with metadata.

    Attributes:
        image: BGR uint8 numpy array of shape (H, W, 3).
        timestamp: Seconds since the source was opened (monotonic for live
            sources, video-time for file sources).
        frame_number: Sequential counter starting from 0.
        source_name: Human-readable identifier, e.g. ``"webcam:0"``,
            ``"file:video.mp4"``, ``"screen:0"``.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    image: np.ndarray
    timestamp: float
    frame_number: int
    source_name: str
    width: int
    height: int

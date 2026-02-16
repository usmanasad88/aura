"""Unified frame source abstraction for the AURA pipeline.

Provides a common interface for feeding frames from webcams,
pre-recorded video files, or live screen capture into AURA monitors.

Quick start::

    from aura.sources import VideoFileSource

    with VideoFileSource("demo.mp4", frame_skip=30) as src:
        for frame in src:
            result = await monitor.update(frame=frame.image)
"""

from aura.sources.frame import Frame
from aura.sources.base import FrameSource
from aura.sources.webcam import WebcamSource
from aura.sources.video_file import VideoFileSource
from aura.sources.screen_capture import ScreenCaptureSource
from aura.sources.realtime_video import RealtimeVideoSource

__all__ = [
    "Frame",
    "FrameSource",
    "WebcamSource",
    "VideoFileSource",
    "ScreenCaptureSource",
    "RealtimeVideoSource",
]

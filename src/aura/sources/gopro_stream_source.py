"""GoPro Max 2 realtime video source via UDP stream.

Launches a long-lived ``ffmpeg`` subprocess that reads the GoPro's HEVC
MPEG-TS preview stream over UDP and pipes decoded BGR frames to stdout.
Each :meth:`read` call returns the next frame with minimal latency.

The preview stream must first be activated via the GoPro HTTP API
(``/gopro/camera/stream/start``).  By default :meth:`open` does this
automatically.

Requirements:
    - GoPro connected and reachable at ``camera_ip`` (default 172.29.170.51)
    - ``ffmpeg`` available on PATH or at ``/usr/bin/ffmpeg``
    - ``requests`` package installed
    - USB interface must have an IP in the camera's subnet

Usage::

    with GoProStreamSource() as src:
        for frame in src:
            result = await monitor.update(frame=frame.image)
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)

_FFMPEG = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"

_EP_STREAM_START = "/gopro/camera/stream/start"
_EP_STREAM_STOP = "/gopro/camera/stream/stop"
_EP_WIRED_USB = "/gopro/camera/control/wired_usb?p=1"
_EP_STATE = "/gopro/camera/state"


class GoProStreamSource(FrameSource):
    """Live realtime frame source from a GoPro Max 2 UDP preview stream.

    Runs ``ffmpeg`` as a subprocess to decode the HEVC stream and pipe
    raw BGR24 frames.  Latency is typically < 200 ms.

    Args:
        camera_ip: GoPro USB IP address (default ``"172.29.170.51"``).
        udp_port: UDP port the camera streams to (default ``8554``).
        target_fps: Target capture rate.  Frames are dropped if the
            consumer is slower than the stream.  Set to ``0`` to read
            every frame without throttling.
        auto_start_stream: If ``True``, automatically send the HTTP
            command to start the preview stream on :meth:`open`.
    """

    def __init__(
        self,
        camera_ip: str = "172.29.170.51",
        udp_port: int = 8554,
        target_fps: float = 0,
        auto_start_stream: bool = True,
    ):
        self._camera_ip = camera_ip
        self._udp_port = udp_port
        self._target_fps = target_fps
        self._auto_start = auto_start_stream

        self._proc: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_read_time: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_size: int = 0  # bytes per frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._proc is not None:
            return

        # Verify camera is reachable
        if not self._ping_camera():
            raise RuntimeError(
                f"GoPro not responding at http://{self._camera_ip}. "
                "Ensure the camera is connected and powered on."
            )

        # Enable wired USB control
        self._http_get(f"http://{self._camera_ip}:8080{_EP_WIRED_USB}", silent=True)
        time.sleep(0.3)

        # Start the preview stream
        if self._auto_start:
            self._http_get(f"http://{self._camera_ip}:8080{_EP_STREAM_START}")
            time.sleep(0.5)

        # Probe the stream to get resolution
        w, h = self._probe_stream()
        if w == 0 or h == 0:
            raise RuntimeError("Failed to probe stream resolution")
        self._width = w
        self._height = h
        self._frame_size = w * h * 3  # BGR24

        # Launch ffmpeg to decode and pipe raw frames
        udp_url = f"udp://{self._camera_ip}:{self._udp_port}"
        cmd = [
            _FFMPEG,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", udp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "error",
            "pipe:1",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 2,
        )

        self._frame_count = 0
        self._start_time = None
        self._last_read_time = 0.0

        logger.info(
            "GoProStreamSource opened: camera=%s stream=%s %dx%d",
            self._camera_ip, udp_url, self._width, self._height,
        )

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
            logger.info("GoProStreamSource closed (camera=%s)", self._camera_ip)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> Optional[Frame]:
        if self._proc is None or self._proc.poll() is not None:
            return None

        if self._start_time is None:
            self._start_time = time.monotonic()

        # Throttle to target FPS if set
        if self._target_fps > 0:
            interval = 1.0 / self._target_fps
            elapsed = time.monotonic() - self._last_read_time
            if elapsed < interval:
                # Drain frames to stay current (drop old frames)
                bytes_to_skip = 0
                while elapsed < interval:
                    time.sleep(0.001)
                    elapsed = time.monotonic() - self._last_read_time
                # After waiting, drain any buffered frames to get the latest
                self._drain_buffer()

        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) != self._frame_size:
            logger.warning(
                "GoProStreamSource: short read (%d/%d bytes)",
                len(raw), self._frame_size,
            )
            return None

        image = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

        self._last_read_time = time.monotonic()
        ts = self._last_read_time - self._start_time

        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"gopro-stream:{self._camera_ip}",
            width=self._width,
            height=self._height,
        )
        self._frame_count += 1
        return frame

    # ------------------------------------------------------------------
    # FrameSource properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._target_fps if self._target_fps > 0 else 30.0

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ping_camera(self) -> bool:
        try:
            resp = requests.get(
                f"http://{self._camera_ip}:8080{_EP_STATE}", timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _http_get(self, url: str, timeout: int = 5, silent: bool = False) -> bool:
        try:
            resp = requests.get(url, timeout=timeout)
            ok = resp.status_code == 200
            if not ok and not silent:
                logger.warning("GoProStreamSource: %s returned HTTP %d", url, resp.status_code)
            return ok
        except Exception as exc:
            if not silent:
                logger.warning("GoProStreamSource: %s failed: %s", url, exc)
            return False

    def _probe_stream(self) -> tuple[int, int]:
        """Probe the UDP stream with ffprobe to get width and height."""
        udp_url = f"udp://{self._camera_ip}:{self._udp_port}"
        cmd = [
            _FFMPEG.replace("ffmpeg", "ffprobe"),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            udp_url,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                return int(parts[0]), int(parts[1])
        except Exception as exc:
            logger.warning("GoProStreamSource: probe failed: %s", exc)

        # Fallback to known GoPro Max 2 stream resolution
        logger.info("GoProStreamSource: using default resolution 1920x1440")
        return 1920, 1440

    def _drain_buffer(self) -> None:
        """Discard buffered frames to get the most recent one."""
        if self._proc is None or self._proc.stdout is None:
            return
        # Read and discard up to a few frames worth of data
        import select
        while True:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0)
            if not ready:
                break
            chunk = self._proc.stdout.read(self._frame_size)
            if len(chunk) < self._frame_size:
                break

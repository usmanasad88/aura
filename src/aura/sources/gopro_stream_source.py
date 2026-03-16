"""GoPro Max 2 realtime video source via UDP preview stream.

Uses ``cv2.VideoCapture`` to read the GoPro's HEVC MPEG-TS preview
stream over UDP.  The stream delivers dual-fisheye frames at 1408x704
@ 30 fps with minimal latency.

The preview stream must first be activated via the GoPro HTTP API
(``/gopro/camera/stream/start``).  By default :meth:`open` does this
automatically.

Requirements:
    - GoPro connected and reachable at ``camera_ip`` (default 172.29.170.51)
    - USB interface must have an IP in the camera's subnet
    - ``requests`` package installed
Migh have to sudo ip addr add 172.29.170.50/24 dev enx04574796c048

Usage::

    with GoProStreamSource() as src:
        for frame in src:
            result = await monitor.update(frame=frame.image)
"""

import logging
import time
from typing import Optional

import cv2
import requests

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)

_EP_STREAM_START = "/gopro/camera/stream/start"
_EP_STREAM_STOP = "/gopro/camera/stream/stop"
_EP_WIRED_USB = "/gopro/camera/control/wired_usb?p=1"
_EP_STATE = "/gopro/camera/state"


class GoProStreamSource(FrameSource):
    """Live realtime frame source from a GoPro Max 2 UDP preview stream.

    Opens the stream with ``cv2.VideoCapture`` which decodes the HEVC
    dual-fisheye feed at native 1408x704 @ ~30 fps.

    Args:
        camera_ip: GoPro USB IP address (default ``"172.29.170.51"``).
        udp_port: UDP port the camera streams to (default ``8554``).
        target_fps: Target capture rate.  If > 0, :meth:`read` sleeps
            between calls to honour this cadence.  Set to ``0`` (default)
            to read every frame at full stream rate.
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

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_read_time: float = 0.0
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._cap is not None:
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

        # Open the UDP stream with OpenCV
        udp_url = f"udp://{self._camera_ip}:{self._udp_port}"
        self._cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Failed to open GoPro stream at {udp_url}"
            )

        # Read initial frames to let the decoder stabilise
        for _ in range(5):
            self._cap.read()

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = 0
        self._start_time = None
        self._last_read_time = 0.0

        logger.info(
            "GoProStreamSource opened: camera=%s stream=%s %dx%d",
            self._camera_ip, udp_url, self._width, self._height,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("GoProStreamSource closed (camera=%s)", self._camera_ip)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._cap.isOpened():
            return None

        if self._start_time is None:
            self._start_time = time.monotonic()

        # Throttle to target FPS if set
        if self._target_fps > 0:
            interval = 1.0 / self._target_fps
            elapsed = time.monotonic() - self._last_read_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            # Grab (discard) buffered frames so we get the latest
            while self._cap.grab():
                # grab() is fast — it discards decoded data
                # Check if we've drained enough
                if not self._cap.grab():
                    break

        ret, image = self._cap.read()
        if not ret or image is None:
            return None

        self._last_read_time = time.monotonic()
        ts = self._last_read_time - self._start_time
        h, w = image.shape[:2]

        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"gopro-stream:{self._camera_ip}",
            width=w,
            height=h,
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
        return self._cap is not None and self._cap.isOpened()

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        with GoProStreamSource() as src:
            logger.info("Starting GoPro Stream. Press 'q' to quit.")
            while True:
                frame = src.read()
                if frame is not None:
                    # Resize for preview if needed.
                    disp = cv2.resize(frame.image, (1024, 512))
                    cv2.imshow("GoPro Stream", disp)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error streaming from GoPro: {e}")
    finally:
        cv2.destroyAllWindows()


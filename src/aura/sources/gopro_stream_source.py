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
import numpy as np
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
        lens: str = "front",
        fisheye_fov_deg: float = 190.0,
        output_fov_deg: float = 120.0,
        output_size: tuple[int, int] = (960, 720),
    ):
        """
        Args:
            lens: ``"front"`` (left half), ``"back"`` (right half),
                or ``"both"`` (full dual-fisheye frame, no remap).
            fisheye_fov_deg: Approximate FOV of each GoPro Max fisheye lens.
            output_fov_deg: Desired rectilinear output FOV (ultrawide).
            output_size: ``(width, height)`` of the remapped output.
        """
        self._camera_ip = camera_ip
        self._udp_port = udp_port
        self._target_fps = target_fps
        self._auto_start = auto_start_stream
        self._lens = lens
        self._fisheye_fov_deg = fisheye_fov_deg
        self._output_fov_deg = output_fov_deg
        self._output_size = output_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_read_time: float = 0.0
        self._width: int = 0
        self._height: int = 0

        # Fisheye remap tables (built on first frame)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None

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

        # Open the UDP stream with OpenCV — listen on 0.0.0.0, not the
        # camera IP; the GoPro *sends* packets to us, we don't pull from it.
        udp_url = f"udp://0.0.0.0:{self._udp_port}"
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

        # Drain buffered frames so we always get the latest.
        # After long pauses (e.g. LLM calls) the buffer may be full.
        drained = 0
        while drained < 120:
            grabbed = self._cap.grab()
            if not grabbed:
                break
            drained += 1

        ret, image = self._cap.read()
        if not ret or image is None:
            # UDP stream may have timed out — attempt reconnect
            logger.warning("GoPro read failed (drained %d). Reconnecting...", drained)
            self._cap.release()
            self._cap = None
            try:
                self._http_get(
                    f"http://{self._camera_ip}:8080{_EP_STREAM_START}",
                    silent=True,
                )
                time.sleep(0.5)
                udp_url = f"udp://0.0.0.0:{self._udp_port}"
                self._cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self._cap.isOpened():
                    # Warm up decoder
                    for _ in range(5):
                        self._cap.read()
                    ret, image = self._cap.read()
                    if not ret or image is None:
                        logger.error("GoPro reconnect failed — no frames after reopen")
                        return None
                    logger.info("GoPro stream reconnected")
                else:
                    logger.error("GoPro reconnect failed — could not reopen stream")
                    return None
            except Exception as exc:
                logger.error("GoPro reconnect error: %s", exc)
                return None

        # Crop to single lens if requested
        if self._lens != "both":
            h, w = image.shape[:2]
            half_w = w // 2
            if self._lens == "front":
                image = image[:, :half_w]
            else:  # back
                image = image[:, half_w:]

            # Apply fisheye → rectilinear remap
            image = self._remap_fisheye(image)

        self._last_read_time = time.monotonic()
        ts = self._last_read_time - self._start_time
        h, w = image.shape[:2]

        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"gopro-stream:{self._camera_ip}:{self._lens}",
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

    def _build_remap_tables(self, src_h: int, src_w: int) -> None:
        """Build fisheye → rectilinear remap tables.

        Uses an equidistant fisheye projection model:
            r_fish = f_fish * theta
        and maps to a rectilinear (pinhole) output:
            r_rect = f_rect * tan(theta)
        """
        out_w, out_h = self._output_size
        cx, cy = src_w / 2.0, src_h / 2.0
        r_max = min(cx, cy)

        fish_fov = np.radians(self._fisheye_fov_deg)
        out_fov = np.radians(self._output_fov_deg)

        # Focal lengths
        f_fish = r_max / (fish_fov / 2.0)
        f_rect = (out_w / 2.0) / np.tan(out_fov / 2.0)

        # Output pixel grid → angles → fisheye source coords
        u = np.arange(out_w, dtype=np.float32) - out_w / 2.0
        v = np.arange(out_h, dtype=np.float32) - out_h / 2.0
        uu, vv = np.meshgrid(u, v)

        # Angle from optical axis in rectilinear projection
        theta = np.arctan2(np.sqrt(uu**2 + vv**2), f_rect)
        phi = np.arctan2(vv, uu)

        # Equidistant fisheye radius
        r_fish_px = f_fish * theta

        # Source pixel coordinates
        src_x = (r_fish_px * np.cos(phi) + cx).astype(np.float32)
        src_y = (r_fish_px * np.sin(phi) + cy).astype(np.float32)

        self._map1 = src_x
        self._map2 = src_y
        logger.info(
            "Fisheye remap built: %dx%d → %dx%d (FOV %.0f° → %.0f°)",
            src_w, src_h, out_w, out_h,
            self._fisheye_fov_deg, self._output_fov_deg,
        )

    def _remap_fisheye(self, image: np.ndarray) -> np.ndarray:
        """Apply fisheye → rectilinear remap to a single-lens crop."""
        h, w = image.shape[:2]
        if self._map1 is None:
            self._build_remap_tables(h, w)
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

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


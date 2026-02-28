"""GoPro frame source using native photo capture and HTTP file download.

Each :meth:`read` call:
1. Switches the camera to photo mode (once, on first capture)
2. Triggers the native shutter via the GoPro HTTP API
3. Downloads the resulting ``.36P`` file (HEIC container, two EAC streams)
4. Extracts stream ``0:v:0`` (front lens) as a JPEG via FFmpeg — a single
   4096×1344 fisheye image, which is immediately usable by a VLM

This avoids the grey/washed-out frames produced by naive UDP stream capture
(the GoPro streams in EAC format which OpenCV cannot decode directly).

Requirements:
    - GoPro connected and reachable at ``camera_ip`` (default 172.29.170.51)
    - ``ffmpeg`` available on PATH or at ``/usr/bin/ffmpeg``
    - ``requests`` package installed

Usage::

    with GoProSource(fps=0.5) as src:
        for frame in src:
            result = await monitor.update(frame=frame.image)
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import requests

from aura.sources.base import FrameSource
from aura.sources.frame import Frame

logger = logging.getLogger(__name__)

_FFMPEG = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"

# GoPro HTTP API endpoints
_EP_STATE      = "/gopro/camera/state"
_EP_WIRED_USB  = "/gopro/camera/control/wired_usb?p=1"
_EP_STREAM_STOP = "/gopro/camera/stream/stop"
_EP_PHOTO_MODE = "/gopro/camera/presets/set_group?id=1001"
_EP_SHUTTER    = "/gopro/camera/shutter/start"
_EP_MEDIA_LIST = "/gopro/media/list"


class GoProSource(FrameSource):
    """Live frame source from a GoPro Max 2 via native photo capture.

    Each :meth:`read` call triggers a native shutter, downloads the raw
    ``.36P`` file, and extracts the front fisheye lens as a BGR numpy array.
    The call blocks for ~3-5 seconds (shutter + download), so this source
    is suited to low-cadence pipelines (0.2–0.5 fps) like the hand-layup
    intent monitor.

    Args:
        camera_ip: GoPro USB IP address (default ``"172.29.170.51"``).
        fps: Target capture rate.  :meth:`read` sleeps between calls to
            honour this cadence.
        capture_timeout: Seconds to wait for the shutter + download before
            giving up on a single frame.
        lens: Which lens stream to extract from the ``.36P`` file.
            ``"front"`` (default) uses stream ``0:v:0``;
            ``"back"`` uses stream ``0:v:1``.
    """

    def __init__(
        self,
        camera_ip: str = "172.29.170.51",
        fps: float = 0.3,
        capture_timeout: int = 30,
        lens: str = "front",
    ):
        self._camera_ip = camera_ip
        self._base = f"http://{camera_ip}"
        self._fps = max(0.01, fps)
        self._capture_timeout = capture_timeout
        self._stream_index = 0 if lens == "front" else 1

        self._opened = False
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_capture_time: float = 0.0
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return

        if not self._ping_camera():
            raise RuntimeError(
                f"GoPro not responding at {self._base}. "
                "Ensure the camera is connected and powered on."
            )

        # Stop any active stream so we can switch to photo mode
        self._get(_EP_STREAM_STOP, timeout=5, silent=True)
        time.sleep(0.5)

        # Enable wired USB control
        self._get(_EP_WIRED_USB, timeout=5, silent=True)
        time.sleep(0.3)

        # Switch to photo mode
        self._get(_EP_PHOTO_MODE, timeout=5)
        time.sleep(1.0)

        self._frame_count = 0
        self._start_time = None
        self._last_capture_time = 0.0
        self._opened = True
        logger.info(
            "GoProSource opened: camera=%s lens=stream%d fps=%.2f",
            self._camera_ip, self._stream_index, self._fps,
        )

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        logger.info("GoProSource closed (camera=%s)", self._camera_ip)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> Optional[Frame]:
        if not self._opened:
            return None

        if self._start_time is None:
            self._start_time = time.monotonic()

        # Throttle to the requested FPS
        interval = 1.0 / self._fps
        elapsed_since_last = time.monotonic() - self._last_capture_time
        if elapsed_since_last < interval:
            time.sleep(interval - elapsed_since_last)

        image = self._native_capture()
        if image is None:
            logger.warning("GoProSource: capture failed")
            return None

        self._last_capture_time = time.monotonic()
        h, w = image.shape[:2]
        self._width = w
        self._height = h

        ts = time.monotonic() - self._start_time
        frame = Frame(
            image=image,
            timestamp=ts,
            frame_number=self._frame_count,
            source_name=f"gopro:{self._camera_ip}",
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
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def is_open(self) -> bool:
        return self._opened

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ping_camera(self) -> bool:
        try:
            resp = requests.get(f"{self._base}{_EP_STATE}", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _get(self, endpoint: str, timeout: int = 5, silent: bool = False) -> bool:
        try:
            resp = requests.get(f"{self._base}{endpoint}", timeout=timeout)
            ok = resp.status_code == 200
            if not ok and not silent:
                logger.warning("GoProSource: %s returned HTTP %d", endpoint, resp.status_code)
            return ok
        except Exception as exc:
            if not silent:
                logger.warning("GoProSource: %s failed: %s", endpoint, exc)
            return False

    def _native_capture(self) -> Optional["np.ndarray"]:
        """Trigger shutter, wait for new file, download .36P, extract one lens."""
        # 1. Record the current newest filename so we can detect the new one
        before = self._newest_media_filename()

        # 2. Trigger shutter
        if not self._get(_EP_SHUTTER, timeout=10):
            return None

        # 3. Poll until a new file appears (up to ~15 s)
        filepath_on_camera = self._wait_for_new_file(previous=before, timeout=15)
        if filepath_on_camera is None:
            logger.warning("GoProSource: timed out waiting for new media file")
            return None

        # 4. Download the .36P to a temp location
        download_url = f"http://{self._camera_ip}:8080/videos/DCIM/{filepath_on_camera}"
        raw_path = self._download_file(download_url)
        if raw_path is None:
            return None

        # 5. Decode: plain image formats (JPG/PNG) read directly;
        #    .36P/.360 containers need ffmpeg to extract a lens stream.
        try:
            suffix = Path(raw_path).suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png"):
                image = cv2.imread(raw_path)
                if image is None:
                    logger.warning("GoProSource: cv2.imread failed on %s", raw_path)
                return image
            else:
                return self._extract_lens(raw_path)
        finally:
            Path(raw_path).unlink(missing_ok=True)

    def _media_list(self) -> Optional[dict]:
        """Fetch and return the parsed media list from the camera, or None."""
        try:
            resp = requests.get(
                f"http://{self._camera_ip}:8080{_EP_MEDIA_LIST}",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.debug("GoProSource: media list failed: %s", exc)
        return None

    def _newest_media_filename(self) -> Optional[str]:
        """Return the filename (not full path) of the newest file on the camera."""
        data = self._media_list()
        if data is None:
            return None
        dirs = data.get("media", [])
        if not dirs:
            return None
        # Files are ordered oldest-first; the last entry in the last dir is newest
        files = dirs[-1].get("fs", [])
        if not files:
            return None
        return files[-1].get("n")

    def _wait_for_new_file(self, previous: Optional[str], timeout: int = 15) -> Optional[str]:
        """Poll until a file newer than ``previous`` appears; return ``"dir/file"``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self._media_list()
            if data:
                dirs = data.get("media", [])
                if dirs:
                    latest_dir = dirs[-1]
                    dir_name = latest_dir.get("d", "")
                    files = latest_dir.get("fs", [])
                    if files:
                        newest = files[-1].get("n", "")
                        if newest and newest != previous:
                            return f"{dir_name}/{newest}"
            time.sleep(1.0)
        return None

    def _download_file(self, url: str) -> Optional[str]:
        """Download ``url`` to a temp file; return the path or None on error."""
        suffix = Path(url.split("?")[0]).suffix or ".36p"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            resp = requests.get(url, stream=True, timeout=self._capture_timeout)
            if resp.status_code != 200:
                logger.warning(
                    "GoProSource: download failed HTTP %d: %s", resp.status_code, url
                )
                Path(tmp_path).unlink(missing_ok=True)
                return None
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = Path(tmp_path).stat().st_size / 1024
            logger.info("GoProSource: downloaded %.1f KB -> %s", size_kb, tmp_path)
            return tmp_path
        except Exception as exc:
            logger.warning("GoProSource: download error: %s", exc)
            Path(tmp_path).unlink(missing_ok=True)
            return None

    def _extract_lens(self, raw_path: str) -> Optional["np.ndarray"]:
        """Extract one fisheye stream from the .36P file as a BGR ndarray."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            jpg_path = tmp.name

        # Select the desired stream (0:v:0 = front, 0:v:1 = back)
        stream_map = f"0:v:{self._stream_index}"
        cmd = [
            _FFMPEG, "-y",
            "-i", raw_path,
            "-map", stream_map,
            "-frames:v", "1",
            "-q:v", "2",
            "-f", "image2",
            jpg_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self._capture_timeout,
            )
            if result.returncode != 0 or not Path(jpg_path).exists():
                logger.warning(
                    "GoProSource: ffmpeg extract failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[-300:] if result.stderr else "",
                )
                return None
            image = cv2.imread(jpg_path)
            if image is None:
                logger.warning("GoProSource: cv2.imread failed on extracted frame")
            return image
        except subprocess.TimeoutExpired:
            logger.warning("GoProSource: ffmpeg extract timed out")
            return None
        except Exception as exc:
            logger.warning("GoProSource: extract error: %s", exc)
            return None
        finally:
            Path(jpg_path).unlink(missing_ok=True)

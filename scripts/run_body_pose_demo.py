#!/usr/bin/env python3
"""Standalone demo: extract body poses from a video using the SAM-3D-Body ZMQ server.

Runs the `BodyPoseMonitor` frame-by-frame on a video file and displays 
bounding boxes and 2D keypoints on the image.

Visualization: serves a live-updating JPEG at `http://localhost:<port>`
via a lightweight HTTP server.

Usage::
    # Ensure the fast sam3d body server is running in another terminal:
    # ./run_aura_server.sh

    uv run python scripts/run_body_pose_demo.py \
        --video demo_data/layup_demo/layup_gesture_demo_stationary_no_gloves_with_overlay.mp4 \
        --server-endpoint tcp://localhost:5556 \
        --frame-skip 2

    # With mesh generation (server saves PLY files):
    uv run python scripts/run_body_pose_demo.py \
        --video demo_data/layup_demo/layup_gesture_demo.mp4 \
        --mesh-output-dir output/demo_meshes
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on sys.path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.aura.monitors.body_pose_monitor import BodyPoseMonitor, BodyPoseMonitorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lightweight live JPEG server ─────────────────────────────────────

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()

_HTML_PAGE = b"""\
<!DOCTYPE html><html><head>
<title>Body Pose Demo</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style>
</head><body>
<img id="f" src="/frame.jpg">
<script>
const img=document.getElementById('f');
setInterval(()=>{img.src='/frame.jpg?t='+Date.now()},100);
</script></body></html>
"""

class _FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML_PAGE)
        elif self.path.startswith("/frame.jpg"):
            with _jpeg_lock:
                data = _latest_jpeg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if data:
                self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress request logs.

def _start_server() -> int:
    """Start the HTTP server on a free port, return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("0.0.0.0", port), _FrameHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port

def _publish_frame(vis_bgr: np.ndarray) -> None:
    """Encode frame as JPEG and publish for the HTTP server."""
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()

# ── Main loop ────────────────────────────────────────────────────────

# MHR70 keypoint indices for wrists (SAM-3D-Body format)
_KPT_LEFT_WRIST = 62
_KPT_RIGHT_WRIST = 41

# Colors for left/right hand trails and velocity display
_COLOR_LEFT = (255, 100, 50)    # blue-ish (BGR)
_COLOR_RIGHT = (50, 100, 255)   # red-ish (BGR)
_COLOR_BODY_VEL = (200, 200, 50)  # cyan-ish

TRAIL_LEN = 5  # number of past predictions to keep


def _extract_wrist_px(
    kpts: Optional[np.ndarray],
    w: int,
    h: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Return (left_wrist_px, right_wrist_px) from a [N,2] keypoints array."""
    if kpts is None:
        return None, None
    if len(kpts.shape) == 3:
        kpts = kpts[0]
    if kpts.shape[0] <= max(_KPT_LEFT_WRIST, _KPT_RIGHT_WRIST):
        return None, None
    lw = kpts[_KPT_LEFT_WRIST]
    rw = kpts[_KPT_RIGHT_WRIST]
    lw_px = (int(np.clip(lw[0], 0, w - 1)), int(np.clip(lw[1], 0, h - 1)))
    rw_px = (int(np.clip(rw[0], 0, w - 1)), int(np.clip(rw[1], 0, h - 1)))
    return lw_px, rw_px


def _body_centroid_px(
    kpts: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return mean of all valid (non-zero) 2D keypoints as a float [x, y] array."""
    if kpts is None:
        return None
    if len(kpts.shape) == 3:
        kpts = kpts[0]
    valid = kpts[(kpts[:, 0] > 0) | (kpts[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return valid.mean(axis=0)


def _draw_trail(
    vis: np.ndarray,
    trail: Deque[Optional[Tuple[int, int]]],
    color: Tuple[int, int, int],
) -> None:
    """Draw a fading polyline for the given trail of (x,y) pixel positions."""
    pts = [p for p in trail if p is not None]
    if len(pts) < 2:
        if len(pts) == 1:
            cv2.circle(vis, pts[0], 5, color, -1)
        return
    for i in range(1, len(pts)):
        alpha = i / (len(pts) - 1)  # 0 → oldest, 1 → newest
        faded = tuple(int(c * (0.25 + 0.75 * alpha)) for c in color)
        thickness = max(1, int(1 + 3 * alpha))
        cv2.line(vis, pts[i - 1], pts[i], faded, thickness)
    # Bright dot at latest position
    cv2.circle(vis, pts[-1], 6, color, -1)


def _px_velocity(
    trail: Deque[Optional[Tuple[int, int]]],
    timestamps: Deque[float],
) -> Optional[float]:
    """Return pixel/s speed between the oldest and newest valid trail entries."""
    valid = [(p, t) for p, t in zip(trail, timestamps) if p is not None]
    if len(valid) < 2:
        return None
    (x0, y0), t0 = valid[0]
    (x1, y1), t1 = valid[-1]
    dt = t1 - t0
    if dt < 1e-6:
        return None
    return float(np.hypot(x1 - x0, y1 - y0)) / dt


def _body_velocity(
    centroids: Deque[Optional[np.ndarray]],
    timestamps: Deque[float],
) -> Optional[float]:
    """Return pixel/s speed of the body centroid between oldest and newest valid entry."""
    valid = [(c, t) for c, t in zip(centroids, timestamps) if c is not None]
    if len(valid) < 2:
        return None
    c0, t0 = valid[0]
    c1, t1 = valid[-1]
    dt = t1 - t0
    if dt < 1e-6:
        return None
    return float(np.linalg.norm(c1 - c0)) / dt


async def run_demo(
    video_path: str,
    server_endpoint: str,
    frame_skip: int = 2,
    headless: bool = False,
    mesh_output_dir: str = "",
):
    config = BodyPoseMonitorConfig(server_endpoint=server_endpoint)
    monitor = BodyPoseMonitor(config=config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %s  |  %dx%d @ %.1f fps  |  %d frames  |  skip=%d",
                video_path, w, h, fps, total, frame_skip)

    generate_mesh = bool(mesh_output_dir)
    if generate_mesh:
        Path(mesh_output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Mesh output directory: %s", mesh_output_dir)

    show = not headless
    if show:
        port = _start_server()
        url = f"http://localhost:{port}/"
        logger.info("Live visualization: %s", url)
        webbrowser.open(url)

    # Per-person history: keyed by person index (0-based).
    # Each entry holds deques of length TRAIL_LEN for left/right wrist positions,
    # body centroids, and timestamps.
    person_lw_trails: dict[int, Deque[Optional[Tuple[int, int]]]] = {}
    person_rw_trails: dict[int, Deque[Optional[Tuple[int, int]]]] = {}
    person_body_cents: dict[int, Deque[Optional[np.ndarray]]] = {}
    person_timestamps: dict[int, Deque[float]] = {}

    frame_num = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % frame_skip != 0:
                continue

            timestamp = frame_num / fps

            t0 = time.perf_counter()
            update_kwargs: dict = {"frame": frame}
            if generate_mesh:
                update_kwargs["generate_mesh"] = True
                update_kwargs["mesh_output_dir"] = mesh_output_dir
                update_kwargs["mesh_prefix"] = f"frame_{frame_num:06d}"

            try:
                result = await monitor.update(**update_kwargs)
            except AttributeError:
                result = await monitor._process(**update_kwargs)
            dt = time.perf_counter() - t0

            if not result.is_valid:
                logger.warning("Frame %d invalid result: %s", frame_num, result.error)
                continue

            processed += 1
            logger.info(
                "Frame %5d | %6.1fs | %.2fs inference | persons=%d",
                frame_num, timestamp, result.inference_time_sec or dt, result.num_persons
            )

            if result.mesh_paths:
                for mp in result.mesh_paths:
                    logger.info("  Mesh saved: %s", mp)

            vis = frame.copy()

            # Update trails and draw persons
            for p_idx, person in enumerate(result.persons):
                # Initialise deques on first appearance of this person
                if p_idx not in person_lw_trails:
                    person_lw_trails[p_idx] = deque(maxlen=TRAIL_LEN)
                    person_rw_trails[p_idx] = deque(maxlen=TRAIL_LEN)
                    person_body_cents[p_idx] = deque(maxlen=TRAIL_LEN)
                    person_timestamps[p_idx] = deque(maxlen=TRAIL_LEN)

                # BBox
                bbox = person.bbox
                if bbox is not None and len(bbox) > 0:
                    if len(bbox.shape) > 1:
                        bbox = bbox[0]
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"Person {p_idx}", (x1, max(y1 - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Keypoints (2D)
                kpts = person.keypoints_2d
                if kpts is not None:
                    if len(kpts.shape) == 3:
                        kpts = kpts[0]
                    for kpt_i, pt in enumerate(kpts):
                        x, y = int(pt[0]), int(pt[1])
                        # Highlight wrist keypoints with larger circles
                        if kpt_i == _KPT_LEFT_WRIST:
                            cv2.circle(vis, (x, y), 6, _COLOR_LEFT, -1)
                        elif kpt_i == _KPT_RIGHT_WRIST:
                            cv2.circle(vis, (x, y), 6, _COLOR_RIGHT, -1)
                        else:
                            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

                # Accumulate trail data
                lw_px, rw_px = _extract_wrist_px(kpts, w, h)
                centroid = _body_centroid_px(kpts)
                person_lw_trails[p_idx].append(lw_px)
                person_rw_trails[p_idx].append(rw_px)
                person_body_cents[p_idx].append(centroid)
                person_timestamps[p_idx].append(timestamp)

                # Draw trails
                _draw_trail(vis, person_lw_trails[p_idx], _COLOR_LEFT)
                _draw_trail(vis, person_rw_trails[p_idx], _COLOR_RIGHT)

                # Velocity metrics
                lw_vel = _px_velocity(person_lw_trails[p_idx], person_timestamps[p_idx])
                rw_vel = _px_velocity(person_rw_trails[p_idx], person_timestamps[p_idx])
                body_vel = _body_velocity(person_body_cents[p_idx], person_timestamps[p_idx])

                # Overlay velocity text near the person bounding box top-right
                bbox_arr = person.bbox
                if bbox_arr is not None and len(bbox_arr) > 0:
                    if len(bbox_arr.shape) > 1:
                        bbox_arr = bbox_arr[0]
                    vx2, vy1 = int(bbox_arr[2]), int(bbox_arr[1])
                else:
                    vx2, vy1 = w - 10, 20

                vel_lines = []
                if lw_vel is not None:
                    vel_lines.append((f"L.hand: {lw_vel:.0f} px/s", _COLOR_LEFT))
                if rw_vel is not None:
                    vel_lines.append((f"R.hand: {rw_vel:.0f} px/s", _COLOR_RIGHT))
                if body_vel is not None:
                    vel_lines.append((f"Body:   {body_vel:.0f} px/s", _COLOR_BODY_VEL))

                for line_i, (text, color) in enumerate(vel_lines):
                    ty = max(vy1 + line_i * 18, 15)
                    # Shadow for readability
                    cv2.putText(vis, text, (vx2 - 145, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(vis, text, (vx2 - 145, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Legend
            cv2.putText(vis, "L.hand trail", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_LEFT, 1)
            cv2.putText(vis, "R.hand trail", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_RIGHT, 1)

            # Timing bar at bottom
            info = (f"Frame {frame_num} | {timestamp:.1f}s | "
                    f"SAM3: {dt:.2f}s | "
                    f"persons: {result.num_persons}")
            cv2.putText(vis, info, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            if show:
                _publish_frame(vis)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
    finally:
        cap.release()
        monitor.stop()

    logger.info("Done. Processed %d / %d frames.", processed, frame_num)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Body Pose Monitor Demo (Server bridge)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--server-endpoint", default="tcp://localhost:5556", help="ZMQ endpoint for fast_sam_3d_body server")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every N-th frame")
    parser.add_argument("--headless", action="store_true", help="Disable visualization server")
    parser.add_argument("--mesh-output-dir", default="", help="Directory to save PLY mesh files (empty = disabled)")

    args = parser.parse_args()

    asyncio.run(run_demo(
        video_path=args.video,
        server_endpoint=args.server_endpoint,
        frame_skip=args.frame_skip,
        headless=args.headless,
        mesh_output_dir=args.mesh_output_dir,
    ))


if __name__ == "__main__":
    main()

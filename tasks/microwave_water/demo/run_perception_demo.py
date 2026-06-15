#!/usr/bin/env python3
"""Standalone demo: track the microwave & cups in a microwave-water video.

Runs the task-specific ``MicrowaveWaterPerceptionMonitor`` frame-by-frame
on a video file and displays masks + cup locations (counter top / on top
of microwave / inside microwave).

Visualization: serves a live-updating JPEG at ``http://localhost:<port>``
via a lightweight HTTP server (avoids OpenCV/matplotlib GUI crashes
caused by SAM3 CUDA + Qt5 conflict).

Outputs are saved to a timestamped run folder under
``logs/microwave_water/perception_demo/<timestamp>/``:
  - ``images/frame_NNNNN.jpg`` — annotated visualization per processed frame
  - ``results.jsonl`` — one JSON object per processed frame with cup
    locations, detections, and timing

Usage::

    uv run python tasks/microwave_water/demo/run_perception_demo.py \\
        --video demo_data/layup_demo/microwave_water.mp4 \\
        --frame-skip 15

    # Also save to file
    uv run python tasks/microwave_water/demo/run_perception_demo.py \\
        --video demo_data/layup_demo/microwave_water.mp4 \\
        --frame-skip 15 --save output_vis.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tasks.microwave_water.perception.microwave_water_perception_monitor import (
    MicrowaveWaterPerceptionMonitor,
    MicrowaveWaterPerceptionConfig,
)

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
<title>Microwave Water Perception Demo</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style>
</head><body>
<img id="f" src="/frame.jpg">
<script>
const img=document.getElementById('f');
setInterval(()=>{img.src='/frame.jpg?t='+Date.now()},300);
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


def _publish_frame(vis_bgr: np.ndarray) -> None:
    """Encode frame as JPEG and publish for the HTTP server."""
    global _latest_jpeg
    _, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _jpeg_lock:
        _latest_jpeg = buf.tobytes()


def _start_server() -> int:
    """Start the HTTP server on a free port, return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("0.0.0.0", port), _FrameHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port


# ── Mask safety ──────────────────────────────────────────────────────

def _ensure_numpy_masks(result: dict) -> None:
    """Force all masks in result to contiguous numpy uint8 on CPU."""
    objs = [result.get("microwave")]
    objs.extend(result.get("cups", {}).values())
    for obj in objs:
        if obj is not None and obj.mask is not None:
            m = obj.mask
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            obj.mask = np.ascontiguousarray(m, dtype=np.uint8)


# ── Main loop ────────────────────────────────────────────────────────

def run_demo(
    video_path: str,
    frame_skip: int = 15,
    headless: bool = False,
    save_path: str | None = None,
) -> None:
    config = MicrowaveWaterPerceptionConfig()
    monitor = MicrowaveWaterPerceptionMonitor(config=config)

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

    # ── Logs folder ─────────────────────────────────────────────────
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _project_root / "logs" / "microwave_water" / "perception_demo" / run_ts
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / "results.jsonl"
    images_dir = log_dir / "images"
    images_dir.mkdir(exist_ok=True)
    logger.info("Logging to %s", log_dir)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps / frame_skip, (w, h))
        logger.info("Saving visualisation to %s", save_path)

    show = not headless
    if show:
        port = _start_server()
        url = f"http://localhost:{port}/"
        logger.info("Live visualization: %s", url)
        import webbrowser
        webbrowser.open(url)

    frame_num = 0
    processed = 0
    results_file = open(results_path, "w")
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
            try:
                result = asyncio.run(monitor.process_frame(frame))
            except Exception as e:
                logger.exception("process_frame error on frame %d: %s",
                                 frame_num, e)
                result = None
            dt = time.perf_counter() - t0

            if result is None:
                if show:
                    _publish_frame(frame)
                continue

            processed += 1
            locs = result["cup_locations"]
            detected = result.get("detected_this_frame", set())
            det_counts = {k: len(v) for k, v in result["detections"].items()}
            logger.info(
                "Frame %5d | %6.1fs | %.2fs | mw=%s | raw_dets=%s",
                frame_num, timestamp, dt,
                "yes" if result.get("microwave") is not None else "no",
                det_counts,
            )
            for cid in sorted(locs):
                tag = "" if cid in detected else " (last known)"
                logger.info("  %s: %s%s", cid, locs[cid], tag)

            if processed <= 3:
                for cls, dets in result["detections"].items():
                    for d in dets:
                        logger.info("  %s: conf=%.3f bbox=%s",
                                    cls, d["confidence"], d["bbox"])

            _ensure_numpy_masks(result)

            try:
                vis = monitor.visualize(frame, result)
            except Exception as e:
                logger.warning("Visualize error: %s", e)
                vis = frame.copy()

            # Add timing bar at bottom (dark strip keeps large text legible).
            info = (f"Frame {frame_num} | {timestamp:.1f}s | "
                    f"SAM3: {dt:.2f}s | "
                    f"black: {locs.get('black_cup', '?')} | "
                    f"white: {locs.get('white_cup', '?')}")
            (tw, th_), _ = cv2.getTextSize(
                info, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3,
            )
            cv2.rectangle(vis, (6, h - th_ - 28), (6 + tw + 16, h - 6),
                          (0, 0, 0), -1)
            cv2.putText(vis, info, (14, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 3)

            # ── Save to logs ────────────────────────────────────────
            img_path = images_dir / f"frame_{frame_num:05d}.jpg"
            cv2.imwrite(str(img_path), vis)

            record = {
                "frame": frame_num,
                "timestamp": round(timestamp, 3),
                "processing_time": round(dt, 4),
                "cup_locations": locs,
                "detected_this_frame": sorted(detected),
                "microwave_detected": result.get("microwave") is not None,
                "detections": det_counts,
            }
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

            if writer:
                writer.write(vis)
            if show:
                _publish_frame(vis)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
    finally:
        cap.release()
        results_file.close()
        if writer:
            writer.release()

    logger.info("Done. Processed %d / %d frames. Logs: %s",
                processed, frame_num, log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microwave water perception demo — track microwave & cups with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to microwave water demo video",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=15,
        help="Process every N-th frame (default: 15)",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Disable visualization (log-only)",
    )
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save visualisation video to PATH (e.g. output.mp4)",
    )
    args = parser.parse_args()
    run_demo(args.video, args.frame_skip, args.headless, args.save)


if __name__ == "__main__":
    main()

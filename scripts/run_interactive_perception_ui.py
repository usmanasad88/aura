#!/usr/bin/env python3
"""Interactive webcam perception UI — click to segment / track with SAM3.

Launches a lightweight browser interface and starts the webcam.  Click anywhere
on the live video to segment whatever object is under the cursor with SAM3, or
switch to *track* mode and click once to have the mask follow the object across
frames.  A text box lets you do open-vocabulary segmentation ("coffee mug",
"hand", …) on demand.

The UI is served as a self-refreshing JPEG over HTTP — the same pattern used by
``tasks/kettle_tea_making/demo/run_perception_demo.py`` — which sidesteps the
OpenCV/Qt GUI crash caused by the SAM3 CUDA + Qt5 conflict.

Controls (in the browser):
  • Mode: **Point** — left-click = keep, right-click = exclude; refine the same
    object with more clicks.  **Box** — drag a rectangle.  **Track** — click an
    object once and it is followed live.  **Text** — type a phrase and segment.
  • **Resume live** clears the current selection and returns to the live feed.

Usage::

    uv run python scripts/run_interactive_perception_ui.py
    uv run python scripts/run_interactive_perception_ui.py --camera 2 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import cv2
import numpy as np

# Ensure project src is importable when run directly.
_project_root = Path(__file__).resolve().parent.parent
for p in (_project_root, _project_root / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aura.monitors.interactive_perception_module import (  # noqa: E402
    InteractivePerceptionMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("interactive_perception_ui")


# ── Shared frame buffers ─────────────────────────────────────────────

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()
_command_queue: "queue.Queue[dict]" = queue.Queue()


def _publish_jpeg(frame_bgr: np.ndarray) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()


# ── Browser UI ───────────────────────────────────────────────────────

_HTML_PAGE = b"""\
<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>Interactive Perception \xe2\x80\x94 SAM3</title>
<style>
  body{margin:0;background:#0d0d0f;color:#e6e6e6;font-family:system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center}
  #bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px;
       background:#17171b;width:100%;box-sizing:border-box;justify-content:center}
  button,select,input{background:#26262d;color:#e6e6e6;border:1px solid #3a3a44;
       border-radius:6px;padding:6px 10px;font-size:14px;cursor:pointer}
  button.active{background:#2d6cdf;border-color:#2d6cdf}
  input[type=text]{cursor:text;min-width:180px}
  #wrap{position:relative;margin-top:10px}
  #view{max-width:96vw;max-height:78vh;display:block;cursor:crosshair;
        border:1px solid #2a2a30}
  #sel{position:absolute;border:2px dashed #2d6cdf;background:rgba(45,108,223,.15);
       pointer-events:none;display:none}
  #hint{padding:6px;color:#9a9aa5;font-size:13px}
</style></head><body>
<div id="bar">
  <button id="m-point" class="active" onclick="setMode('point')">Point</button>
  <button id="m-box" onclick="setMode('box')">Box</button>
  <button id="m-track" onclick="setMode('track')">Track</button>
  <input id="txt" type="text" placeholder="text prompt e.g. coffee mug"
         onkeydown="if(event.key==='Enter')sendText()">
  <button onclick="sendText()">Segment text</button>
  <button onclick="resume()">Resume live</button>
</div>
<div id="hint">Point mode: left-click = keep, right-click = exclude. Box mode: drag. Track mode: click an object.</div>
<div id="wrap">
  <img id="view" src="/frame.jpg" draggable="false">
  <div id="sel"></div>
</div>
<script>
let mode='point';
const img=document.getElementById('view');
const sel=document.getElementById('sel');
function setMode(m){mode=m;
  for(const id of ['point','box','track'])
    document.getElementById('m-'+id).classList.toggle('active',id===m);
}
// Live refresh.
setInterval(()=>{img.src='/frame.jpg?t='+Date.now();},120);
// Normalised coords from a mouse event over the image.
function norm(e){const r=img.getBoundingClientRect();
  return {x:Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),
          y:Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))};}
function post(path,body){fetch(path,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
// Point / track clicks.
img.addEventListener('click',e=>{if(mode==='box')return;
  const p=norm(e);
  if(mode==='track')post('/track',{x:p.x,y:p.y});
  else post('/click',{x:p.x,y:p.y,label:1});});
img.addEventListener('contextmenu',e=>{e.preventDefault();
  if(mode!=='point')return;const p=norm(e);post('/click',{x:p.x,y:p.y,label:0});});
// Box drag.
let drag=null;
img.addEventListener('mousedown',e=>{if(mode!=='box')return;e.preventDefault();
  const r=img.getBoundingClientRect();drag={x0:e.clientX,y0:e.clientY,r:r};
  sel.style.display='block';});
window.addEventListener('mousemove',e=>{if(!drag)return;
  const x=Math.min(e.clientX,drag.x0),y=Math.min(e.clientY,drag.y0);
  sel.style.left=x+'px';sel.style.top=y+'px';
  sel.style.width=Math.abs(e.clientX-drag.x0)+'px';
  sel.style.height=Math.abs(e.clientY-drag.y0)+'px';});
window.addEventListener('mouseup',e=>{if(!drag)return;const r=drag.r;
  const a={x:(drag.x0-r.left)/r.width,y:(drag.y0-r.top)/r.height};
  const b={x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height};
  drag=null;sel.style.display='none';
  post('/box',{x1:a.x,y1:a.y,x2:b.x,y2:b.y});});
function sendText(){const t=document.getElementById('txt').value.trim();
  if(t)post('/text',{prompt:t});}
function resume(){post('/resume',{});}
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html", _HTML_PAGE)
        elif self.path.startswith("/frame.jpg"):
            with _jpeg_lock:
                data = _latest_jpeg
            self._send(200, "image/jpeg", data, cache=False)
        else:
            self.send_error(404)

    def do_POST(self):
        routes = {"/click", "/box", "/text", "/track", "/resume"}
        if self.path not in routes:
            self.send_error(404)
            return
        body = self._json_body()
        body["action"] = self.path.lstrip("/")
        _command_queue.put(body)
        self._send(200, "application/json", b'{"ok":true}', cache=False)

    def _send(self, code: int, ctype: str, data: bytes, cache: bool = True):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if not cache:
            self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        if data:
            self.wfile.write(data)

    def log_message(self, *args):  # silence per-request logging
        pass


def _start_server() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


# ── Webcam capture thread ────────────────────────────────────────────

class _Camera:
    """Background webcam grabber holding only the most recent frame."""

    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        time.sleep(0.05)
        self.cap.release()


# ── Main processing loop (single owner of the SAM3 model) ────────────

def run(camera: int, width: int, height: int) -> None:
    logger.info("Opening camera %d ...", camera)
    cam = _Camera(camera, width, height)

    # Wait for the first frame so the UI has something to show immediately.
    for _ in range(100):
        if cam.read() is not None:
            break
        time.sleep(0.05)
    first = cam.read()
    if first is not None:
        _publish_jpeg(first)

    port = _start_server()
    url = f"http://localhost:{port}/"
    logger.info("Interactive perception UI: %s", url)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    logger.info("Loading SAM3 (first segmentation may take a few seconds)...")
    monitor = InteractivePerceptionMonitor()

    # Selection state owned by this thread.
    mode = "point"
    points: list[tuple[float, float]] = []
    labels: list[int] = []
    work_frame: np.ndarray | None = None   # frozen frame for point/box/text
    frozen_vis: np.ndarray | None = None    # annotated image to display when paused

    def reset_selection():
        nonlocal points, labels, work_frame, frozen_vis
        points, labels, work_frame, frozen_vis = [], [], None, None
        monitor.stop_tracking()

    def to_px(nx: float, ny: float, frame: np.ndarray) -> tuple[float, float]:
        h, w = frame.shape[:2]
        return nx * w, ny * h

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Handle one pending UI command, if any ──────────────────
            try:
                cmd = _command_queue.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                action = cmd.get("action")
                try:
                    if action == "resume":
                        reset_selection()
                        mode = mode  # unchanged
                    elif action == "track":
                        reset_selection()
                        mode = "track"
                        x, y = to_px(cmd["x"], cmd["y"], frame)
                        res = monitor.start_tracking(x, y)
                        if res is None:
                            logger.info("Nothing found to track at click.")
                    elif action == "click":
                        mode = "point"
                        if work_frame is None:
                            work_frame = frame.copy()
                            monitor.set_frame(work_frame)
                        x, y = to_px(cmd["x"], cmd["y"], work_frame)
                        points.append((x, y))
                        labels.append(int(cmd.get("label", 1)))
                        res = monitor.segment_points(points, labels)
                        marks = [(px, py, lb) for (px, py), lb in zip(points, labels)]
                        frozen_vis = monitor.visualize(
                            work_frame, [res] if res else [], marks)
                    elif action == "box":
                        mode = "box"
                        reset_selection()
                        work_frame = frame.copy()
                        monitor.set_frame(work_frame)
                        x1, y1 = to_px(cmd["x1"], cmd["y1"], work_frame)
                        x2, y2 = to_px(cmd["x2"], cmd["y2"], work_frame)
                        res = monitor.segment_box(x1, y1, x2, y2)
                        frozen_vis = monitor.visualize(
                            work_frame, [res] if res else [])
                    elif action == "text":
                        reset_selection()
                        work_frame = frame.copy()
                        monitor.set_frame(work_frame)
                        results = monitor.segment_text(cmd.get("prompt", ""))
                        frozen_vis = monitor.visualize(work_frame, results)
                        logger.info("Text '%s' -> %d instance(s)",
                                    cmd.get("prompt"), len(results))
                except Exception as e:
                    logger.exception("Command %s failed: %s", action, e)

            # ── Render ────────────────────────────────────────────────
            if mode == "track" and monitor.is_tracking:
                res = monitor.track(frame)
                if res is None:
                    logger.info("Track lost — click again to re-acquire.")
                    _publish_jpeg(frame)
                else:
                    _publish_jpeg(monitor.visualize(frame, [res]))
            elif frozen_vis is not None:
                _publish_jpeg(frozen_vis)
            else:
                _publish_jpeg(frame)

            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        cam.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    args = ap.parse_args()
    run(args.camera, args.width, args.height)


if __name__ == "__main__":
    main()

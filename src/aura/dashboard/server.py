"""Flask SSE server for AURA real-time dashboard.

Runs in a background thread alongside the LangGraph workflow.
Publishes workflow state via Server-Sent Events (SSE) so the
browser dashboard refreshes in real time.

Also serves video frames as JPEG snapshots via ``/api/frame``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).parent
_TEMPLATE_DIR = _DASHBOARD_DIR / "templates"
_STATIC_DIR = _DASHBOARD_DIR / "static"

# ── Module-level singleton ──────────────────────────────────────────────────
_instance: Optional["DashboardServer"] = None


def get_dashboard() -> Optional["DashboardServer"]:
    """Return the running DashboardServer singleton (or None)."""
    return _instance


class DashboardServer:
    """Flask-based real-time dashboard with SSE streaming.

    Usage::

        dash = DashboardServer(port=5555)
        dash.start()                     # non-blocking, runs in thread
        dash.publish("run_gesture", {...})  # from workflow nodes
        dash.set_frame(numpy_image)      # latest camera frame
        dash.stop()
    """

    def __init__(self, port: int = 5555, host: str = "0.0.0.0") -> None:
        global _instance
        self.port = port
        self.host = host

        self.app = Flask(
            __name__,
            template_folder=str(_TEMPLATE_DIR),
            static_folder=str(_STATIC_DIR),
        )
        CORS(self.app)

        # SSE subscriber queues (one per connected browser tab)
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

        # Latest state snapshot (for new connections / polling)
        self._state: Dict[str, Any] = {
            "cycle_count": 0,
            "current_frame_num": 0,
            "current_timestamp_sec": 0.0,
            "gesture": {},
            "intent": {},
            "ssg": {},
            "task_state": {},
            "decision": {},
            "actions": [],
            "action_log": [],
            "completed_steps": [],
            "object_locations": {},
            "human_requesting_help": False,
            "is_complete": False,
            "error": None,
            "config": {},
            "node_timings": {},
        }

        # Latest frame as JPEG bytes
        self._frame_jpeg: Optional[bytes] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._register_routes()
        _instance = self

    # ── Publishing API (called from workflow nodes) ─────────────────

    def publish(self, node_name: str, partial_state: Dict[str, Any]) -> None:
        """Publish a node's partial state update to all SSE subscribers."""
        self._merge_state(node_name, partial_state)

        event_data = {
            "node": node_name,
            "time": time.time(),
            "state": self._state,
        }
        msg = f"event: node_update\ndata: {json.dumps(event_data, default=str)}\n\n"

        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def set_frame(self, image: np.ndarray) -> None:
        """Update the latest camera frame (numpy BGR image)."""
        try:
            # Resize for network efficiency
            h, w = image.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / w
                image = cv2.resize(image, (max_w, int(h * scale)))

            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self._frame_jpeg = buf.tobytes()
        except Exception as e:
            logger.debug("Frame encode error: %s", e)

    def get_state(self) -> Dict[str, Any]:
        """Return current aggregated state."""
        return dict(self._state)

    # ── Internal state merging ──────────────────────────────────────

    def _merge_state(self, node_name: str, ps: Dict[str, Any]) -> None:
        """Merge partial state from a node into the aggregated snapshot."""
        s = self._state

        # Track node timing
        s["node_timings"][node_name] = time.time()

        # Per-node merging rules
        if node_name == "capture_frame":
            s["current_frame_num"] = ps.get("current_frame_num", s["current_frame_num"])
            s["current_timestamp_sec"] = ps.get("current_timestamp_sec", s["current_timestamp_sec"])
            if ps.get("is_complete"):
                s["is_complete"] = True

        elif node_name == "run_gesture":
            mo = ps.get("monitor_outputs", {})
            s["gesture"] = mo.get("gesture", s["gesture"])
            s["human_requesting_help"] = ps.get("human_requesting_help", False)

        elif node_name == "run_intent":
            intent = ps.get("intent_result")
            if intent:
                s["intent"] = intent
            mo = ps.get("monitor_outputs", {})
            if mo.get("intent"):
                s["intent"] = mo["intent"]

        elif node_name == "update_ssg":
            s["ssg"] = ps.get("ssg", s["ssg"])
            s["task_state"] = ps.get("task_state", s["task_state"])
            s["completed_steps"] = ps.get("completed_steps", s["completed_steps"])
            s["object_locations"] = ps.get("object_locations", s["object_locations"])

        elif node_name == "decide_action":
            s["decision"] = ps.get("last_decision", s["decision"])
            s["actions"] = ps.get("pending_actions", [])

        elif node_name == "execute_action":
            executed = ps.get("decision_history", [])
            if executed:
                s["action_log"] = (s.get("action_log") or []) + executed
                # Keep last 50 entries
                s["action_log"] = s["action_log"][-50:]
            s["object_locations"] = ps.get("object_locations", s["object_locations"])

        elif node_name == "check_complete":
            s["cycle_count"] = ps.get("cycle_count", s["cycle_count"])
            s["is_complete"] = ps.get("is_complete", False)
            s["human_requesting_help"] = ps.get("human_requesting_help", False)

        # Always propagate errors
        if ps.get("error"):
            s["error"] = ps["error"]

        # Config (set once)
        if ps.get("config") and not s["config"]:
            s["config"] = ps["config"]

    # ── Flask routes ────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            return render_template("dashboard.html")

        @app.route("/api/state")
        def api_state():
            return jsonify(self._state)

        @app.route("/api/frame")
        def api_frame():
            if self._frame_jpeg:
                return Response(self._frame_jpeg, mimetype="image/jpeg")
            # Return 1x1 transparent pixel
            return Response(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
                mimetype="image/png",
            )

        @app.route("/api/frame/stream")
        def api_frame_stream():
            """MJPEG stream of camera frames."""
            def generate():
                while self._running:
                    if self._frame_jpeg:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + self._frame_jpeg
                            + b"\r\n"
                        )
                    time.sleep(0.1)  # 10 fps max
            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/events")
        def api_events():
            """SSE endpoint — streams node_update events."""
            def stream() -> Generator:
                q: queue.Queue = queue.Queue(maxsize=100)
                with self._lock:
                    self._subscribers.append(q)
                try:
                    # Send initial state
                    init = f"event: init\ndata: {json.dumps(self._state, default=str)}\n\n"
                    yield init

                    while self._running:
                        try:
                            msg = q.get(timeout=1.0)
                            yield msg
                        except queue.Empty:
                            # Send keepalive
                            yield ": keepalive\n\n"
                finally:
                    with self._lock:
                        if q in self._subscribers:
                            self._subscribers.remove(q)

            return Response(
                stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="aura-dashboard",
        )
        self._thread.start()
        logger.info("Dashboard started at http://localhost:%d", self.port)

    def _run(self) -> None:
        import werkzeug.serving
        # Suppress werkzeug request logging
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.setLevel(logging.WARNING)

        srv = werkzeug.serving.make_server(
            self.host, self.port, self.app, threaded=True,
        )
        srv.timeout = 1
        while self._running:
            srv.handle_request()

    def stop(self) -> None:
        """Signal the server to shut down."""
        global _instance
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        _instance = None
        logger.info("Dashboard stopped.")

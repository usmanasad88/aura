"""Flask SSE server for AURA real-time dashboard.

Runs in a background thread alongside the LangGraph workflow.
Publishes workflow state via Server-Sent Events (SSE) so the
browser dashboard refreshes in real time.

Also serves video frames as JPEG snapshots via ``/api/frame``.

When started in **launcher mode** (``launcher_mode=True``), the
server also serves a configuration UI at ``/`` that lets the user
pick task, source, model, etc. and launch the workflow from the
browser.
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
from typing import Any, Callable, Dict, Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request
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

    Launcher mode::

        dash = DashboardServer(port=5555, launcher_mode=True,
                               project_root="/path/to/aura")
        dash.set_launch_callback(my_callback)
        dash.start()
        # User configures and clicks Launch in the browser
    """

    def __init__(
        self,
        port: int = 5555,
        host: str = "0.0.0.0",
        *,
        launcher_mode: bool = False,
        project_root: Path | str | None = None,
    ) -> None:
        global _instance
        self.port = port
        self.host = host

        # Launcher mode state
        self._launcher_mode = launcher_mode
        self._project_root = Path(project_root) if project_root else None
        self._workflow_running = False
        self._workflow_thread: Optional[threading.Thread] = None
        self._launch_callback: Optional[Callable[[dict], None]] = None
        self._stop_requested: bool = False

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
        self._state: Dict[str, Any] = self._empty_state()

        # Pre-initialised monitor status (populated by /api/initialize-monitors)
        self._preinit_status: Dict[str, Any] = {
            "perception": {"state": "idle", "task": None, "detail": ""},
            "audio": {"state": "idle", "task": None, "detail": ""},
        }

        # Latest frame as JPEG bytes
        self._frame_jpeg: Optional[bytes] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._register_routes()
        _instance = self

    # ── Launcher configuration ─────────────────────────────────

    def set_launch_callback(self, callback: Callable[[dict], None]) -> None:
        """Set the callback invoked when the user clicks Launch.

        The callback receives a config dict and is called in a
        background thread.  It should block until the workflow
        completes.
        """
        self._launch_callback = callback

    @property
    def workflow_running(self) -> bool:
        return self._workflow_running

    @workflow_running.setter
    def workflow_running(self, value: bool) -> None:
        self._workflow_running = value

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    # ── Publishing API (called from workflow nodes) ─────────────────

    def publish(self, node_name: str, partial_state: Dict[str, Any]) -> None:
        """Publish a node's partial state update to all SSE subscribers."""
        if partial_state is None:
            return
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

        elif node_name == "run_perception":
            mo = ps.get("monitor_outputs", {})
            if mo.get("perception"):
                s["perception"] = mo["perception"]
            s["object_locations"] = ps.get("object_locations", s["object_locations"])

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
            
            # If update_ssg propagates a delayed intent result, pick it up here
            intent = ps.get("intent_result")
            if intent:
                s["intent"] = intent
            mo = ps.get("monitor_outputs", {})
            if mo.get("intent"):
                s["intent"] = mo["intent"]

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

        # Config (set once) — capture task name + active monitors for the UI.
        if ps.get("config") and not s["config"]:
            s["config"] = ps["config"]
            if ps["config"].get("active_monitors"):
                s["active_monitors"] = list(ps["config"]["active_monitors"])

    # ── Flask routes ────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        # ── Page routes ─────────────────────────────────────────

        @app.route("/")
        def index():
            if self._launcher_mode and not self._workflow_running:
                return render_template("launcher.html")
            return render_template("dashboard.html")

        @app.route("/monitor")
        def monitor():
            return render_template("dashboard.html")

        # ── Launcher API ────────────────────────────────────────

        @app.route("/api/tasks")
        def api_tasks():
            tasks = []
            if self._project_root:
                tasks_dir = self._project_root / "tasks"
                if tasks_dir.exists():
                    for d in sorted(tasks_dir.iterdir()):
                        if d.is_dir() and (d / "config").exists():
                            tasks.append(d.name)
            return jsonify(tasks)

        @app.route("/api/videos")
        def api_videos():
            videos = []
            if self._project_root:
                demo_dir = self._project_root / "demo_data"
                if demo_dir.exists():
                    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                        for f in demo_dir.rglob(ext):
                            videos.append(str(f.relative_to(self._project_root)))
            return jsonify(sorted(videos))

        @app.route("/api/task-profile")
        def api_task_profile():
            """Return the selected task's profile (for active_monitors defaults)."""
            task = request.args.get("task", "")
            if not task or not self._project_root:
                return jsonify({"error": "task required"}), 400
            path = self._project_root / "tasks" / task / "config" / "task_profile.json"
            if not path.exists():
                return jsonify({"error": f"profile not found: {task}"}), 404
            try:
                return jsonify(json.loads(path.read_text()))
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/initialize-monitors", methods=["POST"])
        def api_initialize_monitors():
            """Pre-load heavy monitors (perception SAM3, audio session) so they
            are hot when the user hits Launch."""
            cfg = request.json or {}
            task = cfg.get("task")
            if not task or not self._project_root:
                return jsonify({"error": "task required"}), 400
            want_perception = bool(cfg.get("enable_perception"))
            want_audio = bool(cfg.get("enable_audio"))
            if not (want_perception or want_audio):
                return jsonify({"error": "Nothing to initialize"}), 400

            if want_perception:
                self._preinit_status["perception"] = {
                    "state": "loading", "task": task, "detail": "Loading perception model..."
                }
                threading.Thread(
                    target=self._preinit_perception,
                    args=(task,),
                    daemon=True,
                    name="preinit-perception",
                ).start()

            if want_audio:
                self._preinit_status["audio"] = {
                    "state": "loading", "task": task, "detail": "Opening audio session..."
                }
                threading.Thread(
                    target=self._preinit_audio,
                    args=(task, cfg),
                    daemon=True,
                    name="preinit-audio",
                ).start()

            return jsonify({"status": "started", "preinit": self._preinit_status})

        @app.route("/api/initialize-status")
        def api_initialize_status():
            return jsonify(self._preinit_status)

        @app.route("/api/preview", methods=["POST"])
        def api_preview():
            config = request.json or {}
            try:
                frame = self._grab_preview_frame(config)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500
            if frame is None:
                return jsonify({"error": "Could not capture preview frame"}), 404
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return Response(buf.tobytes(), mimetype="image/jpeg")
            return jsonify({"error": "JPEG encoding failed"}), 500

        @app.route("/api/launch", methods=["POST"])
        def api_launch():
            if self._workflow_running:
                return jsonify({"error": "Workflow is already running"}), 409
            if self._launch_callback is None:
                return jsonify({"error": "No launch callback configured"}), 500
            config = request.json or {}

            # Reset dashboard state for fresh run
            self._stop_requested = False
            self._reset_state()

            # Start workflow in background thread
            def _run():
                self._workflow_running = True
                try:
                    self._launch_callback(config)
                except Exception as e:
                    logger.error("Workflow error: %s", e)
                    self._state["error"] = str(e)
                finally:
                    self._workflow_running = False

            self._workflow_thread = threading.Thread(
                target=_run, daemon=True, name="aura-workflow",
            )
            self._workflow_thread.start()
            return jsonify({"status": "started", "monitor_url": "/monitor"})

        @app.route("/api/stop", methods=["POST"])
        def api_stop():
            if not self._workflow_running:
                return jsonify({"status": "not_running"}), 200
            self._stop_requested = True
            return jsonify({"status": "stopping"}), 200

        @app.route("/api/workflow-status")
        def api_workflow_status():
            return jsonify({
                "running": self._workflow_running,
                "cycle_count": self._state.get("cycle_count", 0),
                "is_complete": self._state.get("is_complete", False),
                "error": self._state.get("error"),
            })

        # ── Monitoring API (existing) ───────────────────────────

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

    # ── Preview frame grabbing ─────────────────────────────────────

    def _grab_preview_frame(self, config: dict) -> np.ndarray | None:
        """Grab a single frame from the selected source for preview."""
        source = config.get("source_type", "video")

        if source == "video":
            video_rel = config.get("video_path", "")
            if not video_rel or not self._project_root:
                return None
            full_path = self._project_root / video_rel
            if not full_path.exists():
                raise FileNotFoundError(f"Video not found: {video_rel}")
            cap = cv2.VideoCapture(str(full_path))
            try:
                ok, frame = cap.read()
                return frame if ok else None
            finally:
                cap.release()

        elif source == "webcam":
            device = config.get("webcam_device", 0)
            cap = cv2.VideoCapture(int(device))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open webcam device {device}")
                ok, frame = cap.read()
                return frame if ok else None
            finally:
                cap.release()

        elif source == "screen":
            try:
                import mss
            except ImportError:
                raise RuntimeError("mss package required for screen capture (pip install mss)")
            monitor_idx = config.get("screen_monitor", 1)
            with mss.mss() as sct:
                if monitor_idx >= len(sct.monitors):
                    raise ValueError(f"Monitor {monitor_idx} not available (found {len(sct.monitors) - 1})")
                mon = sct.monitors[int(monitor_idx)]
                region = config.get("screen_region")
                if region and len(region) == 4:
                    mon = {
                        "left": region[0], "top": region[1],
                        "width": region[2], "height": region[3],
                    }
                img = sct.grab(mon)
                frame = np.array(img)
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        elif source == "gopro":
            raise RuntimeError(
                "GoPro preview requires an active UDP stream. "
                "Use the GoPro app to verify the connection first."
            )

        return None

    # ── Pre-initialisation helpers ─────────────────────────────────

    def _preinit_perception(self, task: str) -> None:
        """Instantiate the task-specific perception monitor on a worker
        thread and stash it in the shared singleton dict so the workflow
        skips the (expensive) SAM3 load at launch time."""
        try:
            from aura.workflow.nodes import (
                _create_perception_monitor,
                _perception_monitors,
            )
            if not self._project_root:
                raise RuntimeError("project_root not set")
            config_dir = str(self._project_root / "tasks" / task / "config")
            config = {"task_name": task, "config_dir": config_dir}
            monitor = _create_perception_monitor(task, config)
            if monitor is None:
                self._preinit_status["perception"] = {
                    "state": "error", "task": task,
                    "detail": f"No task-specific perception monitor for '{task}'",
                }
                return
            _perception_monitors[config_dir] = monitor
            self._preinit_status["perception"] = {
                "state": "ready", "task": task,
                "detail": "Perception model loaded.",
            }
            logger.info("Perception monitor pre-initialized for %s", task)
        except Exception as exc:
            logger.exception("Perception pre-init failed")
            self._preinit_status["perception"] = {
                "state": "error", "task": task, "detail": str(exc),
            }

    def _preinit_audio(self, task: str, cfg: Dict[str, Any]) -> None:
        """Validate audio config (device availability, API key). A full
        Gemini Live session is bound to the workflow's event loop, so we
        only do cheap checks here — the session itself opens at launch."""
        try:
            import os
            if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
                self._preinit_status["audio"] = {
                    "state": "error", "task": task,
                    "detail": "GOOGLE_API_KEY / GEMINI_API_KEY not set",
                }
                return
            # Probe that sounddevice can list devices for the requested names.
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                in_name = cfg.get("audio_input_device") or ""
                out_name = cfg.get("audio_output_device") or ""
                names = [str(d.get("name", "")) for d in devices]
                missing = []
                if in_name and not any(in_name.lower() in n.lower() for n in names):
                    missing.append(f"input='{in_name}'")
                if out_name and not any(out_name.lower() in n.lower() for n in names):
                    missing.append(f"output='{out_name}'")
                if missing:
                    self._preinit_status["audio"] = {
                        "state": "error", "task": task,
                        "detail": "Audio device not found: " + ", ".join(missing),
                    }
                    return
            except Exception as exc:
                logger.debug("Audio device probe skipped: %s", exc)
            self._preinit_status["audio"] = {
                "state": "ready", "task": task,
                "detail": "Audio ready — session opens at launch.",
            }
            logger.info("Audio pre-check passed for %s", task)
        except Exception as exc:
            logger.exception("Audio pre-init failed")
            self._preinit_status["audio"] = {
                "state": "error", "task": task, "detail": str(exc),
            }

    # ── State reset ────────────────────────────────────────────────

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        """Return a fresh, empty aggregated-state dict."""
        return {
            "cycle_count": 0,
            "current_frame_num": 0,
            "current_timestamp_sec": 0.0,
            "gesture": {},
            "intent": {},
            "perception": {},
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
            "active_monitors": [],
        }

    def _reset_state(self) -> None:
        """Reset dashboard state for a fresh workflow run."""
        self._state = self._empty_state()
        self._frame_jpeg = None

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

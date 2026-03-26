"""Body Pose Monitor — ZMQ client for SAM-3D-Body inference server.

Sends RGB frames to an external Fast-SAM-3D-Body ZMQ server and
receives back per-person SMPL body pose / mesh results.  The server
runs in a separate conda environment (Python 3.11 / PyTorch 2.5),
completely isolating heavy dependencies (detectron2, smplx, MoGe, etc.)
from aura's runtime.

Usage::

    from aura.monitors.body_pose_monitor import BodyPoseMonitor, BodyPoseMonitorConfig

    monitor = BodyPoseMonitor(BodyPoseMonitorConfig(server_endpoint="tcp://localhost:5556"))
    output = await monitor.update(frame=bgr_image)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

from aura.core import (
    BodyPoseOutput,
    MonitorType,
    PersonBodyPose,
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig

logger = logging.getLogger(__name__)

# Lazy import zmq so the module can be imported without pyzmq installed
_zmq = None


def _get_zmq():
    global _zmq
    if _zmq is None:
        import zmq
        _zmq = zmq
    return _zmq


@dataclass
class BodyPoseMonitorConfig(MonitorConfig):
    """Configuration for the ZMQ body-pose bridge."""
    server_endpoint: str = "tcp://localhost:5556"
    timeout_sec: float = 10.0
    jpeg_quality: int = 85  # JPEG compression for wire transfer
    enabled: bool = True
    update_rate_hz: float = 10.0


class BodyPoseMonitor(BaseMonitor):
    """Aura monitor that delegates body-pose inference to a ZMQ server.

    The server (``sam3d_body_server.py``) runs inside the
    ``fast_sam_3d_body`` conda environment and keeps the model warm.
    Communication is frame-in / results-out over a REQ/REP ZMQ socket.
    """

    def __init__(self, config: Optional[BodyPoseMonitorConfig] = None):
        super().__init__(config or BodyPoseMonitorConfig())
        self.config: BodyPoseMonitorConfig
        self._socket = None
        self._ctx = None

    # ------------------------------------------------------------------
    # BaseMonitor interface
    # ------------------------------------------------------------------

    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.BODY_POSE

    def _ensure_connected(self):
        """Create the ZMQ REQ socket on first use."""
        if self._socket is not None:
            return
        zmq = _get_zmq()
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.config.timeout_sec * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.connect(self.config.server_endpoint)
        logger.info("BodyPoseMonitor connected to %s", self.config.server_endpoint)

    async def _process(self, **inputs) -> BodyPoseOutput:
        """Send a frame to the server and decode the response.

        Expected keyword arguments:
            frame: np.ndarray  — BGR image (H, W, 3), uint8
        """
        frame = inputs.get("frame")
        if frame is None:
            return BodyPoseOutput(is_valid=False, error="no frame provided")

        self._ensure_connected()

        # --- Encode frame as JPEG to reduce wire size ---
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        ok, jpg_buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            return BodyPoseOutput(is_valid=False, error="JPEG encode failed")

        # --- Send / receive in a thread to avoid blocking the loop ---
        t0 = time.monotonic()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._roundtrip, jpg_buf.tobytes()
            )
        except Exception as e:
            self._reset_socket()
            return BodyPoseOutput(is_valid=False, error=str(e))
        elapsed = time.monotonic() - t0

        # --- Unpack response ---
        if result is None:
            return BodyPoseOutput(is_valid=False, error="server timeout")

        return self._decode_response(result, elapsed)

    # ------------------------------------------------------------------
    # ZMQ helpers
    # ------------------------------------------------------------------

    def _roundtrip(self, jpg_bytes: bytes) -> Optional[bytes]:
        """Blocking REQ/REP round-trip (runs in executor thread)."""
        zmq = _get_zmq()
        try:
            self._socket.send(jpg_bytes)
            return self._socket.recv()
        except zmq.Again:
            logger.warning("BodyPoseMonitor: server did not respond in time")
            return None
        except zmq.ZMQError as e:
            logger.error("BodyPoseMonitor ZMQ error: %s", e)
            raise

    def _reset_socket(self):
        """Tear down and recreate socket after an error."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _decode_response(self, raw: bytes, elapsed: float) -> BodyPoseOutput:
        """Decode the msgpack response from the server into BodyPoseOutput."""
        try:
            import msgpack
        except ImportError:
            import json
            data = json.loads(raw)
            return self._build_output(data, elapsed)

        data = msgpack.unpackb(raw, raw=False)
        return self._build_output(data, elapsed)

    @staticmethod
    def _build_output(data: dict, elapsed: float) -> BodyPoseOutput:
        if data.get("error"):
            return BodyPoseOutput(is_valid=False, error=data["error"])

        persons = []
        for p in data.get("persons", []):
            persons.append(PersonBodyPose(
                bbox=np.asarray(p["bbox"], dtype=np.float32),
                keypoints_3d=np.asarray(p["keypoints_3d"], dtype=np.float32),
                keypoints_2d=np.asarray(p["keypoints_2d"], dtype=np.float32),
                vertices=np.asarray(p["vertices"], dtype=np.float32),
                camera_translation=np.asarray(p["camera_translation"], dtype=np.float32),
                body_pose_params=np.asarray(p["body_pose_params"], dtype=np.float32),
                hand_pose_params=np.asarray(p["hand_pose_params"], dtype=np.float32),
                shape_params=np.asarray(p["shape_params"], dtype=np.float32),
                global_rotation=np.asarray(p["global_rotation"], dtype=np.float32),
                joint_global_rotations=(
                    np.asarray(p["joint_global_rotations"], dtype=np.float32)
                    if p.get("joint_global_rotations") is not None else None
                ),
                expression_params=(
                    np.asarray(p["expression_params"], dtype=np.float32)
                    if p.get("expression_params") is not None else None
                ),
                focal_length=(
                    np.asarray(p["focal_length"], dtype=np.float32)
                    if p.get("focal_length") is not None else None
                ),
            ))

        return BodyPoseOutput(
            persons=persons,
            num_persons=len(persons),
            inference_time_sec=data.get("inference_time_sec", elapsed),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self):
        super().stop()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

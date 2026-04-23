"""ZMQ perception monitor for the cuboid_manipulation task.

Queries the ``ros_pose_bridge`` sidecar (``scripts/ros_pose_bridge.py``)
over a ZMQ REQ socket and converts the cached cuboid poses into the
symbolic state AURA's decision engine expects:

* ``cuboid_<color>_location`` ∈ {``table``, ``basket``, ``held``}
* ``cuboid_<color>_xy`` as ``"x,y"`` strings (mirrored into ``task_state``)
* ``held_cuboid`` — id of any cuboid currently lifted off the table

The held check is geometric: any cuboid whose z is above
``held_z_threshold_m`` (default 0.03 m above the resting z baseline) is
considered grasped. The basket containment check is a 2-D radius around
``perception.basket_footprint.center_xy`` from ``task_profile.json``.

Why a bridge? ROS 2 Humble's ``rclpy`` is built for Python 3.10 but
AURA's runtime is 3.12. The 3.10 bridge owns rclpy; this module stays
pure 3.12 (zmq + msgpack). See ``scripts/ros_pose_bridge.py``.

The monitor returns the dict shape expected by
:func:`aura.workflow.nodes.run_perception_node`:

* keys ending in ``_locations`` map ``object_id → region`` and are
  consolidated into ``state["object_locations"]``;
* a ``task_state`` dict is merged into the SSG's task-state.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


CUBOID_ORDER: Tuple[str, ...] = (
    "cuboid_red",
    "cuboid_green",
    "cuboid_blue",
    "cuboid_yellow",
    "cuboid_orange",
)

DEFAULT_BRIDGE_ENDPOINT = "tcp://localhost:5557"


@dataclass
class CuboidPerceptionConfig:
    basket_center_xy: Tuple[float, float] = (0.5, -0.5)
    basket_radius_m: float = 0.12
    # Cuboids spawn on the table at z = -0.05 (see Isaac extension).
    # Anything noticeably above that is being held.
    table_z_m: float = -0.05
    held_z_threshold_m: float = 0.03
    cuboid_order: Tuple[str, ...] = CUBOID_ORDER
    bridge_endpoint: str = DEFAULT_BRIDGE_ENDPOINT
    request_timeout_sec: float = 1.0


class CuboidPerceptionMonitor:
    """ZMQ → symbolic state shim for the cuboid_manipulation task.

    Designed to be polled once per workflow cycle by
    :func:`run_perception_node`. Each poll issues a REQ to the
    ``ros_pose_bridge`` sidecar and converts its response.
    """

    def __init__(self, config: Optional[CuboidPerceptionConfig] = None):
        self.config = config or CuboidPerceptionConfig()
        self._lock = threading.Lock()
        # Latest pose per cuboid: id -> (x, y, z). Kept so callers can
        # introspect raw poses (e.g., the standalone test script).
        self._poses: Dict[str, Tuple[float, float, float]] = {}
        self._ctx = None
        self._socket = None
        self._bridge_stamp: float = 0.0
        self._init_zmq()

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_task_profile(cls, runtime_config: Dict[str, Any]) -> "CuboidPerceptionMonitor":
        """Build a monitor whose footprint comes from ``task_profile.json``."""
        cfg = CuboidPerceptionConfig()
        config_dir = runtime_config.get("config_dir", "")
        if config_dir:
            profile_path = Path(config_dir) / "task_profile.json"
            try:
                profile = json.loads(profile_path.read_text())
                fp = (profile.get("perception") or {}).get("basket_footprint") or {}
                center = fp.get("center_xy")
                radius = fp.get("radius_m")
                if isinstance(center, list) and len(center) == 2:
                    cfg.basket_center_xy = (float(center[0]), float(center[1]))
                if isinstance(radius, (int, float)):
                    cfg.basket_radius_m = float(radius)
                bridge_ep = (profile.get("perception") or {}).get("bridge_endpoint")
                if isinstance(bridge_ep, str) and bridge_ep:
                    cfg.bridge_endpoint = bridge_ep
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not parse perception config from task_profile: %s", exc)
        return cls(cfg)

    # ── ZMQ init ────────────────────────────────────────────────────────

    def _init_zmq(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            logger.error("pyzmq not importable — cuboid perception disabled: %s", exc)
            return

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.config.request_timeout_sec * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self.config.request_timeout_sec * 1000))
        self._socket.connect(self.config.bridge_endpoint)
        logger.info(
            "CuboidPerceptionMonitor REQ connected to %s", self.config.bridge_endpoint
        )

    def _reset_socket(self) -> None:
        """Tear down and recreate socket after a timeout/error (REQ is stuck after a failed recv)."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        self._init_zmq()

    def _fetch_snapshot(self) -> Optional[dict]:
        """Blocking REQ/REP round-trip to the bridge. Returns None on timeout/error."""
        if self._socket is None:
            return None
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            logger.error("pyzmq/msgpack missing: %s", exc)
            return None

        try:
            self._socket.send(b"get")
            raw = self._socket.recv()
        except zmq.Again:
            logger.debug("bridge request timed out (is ros_pose_bridge.py running?)")
            self._reset_socket()
            return None
        except zmq.ZMQError as exc:
            logger.warning("ZMQ error talking to bridge: %s", exc)
            self._reset_socket()
            return None

        try:
            return msgpack.unpackb(raw, raw=False)
        except Exception as exc:
            logger.warning("could not decode bridge response: %s", exc)
            return None

    # ── Per-cycle output ────────────────────────────────────────────────

    async def process_frame(self, frame: Any) -> Dict[str, Any]:
        """Return the latest symbolic state. Frame is ignored."""
        snapshot = self._fetch_snapshot()
        if not snapshot or not snapshot.get("has_data"):
            return {}

        bridge_poses = snapshot.get("cuboids") or {}
        poses: Dict[str, Tuple[float, float, float]] = {}
        for cub_id, pose in bridge_poses.items():
            try:
                poses[cub_id] = (float(pose["x"]), float(pose["y"]), float(pose["z"]))
            except (KeyError, TypeError, ValueError):
                continue

        with self._lock:
            self._poses = dict(poses)
            self._bridge_stamp = float(snapshot.get("stamp", 0.0))

        if not poses:
            return {}

        cuboid_locations: Dict[str, str] = {}
        task_state: Dict[str, str] = {}
        held_cuboid = ""

        cx, cy = self.config.basket_center_xy
        r2 = self.config.basket_radius_m ** 2
        held_z = self.config.table_z_m + self.config.held_z_threshold_m

        for cub_id, (x, y, z) in poses.items():
            task_state[f"{cub_id}_xy"] = f"{x:.4f},{y:.4f}"
            if z > held_z:
                region = "held"
                if not held_cuboid:
                    held_cuboid = cub_id
            elif (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                region = "basket"
            else:
                region = "table"
            cuboid_locations[cub_id] = region
            # Mirror into task_state so the `cuboid_<color>_location` state-schema
            # variables are populated for the decision engine / dashboard.
            task_state[f"{cub_id}_location"] = region

        task_state["held_cuboid"] = held_cuboid
        task_state["cuboids_in_basket"] = sum(
            1 for r in cuboid_locations.values() if r == "basket"
        )

        return {
            "cuboid_locations": cuboid_locations,
            "task_state": task_state,
        }

    # ── Cleanup ─────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        self._ctx = None

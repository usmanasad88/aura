#!/usr/bin/python3.10
"""ZMQ REP sidecar that bridges ROS 2 PoseArray topics into AURA.

AURA's runtime is Python 3.12 (torch / sam3 / mediapipe) but ROS 2 Humble's
``rclpy`` is built against Python 3.10. This script is the small 3.10
process that owns ``rclpy``: it subscribes to the PoseArray topics the
Isaac Sim ``ur_robotiq_cortex`` extension publishes, caches the latest
snapshot, and serves it on a ZMQ REP socket so any AURA-side (3.12)
monitor can query it over msgpack — no rclpy import on that side.

Wire format
-----------
* Request:   empty frame (or b"get") — payload is ignored.
* Response:  msgpack-encoded dict::

    {
      "has_data": bool,
      "stamp": float,          # ROS header stamp, seconds
      "cuboids": {             # id -> pose
        "cuboid_red":   {"x","y","z","qw","qx","qy","qz"},
        ...
      }
    }

Run
---
    source /opt/ros/humble/setup.bash
    ./scripts/ros_pose_bridge.py                       # bind tcp://*:5557
    ./scripts/ros_pose_bridge.py --port 5557 --verbose

Must be launched with ``/usr/bin/python3.10`` (or the system python that
matches the ROS distro). Add to ``launch_all.sh`` alongside the other
services.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Dict, Optional, Tuple

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import PoseArray
except ImportError as exc:
    sys.stderr.write(
        f"rclpy/geometry_msgs not importable ({exc}).\n"
        "Source /opt/ros/<distro>/setup.bash and run this with "
        "/usr/bin/python3.10.\n"
    )
    sys.exit(2)

try:
    import msgpack
    import zmq
except ImportError as exc:
    sys.stderr.write(
        f"pyzmq/msgpack missing ({exc}). Install with:\n"
        "  /usr/bin/python3.10 -m pip install --user pyzmq msgpack\n"
    )
    sys.exit(2)


logger = logging.getLogger("ros_pose_bridge")


# Cuboid ordering in the PoseArray is fixed by the Isaac extension
# (ObjectPosePublisher). Keep in sync with CuboidPerceptionMonitor.
CUBOID_ORDER: Tuple[str, ...] = (
    "cuboid_red",
    "cuboid_green",
    "cuboid_blue",
    "cuboid_yellow",
    "cuboid_orange",
)

DEFAULT_POSES_TOPIC = "/scene/object_poses"
DEFAULT_INITIAL_POSES_TOPIC = "/scene/object_initial_poses"


class PoseCache:
    """Thread-safe latest-wins cache of the most recent PoseArray."""

    def __init__(self, order: Tuple[str, ...]):
        self._lock = threading.Lock()
        self._order = order
        self._cuboids: Dict[str, Dict[str, float]] = {}
        self._stamp: float = 0.0
        self._update_count: int = 0

    def update(self, msg: PoseArray) -> None:
        stamp = 0.0
        try:
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception:
            pass

        cuboids: Dict[str, Dict[str, float]] = {}
        for i, pose in enumerate(msg.poses):
            if i >= len(self._order):
                break
            cuboids[self._order[i]] = {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "qw": float(pose.orientation.w),
                "qx": float(pose.orientation.x),
                "qy": float(pose.orientation.y),
                "qz": float(pose.orientation.z),
            }

        with self._lock:
            self._cuboids = cuboids
            self._stamp = stamp
            self._update_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "has_data": bool(self._cuboids),
                "stamp": self._stamp,
                "cuboids": dict(self._cuboids),
            }

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count


def _run_rclpy(
    cache: PoseCache,
    poses_topic: str,
    initial_poses_topic: str,
    stop_event: threading.Event,
) -> None:
    """Spin rclpy in this thread until ``stop_event`` is set."""
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("aura_ros_pose_bridge")
    node.create_subscription(PoseArray, poses_topic, cache.update, 10)
    node.create_subscription(PoseArray, initial_poses_topic, cache.update, 10)
    logger.info("Subscribed to %s + %s", poses_topic, initial_poses_topic)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not stop_event.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _serve_zmq(cache: PoseCache, endpoint: str, stop_event: threading.Event) -> None:
    """REP loop. Blocks until ``stop_event`` is set."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    logger.info("ZMQ REP bound to %s", endpoint)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=200))
            if sock in events and events[sock] == zmq.POLLIN:
                try:
                    _ = sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                payload = msgpack.packb(cache.snapshot(), use_bin_type=True)
                sock.send(payload)
    finally:
        sock.close(linger=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=5557,
                        help="ZMQ REP port (default: 5557)")
    parser.add_argument("--bind", default="tcp://*",
                        help="ZMQ bind prefix (default: tcp://*)")
    parser.add_argument("--poses-topic", default=DEFAULT_POSES_TOPIC)
    parser.add_argument("--initial-poses-topic", default=DEFAULT_INITIAL_POSES_TOPIC)
    parser.add_argument("--verbose", action="store_true",
                        help="Log every N-th update heartbeat")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    endpoint = f"{args.bind}:{args.port}"
    cache = PoseCache(CUBOID_ORDER)
    stop_event = threading.Event()

    def _shutdown(signum, _frame):
        logger.info("signal %s received — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ros_thread = threading.Thread(
        target=_run_rclpy,
        args=(cache, args.poses_topic, args.initial_poses_topic, stop_event),
        name="ros-bridge-spin",
        daemon=True,
    )
    ros_thread.start()

    if args.verbose:
        def _heartbeat():
            last = -1
            while not stop_event.wait(5.0):
                n = cache.update_count
                if n != last:
                    logger.info("rx updates: %d  (latest snapshot has %d cuboids)",
                                n, len(cache.snapshot()["cuboids"]))
                    last = n
        threading.Thread(target=_heartbeat, name="heartbeat", daemon=True).start()

    try:
        _serve_zmq(cache, endpoint, stop_event)
    finally:
        stop_event.set()
        ros_thread.join(timeout=2.0)
    logger.info("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

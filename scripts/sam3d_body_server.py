#!/usr/bin/env python3
"""ZMQ REQ/REP server for SAM-3D-Body inference.

Run this inside the ``fast_sam_3d_body`` conda environment.  It loads the
model once, then serves per-frame body pose requests from aura's
``BodyPoseMonitor`` over ZMQ.

Usage::

    conda activate fast_sam_3d_body
    cd /home/mani/Repos/Fast-SAM-3D-Body
    python sam3d_body_server.py                    # defaults
    python sam3d_body_server.py --port 5556        # custom port
    python sam3d_body_server.py --image-size 256   # faster, lower quality
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import cv2
import msgpack
import numpy as np
import torch
import trimesh
import zmq

from mocap.core.setup_estimator import build_default_estimator


def _numpy_to_list(arr):
    """Convert numpy array to nested list for msgpack serialization."""
    if arr is None:
        return None
    return arr.tolist()


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def _encode_person(person_dict: dict) -> dict:
    """Convert a single process_one_image output dict to a serializable dict."""
    return {
        "bbox": _numpy_to_list(person_dict["bbox"]),
        "keypoints_3d": _numpy_to_list(person_dict["pred_keypoints_3d"]),
        "keypoints_2d": _numpy_to_list(person_dict["pred_keypoints_2d"]),
        "vertices": _numpy_to_list(person_dict["pred_vertices"]),
        "camera_translation": _numpy_to_list(person_dict["pred_cam_t"]),
        "body_pose_params": _numpy_to_list(person_dict["body_pose_params"]),
        "hand_pose_params": _numpy_to_list(person_dict["hand_pose_params"]),
        "shape_params": _numpy_to_list(person_dict["shape_params"]),
        "global_rotation": _numpy_to_list(person_dict["global_rot"]),
        "joint_global_rotations": _numpy_to_list(person_dict.get("pred_global_rots")),
        "expression_params": _numpy_to_list(person_dict.get("expr_params")),
        "focal_length": _numpy_to_list(person_dict.get("focal_length")),
    }


def _save_meshes(
    outputs: list[dict],
    faces: np.ndarray,
    output_dir: str,
    mesh_prefix: str,
) -> list[str]:
    """Save PLY meshes for each detected person. Returns list of saved file paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for pid, person in enumerate(outputs):
        vertices = person["pred_vertices"]
        cam_t = person["pred_cam_t"]
        if vertices is None or len(vertices) == 0:
            continue
        if np.any(np.isnan(vertices)) or np.any(np.isnan(cam_t)):
            continue
        vertex_colors = np.array([(*LIGHT_BLUE, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(
            vertices.copy() + cam_t,
            faces.copy(),
            vertex_colors=vertex_colors,
        )
        # Flip to standard coordinate system (same as Renderer.vertices_to_trimesh)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        filename = f"{mesh_prefix}_person_{pid:03d}.ply"
        filepath = os.path.join(output_dir, filename)
        mesh.export(filepath)
        paths.append(filepath)
    return paths


def serve(port: int, image_size: int, yolo_model: str):
    print(f"Loading SAM-3D-Body model (image_size={image_size}) ...")
    estimator = build_default_estimator(
        image_size=image_size,
        yolo_model_path=yolo_model,
    )
    print("Model loaded. Warming up ...")

    # Warmup with a dummy frame
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = np.array([[0.0, 0.0, 639.0, 479.0]], dtype=np.float32)
    for _ in range(2):
        estimator.process_one_image(dummy, bboxes=dummy_bbox, hand_box_source="body_decoder")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("Warmup done.")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"Listening on tcp://*:{port}  (Ctrl+C to stop)")

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        print("\nShutting down ...")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Use a poller so we can check `running` between polls
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    request_count = 0
    while running:
        events = dict(poller.poll(timeout=500))  # 500 ms
        if sock not in events:
            continue

        raw = sock.recv()
        t0 = time.monotonic()

        try:
            # Try to decode as msgpack structured request first.
            # Falls back to treating raw bytes as JPEG for backward compat.
            generate_mesh = False
            mesh_output_dir = ""
            mesh_prefix = "mesh"
            try:
                request = msgpack.unpackb(raw, raw=False)
                if isinstance(request, dict):
                    jpg_bytes = request["jpeg"]
                    generate_mesh = request.get("generate_mesh", False)
                    mesh_output_dir = request.get("mesh_output_dir", "")
                    mesh_prefix = request.get("mesh_prefix", "mesh")
                else:
                    jpg_bytes = raw
            except (msgpack.UnpackValueError, msgpack.UnpackException, KeyError):
                jpg_bytes = raw

            # Decode JPEG -> BGR -> RGB
            buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError("Failed to decode JPEG")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            outputs = estimator.process_one_image(
                frame_rgb,
                hand_box_source="yolo_pose",
            )

            # Optionally save meshes
            mesh_paths = []
            if generate_mesh and mesh_output_dir and outputs:
                mesh_paths = _save_meshes(
                    outputs, estimator.faces, mesh_output_dir, mesh_prefix,
                )

            persons = [_encode_person(p) for p in outputs]
            elapsed = time.monotonic() - t0
            response = {
                "persons": persons,
                "num_persons": len(persons),
                "inference_time_sec": elapsed,
                "mesh_paths": mesh_paths,
                "error": None,
            }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            response = {
                "persons": [],
                "num_persons": 0,
                "inference_time_sec": elapsed,
                "mesh_paths": [],
                "error": str(exc),
            }
            print(f"  ERROR: {exc}")

        sock.send(msgpack.packb(response, use_bin_type=True))
        request_count += 1
        if request_count % 10 == 0:
            print(f"  Served {request_count} requests  (last: {elapsed:.3f}s, {len(response['persons'])} persons)")

    sock.close()
    ctx.term()
    print(f"Server stopped after {request_count} requests.")


def main():
    parser = argparse.ArgumentParser(description="SAM-3D-Body ZMQ inference server")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port (default: 5556)")
    parser.add_argument(
        "--image-size", type=int, default=512, choices=[256, 384, 512],
        help="Model input size (default: 512)",
    )
    parser.add_argument(
        "--yolo-model", type=str, default="checkpoints/yolo/yolo11m-pose.engine",
        help="YOLO pose model path",
    )
    args = parser.parse_args()
    serve(port=args.port, image_size=args.image_size, yolo_model=args.yolo_model)


if __name__ == "__main__":
    main()

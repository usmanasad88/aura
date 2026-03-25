#!/usr/bin/env python3
"""Live webcam gesture monitor test with hand pose + gesture overlay.

Shows a real-time window with:
- Hand landmark skeleton (21 keypoints + connections)
- Gesture label per hand with confidence
- Large dominant gesture label
- Safety status bar

Usage:
    .venv/bin/python scripts/test_gesture_live.py
    .venv/bin/python scripts/test_gesture_live.py --camera 0
    .venv/bin/python scripts/test_gesture_live.py --camera 2 --num_hands 1

Keys:
    q / ESC  — quit
    r        — reset safety state
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from aura.monitors.gesture_monitor import (
    GestureMonitor,
    GestureMonitorConfig,
    MEDIAPIPE_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# MediaPipe hand connections (21 landmarks)
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (fixed: 0→9 not 5→9)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring    (fixed: 0→13)
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky   (fixed: 0→17)
    (5, 9), (9, 13), (13, 17),             # palm
]

# Distinct colours per finger group
FINGER_COLORS = {
    "thumb":  (0, 255, 255),   # yellow
    "index":  (0, 255, 0),     # green
    "middle": (255, 255, 0),   # cyan
    "ring":   (255, 0, 255),   # magenta
    "pinky":  (255, 0, 0),     # blue
    "palm":   (200, 200, 200), # grey
}

def _connection_color(a: int, b: int):
    if max(a, b) <= 4:
        return FINGER_COLORS["thumb"]
    if max(a, b) <= 8:
        return FINGER_COLORS["index"]
    if max(a, b) <= 12:
        return FINGER_COLORS["middle"]
    if max(a, b) <= 16:
        return FINGER_COLORS["ring"]
    if max(a, b) <= 20:
        return FINGER_COLORS["pinky"]
    return FINGER_COLORS["palm"]


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks,
    w: int,
    h: int,
    handedness: str,
) -> None:
    """Draw 21-point hand skeleton with coloured finger groups."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            color = _connection_color(a, b)
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # Draw landmarks
    for i, (cx, cy) in enumerate(pts):
        # Wrist and fingertips slightly larger
        r = 6 if i in (0, 4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 1, cv2.LINE_AA)

    # Handedness label near wrist
    wx, wy = pts[0]
    cv2.putText(
        frame, handedness, (wx + 10, wy + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    output,
    monitor: GestureMonitor,
) -> np.ndarray:
    """Full overlay: skeleton + gesture labels + status bar."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Person bounding box ──
    if output.person_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in output.person_bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis, "Person", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA,
        )

    # ── Hand skeletons ──
    for g in output.gestures:
        if g.hand_landmarks:
            draw_hand_skeleton(vis, g.hand_landmarks, w, h, g.handedness)

    # ── Status bar (top) ──
    bar_h = 80
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    safety_color = (0, 0, 255) if output.safety_triggered else (0, 255, 0)
    safety_text = "STOP" if output.safety_triggered else "SAFE"
    cv2.putText(
        vis, safety_text, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, safety_color, 2, cv2.LINE_AA,
    )

    # Per-hand gesture list
    if output.gestures:
        parts = [
            f"{g.handedness}: {g.gesture_name} ({g.confidence:.0%})"
            for g in output.gestures
        ]
        cv2.putText(
            vis, " | ".join(parts), (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ── Large dominant gesture label (centre) ──
    gesture = output.dominant_gesture
    if gesture and gesture != "None":
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 2.0, 4
        (tw, th), baseline = cv2.getTextSize(gesture, font, scale, thick)
        tx = (w - tw) // 2
        ty = bar_h + th + 30
        # Pill background
        pad = 15
        cv2.rectangle(
            vis,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + baseline + pad),
            (0, 0, 0), -1,
        )
        color = (0, 0, 255) if output.safety_triggered else (0, 255, 0)
        cv2.putText(vis, gesture, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

    # ── Instructions (bottom) ──
    cv2.rectangle(vis, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        vis,
        "STOP: Open_Palm / Pointing_Up  |  RESUME: Thumb_Up / Victory  |  q: quit  r: reset",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return vis


async def run_live(camera_id: int, num_hands: int, use_yolo: bool = False, yolo_model: str = "") -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    config = GestureMonitorConfig(
        enabled=True,
        num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        gesture_hold_frames=3,
        stop_gestures={"Open_Palm", "Pointing_Up"},
        resume_gestures={"Thumb_Up", "Victory"},
        enable_intent_mapping=True,
        use_person_detector=use_yolo,
        yolo_model_path=yolo_model,
    )
    monitor = GestureMonitor(config)

    logger.info(f"Camera {camera_id} opened — YOLO={'ON' if use_yolo else 'OFF'} — press q/ESC to quit, r to reset safety")

    window_name = "Gesture Monitor — Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_gesture = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed")
                await asyncio.sleep(0.01)
                continue

            output = await monitor.update(frame=frame)

            if output and output.is_valid:
                vis = draw_overlay(frame, output, monitor)

                # Log gesture transitions
                cur = output.dominant_gesture
                if cur and cur != "None" and cur != prev_gesture:
                    logger.info(f"New gesture: {cur}")
                prev_gesture = cur
            else:
                vis = frame

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("r"):
                monitor.reset_safety()
                logger.info("Safety state reset")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stats = monitor.get_gesture_statistics()
        if stats:
            logger.info(f"Session stats: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam gesture test")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--num_hands", type=int, default=2, help="Max hands to detect")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO person detection (crop to person)")
    parser.add_argument("--yolo-model", default="", help="Path to YOLO weights (default: GVHMR checkpoint)")
    args = parser.parse_args()
    asyncio.run(run_live(args.camera, args.num_hands, args.yolo, args.yolo_model))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test gesture monitor on a video file and produce annotated output.

Runs the GestureMonitor on each frame, overlays detected gestures,
and pauses (freezes) for 1 second at the first frame of each new gesture.

Usage:
    uv run python scripts/test_gesture_video.py \
        --video demo_data/layup_demo/layup_gesture_demo.mp4
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

from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hand skeleton constants ─────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

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


def _draw_hand_skeleton(
    frame: np.ndarray, landmarks, w: int, h: int, handedness: str,
) -> None:
    """Draw 21-point hand skeleton with coloured finger groups."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], _connection_color(a, b), 2, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(pts):
        r = 6 if i in (0, 4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 1, cv2.LINE_AA)

    wx, wy = pts[0]
    cv2.putText(
        frame, handedness, (wx + 10, wy + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


async def process_video(
    video_path: str,
    output_path: str | None,
    use_yolo: bool = True,
    yolo_model: str = "",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.with_stem(p.stem + "_gesture_overlay"))

    # Ensure .mp4 extension
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    config = GestureMonitorConfig(
        enabled=True,
        num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        gesture_hold_frames=3,  # Require 3 consecutive frames to confirm gesture
        stop_gestures={"Open_Palm", "Pointing_Up"},
        resume_gestures={"Thumb_Up", "Victory"},
        enable_intent_mapping=True,
        use_person_detector=use_yolo,
        yolo_model_path=yolo_model,
    )
    monitor = GestureMonitor(config)

    pause_frames = int(fps)  # 1 second worth of frames
    prev_dominant_gesture: str | None = None
    frame_idx = 0

    logger.info(
        f"Processing {total_frames} frames at {fps:.1f} fps "
        f"({width}x{height}) → {output_path}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = await monitor.update(frame=frame)
        frame_idx += 1

        if output is None or not output.is_valid:
            writer.write(frame)
            continue

        # Draw the annotated frame
        annotated = _draw_overlay(frame, output)

        # Detect new gesture transition
        current_gesture = output.dominant_gesture
        is_new_gesture = (
            current_gesture is not None
            and current_gesture != "None"
            and current_gesture != prev_dominant_gesture
        )

        if is_new_gesture:
            logger.info(
                f"[frame {frame_idx}/{total_frames}] "
                f"New gesture: {current_gesture} (was {prev_dominant_gesture})"
            )
            # Write the freeze frame for 1 second
            for _ in range(pause_frames):
                writer.write(annotated)

        # Always write the current annotated frame
        writer.write(annotated)
        prev_dominant_gesture = current_gesture

        if frame_idx % 100 == 0:
            logger.info(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    logger.info(f"Done. Output saved to {output_path}")


def _draw_overlay(
    frame: np.ndarray,
    output,
) -> np.ndarray:
    """Draw hand skeleton + gesture labels + status bar + person bbox on frame."""
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
            _draw_hand_skeleton(vis, g.hand_landmarks, w, h, g.handedness)

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
        pad = 15
        cv2.rectangle(
            vis,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + baseline + pad),
            (0, 0, 0), -1,
        )
        color = (0, 0, 255) if output.safety_triggered else (0, 255, 0)
        cv2.putText(vis, gesture, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Test gesture monitor on video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path (default: <input>_gesture_overlay.mp4)")
    parser.add_argument("--no-yolo", dest="use_yolo", action="store_false",
                        help="Disable YOLO person detection (run on full frame)")
    parser.add_argument("--yolo-model", default="", help="Path to YOLO weights (default: GVHMR checkpoint)")
    args = parser.parse_args()

    asyncio.run(process_video(args.video, args.output, args.use_yolo, args.yolo_model))


if __name__ == "__main__":
    main()

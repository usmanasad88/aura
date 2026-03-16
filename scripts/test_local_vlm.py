#!/usr/bin/env python3
"""Standalone test for LocalVLMMonitor on a video file.

Reads frames from a video, asks the VLM a question each frame, and
prints the structured response.

Usage::

    # SGLang backend (start server first: ./scripts/start_sglang_server.sh)
    uv run python scripts/test_local_vlm.py --backend sglang

    # SGLang with preview window
    uv run python scripts/test_local_vlm.py --backend sglang --show

    # transformers backend (in-process, slower)
    uv run python scripts/test_local_vlm.py --backend transformers \\
        --model HuggingFaceTB/SmolVLM2-2.2B-Instruct

    # Custom question
    uv run python scripts/test_local_vlm.py --question "Describe what the person is doing"

    # Process only first N frames
    uv run python scripts/test_local_vlm.py --max-frames 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

from aura.monitors.local_vlm_monitor import (
    LocalVLMConfig,
    LocalVLMMonitor,
    LocalVLMOutput,
)

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_VIDEO = str(
    _project_root
    / "demo_data"
    / "layup_demo"
    / "layup_dummy_demo_crop_1080.mp4"
)
DEFAULT_QUESTION = "Is the robot gripper holding anything? Output Nothing if the robot isn't holding anything."


# ── pretty printer ──────────────────────────────────────────────────────────

def print_result(output: LocalVLMOutput, frame_num: int, timestamp: float) -> None:
    p = output.perception
    if p is None:
        print(f"  [frame {frame_num} @ {timestamp:.1f}s] — no result")
        return

    held = [o.name for o in p.objects if o.held_by_human]
    objs = [o.name for o in p.objects]

    print(
        f"\n{'─' * 60}\n"
        f"  Frame {frame_num}  |  {timestamp:.1f}s  |  {output.processing_time_sec:.2f}s inference\n"
        f"  Answer : {p.answer}\n"
        f"  Objects: {objs}\n"
        f"  Held   : {held}\n"
        f"  Conf   : {p.confidence:.2f}\n"
        f"  Scene  : {p.scene_description}\n"
        f"{'─' * 60}"
    )


# ── main loop ───────────────────────────────────────────────────────────────

async def run_test(
    video_path: str,
    question: str,
    show: bool,
    max_frames: int,
    sample_fps: float,
    model_id: str,
    backend: str = "sglang",
    sglang_url: str = "http://localhost:8100/v1",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"Video   : {video_path}")
    print(f"Frames  : {total_frames}  |  FPS: {video_fps:.1f}  |  Duration: {duration:.1f}s")
    print(f"Backend : {backend}")
    print(f"Model   : {model_id}")
    if backend == "sglang":
        print(f"SGLang URL: {sglang_url}")
    print(f"Question: {question}")
    print(f"Sampling at {sample_fps} FPS — ~{int(duration * sample_fps)} queries\n")

    config = LocalVLMConfig(
        backend=backend,
        model_id=model_id,
        sglang_base_url=sglang_url,
        question=question,
        max_image_dimension=512,
        update_rate_hz=sample_fps,
    )
    monitor = LocalVLMMonitor(config)

    # Initialise backend
    print("Connecting to backend...")
    monitor._ensure_backend()
    print("Backend ready.\n")

    frame_interval = int(video_fps / sample_fps)
    frame_idx = 0
    query_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps
        query_count += 1

        output = await monitor.process_frame(frame, question=question)
        print_result(output, frame_idx, timestamp)

        if show:
            vis = LocalVLMMonitor.visualize(frame, output)
            # Resize for display
            h, w = vis.shape[:2]
            if w > 1280:
                scale = 1280 / w
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow("LocalVLM Monitor Test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuit requested.")
                break

        frame_idx += 1
        if 0 < max_frames <= query_count:
            print(f"\nReached --max-frames={max_frames}, stopping.")
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone — processed {query_count} frames.")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test LocalVLMMonitor on a video file",
    )
    parser.add_argument(
        "--video", default=DEFAULT_VIDEO,
        help="Path to video file",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION,
        help="Question to ask the VLM each frame",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show annotated preview window (requires display)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10,
        help="Max frames to process (0 = all)",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=0.5,
        help="How many frames per second to sample from the video",
    )
    parser.add_argument(
        "--backend", default="sglang", choices=["sglang", "transformers"],
        help="Inference backend (default: sglang)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--sglang-url", default="http://localhost:8100/v1",
        help="SGLang server base URL",
    )
    args = parser.parse_args()

    asyncio.run(
        run_test(
            video_path=args.video,
            question=args.question,
            show=args.show,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            model_id=args.model,
            backend=args.backend,
            sglang_url=args.sglang_url,
        )
    )


if __name__ == "__main__":
    main()

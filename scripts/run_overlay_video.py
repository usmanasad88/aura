#!/usr/bin/env python
"""Generate an overlay video with 3D mesh tracking.

Usage:
    PYTHONPATH="" .venv/bin/python scripts/run_overlay_video.py
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
VIDEO = "demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4"
BOTTLE_GLB = "demo_data/layup_demo/mesh3d/bottle.glb"
SCALE_GLB = "demo_data/layup_demo/mesh3d/scale.glb"
ANY6D_ROOT = "third_party/any6d"
OUTPUT = "results/pose_tracking/overlay_output.mp4"
SAVE_DIR = "results/pose_tracking"
SKIP_FRAMES = 100  # Process ~33 frames from a 3371-frame video
DA3_MODEL = "da3-small"


def main():
    from aura.monitors.pose_tracking_monitor import PoseTrackingMonitor
    from aura.utils.config import PoseTrackingConfig

    cfg = PoseTrackingConfig(
        mesh_map={
            "bottle": BOTTLE_GLB,
            "scale": SCALE_GLB,
        },
        any6d_root=ANY6D_ROOT,
        da3_model=DA3_MODEL,
        da3_batch_size=4,
        save_dir=SAVE_DIR,
        render_overlay=True,
        render_axes=True,
        debug_level=0,
        sam3_confidence=0.3,
    )

    monitor = PoseTrackingMonitor(cfg)

    logger.info("Starting overlay video generation...")
    result = monitor.process_video(
        VIDEO,
        output_path=OUTPUT,
        skip_frames=SKIP_FRAMES,
    )

    logger.info("Output video: %s", result)
    logger.info("File size: %.1f MB", os.path.getsize(result) / 1e6)
    monitor.shutdown()
    logger.info("Done.")


if __name__ == "__main__":
    main()

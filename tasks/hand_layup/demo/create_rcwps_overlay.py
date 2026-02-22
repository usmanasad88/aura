#!/usr/bin/env python3
"""Create a rich overlay video from saved RCWPS results + re-run perception for masks.

This reads the results JSON from a previous ``run_integrated_demo`` session,
re-opens the source video, runs perception (SAM3) at 5 FPS for object segmentation
masks, and composites an overlay video with:

  * Base layer – perception visualisation (SAM3 segmentation masks only)
  * Right panel – RCWPS state, step progress, robot instructions (shown on all frames)
  * Bottom bar  – phase progress bar

No Gemini RCWPS calls are made — all intent data comes from the JSON.

Usage:
    python -m tasks.hand_layup.demo.create_rcwps_overlay \
        --results tasks/hand_layup/demo/rcwps_120_results.json

    # Custom output path
    python -m tasks.hand_layup.demo.create_rcwps_overlay \
        --results rcwps_120_results.json \
        --output my_overlay.mp4
"""

import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Objects we want to detect + segment
OBJECTS_OF_INTEREST = [
    "fiberglass sheet", "metal mold", "resin bottle", "hardener bottle",
    "cup", "mixing stick", "weigh scale", "small brush", "medium brush",
    "roller", "person", "hand", "gloves", "table",
]

PHASES = [
    ("init", "initialization"),
    ("prep", "resin_preparation"),
    ("mix", "mixing"),
    ("L1p", "layer_1_placement"),
    ("L1r", "layer_1_resin"),
    ("L2p", "layer_2_placement"),
    ("L2r", "layer_2_resin"),
    ("L3p", "layer_3_placement"),
    ("L3r", "layer_3_resin"),
    ("L4p", "layer_4_placement"),
    ("L4r", "layer_4_resin"),
    ("roll", "consolidation"),
    ("done", "cleanup"),
]

# Preset colors for different object classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
]


def _visualize_masks_only(frame: np.ndarray, perc_output, prompts: List[str]) -> np.ndarray:
    """Visualize only masks (no bounding boxes) from perception output."""
    vis_frame = frame.copy()
    
    # Draw each detected object's mask
    for obj in perc_output.objects:
        # Get color for this object class
        prompt_idx = prompts.index(obj.name) if obj.name in prompts else 0
        color = COLORS[prompt_idx % len(COLORS)]
        
        # Draw mask if available
        if obj.mask is not None:
            colored_mask = np.zeros_like(vis_frame)
            colored_mask[:, :, 0] = obj.mask * (color[0] // 2)
            colored_mask[:, :, 1] = obj.mask * (color[1] // 2)
            colored_mask[:, :, 2] = obj.mask * (color[2] // 2)
            vis_frame = cv2.addWeighted(vis_frame, 1, colored_mask, 0.5, 0)
            
            # Draw label near mask center
            if obj.mask.sum() > 0:
                mask_coords = np.argwhere(obj.mask > 0)
                center_y = int(mask_coords[:, 0].mean())
                center_x = int(mask_coords[:, 1].mean())
                label = f"{obj.name}"
                cv2.putText(vis_frame, label, (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_frame


def _draw_rcwps_panel(vis: np.ndarray, fa: Dict[str, Any], w: int, h: int):
    """Draw the RCWPS state + robot/safety overlay panels on *vis* in-place."""
    margin = 10
    panel_w = 420
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Left panel: RCWPS state ----
    panel_h = 360
    overlay = vis.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + panel_w, margin + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    x, y = margin + 12, margin + 24

    cv2.putText(vis, "AURA - RCWPS Intent Monitor",
                (x, y), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(vis, f"Time: {fa['timestamp_sec']:.1f}s  |  Frame: {fa['frame_num']}",
                (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    phase = fa.get("current_phase", "initialization")
    cv2.putText(vis, f"Phase: {phase.replace('_', ' ').title()}",
                (x, y), font, 0.50, (0, 230, 0), 1, cv2.LINE_AA)
    y += 22
    action = fa.get("current_action", "idle")
    cv2.putText(vis, f"Action: {action.replace('_', ' ')}",
                (x, y), font, 0.50, (0, 230, 255), 1, cv2.LINE_AA)
    y += 22
    human_state = fa.get("human_state", "idle")
    cv2.putText(vis, f"Human: {human_state.replace('_', ' ')}",
                (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += 24

    # Layer boxes
    cv2.putText(vis, "Layer Progress:",
                (x, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    layers_placed = fa.get("layers_placed", 0)
    layers_resined = fa.get("layers_resined", 0)
    for ln in range(1, 5):
        lx = x + (ln - 1) * 95
        placed = layers_placed >= ln
        resined = layers_resined >= ln
        bw, bh = 85, 22
        if placed and resined:
            cv2.rectangle(vis, (lx, y), (lx + bw, y + bh), (0, 180, 0), -1)
            lbl, lc = f"L{ln} OK", (255, 255, 255)
        elif placed:
            cv2.rectangle(vis, (lx, y), (lx + bw, y + bh), (0, 180, 255), -1)
            lbl, lc = f"L{ln} dry", (0, 0, 0)
        else:
            cv2.rectangle(vis, (lx, y), (lx + bw, y + bh), (80, 80, 80), -1)
            lbl, lc = f"L{ln} --", (180, 180, 180)
        cv2.putText(vis, lbl, (lx + 5, y + 16), font, 0.38, lc, 1, cv2.LINE_AA)
    y += 30

    mix_icon = "[x]" if fa.get("mixture_mixed") else "[ ]"
    cons_icon = "[x]" if fa.get("consolidated") else "[ ]"
    cv2.putText(vis, f"Mixed: {mix_icon}  Consolidated: {cons_icon}",
                (x, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22

    comp = len(fa.get("steps_completed", []))
    prog = len(fa.get("steps_in_progress", []))
    pend = len(fa.get("steps_pending", []))
    cv2.putText(vis, f"Steps  Done: {comp}  Prog: {prog}  Pend: {pend}",
                (x, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y += 22

    next_action = fa.get("predicted_next_action", "unknown")
    cv2.putText(vis, f"Next: {next_action.replace('_', ' ')}",
                (x, y), font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    y += 22

    gt_action = fa.get("gt_action") or "N/A"
    cv2.putText(vis, f"GT: {gt_action.replace('_', ' ')}",
                (x, y), font, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
    y += 22

    # Reasoning text (word-wrap)
    reasoning = fa.get("intent_reasoning", "")
    if reasoning:
        words = reasoning.split()
        line = ""
        for wd in words:
            if len(line + " " + wd) > 55:
                cv2.putText(vis, line, (x, y), font, 0.35,
                            (180, 180, 180), 1, cv2.LINE_AA)
                y += 16
                line = wd
            else:
                line = (line + " " + wd).strip()
        if line:
            cv2.putText(vis, line, (x, y), font, 0.35,
                        (180, 180, 180), 1, cv2.LINE_AA)

    # ---- Right panel: Robot + Safety ----
    rpx = w - margin - panel_w
    rph = 200
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (rpx, margin),
                  (rpx + panel_w, margin + rph), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.75, vis, 0.25, 0, vis)

    rx, ry = rpx + 12, margin + 24
    cv2.putText(vis, "Robot Assistant",
                (rx, ry), font, 0.55, (255, 150, 50), 1, cv2.LINE_AA)
    ry += 24
    robot_instr = fa.get("robot_instruction")
    if robot_instr:
        cv2.putText(vis, f"CMD: {robot_instr}",
                    (rx, ry), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        ry += 18
        reason = fa.get("robot_instruction_reasoning", "")
        if reason:
            cv2.putText(vis, reason[:55],
                        (rx + 8, ry), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            ry += 18
    else:
        cv2.putText(vis, "Status: Standby",
                    (rx, ry), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        ry += 22

    ry += 6
    alerts = fa.get("safety_alerts", [])
    has_alerts = bool(alerts)
    cv2.putText(vis, "Safety Monitor",
                (rx, ry), font, 0.50,
                (0, 0, 255) if has_alerts else (0, 230, 0), 1, cv2.LINE_AA)
    ry += 22
    if alerts:
        for alert in alerts[:3]:
            ac = (0, 0, 255) if "CRITICAL" in alert else (0, 165, 255)
            cv2.putText(vis, alert[:55], (rx, ry), font, 0.35, ac, 1, cv2.LINE_AA)
            ry += 16
    else:
        cv2.putText(vis, "No alerts", (rx, ry), font, 0.38,
                    (0, 230, 0), 1, cv2.LINE_AA)

    # ---- Detected objects list (right side, below robot panel) ----
    obj_panel_y = margin + rph + 10
    detected = fa.get("detected_objects", [])
    if detected:
        obj_panel_h = 30 + min(len(detected), 10) * 18
        overlay_obj = vis.copy()
        cv2.rectangle(overlay_obj, (rpx, obj_panel_y),
                      (rpx + panel_w, obj_panel_y + obj_panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay_obj, 0.75, vis, 0.25, 0, vis)

        oy = obj_panel_y + 20
        cv2.putText(vis, f"Detected Objects ({len(detected)})",
                    (rpx + 12, oy), font, 0.45, (200, 200, 0), 1, cv2.LINE_AA)
        oy += 20
        for obj_name in detected[:10]:
            cv2.putText(vis, f"  • {obj_name}",
                        (rpx + 12, oy), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            oy += 18

    # ---- Bottom: phase progress bar ----
    bar_h = 50
    by = h - margin - bar_h
    bw_total = w - 2 * margin
    overlay3 = vis.copy()
    cv2.rectangle(overlay3, (margin, by),
                  (margin + bw_total, by + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay3, 0.75, vis, 0.25, 0, vis)

    n_phases = len(PHASES)
    seg_w = (bw_total - 16) // n_phases
    found_cur = False
    for pi, (plbl, pid) in enumerate(PHASES):
        px = margin + 8 + pi * seg_w
        py = by + 8
        if pid == phase:
            pc, tc, found_cur = (0, 255, 100), (0, 0, 0), True
        elif not found_cur:
            pc, tc = (60, 140, 60), (220, 220, 220)
        else:
            pc, tc = (80, 80, 80), (150, 150, 150)
        cv2.rectangle(vis, (px, py), (px + seg_w - 4, py + bar_h - 20), pc, -1)
        cv2.putText(vis, plbl, (px + 3, py + bar_h - 28),
                    font, 0.30, tc, 1, cv2.LINE_AA)

    # Gemini time badge
    gemini_t = fa.get("gemini_time_sec", 0)
    badge = f"Gemini: {gemini_t:.1f}s"
    cv2.putText(vis, badge, (w - 180, h - margin - bar_h - 10),
                font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)


async def create_overlay_video(
    results_path: str,
    output_path: Optional[str] = None,
    use_sam3: bool = True,
):
    """Build the overlay video.

    Args:
        results_path: Path to the ``rcwps_*_results.json``.
        output_path: Where to write the MP4. Auto-derived if None.
        use_sam3: Run SAM3 segmentation for masks (True) or Gemini-only boxes.
    """
    # ---- Load results ----
    with open(results_path) as f:
        results = json.load(f)

    video_path = results["video_path"]
    frame_analyses: List[Dict[str, Any]] = results["frame_analyses"]
    logger.info(f"Loaded {len(frame_analyses)} frame analyses from {results_path}")
    logger.info(f"Source video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {vid_w}x{vid_h} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Build lookup: frame_num -> analysis
    analysis_by_frame = {fa["frame_num"]: fa for fa in frame_analyses}
    logger.info(f"Analyzed frames: {len(analysis_by_frame)} of {total_frames}")

    # ---- Init perception module ----
    from aura.monitors.perception_module import PerceptionModule
    from aura.utils.config import PerceptionConfig
    from aura.core import MonitorType

    perception = PerceptionModule(PerceptionConfig(
        monitor_type=MonitorType.PERCEPTION,
        enabled=True,
        use_sam3=use_sam3,
        use_gemini_detection=True,
        default_prompts=OBJECTS_OF_INTEREST,
    ))
    logger.info(f"Perception module ready (SAM3={use_sam3})")

    # ---- Output path ----
    if output_path is None:
        output_path = str(TASK_DIR / "demo" / "hand_layup_full_overlay.mp4")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))
    logger.info(f"Writing overlay video to: {output_path}")
    logger.info(f"Running SAM3 at 5 FPS (every 6 frames), RCWPS panel on all frames")

    # ---- Process ALL frames from source video ----
    # Run SAM3 at 5 FPS: video is 30 FPS, so every 6 frames
    sam3_interval = int(fps / 5.0)
    logger.info(f"SAM3 interval: every {sam3_interval} frames")
    
    last_analysis = None  # Track last RCWPS analysis for panel display
    last_perc_vis = None  # Cache last perception visualization
    processed_count = 0
    sam3_count = 0
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_num}")
            break
        
        # Update last analysis if we have one for this frame
        fa = analysis_by_frame.get(frame_num)
        if fa is not None:
            last_analysis = fa
            processed_count += 1
        
        # Run SAM3 at 5 FPS
        if frame_num % sam3_interval == 0:
            sam3_count += 1
            try:
                perc_output = await perception.process_frame(frame)
                if perc_output:
                    last_perc_vis = _visualize_masks_only(frame, perc_output, OBJECTS_OF_INTEREST)
                else:
                    last_perc_vis = frame.copy()
            except Exception as e:
                logger.warning(f"Perception failed on frame {frame_num}: {e}")
                last_perc_vis = frame.copy()
            
            if sam3_count % 10 == 0:
                logger.info(f"  SAM3 processed: {sam3_count} frames")
        
        # Use last perception vis or original frame
        if last_perc_vis is not None:
            vis = last_perc_vis.copy()
        else:
            vis = frame.copy()
        
        # Draw RCWPS panel on ALL frames using last available analysis
        if last_analysis is not None:
            _draw_rcwps_panel(vis, last_analysis, vid_w, vid_h)

        writer.write(vis)
        
        if frame_num % 500 == 0:
            phase_str = last_analysis['current_phase'] if last_analysis else 'no data'
            logger.info(
                f"  Frame {frame_num}/{total_frames} – Phase: {phase_str} "
                f"(SAM3: {sam3_count}, Analyses: {processed_count})"
            )

    writer.release()
    cap.release()
    logger.info(
        f"✅ Overlay video saved: {output_path} "
        f"({total_frames} total frames, {sam3_count} SAM3 runs, {processed_count} RCWPS analyses)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create overlay video from saved RCWPS results + SAM3 perception"
    )
    parser.add_argument(
        "--results", "-r", type=str, required=True,
        help="Path to rcwps_*_results.json from run_integrated_demo",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output MP4 path (default: auto)",
    )
    parser.add_argument(
        "--no-sam3", action="store_true",
        help="Skip SAM3 masks, use Gemini bounding boxes only",
    )
    args = parser.parse_args()

    asyncio.run(create_overlay_video(
        results_path=args.results,
        output_path=args.output,
        use_sam3=not args.no_sam3,
    ))


if __name__ == "__main__":
    main()

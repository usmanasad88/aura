#!/usr/bin/env python3
"""Create visualization overlay video for Hand Layup task.

Overlays key module outputs on the input video:
- Task phase and current action
- Layer progress (placed / resined)
- Object detections
- Robot instructions (what the robot would do)
- Safety alerts (gloves, shelf life)
- Task DAG progress bar

Usage:
    # Basic overlay with ground truth timeline
    python -m tasks.hand_layup.demo.create_overlay_video

    # With custom output path
    python -m tasks.hand_layup.demo.create_overlay_video --output my_overlay.mp4

    # Overlay from previously computed results JSON
    python -m tasks.hand_layup.demo.create_overlay_video --results results.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Color scheme
# =============================================================================

COLORS = {
    "bg": (40, 40, 40),
    "bg_alpha": 0.7,
    "text_white": (255, 255, 255),
    "text_green": (0, 230, 0),
    "text_yellow": (0, 230, 255),
    "text_cyan": (255, 200, 0),
    "text_orange": (0, 165, 255),
    "text_red": (0, 0, 255),
    "progress_bg": (80, 80, 80),
    "progress_fill": (0, 200, 0),
    "progress_resin": (255, 180, 0),
    "robot_action": (255, 150, 50),
    "safety_warning": (0, 165, 255),
    "safety_critical": (0, 0, 255),
    "phase_active": (0, 255, 100),
    "phase_done": (100, 100, 100),
}

# Task phases in order (for progress bar)
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


def draw_rounded_rect(img, pt1, pt2, color, alpha=0.7, radius=8):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_text_with_bg(img, text, pos, font_scale=0.55, color=(255, 255, 255),
                       bg_color=(40, 40, 40), thickness=1, padding=4):
    """Draw text with a background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    draw_rounded_rect(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color, alpha=0.7
    )
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 2 * padding


class OverlayRenderer:
    """Renders overlay panels on video frames."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Layout: left panel for state, right panel for robot, bottom for progress
        self.panel_margin = 10
        self.left_panel_w = 380
        self.right_panel_w = 380
        self.bottom_panel_h = 60
    
    def render(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_num: int,
        gt_event: Optional[Dict[str, Any]],
        current_phase: str,
        layers_placed: int,
        layers_resined: int,
        mixture_mixed: bool,
        consolidated: bool,
        robot_instruction: Optional[str],
        robot_instruction_reason: str,
        safety_alerts: List[str],
        shelf_life_min: Optional[float],
        detected_objects: List[str],
        active_robot_instructions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Render all overlay panels on a frame."""
        vis = frame.copy()
        
        # --- Left Panel: Task State ---
        self._draw_left_panel(
            vis, timestamp, frame_num, gt_event,
            current_phase, layers_placed, layers_resined,
            mixture_mixed, consolidated, detected_objects,
        )
        
        # --- Right Panel: Robot & Safety ---
        self._draw_right_panel(
            vis, robot_instruction, robot_instruction_reason,
            safety_alerts, shelf_life_min, active_robot_instructions,
        )
        
        # --- Bottom Panel: Phase Progress Bar ---
        self._draw_progress_bar(vis, current_phase)
        
        return vis
    
    def _draw_left_panel(
        self, vis, timestamp, frame_num, gt_event,
        current_phase, layers_placed, layers_resined,
        mixture_mixed, consolidated, detected_objects,
    ):
        m = self.panel_margin
        pw = self.left_panel_w
        
        # Background
        draw_rounded_rect(vis, (m, m), (m + pw, 300), COLORS["bg"], alpha=0.75)
        
        y = m + 22
        x = m + 10
        
        # Title
        cv2.putText(vis, "AURA - Hand Layup Monitor",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text_cyan"], 1, cv2.LINE_AA)
        y += 28
        
        # Time
        cv2.putText(vis, f"Time: {timestamp:.1f}s  |  Frame: {frame_num}",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_white"], 1, cv2.LINE_AA)
        y += 24
        
        # Phase
        cv2.putText(vis, f"Phase: {current_phase.replace('_', ' ').title()}",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_green"], 1, cv2.LINE_AA)
        y += 24
        
        # GT Action
        gt_text = gt_event.get("action", "N/A") if gt_event else "N/A"
        cv2.putText(vis, f"Action: {gt_text.replace('_', ' ')}",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_yellow"], 1, cv2.LINE_AA)
        y += 28
        
        # Layer progress
        cv2.putText(vis, "Layer Progress:",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_white"], 1, cv2.LINE_AA)
        y += 22
        
        for layer_num in range(1, 5):
            lx = x + (layer_num - 1) * 85
            placed = layers_placed >= layer_num
            resined = layers_resined >= layer_num
            
            # Draw layer box
            box_w, box_h = 75, 22
            if placed and resined:
                cv2.rectangle(vis, (lx, y), (lx + box_w, y + box_h), (0, 180, 0), -1)
                label = f"L{layer_num} OK"
                label_color = (255, 255, 255)
            elif placed:
                cv2.rectangle(vis, (lx, y), (lx + box_w, y + box_h), (0, 180, 255), -1)
                label = f"L{layer_num} dry"
                label_color = (0, 0, 0)
            else:
                cv2.rectangle(vis, (lx, y), (lx + box_w, y + box_h), (80, 80, 80), -1)
                label = f"L{layer_num} --"
                label_color = (180, 180, 180)
            
            cv2.putText(vis, label, (lx + 5, y + 16),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1, cv2.LINE_AA)
        
        y += 32
        
        # Status flags
        mix_icon = "[x]" if mixture_mixed else "[ ]"
        cons_icon = "[x]" if consolidated else "[ ]"
        cv2.putText(vis, f"Mixed: {mix_icon}  Consolidated: {cons_icon}",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["text_white"], 1, cv2.LINE_AA)
        y += 24
        
        # Detected objects (compact)
        if detected_objects:
            obj_text = ", ".join(detected_objects[:5])
            if len(detected_objects) > 5:
                obj_text += f" +{len(detected_objects)-5}"
            cv2.putText(vis, f"Objects: {obj_text}",
                         (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_white"], 1, cv2.LINE_AA)
    
    def _draw_right_panel(
        self, vis, robot_instruction, robot_instruction_reason,
        safety_alerts, shelf_life_min, active_robot_instructions,
    ):
        m = self.panel_margin
        pw = self.right_panel_w
        rx = self.width - m - pw
        
        panel_h = 200
        draw_rounded_rect(vis, (rx, m), (rx + pw, m + panel_h), COLORS["bg"], alpha=0.75)
        
        y = m + 22
        x = rx + 10
        
        # Robot section
        cv2.putText(vis, "Robot Assistant",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["robot_action"], 1, cv2.LINE_AA)
        y += 26
        
        if robot_instruction:
            cv2.putText(vis, f"CMD: {robot_instruction}",
                         (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["text_white"], 1, cv2.LINE_AA)
            y += 20
            # Wrap reason text
            reason_words = robot_instruction_reason.split()
            line = ""
            for w in reason_words:
                if len(line + " " + w) > 45:
                    cv2.putText(vis, line, (x + 10, y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                    y += 16
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                cv2.putText(vis, line, (x + 10, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                y += 20
        else:
            cv2.putText(vis, "Status: Standby (no action needed)",
                         (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            y += 24
        
        # All robot instructions issued so far
        if active_robot_instructions:
            cv2.putText(vis, f"Instructions issued: {len(active_robot_instructions)}",
                         (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_white"], 1, cv2.LINE_AA)
            y += 20
        
        # Safety section
        y += 5
        cv2.putText(vis, "Safety Monitor",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     COLORS["safety_critical"] if safety_alerts else COLORS["text_green"], 1, cv2.LINE_AA)
        y += 22
        
        if shelf_life_min is not None:
            sl_color = COLORS["text_green"]
            if shelf_life_min > 20:
                sl_color = COLORS["safety_warning"]
            if shelf_life_min > 28:
                sl_color = COLORS["safety_critical"]
            cv2.putText(vis, f"Shelf life: {shelf_life_min:.1f} / 30.0 min",
                         (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, sl_color, 1, cv2.LINE_AA)
            y += 18
        
        for alert in safety_alerts[:3]:
            color = COLORS["safety_critical"] if "CRITICAL" in alert else COLORS["safety_warning"]
            # Truncate long alerts
            display_alert = alert[:50] + "..." if len(alert) > 50 else alert
            cv2.putText(vis, display_alert, (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            y += 16
        
        if not safety_alerts:
            cv2.putText(vis, "No alerts", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_green"], 1, cv2.LINE_AA)
    
    def _draw_progress_bar(self, vis, current_phase):
        """Draw phase progress bar at bottom."""
        m = self.panel_margin
        bh = self.bottom_panel_h
        by = self.height - m - bh
        bw = self.width - 2 * m
        
        draw_rounded_rect(vis, (m, by), (m + bw, by + bh), COLORS["bg"], alpha=0.75)
        
        # Phase boxes
        n = len(PHASES)
        box_w = (bw - 20) // n
        bx = m + 10
        
        found_current = False
        for i, (label, phase_id) in enumerate(PHASES):
            x = bx + i * box_w
            y = by + 8
            
            if phase_id == current_phase:
                color = COLORS["phase_active"]
                found_current = True
                text_color = (0, 0, 0)
            elif not found_current:
                color = (60, 140, 60)  # completed
                text_color = (220, 220, 220)
            else:
                color = COLORS["phase_done"]
                text_color = (150, 150, 150)
            
            cv2.rectangle(vis, (x, y), (x + box_w - 4, y + bh - 20), color, -1)
            cv2.putText(vis, label, (x + 4, y + bh - 28),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1, cv2.LINE_AA)


def create_overlay_video(
    video_path: str,
    output_path: str,
    ground_truth_path: str,
    results_path: Optional[str] = None,
    frame_skip: int = 1,
):
    """Create overlay video.
    
    Args:
        video_path: Input video path
        output_path: Output overlay video path
        ground_truth_path: Ground truth JSON path
        results_path: Optional pre-computed results JSON
        frame_skip: Write every Nth frame (1 = all frames)
    """
    # Load ground truth
    gt_events = []
    if Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            data = json.load(f)
            gt_events = data.get("events", [])
    
    # Load pre-computed results if available
    frame_results = {}
    if results_path and Path(results_path).exists():
        with open(results_path) as f:
            results_data = json.load(f)
            for fa in results_data.get("frame_analyses", []):
                frame_results[fa["frame_num"]] = fa
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Input: {video_path}")
    logger.info(f"  {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / frame_skip
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    
    renderer = OverlayRenderer(width, height)
    
    # Task state tracking from ground truth
    ACTION_TO_STATE = {
        "recording_start": ("initialization", 0, 0, False, False),
        "place_cup_on_scale": ("resin_preparation", 0, 0, False, False),
        "add_resin": ("resin_preparation", 0, 0, False, False),
        "add_hardener": ("resin_preparation", 0, 0, False, False),
        "weigh_mixture": ("resin_preparation", 0, 0, False, False),
        "mix_resin_hardener": ("mixing", 0, 0, True, False),
        "place_layer_1": ("layer_1_placement", 1, 0, True, False),
        "apply_resin_layer_1": ("layer_1_resin", 1, 1, True, False),
        "place_layer_2": ("layer_2_placement", 2, 1, True, False),
        "apply_resin_layer_2": ("layer_2_resin", 2, 2, True, False),
        "place_layer_3": ("layer_3_placement", 3, 2, True, False),
        "apply_resin_layer_3": ("layer_3_resin", 3, 3, True, False),
        "place_layer_4": ("layer_4_placement", 4, 3, True, False),
        "apply_resin_layer_4": ("layer_4_resin", 4, 4, True, False),
        "consolidate_with_roller": ("consolidation", 4, 4, True, True),
        "cleanup": ("cleanup", 4, 4, True, True),
        "task_complete": ("complete", 4, 4, True, True),
    }
    
    current_phase = "initialization"
    layers_placed = 0
    layers_resined = 0
    mixture_mixed = False
    consolidated = False
    shelf_life_start = None
    robot_instructions_so_far = []
    
    written = 0
    for frame_num in range(0, total_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps
        
        # Get ground truth event
        gt_event = None
        for event in gt_events:
            if event.get("timestamp", 0) <= timestamp:
                gt_event = event
            else:
                break
        
        # Update state from GT
        if gt_event:
            action = gt_event.get("action", "")
            if action in ACTION_TO_STATE:
                current_phase, layers_placed, layers_resined, mixture_mixed, consolidated = (
                    ACTION_TO_STATE[action]
                )
                if action == "mix_resin_hardener" and shelf_life_start is None:
                    shelf_life_start = timestamp
        
        # Robot instruction
        robot_instruction = None
        robot_reason = ""
        if gt_event and gt_event.get("robot_action"):
            robot_instruction = gt_event["robot_action"]
            robot_reason = gt_event.get("robot_action_reason", "")
            if robot_instruction not in [ri.get("action") for ri in robot_instructions_so_far]:
                robot_instructions_so_far.append({
                    "action": robot_instruction,
                    "timestamp": timestamp,
                })
        
        # Safety
        safety_alerts = []
        shelf_life_min = None
        if shelf_life_start is not None:
            shelf_life_min = (timestamp - shelf_life_start) / 60.0
            if shelf_life_min > 30:
                safety_alerts.append(f"CRITICAL: Shelf life exceeded ({shelf_life_min:.0f} min)")
            elif shelf_life_min > 20:
                safety_alerts.append(f"WARNING: Shelf life at {shelf_life_min:.0f} min")
        
        # Use pre-computed results if available
        detected_objects = []
        if frame_num in frame_results:
            fr = frame_results[frame_num]
            detected_objects = fr.get("detected_objects", [])
        
        # Render overlay
        vis = renderer.render(
            frame=frame,
            timestamp=timestamp,
            frame_num=frame_num,
            gt_event=gt_event,
            current_phase=current_phase,
            layers_placed=layers_placed,
            layers_resined=layers_resined,
            mixture_mixed=mixture_mixed,
            consolidated=consolidated,
            robot_instruction=robot_instruction,
            robot_instruction_reason=robot_reason,
            safety_alerts=safety_alerts,
            shelf_life_min=shelf_life_min,
            detected_objects=detected_objects,
            active_robot_instructions=robot_instructions_so_far,
        )
        
        out.write(vis)
        written += 1
        
        if written % 100 == 0:
            logger.info(f"Written {written} frames ({timestamp:.1f}s)")
    
    cap.release()
    out.release()
    
    logger.info(f"Overlay video saved: {output_path}")
    logger.info(f"  Written {written} frames at {out_fps:.1f} FPS")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create visualization overlay video for hand layup task"
    )
    parser.add_argument(
        "--video", "-v", type=str,
        default=str(AURA_ROOT / "demo_data" / "layup_demo" / "layup_dummy_demo_crop_1080.mp4"),
        help="Input video path"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(TASK_DIR / "demo" / "hand_layup_overlay.mp4"),
        help="Output overlay video path"
    )
    parser.add_argument(
        "--ground-truth", "-gt", type=str,
        default=str(TASK_DIR / "config" / "ground_truth.json"),
        help="Path to ground_truth.json"
    )
    parser.add_argument(
        "--results", "-r", type=str,
        default=None,
        help="Path to pre-computed results JSON (from run_integrated_demo.py)"
    )
    parser.add_argument(
        "--frame-skip", "-s", type=int,
        default=1,
        help="Write every Nth frame (default: 1 = all frames)"
    )
    
    args = parser.parse_args()
    
    create_overlay_video(
        video_path=args.video,
        output_path=args.output,
        ground_truth_path=args.ground_truth,
        results_path=args.results,
        frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()

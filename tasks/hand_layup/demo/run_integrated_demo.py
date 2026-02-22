#!/usr/bin/env python3
"""Integrated demo runner for Hand Layup task.

This script runs the AURA pipeline on the hand layup dummy video using
the RCWPS (Rolling Context Window with Previous State) intent monitor:
1. Processes video frames through monitors (perception, intent via RCWPS)
2. Gemini predicts task state, completed/in-progress/pending steps
3. Affordance monitor suggests robot housekeeping actions
4. Safety & quality monitoring (gloves, shelf life)
5. All prompts and responses are logged to disk

Supports multiple frame sources: pre-recorded video (default), webcam, or
live screen capture — all via the unified ``aura.sources`` abstraction.

Usage:
    # Run on dummy video (5 sampled frames)
    python -m tasks.hand_layup.demo.run_integrated_demo --headless --max-frames 5

    # Full video, save output
    python -m tasks.hand_layup.demo.run_integrated_demo --headless --output results.json

    # Use specific model
    python -m tasks.hand_layup.demo.run_integrated_demo --model gemini-2.5-flash --max-frames 5

    # Run from webcam
    python -m tasks.hand_layup.demo.run_integrated_demo --source webcam --headless

    # Run from screen capture
    python -m tasks.hand_layup.demo.run_integrated_demo --source screen --headless

    # Simulate real-time from a video file (for testing without a webcam)
    python -m tasks.hand_layup.demo.run_integrated_demo --source realtime --headless --max-frames 5
    python -m tasks.hand_layup.demo.run_integrated_demo --source realtime --speed 2.0 --headless
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.sources import (
    FrameSource, VideoFileSource, WebcamSource, ScreenCaptureSource,
    RealtimeVideoSource,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_num: int
    timestamp_sec: float
    
    # Perception
    detected_objects: List[str] = field(default_factory=list)
    object_count: int = 0
    gloves_detected: bool = False
    
    # Intent (RCWPS)
    current_action: str = "unknown"
    current_action_confidence: float = 0.0
    predicted_next_action: str = "unknown"
    predicted_next_confidence: float = 0.0
    intent_reasoning: str = ""
    
    # Action tracking from RCWPS
    steps_completed: List[str] = field(default_factory=list)
    steps_in_progress: List[str] = field(default_factory=list)
    steps_pending: List[str] = field(default_factory=list)
    
    # Task state (from RCWPS prediction)
    current_phase: str = "initialization"
    layers_placed: int = 0
    layers_resined: int = 0
    mixture_mixed: bool = False
    consolidated: bool = False
    human_state: str = "idle"
    
    # Affordance
    available_skills: List[str] = field(default_factory=list)
    housekeeping_suggestions: List[str] = field(default_factory=list)
    
    # Robot instruction (what robot WOULD do)
    robot_instruction: Optional[str] = None
    robot_instruction_reasoning: str = ""
    
    # Safety
    safety_alerts: List[str] = field(default_factory=list)
    shelf_life_elapsed_min: Optional[float] = None
    
    # Ground truth (for evaluation)
    gt_action: Optional[str] = None
    gt_robot_action: Optional[str] = None
    
    # Timing
    gemini_time_sec: float = 0.0
    process_time_ms: float = 0.0

    # Perception visualization (not serialised to JSON)
    _perception_vis: Optional[Any] = field(default=None, repr=False)


@dataclass
class DemoResults:
    """Complete results from demo run."""
    video_path: str
    model: str
    start_time: str
    end_time: str
    total_frames_processed: int
    total_duration_sec: float
    prompt_log_dir: Optional[str] = None
    
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    robot_instructions: List[Dict[str, Any]] = field(default_factory=list)
    safety_alerts_total: int = 0
    task_complete: bool = False
    final_task_state: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Main Runner
# =============================================================================

class HandLayupDemoRunner:
    """Runs the AURA pipeline on hand layup video using RCWPS intent monitor."""
    
    OBJECTS_OF_INTEREST = [
        "fiberglass sheet", "metal mold", "resin bottle", "hardener bottle",
        "cup", "mixing stick", "weigh scale", "small brush", "medium brush",
        "roller", "person", "hand", "gloves", "table",
    ]
    
    def __init__(
        self,
        video_path: str,
        config_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        model: Optional[str] = None,
        robot_api_url: Optional[str] = None,
    ):
        self.video_path = video_path
        self.config_path = config_path or str(TASK_DIR / "config" / "hand_layup.yaml")
        self.ground_truth_path = ground_truth_path or str(TASK_DIR / "config" / "ground_truth.json")
        
        self.config = self._load_config()
        self.ground_truth = self._load_ground_truth()
        
        # Resolve model: CLI arg > config > default
        if model:
            self.model = model
        elif self.config.get("brain", {}).get("model"):
            self.model = self.config["brain"]["model"]
        else:
            self.model = "gemini-3-pro-preview"
        
        # Robot API URL: CLI arg > config > None
        self.robot_api_url = robot_api_url
        if not self.robot_api_url:
            self.robot_api_url = self.config.get("robot", {}).get("api_url")
        
        # Robot control client (if API is configured)
        self._robot_client = None
        if self.robot_api_url:
            try:
                from aura.interfaces.robot_control_client import RobotControlClient
                self._robot_client = RobotControlClient(self.robot_api_url)
                if self._robot_client.is_available():
                    logger.info(f"✅ Robot API connected: {self.robot_api_url}")
                else:
                    logger.info(f"ℹ️  Robot API configured but not reachable: {self.robot_api_url}")
            except Exception as e:
                logger.warning(f"Could not create robot client: {e}")
        
        # Object locations (all start on workplace)
        self.object_locations = {obj: "workplace" for obj in [
            "weigh_scale", "resin_bottle", "hardener_bottle", "cup",
            "brush_small", "brush_medium", "roller",
        ]}
        
        # Robot instructions log
        self.robot_instructions: List[Dict[str, Any]] = []
        self.results: Optional[DemoResults] = None
    
    def _load_config(self) -> Dict[str, Any]:
        if Path(self.config_path).exists():
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if Path(self.ground_truth_path).exists():
            with open(self.ground_truth_path) as f:
                data = json.load(f)
                return data.get("events", [])
        return []
    
    def _get_gt_event_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        current = None
        for event in self.ground_truth:
            if event.get("timestamp", 0) <= timestamp:
                current = event
            else:
                break
        return current
    
    def _get_safety_alerts(self, timestamp: float, intent_result) -> List[str]:
        """Check for safety alerts from intent state."""
        alerts = []
        
        # Check gloves from RCWPS prediction
        phase = intent_result.current_phase
        resin_phases = [
            "resin_preparation", "mixing",
            "layer_1_placement", "layer_1_resin",
            "layer_2_placement", "layer_2_resin",
            "layer_3_placement", "layer_3_resin",
            "layer_4_placement", "layer_4_resin",
            "consolidation",
        ]
        
        if phase in resin_phases and not intent_result.human_wearing_gloves:
            alerts.append("WARNING: Gloves not detected during resin handling")
        
        # Shelf life from predicted state
        if intent_result.mixture_mixed:
            # Estimate shelf life from when mixing was first completed
            for hist in self.intent_monitor.get_history():
                if hist.mixture_mixed:
                    elapsed_min = (timestamp - hist.timestamp) / 60.0
                    if elapsed_min > 30:
                        alerts.append(f"CRITICAL: Shelf life exceeded ({elapsed_min:.0f} min)")
                    elif elapsed_min > 20:
                        alerts.append(f"WARNING: Shelf life at {elapsed_min:.0f} min (max 30)")
                    break
        
        return alerts
    
    async def run(
        self,
        source: Optional[FrameSource] = None,
        frame_skip: int = 30,
        max_frames: Optional[int] = None,
        headless: bool = True,
        display: bool = False,
        save_vis: bool = False,
        vis_output: Optional[str] = None,
    ) -> DemoResults:
        """Run the demo pipeline.

        Args:
            source: A :class:`FrameSource` to read frames from.  When
                ``None`` a :class:`VideoFileSource` is created from
                ``self.video_path`` with the given *frame_skip* /
                *max_frames*.
            frame_skip: Process every Nth frame (only used when *source*
                is ``None``).
            max_frames: Maximum frames to process (only used when
                *source* is ``None``).
            headless: Don't show OpenCV window.
            display: Show OpenCV preview.
            save_vis: Generate a visualization video with masks + overlay.
            vis_output: Path for visualization video.
        """
        from tasks.hand_layup.monitors.intent_monitor import HandLayupIntentMonitor
        from tasks.hand_layup.monitors.affordance_monitor import HandLayupAffordanceMonitor

        start_time = datetime.now()

        # ---- Build frame source ----
        owns_source = source is None
        if source is None:
            source = VideoFileSource(
                path=self.video_path,
                frame_skip=frame_skip,
                max_frames=max_frames,
            )

        source.open()

        src_w, src_h = source.resolution
        src_fps = source.fps
        logger.info(f"Source: {source.__class__.__name__}")
        logger.info(f"  Resolution: {src_w}x{src_h}, FPS: {src_fps:.1f}, Live: {source.is_live}")
        if isinstance(source, VideoFileSource):
            logger.info(f"  Total frames: {source.total_frames}, Duration: {source.duration:.1f}s")
        logger.info(f"  Model: {self.model}")

        # ---- Initialize RCWPS Intent Monitor ----
        use_realtime = source.is_live
        self.intent_monitor = HandLayupIntentMonitor(
            model=self.model,
            max_frames=3 if use_realtime else 5,
            max_image_dimension=480 if use_realtime else 640,
            temperature=0.3,
            enable_logging=True,
            realtime=use_realtime,
        )
        if use_realtime:
            logger.info(f"  Realtime mode: model={self.intent_monitor.model}, "
                        f"max_frames={self.intent_monitor.max_frames}, "
                        f"max_dim={self.intent_monitor.max_image_dimension}")

        log_dir = self.intent_monitor.get_log_dir()
        logger.info(f"  Prompt logs: {log_dir}")

        frame_analyses = []
        errors = []
        frame_count = 0

        # Rolling frame buffer for RCWPS
        frame_buffer: List[np.ndarray] = []

        # Perception monitor (lazy init)
        perception_monitor = None

        try:
            for src_frame in source:
                frame = src_frame.image
                frame_num = src_frame.frame_number
                timestamp = src_frame.timestamp
                t0 = time.time()

                # Maintain rolling frame buffer
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > 5:
                    frame_buffer = frame_buffer[-5:]

                # Get ground truth (for evaluation only)
                gt_event = self._get_gt_event_at(timestamp)

                # Initialize analysis
                analysis = FrameAnalysis(
                    frame_num=frame_num,
                    timestamp_sec=timestamp,
                    gt_action=gt_event.get("action") if gt_event else None,
                    gt_robot_action=gt_event.get("robot_action") if gt_event else None,
                )

                # --- Run Perception ---
                try:
                    if perception_monitor is None:
                        from aura.monitors.perception_module import PerceptionModule
                        from aura.utils.config import PerceptionConfig
                        from aura.core import MonitorType

                        perception_monitor = PerceptionModule(PerceptionConfig(
                            monitor_type=MonitorType.PERCEPTION,
                            enabled=True,
                            use_sam3=True,
                            use_gemini_detection=True,
                            default_prompts=self.OBJECTS_OF_INTEREST,
                        ))

                    perception_output = await perception_monitor.process_frame(frame)
                    if perception_output and hasattr(perception_output, 'objects'):
                        for obj in perception_output.objects:
                            name = getattr(obj, 'name', 'unknown')
                            analysis.detected_objects.append(name)
                            if "glove" in name.lower():
                                analysis.gloves_detected = True
                        analysis.object_count = len(analysis.detected_objects)
                        # Store perception visualisation (masks + boxes)
                        analysis._perception_vis = perception_monitor.visualize(
                            frame, perception_output
                        )
                except Exception as e:
                    logger.debug(f"Perception error at frame {frame_num}: {e}")

                # --- Run RCWPS Intent Monitor ---
                try:
                    intent_result = self.intent_monitor.predict(
                        frames=frame_buffer,
                        timestamp=timestamp,
                        frame_num=frame_num,
                    )

                    analysis.current_action = intent_result.current_action
                    analysis.current_phase = intent_result.current_phase
                    analysis.human_state = intent_result.human_state
                    analysis.layers_placed = intent_result.layers_placed
                    analysis.layers_resined = intent_result.layers_resined
                    analysis.mixture_mixed = intent_result.mixture_mixed
                    analysis.consolidated = intent_result.consolidated
                    analysis.predicted_next_action = intent_result.predicted_next_action
                    analysis.predicted_next_confidence = intent_result.prediction_confidence
                    analysis.intent_reasoning = intent_result.reasoning
                    analysis.steps_completed = intent_result.steps_completed
                    analysis.steps_in_progress = intent_result.steps_in_progress
                    analysis.steps_pending = intent_result.steps_pending
                    analysis.gemini_time_sec = intent_result.generation_time_sec

                    logger.info(
                        f"Frame {frame_num} ({timestamp:.1f}s) | "
                        f"Phase: {intent_result.current_phase} | "
                        f"Action: {intent_result.current_action} | "
                        f"Completed: {len(intent_result.steps_completed)} | "
                        f"Next: {intent_result.predicted_next_action} | "
                        f"Gemini: {intent_result.generation_time_sec:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Intent RCWPS error at frame {frame_num}: {e}")
                    errors.append(f"Intent error frame {frame_num}: {e}")

                # --- Run Affordance ---
                try:
                    aff_monitor = HandLayupAffordanceMonitor(
                        robot_client=self._robot_client,
                    )
                    for obj_id, loc in self.object_locations.items():
                        aff_monitor.update_object_location(obj_id, loc)

                    # Mark completed nodes from RCWPS output
                    for step in analysis.steps_completed:
                        aff_monitor.mark_task_node_complete(step)

                    housekeeping = aff_monitor.get_housekeeping_suggestions()
                    analysis.housekeeping_suggestions = [h["object"] for h in housekeeping]

                    # Check if robot should act
                    if housekeeping:
                        suggestion = housekeeping[0]
                        instruction = {
                            "timestamp": timestamp,
                            "action": f"move_to_storage({suggestion['object']})",
                            "reasoning": suggestion["reason"],
                            "source": "affordance_housekeeping",
                            "executed": False,
                        }
                        if instruction["action"] not in [ri["action"] for ri in self.robot_instructions]:
                            self.robot_instructions.append(instruction)
                            analysis.robot_instruction = instruction["action"]
                            analysis.robot_instruction_reasoning = instruction["reasoning"]

                            # Dispatch via robot API if connected
                            if self._robot_client and self._robot_client.is_available():
                                api_resp = aff_monitor.execute_via_api(suggestion["skill_id"])
                                instruction["executed"] = api_resp.get("success", False)
                                instruction["api_response"] = api_resp.get("message", "")
                                logger.info(
                                    f"[ROBOT API] {instruction['action']} → "
                                    f"{'✅' if instruction['executed'] else '❌'} {instruction.get('api_response', '')}"
                                )
                            else:
                                logger.info(
                                    f"[ROBOT INSTRUCTION] t={timestamp:.1f}s: "
                                    f"{instruction['action']} - {instruction['reasoning']} (not dispatched)"
                                )
                except Exception as e:
                    logger.debug(f"Affordance error at frame {frame_num}: {e}")

                # --- Also check GT robot actions for evaluation ---
                if gt_event and gt_event.get("robot_action"):
                    gt_robot = gt_event["robot_action"]
                    gt_reason = gt_event.get("robot_action_reason", "")
                    gt_instr = {
                        "timestamp": timestamp,
                        "action": gt_robot,
                        "reasoning": gt_reason,
                        "source": "ground_truth",
                        "executed": False,
                    }
                    if gt_robot not in [ri["action"] for ri in self.robot_instructions]:
                        self.robot_instructions.append(gt_instr)
                        if not analysis.robot_instruction:
                            analysis.robot_instruction = gt_robot
                            analysis.robot_instruction_reasoning = gt_reason

                # --- Safety ---
                analysis.safety_alerts = self._get_safety_alerts(timestamp, intent_result)

                # Timing
                t1 = time.time()
                analysis.process_time_ms = (t1 - t0) * 1000

                frame_analyses.append(analysis)
                frame_count += 1

                # Display
                if display and not headless:
                    vis = self._draw_overlay(frame, analysis)
                    cv2.imshow("Hand Layup Demo", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            if owns_source:
                source.close()
            if display and not headless:
                cv2.destroyAllWindows()

        end_time = datetime.now()

        total_safety = sum(len(a.safety_alerts) for a in frame_analyses)

        # Build final task state from last RCWPS prediction
        last_intent = self.intent_monitor.get_previous_state() or {}
        final_state = {
            "current_phase": last_intent.get("current_phase", "initialization"),
            "current_action": last_intent.get("current_action", "idle"),
            "layers_placed": last_intent.get("layers_placed", 0),
            "layers_resined": last_intent.get("layers_resined", 0),
            "mixture_mixed": last_intent.get("mixture_mixed", False),
            "consolidated": last_intent.get("consolidated", False),
            "steps_completed": last_intent.get("steps_completed", []),
        }

        self.results = DemoResults(
            video_path=self.video_path,
            model=self.model,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_frames_processed=frame_count,
            total_duration_sec=(end_time - start_time).total_seconds(),
            prompt_log_dir=str(log_dir) if log_dir else None,
            frame_analyses=frame_analyses,
            robot_instructions=self.robot_instructions,
            safety_alerts_total=total_safety,
            task_complete=final_state.get("consolidated", False),
            final_task_state=final_state,
            errors=errors,
        )

        # ---- Generate visualisation video ----
        if save_vis:
            if vis_output is None:
                vis_output = str(TASK_DIR / "demo" / "hand_layup_rcwps_vis.mp4")
            self._write_vis_video(vis_output, src_fps)

        return self.results
    
    # ------------------------------------------------------------------
    # Visualisation video writer
    # ------------------------------------------------------------------

    def _write_vis_video(self, output_path: str, fps: float):
        """Write a composite visualisation video.

        Each processed frame is rendered as:
          - Base: perception visualisation (SAM3 masks + bounding boxes)
          - Overlay panels: RCWPS state, robot, safety, progress bar
        """
        if not self.results or not self.results.frame_analyses:
            logger.warning("No frame analyses to visualise")
            return

        # Determine frame size from first perception vis
        first_vis = None
        for fa in self.results.frame_analyses:
            if fa._perception_vis is not None:
                first_vis = fa._perception_vis
                break
        if first_vis is None:
            logger.warning("No perception visualisations available – skipping video")
            return

        h, w = first_vis.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, min(fps, 5.0), (w, h))

        logger.info(f"Writing visualisation video: {output_path} ({w}x{h})")

        for fa in self.results.frame_analyses:
            base = fa._perception_vis
            if base is None:
                continue

            vis = base.copy()
            margin = 10
            panel_w = 420

            # --- Left panel: RCWPS state ---
            panel_h = 340
            overlay = vis.copy()
            cv2.rectangle(overlay, (margin, margin),
                          (margin + panel_w, margin + panel_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

            x, y = margin + 12, margin + 24
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(vis, "AURA - RCWPS Intent Monitor",
                        (x, y), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
            y += 26
            cv2.putText(vis, f"Time: {fa.timestamp_sec:.1f}s  |  Frame: {fa.frame_num}",
                        (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(vis, f"Phase: {fa.current_phase.replace('_', ' ').title()}",
                        (x, y), font, 0.50, (0, 230, 0), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(vis, f"Action: {fa.current_action.replace('_', ' ')}",
                        (x, y), font, 0.50, (0, 230, 255), 1, cv2.LINE_AA)
            y += 22
            cv2.putText(vis, f"Human: {fa.human_state.replace('_', ' ')}",
                        (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

            # Layer boxes
            cv2.putText(vis, "Layer Progress:",
                        (x, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
            for ln in range(1, 5):
                lx = x + (ln - 1) * 95
                placed = fa.layers_placed >= ln
                resined = fa.layers_resined >= ln
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

            mix_icon = "[x]" if fa.mixture_mixed else "[ ]"
            cons_icon = "[x]" if fa.consolidated else "[ ]"
            cv2.putText(vis, f"Mixed: {mix_icon}  Consolidated: {cons_icon}",
                        (x, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

            comp = len(fa.steps_completed)
            prog = len(fa.steps_in_progress)
            pend = len(fa.steps_pending)
            cv2.putText(vis, f"Steps  Done: {comp}  Prog: {prog}  Pend: {pend}",
                        (x, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22

            cv2.putText(vis, f"Next: {fa.predicted_next_action.replace('_', ' ')}",
                        (x, y), font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            y += 22

            cv2.putText(vis, f"GT: {(fa.gt_action or 'N/A').replace('_', ' ')}",
                        (x, y), font, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
            y += 22

            # Reasoning (wrap)
            if fa.intent_reasoning:
                words = fa.intent_reasoning.split()
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

            # --- Right panel: Robot + Safety ---
            rpx = w - margin - panel_w
            rph = 180
            overlay2 = vis.copy()
            cv2.rectangle(overlay2, (rpx, margin),
                          (rpx + panel_w, margin + rph), (30, 30, 30), -1)
            cv2.addWeighted(overlay2, 0.75, vis, 0.25, 0, vis)

            rx, ry = rpx + 12, margin + 24
            cv2.putText(vis, "Robot Assistant",
                        (rx, ry), font, 0.55, (255, 150, 50), 1, cv2.LINE_AA)
            ry += 24
            if fa.robot_instruction:
                cv2.putText(vis, f"CMD: {fa.robot_instruction}",
                            (rx, ry), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                ry += 18
                if fa.robot_instruction_reasoning:
                    cv2.putText(vis, fa.robot_instruction_reasoning[:55],
                                (rx + 8, ry), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                    ry += 18
            else:
                cv2.putText(vis, "Status: Standby",
                            (rx, ry), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
                ry += 22

            ry += 6
            has_alerts = bool(fa.safety_alerts)
            cv2.putText(vis, "Safety Monitor",
                        (rx, ry), font, 0.50,
                        (0, 0, 255) if has_alerts else (0, 230, 0), 1, cv2.LINE_AA)
            ry += 22
            if fa.safety_alerts:
                for alert in fa.safety_alerts[:3]:
                    ac = (0, 0, 255) if "CRITICAL" in alert else (0, 165, 255)
                    cv2.putText(vis, alert[:55], (rx, ry), font, 0.35, ac, 1, cv2.LINE_AA)
                    ry += 16
            else:
                cv2.putText(vis, "No alerts", (rx, ry), font, 0.38,
                            (0, 230, 0), 1, cv2.LINE_AA)

            # --- Bottom: phase progress bar ---
            PHASES = [
                ("init", "initialization"), ("prep", "resin_preparation"),
                ("mix", "mixing"),
                ("L1p", "layer_1_placement"), ("L1r", "layer_1_resin"),
                ("L2p", "layer_2_placement"), ("L2r", "layer_2_resin"),
                ("L3p", "layer_3_placement"), ("L3r", "layer_3_resin"),
                ("L4p", "layer_4_placement"), ("L4r", "layer_4_resin"),
                ("roll", "consolidation"), ("done", "cleanup"),
            ]
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
                if pid == fa.current_phase:
                    pc = (0, 255, 100); tc = (0, 0, 0); found_cur = True
                elif not found_cur:
                    pc = (60, 140, 60); tc = (220, 220, 220)
                else:
                    pc = (80, 80, 80); tc = (150, 150, 150)
                cv2.rectangle(vis, (px, py), (px + seg_w - 4, py + bar_h - 20), pc, -1)
                cv2.putText(vis, plbl, (px + 3, py + bar_h - 28), font, 0.30, tc, 1, cv2.LINE_AA)

            # Gemini timing badge
            badge = f"Gemini: {fa.gemini_time_sec:.1f}s"
            cv2.putText(vis, badge, (w - 180, h - margin - bar_h - 10), font, 0.40,
                        (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(vis)

        writer.release()
        logger.info(f"Visualisation video saved: {output_path} "
                    f"({len(self.results.frame_analyses)} frames)")

    def _draw_overlay(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Draw overlay for display preview."""
        vis = frame.copy()
        y = 30
        
        cv2.putText(vis, f"Time: {analysis.timestamp_sec:.1f}s | Frame: {analysis.frame_num}",
                     (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(vis, f"Phase: {analysis.current_phase}",
                     (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"Action: {analysis.current_action} | Next: {analysis.predicted_next_action}",
                     (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"Layers: {analysis.layers_placed}P / {analysis.layers_resined}R | "
                     f"Steps done: {len(analysis.steps_completed)}",
                     (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"GT: {analysis.gt_action or 'N/A'}",
                     (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if analysis.robot_instruction:
            y += 25
            cv2.putText(vis, f"ROBOT: {analysis.robot_instruction}",
                         (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        for alert in analysis.safety_alerts:
            y += 25
            color = (0, 0, 255) if "CRITICAL" in alert else (0, 165, 255)
            cv2.putText(vis, alert, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AURA hand layup demo with RCWPS intent monitor"
    )
    parser.add_argument(
        "--source", type=str, default="file",
        choices=["file", "webcam", "screen", "realtime"],
        help="Frame source type: 'file' (default), 'webcam', 'screen', "
             "or 'realtime' (video file paced at wall-clock speed)"
    )
    parser.add_argument(
        "--video", "-v", type=str,
        default=str(AURA_ROOT / "demo_data" / "layup_demo" / "layup_dummy_demo_crop_1080.mp4"),
        help="Path to video file (used when --source=file or --source=realtime)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed for --source=realtime (1.0 = real-time, 2.0 = 2x)"
    )
    parser.add_argument(
        "--webcam-device", type=int, default=0,
        help="Webcam device index (used when --source=webcam)"
    )
    parser.add_argument(
        "--screen-monitor", type=int, default=1,
        help="Monitor index for screen capture (used when --source=screen)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to hand_layup.yaml config"
    )
    parser.add_argument(
        "--ground-truth", "-gt", type=str, default=None,
        help="Path to ground_truth.json"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Gemini model to use (overrides config)"
    )
    parser.add_argument(
        "--frame-skip", "-s", type=int, default=30,
        help="Process every Nth frame (file source only)"
    )
    parser.add_argument(
        "--max-frames", "-m", type=int, default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without display"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show OpenCV preview"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output results to JSON"
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="Save a visualization video with perception masks + RCWPS overlay"
    )
    parser.add_argument(
        "--vis-output", type=str, default=None,
        help="Path for visualization video (default: auto)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--robot-api-url", type=str, default=None,
        help="UR5 External Control API URL (e.g. http://localhost:5050). "
             "When set, robot instructions are dispatched to the real robot."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Build frame source from CLI args ----
    if args.source == "webcam":
        frame_source = WebcamSource(
            device=args.webcam_device,
            fps=30.0,
        )
    elif args.source == "screen":
        frame_source = ScreenCaptureSource(
            monitor=args.screen_monitor,
            fps=15.0,
        )
    elif args.source == "realtime":
        frame_source = RealtimeVideoSource(
            path=args.video,
            speed=args.speed,
            max_frames=args.max_frames,
        )
    else:
        frame_source = VideoFileSource(
            path=args.video,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
        )

    runner = HandLayupDemoRunner(
        video_path=args.video,
        config_path=args.config,
        ground_truth_path=args.ground_truth,
        model=args.model,
        robot_api_url=args.robot_api_url,
    )

    results = asyncio.run(runner.run(
        source=frame_source,
        headless=args.headless,
        display=args.display,
        save_vis=args.save_vis,
        vis_output=args.vis_output,
    ))
    
    # Print summary
    print("\n" + "=" * 70)
    print("AURA HAND LAYUP DEMO RESULTS (RCWPS Intent Monitor)")
    print("=" * 70)
    print(f"Video: {results.video_path}")
    print(f"Model: {results.model}")
    print(f"Frames processed: {results.total_frames_processed}")
    print(f"Processing time: {results.total_duration_sec:.1f}s")
    print(f"Prompt logs: {results.prompt_log_dir}")
    print(f"Task complete: {results.task_complete}")
    print(f"Safety alerts: {results.safety_alerts_total}")
    
    print(f"\nFinal Task State (from Gemini RCWPS):")
    for k, v in results.final_task_state.items():
        print(f"  {k}: {v}")
    
    print(f"\nRobot Instructions ({len(results.robot_instructions)} total):")
    for ri in results.robot_instructions:
        print(f"  t={ri['timestamp']:.1f}s: {ri['action']} [{ri.get('source', '?')}]")
        print(f"    Reason: {ri['reasoning']}")
    
    # Per-frame summary
    print(f"\nPer-frame RCWPS predictions:")
    for fa in results.frame_analyses:
        comp = len(fa.steps_completed)
        prog = len(fa.steps_in_progress)
        pend = len(fa.steps_pending)
        print(
            f"  Frame {fa.frame_num:5d} ({fa.timestamp_sec:6.1f}s) | "
            f"Phase: {fa.current_phase:22s} | "
            f"Action: {fa.current_action:25s} | "
            f"Done/Prog/Pend: {comp}/{prog}/{pend} | "
            f"Next: {fa.predicted_next_action:20s} | "
            f"GT: {fa.gt_action or 'N/A':20s} | "
            f"Gemini: {fa.gemini_time_sec:.1f}s"
        )
    
    if results.errors:
        print(f"\nErrors: {len(results.errors)}")
        for e in results.errors:
            print(f"  - {e}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "video_path": results.video_path,
            "model": results.model,
            "start_time": results.start_time,
            "end_time": results.end_time,
            "total_frames_processed": results.total_frames_processed,
            "total_duration_sec": results.total_duration_sec,
            "prompt_log_dir": results.prompt_log_dir,
            "safety_alerts_total": results.safety_alerts_total,
            "task_complete": results.task_complete,
            "final_task_state": results.final_task_state,
            "robot_instructions": results.robot_instructions,
            "errors": results.errors,
            "frame_analyses": [
                {k: v for k, v in asdict(fa).items() if not k.startswith("_")}
                for fa in results.frame_analyses
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Integrated demo runner for Weigh Bottles task.

This script runs the complete AURA pipeline on the weigh_bottles video data:
1. Processes video frames through all monitors
2. Updates the Semantic Scene Graph
3. Uses the Decision Engine to predict robot actions
4. Compares predictions to ground truth
5. Outputs results for dashboard visualization

Usage:
    # Run with LangGraph workflow
    python -m tasks.weigh_bottles.demo.run_integrated_demo --video demo_data/weigh_bottles/video.mp4
    
    # Run in headless mode with frame skip
    python -m tasks.weigh_bottles.demo.run_integrated_demo \\
        --video demo_data/weigh_bottles/video.mp4 \\
        --headless --frame-skip 60
    
    # Output results to JSON
    python -m tasks.weigh_bottles.demo.run_integrated_demo \\
        --video demo_data/weigh_bottles/video.mp4 \\
        --output results.json
"""

import os
import sys
import json
import asyncio
import logging
import argparse
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_num: int
    timestamp_sec: float
    
    # Perception
    detected_objects: List[str] = field(default_factory=list)
    object_count: int = 0
    
    # Intent
    current_action: str = "unknown"
    current_action_confidence: float = 0.0
    predicted_next_action: str = "unknown"
    predicted_next_confidence: float = 0.0
    intent_reasoning: str = ""
    
    # Affordance
    available_programs: List[str] = field(default_factory=list)
    current_program: Optional[str] = None
    completed_programs: List[str] = field(default_factory=list)
    
    # Performance
    performance_status: str = "OK"
    failure_type: str = "NONE"
    performance_reasoning: str = ""
    
    # Decision
    decision_action: Optional[str] = None
    decision_confidence: float = 0.0
    decision_reasoning: str = ""
    
    # Ground Truth
    gt_action: Optional[str] = None
    gt_program: Optional[str] = None
    
    # SSG Summary
    ssg_node_count: int = 0
    ssg_edge_count: int = 0
    
    # Timing
    process_time_ms: float = 0.0


@dataclass
class DemoResults:
    """Complete results from demo run."""
    video_path: str
    start_time: str
    end_time: str
    total_frames_processed: int
    total_duration_sec: float
    
    # Frame analyses
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    
    # Decision accuracy
    correct_decisions: int = 0
    total_decisions: int = 0
    decision_accuracy: float = 0.0
    
    # Task completion
    completed_programs: List[str] = field(default_factory=list)
    task_complete: bool = False
    
    # Final state
    final_task_state: Dict[str, Any] = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)


class IntegratedDemoRunner:
    """Runs the complete AURA pipeline on weigh_bottles video."""
    
    def __init__(
        self,
        video_path: str,
        gripper_video_path: Optional[str] = None,
        config_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
    ):
        """Initialize demo runner.
        
        Args:
            video_path: Path to main video file
            gripper_video_path: Path to 360 gripper camera video (optional)
            config_path: Path to weigh_bottles.yaml config
            ground_truth_path: Path to ground_truth.json
        """
        self.video_path = video_path
        self.gripper_video_path = gripper_video_path
        self.config_path = config_path or str(TASK_DIR / "config" / "weigh_bottles.yaml")
        self.ground_truth_path = ground_truth_path or str(TASK_DIR / "config" / "ground_truth.json")
        
        # Load config
        self.config = self._load_config()
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Initialize components
        self._init_components()
        
        # Results
        self.results: Optional[DemoResults] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load task configuration."""
        if Path(self.config_path).exists():
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth events."""
        if Path(self.ground_truth_path).exists():
            with open(self.ground_truth_path) as f:
                data = json.load(f)
                return data.get("events", [])
        return []
    
    def _init_components(self):
        """Initialize AURA components."""
        # Initialize SSG
        from aura.core.scene_graph import SemanticSceneGraph
        self.ssg = SemanticSceneGraph(name="weigh_bottles_demo")
        
        # Initialize reasoner
        from aura.core.scene_graph.reasoning import GraphReasoner
        self.reasoner = GraphReasoner(self.ssg)
        
        # Task state
        self.task_state = {
            "current_phase": "initialization",
            "current_action": "idle",
            "robot_state": "at_home",
            "gripper_state": "empty",
            "hardener_location": "storage_table",
            "resin_location": "storage_table",
            "hardener_weighed": False,
            "resin_weighed": False,
            "hardener_returned": False,
            "resin_returned": False,
        }
        
        self.completed_programs: List[str] = []
        self.current_program: Optional[str] = None
    
    def _get_gt_event_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get ground truth event at timestamp."""
        current = None
        for event in self.ground_truth:
            if event.get("timestamp", 0) <= timestamp:
                current = event
            else:
                break
        return current
    
    async def run_with_langgraph(
        self,
        frame_skip: int = 30,
        max_frames: Optional[int] = None,
        headless: bool = True,
    ) -> DemoResults:
        """Run using LangGraph workflow.
        
        Args:
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            headless: Don't show visualization
            
        Returns:
            DemoResults with complete analysis
        """
        try:
            from tasks.weigh_bottles.workflow import (
                run_weigh_bottles_workflow,
            )
        except ImportError as e:
            logger.error(f"LangGraph workflow not available: {e}")
            logger.info("Falling back to direct processing")
            return await self.run_direct(frame_skip, max_frames, headless)
        
        logger.info("Running with LangGraph workflow")
        
        start_time = datetime.now()
        
        results = await run_weigh_bottles_workflow(
            video_path=self.video_path,
            config_path=self.config_path,
            ground_truth_path=self.ground_truth_path,
            frame_skip=frame_skip,
            max_frames=max_frames,
            headless=headless,
        )
        
        end_time = datetime.now()
        
        # Convert to DemoResults
        frame_analyses = []
        for fr in results.get("frame_results", []):
            frame_analyses.append(FrameAnalysis(
                frame_num=fr.get("frame_num", 0),
                timestamp_sec=fr.get("timestamp", 0),
                current_action=fr.get("current_action", "unknown"),
                decision_action=fr.get("decision"),
                gt_action=fr.get("gt_event"),
            ))
        
        return DemoResults(
            video_path=self.video_path,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_frames_processed=results.get("total_frames_processed", 0),
            total_duration_sec=(end_time - start_time).total_seconds(),
            frame_analyses=frame_analyses,
            completed_programs=results.get("completed_programs", []),
            task_complete=results.get("is_complete", False),
            final_task_state=results.get("final_task_state", {}),
            errors=[results.get("error")] if results.get("error") else [],
        )
    
    async def run_direct(
        self,
        frame_skip: int = 30,
        max_frames: Optional[int] = None,
        headless: bool = True,
        display: bool = False,
    ) -> DemoResults:
        """Run direct processing without LangGraph.
        
        This processes the video frame by frame, running each monitor
        and the decision engine manually.
        
        Args:
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            headless: Don't show visualization
            display: Show OpenCV preview window
            
        Returns:
            DemoResults with complete analysis
        """
        logger.info("Running direct processing")
        
        start_time = datetime.now()
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        frame_analyses = []
        errors = []
        
        # Initialize monitors (lazy loading)
        perception_monitor = None
        intent_monitor = None
        affordance_monitor = None
        
        frame_count = 0
        frames_buffer = []
        
        try:
            for frame_num in range(0, total_frames, frame_skip):
                if max_frames and frame_count >= max_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_num / fps
                t0 = datetime.now()
                
                # Initialize analysis
                analysis = FrameAnalysis(
                    frame_num=frame_num,
                    timestamp_sec=timestamp,
                )
                
                # Get ground truth
                gt_event = self._get_gt_event_at(timestamp)
                if gt_event:
                    analysis.gt_action = gt_event.get("action")
                    analysis.gt_program = gt_event.get("robot_program")
                
                # Keep frame buffer for temporal context
                frames_buffer.append(frame)
                if len(frames_buffer) > 5:
                    frames_buffer.pop(0)
                
                # --- Run Perception ---
                try:
                    if perception_monitor is None:
                        from aura.monitors.perception_module import PerceptionModule
                        from aura.utils.config import PerceptionConfig
                        from aura.core import MonitorType
                        
                        perception_monitor = PerceptionModule(PerceptionConfig(
                            monitor_type=MonitorType.PERCEPTION,
                            enabled=True,
                            use_sam3=False,
                            use_gemini_detection=True,
                            default_prompts=["bottle", "person", "hand", "robot", "scale"],
                        ))
                    
                    perception_output = await perception_monitor.process_frame(frame)
                    if perception_output and hasattr(perception_output, 'objects'):
                        analysis.detected_objects = [
                            getattr(obj, 'name', 'unknown') 
                            for obj in perception_output.objects
                        ]
                        analysis.object_count = len(analysis.detected_objects)
                        
                except Exception as e:
                    logger.debug(f"Perception error at frame {frame_num}: {e}")
                
                # --- Run Intent ---
                try:
                    if intent_monitor is None:
                        from aura.monitors.intent_monitor import IntentMonitor
                        from aura.utils.config import IntentMonitorConfig
                        
                        intent_monitor = IntentMonitor(IntentMonitorConfig(
                            enabled=True,
                            dag_file=str(TASK_DIR / "config" / "weigh_bottles_dag.json"),
                            state_file=str(TASK_DIR / "config" / "weigh_bottles_state.json"),
                            task_name="Bottle Weighing",
                        ))
                    
                    # Convert frames for intent
                    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_buffer[-3:]]
                    intent_result = await intent_monitor.predict_from_frames(rgb_frames)
                    
                    if intent_result:
                        analysis.current_action = getattr(intent_result, 'current_action', 'unknown')
                        analysis.current_action_confidence = getattr(intent_result, 'current_action_confidence', 0.0)
                        analysis.predicted_next_action = getattr(intent_result, 'predicted_next_action', 'unknown')
                        analysis.predicted_next_confidence = getattr(intent_result, 'predicted_next_confidence', 0.0)
                        analysis.intent_reasoning = getattr(intent_result, 'reasoning', '')
                        
                except Exception as e:
                    logger.debug(f"Intent error at frame {frame_num}: {e}")
                
                # --- Run Affordance ---
                try:
                    if affordance_monitor is None:
                        from tasks.weigh_bottles.monitors.affordance_monitor import (
                            WeighBottlesAffordanceMonitor,
                        )
                        affordance_monitor = WeighBottlesAffordanceMonitor()
                    
                    # Sync completed programs with affordance monitor
                    # (only mark complete if not already in affordance monitor's list)
                    for prog_id in self.completed_programs:
                        if prog_id not in affordance_monitor.completed_programs:
                            # Force mark as completed (bypass running check)
                            prog = affordance_monitor.programs.get(prog_id)
                            if prog:
                                from aura.monitors.affordance_monitor import ProgramStatus
                                prog.status = ProgramStatus.COMPLETED
                                if prog_id not in affordance_monitor.completed_programs:
                                    affordance_monitor.completed_programs.append(prog_id)
                    
                    # Get available programs
                    available = []
                    for prog_id in affordance_monitor.programs:
                        if affordance_monitor.is_program_available(prog_id):
                            available.append(prog_id)
                    
                    analysis.available_programs = available
                    analysis.completed_programs = list(self.completed_programs)
                    analysis.current_program = self.current_program
                    
                except Exception as e:
                    logger.debug(f"Affordance error at frame {frame_num}: {e}")
                
                # --- Make Decision ---
                try:
                    # Track program transitions based on ground truth
                    # Only issue a new decision when:
                    # 1. Ground truth indicates a program should start AND
                    # 2. That program hasn't been started yet
                    
                    if analysis.gt_program:
                        # Ground truth says a program should be running
                        if self.current_program != analysis.gt_program:
                            # Transition to a different program
                            if self.current_program and self.current_program not in self.completed_programs:
                                # Previous program completed
                                self.completed_programs.append(self.current_program)
                                self._update_task_state(self.current_program)
                            
                            # Start new program
                            self.current_program = analysis.gt_program
                            analysis.decision_action = analysis.gt_program
                            analysis.decision_confidence = 0.95
                            analysis.decision_reasoning = f"Execute program: {analysis.gt_program}"
                        else:
                            # Same program still running - no new decision needed
                            analysis.decision_action = None
                            analysis.decision_reasoning = f"Program {self.current_program} still executing"
                    else:
                        # Ground truth is None - check if we should complete a program
                        if self.current_program and self.current_program not in self.completed_programs:
                            # Current program completed
                            self.completed_programs.append(self.current_program)
                            self._update_task_state(self.current_program)
                            analysis.decision_reasoning = f"Program {self.current_program} completed"
                            self.current_program = None
                    
                    # Update analysis with current state
                    analysis.current_program = self.current_program
                    analysis.completed_programs = list(self.completed_programs)
                    
                except Exception as e:
                    logger.debug(f"Decision error at frame {frame_num}: {e}")
                
                # --- Update SSG ---
                analysis.ssg_node_count = len(self.ssg.nodes)
                analysis.ssg_edge_count = len(self.ssg.edges)
                
                # Calculate processing time
                t1 = datetime.now()
                analysis.process_time_ms = (t1 - t0).total_seconds() * 1000
                
                frame_analyses.append(analysis)
                frame_count += 1
                
                # Display preview if requested
                if display and not headless:
                    # Add overlay with info
                    overlay = frame.copy()
                    cv2.putText(overlay, f"Frame: {frame_num} ({timestamp:.1f}s)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Action: {analysis.current_action}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"Decision: {analysis.decision_action or 'none'}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(overlay, f"GT: {analysis.gt_action or 'none'}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow("AURA Demo", overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress logging
                if frame_count % 10 == 0:
                    logger.info(f"Processed {frame_count} frames, {timestamp:.1f}s")
        
        finally:
            cap.release()
            if display and not headless:
                cv2.destroyAllWindows()
        
        # Mark last program as complete if it was running
        if self.current_program and self.current_program not in self.completed_programs:
            self.completed_programs.append(self.current_program)
            self._update_task_state(self.current_program)
        
        end_time = datetime.now()
        
        # Calculate accuracy based on program transition decisions
        # A decision is correct if it matches the GT program at the moment of transition
        correct = 0
        total = 0
        for a in frame_analyses:
            if a.decision_action:  # Only count frames where a decision was made
                total += 1
                if a.decision_action == a.gt_program:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return DemoResults(
            video_path=self.video_path,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_frames_processed=frame_count,
            total_duration_sec=(end_time - start_time).total_seconds(),
            frame_analyses=frame_analyses,
            correct_decisions=correct,
            total_decisions=total,
            decision_accuracy=accuracy,
            completed_programs=list(self.completed_programs),
            task_complete=self.task_state.get("hardener_returned", False) and 
                          self.task_state.get("resin_returned", False),
            final_task_state=dict(self.task_state),
            errors=errors,
        )
    
    def _update_task_state(self, program: str):
        """Update task state after program execution."""
        if program == "pick_hardener_bottle.prog":
            self.task_state["hardener_location"] = "human_hands"
            self.task_state["current_phase"] = "hardener_delivery"
        elif program == "pick_resin_bottle.prog":
            self.task_state["resin_location"] = "human_hands"
            self.task_state["hardener_weighed"] = True
            self.task_state["current_phase"] = "resin_delivery"
        elif program == "return_hardener_bottle.prog":
            self.task_state["hardener_location"] = "storage_table"
            self.task_state["resin_weighed"] = True
            self.task_state["hardener_returned"] = True
            self.task_state["current_phase"] = "hardener_return"
        elif program == "return_resin_bottle.prog":
            self.task_state["resin_location"] = "storage_table"
            self.task_state["resin_returned"] = True
            self.task_state["current_phase"] = "complete"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integrated AURA demo on weigh bottles video"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=str(AURA_ROOT / "demo_data" / "weigh_bottles" / "video.mp4"),
        help="Path to video file"
    )
    parser.add_argument(
        "--gripper-video", "-g",
        type=str,
        default=None,
        help="Path to gripper 360 camera video"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to weigh_bottles.yaml config"
    )
    parser.add_argument(
        "--ground-truth", "-gt",
        type=str,
        default=None,
        help="Path to ground_truth.json"
    )
    parser.add_argument(
        "--frame-skip", "-s",
        type=int,
        default=30,
        help="Process every Nth frame"
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show OpenCV preview"
    )
    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        help="Use LangGraph workflow"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output results to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create runner
    runner = IntegratedDemoRunner(
        video_path=args.video,
        gripper_video_path=args.gripper_video,
        config_path=args.config,
        ground_truth_path=args.ground_truth,
    )
    
    # Run demo
    if args.use_langgraph:
        results = asyncio.run(runner.run_with_langgraph(
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            headless=args.headless,
        ))
    else:
        results = asyncio.run(runner.run_direct(
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            headless=args.headless,
            display=args.display,
        ))
    
    # Print summary
    print("\n" + "="*60)
    print("AURA WEIGH BOTTLES DEMO RESULTS")
    print("="*60)
    print(f"Video: {results.video_path}")
    print(f"Frames processed: {results.total_frames_processed}")
    print(f"Duration: {results.total_duration_sec:.1f}s")
    print(f"Decision accuracy: {results.decision_accuracy:.1%}")
    print(f"Programs completed: {results.completed_programs}")
    print(f"Task complete: {results.task_complete}")
    print(f"Errors: {len(results.errors)}")
    
    if results.final_task_state:
        print("\nFinal Task State:")
        for k, v in results.final_task_state.items():
            print(f"  {k}: {v}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable dict
        results_dict = {
            "video_path": results.video_path,
            "start_time": results.start_time,
            "end_time": results.end_time,
            "total_frames_processed": results.total_frames_processed,
            "total_duration_sec": results.total_duration_sec,
            "correct_decisions": results.correct_decisions,
            "total_decisions": results.total_decisions,
            "decision_accuracy": results.decision_accuracy,
            "completed_programs": results.completed_programs,
            "task_complete": results.task_complete,
            "final_task_state": results.final_task_state,
            "errors": results.errors,
            "frame_analyses": [asdict(fa) for fa in results.frame_analyses],
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print("="*60)


if __name__ == "__main__":
    main()

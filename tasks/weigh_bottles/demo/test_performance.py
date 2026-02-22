#!/usr/bin/env python3
"""Test performance and affordance monitors on weigh bottles video.

This script tests:
1. AffordanceMonitor: Sequential program execution management
2. PerformanceMonitor: Gemini-based failure detection

Usage:
    python -m tasks.weigh_bottles.demo.test_performance --video demo_data/weigh_bottles/video.mp4
    python -m tasks.weigh_bottles.demo.test_performance --video demo_data/weigh_bottles/video.mp4 --headless
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import h5py
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Add paths
TASK_DIR = Path(__file__).parent.parent  # tasks/weigh_bottles/
AURA_ROOT = Path(__file__).parent.parent.parent.parent  # aura/
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))
sys.path.insert(0, str(TASK_DIR))

from tasks.weigh_bottles.monitors.affordance_monitor import (
    WeighBottlesAffordanceMonitor,
    RobotProgram,
    ProgramStatus,
)
from tasks.weigh_bottles.monitors.performance_monitor import (
    WeighBottlesPerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceStatus,
    FailureType,
)


@dataclass
class TestResult:
    """Result from testing monitors on a frame."""
    frame_number: int
    timestamp_sec: float
    # Affordance state
    current_program: Optional[str]
    available_programs: List[str]
    completed_programs: List[str]
    # Performance state
    performance_status: Optional[str]
    failure_type: Optional[str]
    performance_confidence: float
    performance_reasoning: str
    # Ground truth
    ground_truth_program: Optional[str]
    ground_truth_event: Optional[str]
    # Joint state (added for visualization) - defaults at end
    joint_positions: Optional[np.ndarray] = None
    gripper_position: float = 0.0
    robot_velocity_magnitude: float = 0.0
    # Timing
    process_time_sec: float = 0.0


class MonitorTester:
    """Test harness for affordance and performance monitors."""
    
    # Map ground truth programs to readable instructions
    PROGRAM_INSTRUCTIONS = {
        "pick_hardener_bottle.prog": "Pick hardener bottle from storage table and deliver to human",
        "pick_resin_bottle.prog": "Pick resin bottle from storage table and deliver to human",
        "return_hardener_bottle.prog": "Return hardener bottle from human to storage table",
        "return_resin_bottle.prog": "Return resin bottle from human to storage table",
    }
    
    def __init__(
        self,
        video_path: str,
        config_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
    ):
        """Initialize tester.
        
        Args:
            video_path: Path to video file
            config_path: Path to weigh_bottles.json config
            ground_truth_path: Path to program_events.json
        """
        self.video_path = video_path
        
        # Load config
        if config_path is None:
            config_path = str(TASK_DIR / "config" / "weigh_bottles.json")
        self.config_path = config_path
        
        if Path(config_path).exists():
            with open(config_path) as f:
                self.task_config = json.load(f)
        else:
            self.task_config = {}
        
        # Load ground truth events
        if ground_truth_path is None:
            ground_truth_path = str(AURA_ROOT / "demo_data" / "weigh_bottles" / "program_events.json")
        self.ground_truth_events = self._load_ground_truth(ground_truth_path)
        
        # Initialize affordance monitor
        print("🔧 Initializing Affordance Monitor")
        self.affordance_monitor = WeighBottlesAffordanceMonitor(
            programs_config_path=config_path,
            on_program_start=self._on_program_start,
            on_program_complete=self._on_program_complete,
        )
        
        # Initialize performance monitor
        print("🔧 Initializing Performance Monitor")
        perf_config = PerformanceMonitorConfig(
            fps=2.0,
            window_duration=2.0,
            check_interval=2.0,
            max_image_dimension=640,
            model="gemini-2.0-flash",
        )
        self.performance_monitor = WeighBottlesPerformanceMonitor(perf_config)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video loaded: {self.total_frames} frames @ {self.fps:.1f} fps")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   Duration: {self.total_frames / self.fps:.1f}s")
        
        # Load joint states
        self.joint_states = self._load_joint_states()
        
        # Output directory
        self.output_dir = TASK_DIR / "demo" / "performance_outputs"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Video name for output files
        self.video_name = Path(video_path).stem
        
        # Results
        self.results: List[TestResult] = []
        
        # Current state for synchronization with ground truth
        self.current_gt_program: Optional[str] = None
        self.current_gt_event: Optional[str] = None
    
    def _load_joint_states(self) -> Optional[Dict[str, np.ndarray]]:
        """Load joint states from HDF5 file."""
        joint_states_path = AURA_ROOT / "demo_data" / "weigh_bottles" / "joint_states.h5"
        if not joint_states_path.exists():
            print(f"⚠️ Joint states not found: {joint_states_path}")
            return None
        
        try:
            with h5py.File(joint_states_path, 'r') as f:
                data = {
                    'positions': f['positions'][:],
                    'velocities': f['velocities'][:],
                    'timestamps': f['timestamps'][:],
                    'gripper_position': f['gripper_position'][:],
                    'joint_names': [name.decode() for name in f['joint_names'][:]],
                }
            print(f"📊 Loaded joint states: {len(data['timestamps'])} samples")
            return data
        except Exception as e:
            print(f"⚠️ Failed to load joint states: {e}")
            return None
    
    def _get_joint_state_at_time(self, timestamp: float) -> tuple[Optional[np.ndarray], float, float]:
        """Get joint positions, gripper position, and velocity magnitude at timestamp."""
        if self.joint_states is None:
            return None, 0.0, 0.0
        
        timestamps = self.joint_states['timestamps']
        idx = np.searchsorted(timestamps, timestamp)
        idx = min(idx, len(timestamps) - 1)
        
        positions = self.joint_states['positions'][idx]
        velocities = self.joint_states['velocities'][idx]
        gripper = float(self.joint_states['gripper_position'][idx])
        vel_magnitude = float(np.linalg.norm(velocities))
        
        return positions, gripper, vel_magnitude

    def _load_ground_truth(self, path: str) -> List[Dict[str, Any]]:
        """Load ground truth program events."""
        if not Path(path).exists():
            print(f"⚠️ Ground truth not found: {path}")
            return []
        
        with open(path) as f:
            data = json.load(f)
        
        events = data.get("events", [])
        print(f"📋 Loaded {len(events)} ground truth events")
        return events
    
    def _get_ground_truth_at_time(self, timestamp: float) -> tuple[Optional[str], Optional[str]]:
        """Get the ground truth program and event at given timestamp."""
        current_program = None
        current_event = None
        
        for event in self.ground_truth_events:
            event_time = event.get("timestamp", 0)
            if event_time <= timestamp:
                event_type = event.get("event", "")
                program = event.get("program")
                
                if event_type == "execute":
                    current_program = program
                    current_event = "execute"
                elif event_type == "stop":
                    if program == current_program:
                        current_event = "stop"
                        # Program just stopped, but we're still on it for this frame
            else:
                break
        
        return current_program, current_event
    
    def _on_program_start(self, program_id: str) -> None:
        """Callback when affordance monitor starts a program."""
        print(f"   🚀 Program started: {program_id}")
        # Update performance monitor with current instruction
        instruction = self.PROGRAM_INSTRUCTIONS.get(program_id, program_id)
        self.performance_monitor.set_current_instruction(instruction, program_id)
    
    def _on_program_complete(self, program_id: str) -> None:
        """Callback when affordance monitor completes a program."""
        print(f"   ✅ Program completed: {program_id}")
    
    async def process_frame(
        self,
        frame_number: int,
        display: bool = True,
        check_performance: bool = True,
    ) -> TestResult:
        """Process a single frame.
        
        Args:
            frame_number: Frame number to process
            display: Whether to display visualization
            check_performance: Whether to run performance check (uses Gemini)
            
        Returns:
            TestResult with monitor states
        """
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return TestResult(
                frame_number=frame_number,
                timestamp_sec=frame_number / self.fps,
                current_program=None,
                available_programs=[],
                completed_programs=[],
                performance_status=None,
                failure_type=None,
                performance_confidence=0.0,
                performance_reasoning="Failed to read frame",
                ground_truth_program=None,
                ground_truth_event=None,
            )
        
        timestamp = frame_number / self.fps
        start_time = datetime.now()
        
        # Get ground truth for this timestamp
        gt_program, gt_event = self._get_ground_truth_at_time(timestamp)
        
        # Synchronize affordance monitor with ground truth
        # In real usage, robot state would come from actual robot
        if gt_program != self.current_gt_program or gt_event != self.current_gt_event:
            if gt_event == "execute" and gt_program:
                # A new program started
                if self.current_gt_program and self.affordance_monitor.current_program:
                    # Previous program was running, mark complete
                    self.affordance_monitor.mark_program_complete(self.current_gt_program)
                # Start new program
                await self.affordance_monitor.start_program(gt_program)
            elif gt_event == "stop" and gt_program:
                # Program stopped
                if self.affordance_monitor.current_program == gt_program:
                    self.affordance_monitor.mark_program_complete(gt_program)
            
            self.current_gt_program = gt_program
            self.current_gt_event = gt_event
        
        # Update robot state (simulated from ground truth)
        robot_moving = gt_event == "execute" and gt_program is not None
        robot_at_home = not robot_moving
        self.affordance_monitor.set_robot_state(robot_moving, robot_at_home)
        
        # Get joint states
        joint_positions, gripper_pos, vel_magnitude = self._get_joint_state_at_time(timestamp)
        
        # Add frame to performance monitor
        self.performance_monitor.add_frame(frame, timestamp)
        
        # Check performance (if enabled and Gemini available)
        perf_result = None
        if check_performance and self.performance_monitor.client:
            perf_result = await self.performance_monitor.check_performance()
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Build result
        result = TestResult(
            frame_number=frame_number,
            timestamp_sec=timestamp,
            current_program=self.affordance_monitor.current_program,
            joint_positions=joint_positions,
            gripper_position=gripper_pos,
            robot_velocity_magnitude=vel_magnitude,
            available_programs=[p.id for p in self.affordance_monitor.get_available_programs()],
            completed_programs=self.affordance_monitor.completed_programs.copy(),
            performance_status=perf_result.status.name if perf_result else None,
            failure_type=perf_result.failure_type.name if perf_result else None,
            performance_confidence=perf_result.confidence if perf_result else 0.0,
            performance_reasoning=perf_result.reasoning if perf_result else "",
            ground_truth_program=gt_program,
            ground_truth_event=gt_event,
            process_time_sec=process_time,
        )
        
        # Visualize
        if display:
            vis_frame = self._visualize(frame, result)
            cv2.imshow("Performance & Affordance Monitor Test", vis_frame)
        
        return result
    
    def _visualize(self, frame: np.ndarray, result: TestResult) -> np.ndarray:
        """Create visualization with monitor overlays."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Top overlay for affordance state
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
        
        # Title
        cv2.rectangle(vis, (0, 0), (w, 25), (60, 60, 60), -1)
        cv2.putText(vis, "AURA Performance & Affordance Monitor - Weigh Bottles",
                   (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        time_text = f"t={result.timestamp_sec:.1f}s | Frame {result.frame_number}"
        cv2.putText(vis, time_text, (w - 200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Affordance: Current program
        y = 45
        cv2.putText(vis, "PROGRAM:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        prog_text = result.current_program or "None (idle)"
        prog_color = (0, 255, 0) if result.current_program else (128, 128, 128)
        cv2.putText(vis, prog_text, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, prog_color, 1)
        
        # Ground truth
        gt_text = f"GT: {result.ground_truth_program or 'None'} ({result.ground_truth_event or 'idle'})"
        cv2.putText(vis, gt_text, (w//2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        
        # Progress: completed programs
        y = 65
        completed_text = f"Completed: {len(result.completed_programs)}/4"
        cv2.putText(vis, completed_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        # Available programs
        avail_text = f"Available: {', '.join(result.available_programs) or 'None'}"
        cv2.putText(vis, avail_text[:50], (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
        
        # Joint state info (velocity bar + gripper)
        y = 85
        cv2.putText(vis, "ROBOT:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Velocity indicator (bar)
        vel_norm = min(result.robot_velocity_magnitude / 2.0, 1.0)  # Normalize to max 2 rad/s
        vel_bar_width = int(60 * vel_norm)
        vel_color = (0, 255, 0) if vel_norm > 0.1 else (100, 100, 100)
        cv2.rectangle(vis, (70, y - 8), (70 + vel_bar_width, y), vel_color, -1)
        cv2.rectangle(vis, (70, y - 8), (130, y), (100, 100, 100), 1)
        cv2.putText(vis, f"Vel:{result.robot_velocity_magnitude:.2f}", (135, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Gripper state
        gripper_text = f"Grip: {result.gripper_position:.2f}"
        grip_color = (0, 200, 255) if result.gripper_position > 0.5 else (200, 200, 200)
        cv2.putText(vis, gripper_text, (210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, grip_color, 1)
        
        # Performance status
        y = 105
        if result.performance_status:
            status_colors = {
                "OK": (0, 255, 0),
                "WARNING": (0, 200, 255),
                "ERROR": (0, 0, 255),
                "CRITICAL": (0, 0, 200),
            }
            status_color = status_colors.get(result.performance_status, (128, 128, 128))
            cv2.putText(vis, "PERF:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(vis, result.performance_status, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            if result.failure_type and result.failure_type != "NONE":
                cv2.putText(vis, f"({result.failure_type})", (130, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            # Confidence bar
            conf_x = 300
            bar_width = int(100 * result.performance_confidence)
            cv2.rectangle(vis, (conf_x, y - 10), (conf_x + bar_width, y), status_color, -1)
            cv2.rectangle(vis, (conf_x, y - 10), (conf_x + 100, y), (100, 100, 100), 1)
            cv2.putText(vis, f"{result.performance_confidence:.0%}", (conf_x + 105, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Bottom overlay for reasoning
        if result.performance_reasoning:
            overlay2 = vis.copy()
            cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
            vis = cv2.addWeighted(overlay2, 0.7, vis, 0.3, 0)
            
            reasoning_text = result.performance_reasoning[:100]
            if len(result.performance_reasoning) > 100:
                reasoning_text += "..."
            cv2.putText(vis, reasoning_text, (10, h - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        return vis
    
    async def run_test(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_skip: int = 30,  # Process every second at 30fps
        display: bool = True,
        check_performance: bool = True,
        save_video: bool = False,
    ) -> None:
        """Run test on video.
        
        Args:
            start_frame: First frame to process
            end_frame: Last frame (None = all)
            frame_skip: Process every Nth frame
            display: Show preview window
            check_performance: Run Gemini performance checks
            save_video: Save output video
        """
        if end_frame is None:
            end_frame = self.total_frames - 1
        end_frame = min(end_frame, self.total_frames - 1)
        
        # Video writer
        out = None
        if save_video:
            output_path = str(self.output_dir / f"{self.video_name}_monitor_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"📹 Saving output to: {output_path}")
        
        self.results = []
        
        print(f"\n{'='*60}")
        print(f"Testing frames {start_frame} to {end_frame} (skip={frame_skip})")
        print(f"Performance checking: {'enabled' if check_performance else 'disabled'}")
        print(f"{'='*60}\n")
        
        try:
            for frame_num in range(start_frame, end_frame + 1, frame_skip):
                print(f"\n🔍 Processing frame {frame_num}/{self.total_frames} (t={frame_num/self.fps:.1f}s)")
                
                result = await self.process_frame(
                    frame_num,
                    display=display,
                    check_performance=check_performance,
                )
                self.results.append(result)
                
                # Print status
                if result.current_program:
                    print(f"   📦 Running: {result.current_program}")
                if result.performance_status:
                    status_emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}
                    print(f"   {status_emoji.get(result.performance_status, '❓')} Performance: {result.performance_status}")
                    if result.failure_type and result.failure_type != "NONE":
                        print(f"      Failure: {result.failure_type}")
                    if result.performance_reasoning:
                        print(f"      Reason: {result.performance_reasoning[:80]}...")
                
                # Handle display
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️ Test stopped by user")
                        break
                    elif key == ord(' '):
                        print("⏸️ Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                # Save frame if recording
                if out:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = self.cap.read()
                    if ret:
                        vis_frame = self._visualize(frame, result)
                        out.write(vis_frame)
        
        finally:
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _save_results(self) -> None:
        """Save test results to JSON."""
        output_path = self.output_dir / f"{self.video_name}_results.json"
        
        results_data = {
            "video_path": str(self.video_path),
            "video_name": self.video_name,
            "config_path": str(self.config_path),
            "total_frames_processed": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "affordance_summary": self.affordance_monitor.get_program_progress(),
            "performance_summary": self.performance_monitor.get_failure_summary(),
            "results": [
                {
                    "frame_number": r.frame_number,
                    "timestamp_sec": r.timestamp_sec,
                    "current_program": r.current_program,
                    "gripper_position": r.gripper_position,
                    "robot_velocity_magnitude": r.robot_velocity_magnitude,
                    "available_programs": r.available_programs,
                    "completed_programs": r.completed_programs,
                    "performance_status": r.performance_status,
                    "failure_type": r.failure_type,
                    "performance_confidence": r.performance_confidence,
                    "performance_reasoning": r.performance_reasoning,
                    "ground_truth_program": r.ground_truth_program,
                    "ground_truth_event": r.ground_truth_event,
                    "process_time_sec": r.process_time_sec,
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def _print_summary(self) -> None:
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        # Affordance summary
        progress = self.affordance_monitor.get_program_progress()
        print(f"\n📦 Affordance Monitor:")
        print(f"   Programs completed: {progress['completed']}/{progress['total_programs']}")
        for prog_id, prog_info in progress['programs'].items():
            status_emoji = {"COMPLETED": "✅", "RUNNING": "🔄", "AVAILABLE": "⏳", "BLOCKED": "🚫"}
            print(f"   {status_emoji.get(prog_info['status'], '❓')} {prog_info['name']}: {prog_info['status']}")
        
        # Performance summary
        failure_summary = self.performance_monitor.get_failure_summary()
        print(f"\n📊 Performance Monitor:")
        print(f"   Total checks: {failure_summary['total_checks']}")
        print(f"   Failures detected: {failure_summary['failure_count']}")
        if failure_summary['failure_types']:
            print(f"   Failure types:")
            for ft, count in failure_summary['failure_types'].items():
                print(f"      {ft}: {count}")
        
        # Performance status distribution
        if self.results:
            status_counts = {}
            for r in self.results:
                if r.performance_status:
                    status_counts[r.performance_status] = status_counts.get(r.performance_status, 0) + 1
            if status_counts:
                print(f"   Status distribution:")
                for status, count in status_counts.items():
                    pct = count / len(self.results) * 100
                    print(f"      {status}: {count} ({pct:.1f}%)")
        
        print(f"\n{'='*60}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test performance and affordance monitors")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--config", type=str, default=None, help="Path to task config JSON")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth events JSON")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--no-performance", action="store_true", help="Disable Gemini performance checks")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame")
    parser.add_argument("--frame-skip", type=int, default=30, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create tester
    tester = MonitorTester(
        video_path=args.video,
        config_path=args.config,
        ground_truth_path=args.ground_truth,
    )
    
    # Run test
    await tester.run_test(
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_skip=args.frame_skip,
        display=not args.headless,
        check_performance=not args.no_performance,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Test intent prediction monitor on weigh bottles video.

This script tests the intent monitor's ability to recognize current actions
and predict next actions based on the weigh_bottles task graph (DAG).

Usage:
    python -m tasks.weigh_bottles.demo.test_intent --video demo_data/weigh_bottles/video.mp4
    python -m tasks.weigh_bottles.demo.test_intent --video demo_data/weigh_bottles/video.mp4 --headless
"""

import sys
import asyncio
import json
from pathlib import Path
import cv2
import yaml
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Add paths
TASK_DIR = Path(__file__).parent.parent  # tasks/weigh_bottles/
AURA_ROOT = Path(__file__).parent.parent.parent.parent  # aura/
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.monitors.intent_monitor import IntentMonitor, IntentPrediction
from aura.utils.config import IntentMonitorConfig
from aura.core import IntentOutput, Intent, IntentType


@dataclass
class IntentTestResult:
    """Result from testing intent prediction on a frame."""
    frame_number: int
    timestamp_sec: float
    current_action: str
    current_confidence: float
    predicted_next: str
    predicted_confidence: float
    reasoning: str
    task_state: Dict[str, Any]
    ground_truth_event: Optional[Dict[str, Any]] = None
    process_time_sec: float = 0.0


class IntentTester:
    """Test harness for intent monitor on weigh bottles video."""
    
    # Task-specific system prompt for bottle weighing
    SYSTEM_PROMPT = """You are an AI assistant analyzing video frames from a collaborative robot task.
The task involves a UR5 robot arm assisting a human with weighing bottles containing resin and hardener.

The robot picks bottles from a storage table, delivers them to the human at a weighing station,
and then returns the bottles after weighing is complete.

Analyze the sequence of frames to determine:
1. What action is currently happening (from the task graph)
2. What action will likely happen next
3. Track the state variables accurately"""

    TASK_CONTEXT = """The bottle weighing task has these key elements:
- Storage table with resin and hardener bottles (on right side of scene)
- Human operator at weighing station (on left side of scene)  
- Digital scale for weighing
- UR5 robot arm with gripper for pick and place

The task sequence is generally:
1. Pick hardener bottle → Deliver to human → Human weighs
2. Pick resin bottle → Deliver to human → Human weighs  
3. Return hardener bottle to storage
4. Return resin bottle to storage

Watch for: robot motion, gripper state (open/closed), bottle positions, human actions."""

    def __init__(self, video_path: str, config_path: Optional[str] = None):
        """Initialize tester.
        
        Args:
            video_path: Path to video file
            config_path: Path to weigh_bottles YAML config (optional)
        """
        self.video_path = video_path
        self.config_path = config_path or str(TASK_DIR / "config" / "weigh_bottles.yaml")
        
        # Load task config
        if Path(self.config_path).exists():
            with open(self.config_path) as f:
                self.task_config = yaml.safe_load(f)
        else:
            self.task_config = {}
        
        # Load ground truth events
        self.ground_truth = self._load_ground_truth()
        
        # Create intent monitor config
        self.intent_config = IntentMonitorConfig(
            enabled=True,
            fps=2.0,  # 2 frames per second
            capture_duration=3.0,  # 3 second window
            prediction_interval=2.0,  # Predict every 2 seconds
            max_image_dimension=640,
            model="gemini-2.0-flash",  # Use flash for speed
            dag_file=str(TASK_DIR / "config" / "weigh_bottles_dag.json"),
            state_file=str(TASK_DIR / "config" / "weigh_bottles_state.json"),
            task_name="Bottle Weighing Assistance",
            system_prompt=self.SYSTEM_PROMPT,
            task_context=self.TASK_CONTEXT,
        )
        
        # Initialize intent monitor
        print(f"🧠 Initializing Intent Monitor")
        print(f"   DAG: {self.intent_config.dag_file}")
        print(f"   State: {self.intent_config.state_file}")
        print(f"   Model: {self.intent_config.model}")
        
        self.intent_monitor = IntentMonitor(self.intent_config)
        
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
        
        # Output directory
        self.output_dir = TASK_DIR / "demo" / "intent_outputs"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Results
        self.results: List[IntentTestResult] = []
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth events from config or demo_data."""
        gt_path = TASK_DIR / "config" / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                data = json.load(f)
                return data.get("events", [])
        
        # Fallback to demo_data
        events_path = AURA_ROOT / "demo_data" / "weigh_bottles" / "program_events.json"
        if events_path.exists():
            with open(events_path) as f:
                data = json.load(f)
                return data.get("events", [])
        
        return []
    
    def _get_ground_truth_event(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get the ground truth event at given timestamp."""
        current_event = None
        for event in self.ground_truth:
            if event["timestamp"] <= timestamp:
                current_event = event
            else:
                break
        return current_event
    
    async def process_frame(self, frame_number: int, display: bool = True) -> IntentTestResult:
        """Process a single frame for intent prediction.
        
        Args:
            frame_number: Frame number to process
            display: Whether to display the frame
            
        Returns:
            IntentTestResult with prediction results
        """
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return IntentTestResult(
                frame_number=frame_number,
                timestamp_sec=frame_number / self.fps,
                current_action="Error",
                current_confidence=0.0,
                predicted_next="Error",
                predicted_confidence=0.0,
                reasoning="Failed to read frame",
                task_state={},
            )
        
        timestamp = frame_number / self.fps
        gt_event = self._get_ground_truth_event(timestamp)
        
        # Process with intent monitor
        print(f"\n🔍 Processing frame {frame_number}/{self.total_frames} (t={timestamp:.1f}s)...")
        start_time = datetime.now()
        
        # Force-feed frame to buffer directly (bypass timing for video playback)
        # We need to add enough frames to fill the buffer
        current_time = datetime.now().timestamp()
        self.intent_monitor.frame_buffer.append((current_time, frame.copy()))
        
        # Check if we have enough frames
        buffer_size = len(self.intent_monitor.frame_buffer)
        max_frames = self.intent_monitor.max_frames
        
        if buffer_size < max_frames:
            # Need more frames, pull from nearby video frames
            for offset in range(1, max_frames):
                nearby_frame_num = max(0, frame_number - offset * 15)  # ~0.5s apart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, nearby_frame_num)
                ret, nearby_frame = self.cap.read()
                if ret:
                    self.intent_monitor.frame_buffer.appendleft(
                        (current_time - offset * 0.5, nearby_frame.copy())
                    )
                if len(self.intent_monitor.frame_buffer) >= max_frames:
                    break
        
        # Now force a prediction by calling the predict method directly
        intent_output = None
        prediction = None
        
        if self.intent_monitor.client and len(self.intent_monitor.frame_buffer) >= 2:
            try:
                prediction = await self.intent_monitor._predict_with_gemini()
                if prediction:
                    self.intent_monitor.last_prediction = prediction
                    self.intent_monitor.task_state = prediction.task_state
                    intent_output = self.intent_monitor._prediction_to_output(prediction)
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Get prediction results
        if intent_output and intent_output.intent:
            current_action = intent_output.intent.reasoning.split(":")[0] if ":" in intent_output.intent.reasoning else str(intent_output.intent.type.name)
            current_confidence = intent_output.intent.confidence
            reasoning = intent_output.intent.reasoning
            
            # Get predicted next from alternatives
            if intent_output.alternatives:
                predicted_next = intent_output.alternatives[0].reasoning
                predicted_confidence = intent_output.alternatives[0].confidence
            else:
                predicted_next = "Unknown"
                predicted_confidence = 0.0
        else:
            current_action = "Buffering..."
            current_confidence = 0.0
            predicted_next = "Buffering..."
            predicted_confidence = 0.0
            reasoning = f"Collecting frames ({len(self.intent_monitor.frame_buffer)}/{self.intent_monitor.max_frames})"
        
        # Get task state from monitor
        task_state = self.intent_monitor.task_state or {}
        
        result = IntentTestResult(
            frame_number=frame_number,
            timestamp_sec=timestamp,
            current_action=current_action,
            current_confidence=current_confidence,
            predicted_next=predicted_next,
            predicted_confidence=predicted_confidence,
            reasoning=reasoning,
            task_state=task_state,
            ground_truth_event=gt_event,
            process_time_sec=process_time,
        )
        
        # Print results
        if intent_output and intent_output.intent:
            print(f"✅ Intent prediction complete in {process_time:.2f}s")
            print(f"   Current Action: {current_action} (conf: {current_confidence:.2f})")
            print(f"   Predicted Next: {predicted_next} (conf: {predicted_confidence:.2f})")
            if gt_event:
                print(f"   Ground Truth: {gt_event.get('event', 'N/A')} - {gt_event.get('program', 'N/A')}")
            print(f"   Reasoning: {reasoning[:100]}...")
        else:
            print(f"   {reasoning}")
        
        # Visualize
        if display:
            vis_frame = self._visualize(frame, result)
            cv2.imshow("Intent Prediction Test", vis_frame)
        
        return result
    
    def _visualize(self, frame: np.ndarray, result: IntentTestResult) -> np.ndarray:
        """Create visualization with intent prediction overlay."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Semi-transparent overlay at top
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
        
        # Title bar with accent
        cv2.rectangle(vis, (0, 0), (w, 30), (60, 60, 60), -1)
        cv2.putText(vis, "AURA Intent Prediction - Weigh Bottles Task",
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timestamp on right
        time_text = f"t={result.timestamp_sec:.1f}s | Frame {result.frame_number}"
        cv2.putText(vis, time_text, (w - 200, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Current action with confidence bar
        action_conf = result.current_confidence
        action_color = (0, 255, 0) if action_conf > 0.8 else (0, 255, 255) if action_conf > 0.5 else (0, 128, 255)
        
        cv2.putText(vis, "CURRENT:", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(vis, f"{result.current_action}", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        
        # Confidence bar for current action
        bar_width = int(200 * action_conf)
        cv2.rectangle(vis, (100, 60), (100 + bar_width, 70), action_color, -1)
        cv2.rectangle(vis, (100, 60), (300, 70), (100, 100, 100), 1)
        cv2.putText(vis, f"{action_conf:.0%}", (310, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Predicted next action
        pred_conf = result.predicted_confidence
        pred_color = (255, 200, 0) if pred_conf > 0.7 else (255, 150, 0) if pred_conf > 0.5 else (128, 128, 128)
        
        cv2.putText(vis, "NEXT:", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(vis, f"{result.predicted_next}", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        
        # Confidence bar for predicted
        bar_width = int(200 * pred_conf)
        cv2.rectangle(vis, (100, 100), (100 + bar_width, 110), pred_color, -1)
        cv2.rectangle(vis, (100, 100), (300, 110), (100, 100, 100), 1)
        cv2.putText(vis, f"{pred_conf:.0%}", (310, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Ground truth (if available)
        if result.ground_truth_event:
            gt_program = result.ground_truth_event.get('program', 'N/A')
            cv2.putText(vis, "GT:", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(vis, gt_program, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Reasoning snippet at bottom
        if result.reasoning and result.current_action != "Buffering...":
            # Bottom overlay
            overlay2 = vis.copy()
            cv2.rectangle(overlay2, (0, h - 40), (w, h), (0, 0, 0), -1)
            vis = cv2.addWeighted(overlay2, 0.7, vis, 0.3, 0)
            
            # Truncate reasoning to fit
            reasoning_text = result.reasoning[:80] + "..." if len(result.reasoning) > 80 else result.reasoning
            cv2.putText(vis, reasoning_text, (10, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return vis
    
    async def run_continuous(self, 
                            start_frame: int = 0,
                            end_frame: Optional[int] = None,
                            frame_skip: int = 15,  # Process every 0.5 seconds at 30fps
                            display: bool = True,
                            save_video: bool = False,
                            output_video_path: Optional[str] = None):
        """Run intent prediction on multiple frames.
        
        Args:
            start_frame: First frame to process
            end_frame: Last frame (None = all)
            frame_skip: Process every Nth frame
            display: Show preview window
            save_video: Save output video
            output_video_path: Path for output video
        """
        if end_frame is None:
            end_frame = self.total_frames - 1
        end_frame = min(end_frame, self.total_frames - 1)
        
        # Video writer - saves ALL frames with interpolated predictions
        out = None
        if save_video:
            if output_video_path is None:
                output_video_path = str(self.output_dir / "intent_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Write at original FPS for smooth playback
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
            print(f"📹 Saving output to: {output_video_path}")
        
        self.results = []
        predictions_made = 0
        current_result = None  # Track current prediction for interpolation
        
        print(f"\n{'='*60}")
        print(f"Processing frames {start_frame} to {end_frame} (skip={frame_skip})")
        print(f"{'='*60}")
        
        try:
            for frame_num in range(start_frame, end_frame + 1, frame_skip):
                result = await self.process_frame(frame_num, display=display)
                self.results.append(result)
                
                if result.current_action != "Buffering...":
                    predictions_made += 1
                    current_result = result
                
                # Write ALL intermediate frames to video with current prediction overlay
                if out:
                    # Calculate next prediction frame
                    next_frame_num = min(frame_num + frame_skip, end_frame + 1)
                    
                    # Write all frames from current to next
                    for f in range(frame_num, next_frame_num):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                        ret, frame = self.cap.read()
                        if ret:
                            # Use current result (or create buffering result)
                            if current_result:
                                # Update timestamp for this frame
                                interpolated_result = IntentTestResult(
                                    frame_number=f,
                                    timestamp_sec=f / self.fps,
                                    current_action=current_result.current_action,
                                    current_confidence=current_result.current_confidence,
                                    predicted_next=current_result.predicted_next,
                                    predicted_confidence=current_result.predicted_confidence,
                                    reasoning=current_result.reasoning,
                                    task_state=current_result.task_state,
                                    ground_truth_event=self._get_ground_truth_event(f / self.fps),
                                )
                            else:
                                interpolated_result = IntentTestResult(
                                    frame_number=f,
                                    timestamp_sec=f / self.fps,
                                    current_action="Initializing...",
                                    current_confidence=0.0,
                                    predicted_next="",
                                    predicted_confidence=0.0,
                                    reasoning="Collecting frames for prediction",
                                    task_state={},
                                )
                            
                            vis_frame = self._visualize(frame, interpolated_result)
                            out.write(vis_frame)
                
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[User quit]")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
        
        finally:
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Total frames processed: {len(self.results)}")
        print(f"Predictions made: {predictions_made}")
        
        if predictions_made > 0:
            valid_results = [r for r in self.results if r.current_action != "Buffering..."]
            avg_confidence = np.mean([r.current_confidence for r in valid_results])
            print(f"Average confidence: {avg_confidence:.2f}")
            
            # Action distribution
            actions = {}
            for r in valid_results:
                action = r.current_action
                if action not in actions:
                    actions[action] = 0
                actions[action] += 1
            
            print(f"\nAction distribution:")
            for action, count in sorted(actions.items(), key=lambda x: -x[1]):
                print(f"  - {action}: {count}")
        
        return self.results
    
    def save_results(self, output_path: Optional[str] = None):
        """Save results to JSON file."""
        if output_path is None:
            output_path = str(self.output_dir / "intent_results.json")
        
        data = {
            "video": self.video_path,
            "config": self.config_path,
            "timestamp": datetime.now().isoformat(),
            "num_frames": len(self.results),
            "results": [
                {
                    "frame": r.frame_number,
                    "timestamp": r.timestamp_sec,
                    "current_action": r.current_action,
                    "current_confidence": r.current_confidence,
                    "predicted_next": r.predicted_next,
                    "predicted_confidence": r.predicted_confidence,
                    "reasoning": r.reasoning,
                    "task_state": r.task_state,
                    "ground_truth": r.ground_truth_event,
                    "process_time": r.process_time_sec,
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_path}")
    
    def close(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test intent prediction on weigh bottles video"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=None,
        help="Process single frame"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start frame"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End frame"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=30,  # ~1 second at 30fps
        help="Frame skip (default: 30)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path"
    )
    
    args = parser.parse_args()
    
    # Default video path
    if args.video is None:
        args.video = str(AURA_ROOT / "demo_data" / "weigh_bottles" / "video.mp4")
    
    # Initialize tester
    tester = IntentTester(video_path=args.video, config_path=args.config)
    
    try:
        if args.frame is not None:
            # Single frame mode
            result = await tester.process_frame(args.frame, display=not args.headless)
            if not args.headless:
                cv2.waitKey(0)
        else:
            # Continuous mode
            await tester.run_continuous(
                start_frame=args.start,
                end_frame=args.end,
                frame_skip=args.skip,
                display=not args.headless,
                save_video=args.save_video,
                output_video_path=args.output
            )
        
        # Save results
        if args.save_results:
            tester.save_results()
    
    finally:
        tester.close()


if __name__ == "__main__":
    asyncio.run(main())

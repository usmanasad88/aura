#!/usr/bin/env python3
"""Run perception and gesture monitors for the weigh_bottles task.

This script processes the recorded video data from the Wizard-of-Oz experiment
and runs both the perception and gesture recognition systems.

Usage:
    # Run on recorded video files
    python run_weigh_bottles_monitors.py --video demo_data/weigh_bottles/video.mp4
    
    # Run on both videos side by side
    python run_weigh_bottles_monitors.py --video demo_data/weigh_bottles/video.mp4 \
                                         --gripper-video demo_data/weigh_bottles/exp.mp4
    
    # Run with live webcam
    python run_weigh_bottles_monitors.py --live
    
    # Run headless (no display, save output video)
    python run_weigh_bottles_monitors.py --video demo_data/weigh_bottles/video.mp4 \
                                         --headless --output output.mp4
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aura.utils.config import load_config
from aura.monitors.perception_module import PerceptionModule
from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

# Import task-specific monitors (these can be removed without breaking base code)
try:
    from aura.tasks.weigh_bottles.monitors import (
        WeighBottlesPerceptionMonitor,
        WeighBottlesGestureMonitor,
        WeighBottlesPerceptionConfig,
        WeighBottlesGestureConfig,
    )
    TASK_MONITORS_AVAILABLE = True
except ImportError:
    TASK_MONITORS_AVAILABLE = False
    print("Task-specific monitors not available, using base monitors")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeighBottlesMonitorRunner:
    """Runner for perception and gesture monitors on weigh_bottles task."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_task_monitors: bool = True,
    ):
        """Initialize the monitor runner.
        
        Args:
            config_path: Path to weigh_bottles.yaml config
            use_task_monitors: Use task-specific monitors (True) or base monitors (False)
        """
        self.config_path = config_path or str(REPO_ROOT / "config" / "weigh_bottles.yaml")
        self.use_task_monitors = use_task_monitors and TASK_MONITORS_AVAILABLE
        
        # Load config
        self.config = load_config(self.config_path)
        logger.info(f"Loaded config from {self.config_path}")
        
        # Load ground truth events
        self.events = self._load_events()
        
        # Initialize monitors
        self._init_monitors()
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        
    def _load_events(self) -> List[Dict[str, Any]]:
        """Load program events for ground truth comparison."""
        events_path = REPO_ROOT / "demo_data" / "weigh_bottles" / "program_events.json"
        if events_path.exists():
            with open(events_path) as f:
                data = json.load(f)
                return data.get("events", [])
        return []
    
    def _init_monitors(self):
        """Initialize perception and gesture monitors."""
        if self.use_task_monitors:
            logger.info("Initializing task-specific monitors...")
            self.perception = WeighBottlesPerceptionMonitor(
                config_path=self.config_path
            )
            self.gesture = WeighBottlesGestureMonitor()
        else:
            logger.info("Initializing base monitors...")
            # Use base monitors with config from file
            from aura.utils.config import PerceptionConfig
            
            self.perception = PerceptionModule(
                PerceptionConfig(
                    enabled=True,
                    use_sam3=False,
                    use_gemini_detection=True,
                    max_objects=10,
                    default_prompts=["bottle", "person", "hand", "scale"],
                )
            )
            self.gesture = GestureMonitor(
                GestureMonitorConfig(
                    enabled=True,
                    min_detection_confidence=0.5,
                    gesture_hold_frames=3,
                )
            )
        
        logger.info("Monitors initialized successfully")
    
    def _get_current_event(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get the current program event based on timestamp."""
        current_event = None
        for event in self.events:
            if event["timestamp"] <= timestamp:
                current_event = event
            else:
                break
        return current_event
    
    async def process_video(
        self,
        video_path: str,
        gripper_video_path: Optional[str] = None,
        display: bool = True,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        """Process recorded video through monitors.
        
        Args:
            video_path: Path to main video (3rd person view)
            gripper_video_path: Optional path to gripper camera video (360)
            display: Show visualization window
            output_path: Save output video to this path
            max_frames: Maximum frames to process (None = all)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.1f}")
        logger.info(f"  Duration: {duration:.1f}s, Frames: {total_frames}")
        
        # Open gripper video if provided
        cap_gripper = None
        if gripper_video_path:
            cap_gripper = cv2.VideoCapture(gripper_video_path)
            if cap_gripper.isOpened():
                logger.info(f"Gripper video: {gripper_video_path}")
            else:
                logger.warning(f"Could not open gripper video: {gripper_video_path}")
                cap_gripper = None
        
        # Setup output video writer
        out = None
        if output_path:
            output_width = width * 2 if cap_gripper else width
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
            logger.info(f"Saving output to: {output_path}")
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                video_timestamp = frame_count / fps
                
                if max_frames and frame_count > max_frames:
                    break
                
                # Read gripper frame if available
                gripper_frame = None
                if cap_gripper:
                    ret_g, gripper_frame = cap_gripper.read()
                    if not ret_g:
                        gripper_frame = None
                
                # Get current ground truth event
                current_event = self._get_current_event(video_timestamp)
                
                # Process through perception
                try:
                    if self.use_task_monitors:
                        perception_output = await self.perception.process_frame(frame)
                    else:
                        perception_output = await self.perception.process_frame(frame)
                except Exception as e:
                    logger.error(f"Perception error at frame {frame_count}: {e}")
                    perception_output = None
                
                # Process through gesture recognition
                try:
                    if self.use_task_monitors:
                        gesture_output = await self.gesture.update(frame=frame)
                    else:
                        gesture_output = await self.gesture.update(frame=frame)
                except Exception as e:
                    logger.error(f"Gesture error at frame {frame_count}: {e}")
                    gesture_output = None
                
                # Record results
                result = {
                    "frame": frame_count,
                    "timestamp": video_timestamp,
                    "ground_truth_event": current_event,
                    "perception": self._extract_perception_result(perception_output),
                    "gesture": self._extract_gesture_result(gesture_output),
                }
                self.results.append(result)
                
                # Visualize
                vis_frame = self._visualize(
                    frame, 
                    gripper_frame,
                    perception_output, 
                    gesture_output,
                    video_timestamp,
                    current_event,
                )
                
                # Display or save
                if display:
                    cv2.imshow("Weigh Bottles Monitor", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord(' '):
                        # Pause on space
                        cv2.waitKey(0)
                
                if out:
                    out.write(vis_frame)
                
                # Progress logging
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = frame_count / total_frames * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                               f"Time: {elapsed:.1f}s, Video time: {video_timestamp:.1f}s")
        
        finally:
            cap.release()
            if cap_gripper:
                cap_gripper.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Summary
        elapsed = time.time() - start_time
        logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s "
                   f"({frame_count/elapsed:.1f} fps)")
        
        return self.results
    
    async def run_live(
        self,
        camera_id: int = 0,
        duration: float = 60.0,
    ):
        """Run monitors on live webcam feed.
        
        Args:
            camera_id: Camera device ID
            duration: How long to run (seconds)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info(f"Running live on camera {camera_id} for {duration}s")
        logger.info("Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                elapsed = time.time() - start_time
                
                if elapsed > duration:
                    break
                
                # Process through monitors
                try:
                    if self.use_task_monitors:
                        perception_output = await self.perception.process_frame(frame)
                        gesture_output = await self.gesture.update(frame=frame)
                    else:
                        perception_output = await self.perception.process_frame(frame)
                        gesture_output = await self.gesture.update(frame=frame)
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    perception_output = None
                    gesture_output = None
                
                # Visualize
                vis_frame = self._visualize(
                    frame,
                    None,
                    perception_output,
                    gesture_output,
                    elapsed,
                    None,
                )
                
                cv2.imshow("Weigh Bottles Monitor (Live)", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                await asyncio.sleep(0.01)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s")
    
    def _extract_perception_result(self, output) -> Dict[str, Any]:
        """Extract key data from perception output."""
        if output is None:
            return {"detected_objects": [], "count": 0}
        
        objects = []
        # Support both 'objects' and 'tracked_objects' attribute names
        obj_list = getattr(output, 'objects', None) or getattr(output, 'tracked_objects', [])
        for obj in obj_list:
            objects.append({
                "label": getattr(obj, 'label', str(obj)),
                "confidence": getattr(obj, 'confidence', 0.0),
            })
        
        return {
            "detected_objects": objects,
            "count": len(objects),
        }
    
    def _extract_gesture_result(self, output) -> Dict[str, Any]:
        """Extract key data from gesture output."""
        if output is None:
            return {"gesture": None, "safety_triggered": False}
        
        return {
            "gesture": output.dominant_gesture,
            "safety_triggered": output.safety_triggered,
            "gestures": [
                {"name": g.gesture_name, "confidence": g.confidence}
                for g in getattr(output, 'gestures', [])
            ],
        }
    
    def _visualize(
        self,
        frame: np.ndarray,
        gripper_frame: Optional[np.ndarray],
        perception_output,
        gesture_output,
        timestamp: float,
        current_event: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Create visualization frame."""
        # Start with perception visualization
        if perception_output is not None:
            if self.use_task_monitors:
                vis = self.perception.visualize(frame, perception_output)
            else:
                vis = self.perception.visualize(frame, perception_output)
        else:
            vis = frame.copy()
        
        # Add gesture visualization
        if gesture_output is not None and self.use_task_monitors:
            vis = self.gesture.visualize(vis, gesture_output)
        elif gesture_output and gesture_output.dominant_gesture:
            cv2.putText(vis, f"Gesture: {gesture_output.dominant_gesture}",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add timestamp and event info
        cv2.putText(vis, f"Time: {timestamp:.2f}s", 
                   (10, vis.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if current_event:
            event_text = f"Event: {current_event.get('event', 'N/A')}"
            program = current_event.get('program')
            if program:
                event_text += f" - {program}"
            cv2.putText(vis, event_text,
                       (10, vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Combine with gripper view if available
        if gripper_frame is not None:
            # Resize gripper frame to match height
            h, w = vis.shape[:2]
            gripper_resized = cv2.resize(gripper_frame, (w, h))
            vis = np.hstack([vis, gripper_resized])
        
        return vis
    
    def save_results(self, output_path: str):
        """Save processing results to JSON."""
        with open(output_path, 'w') as f:
            json.dump({
                "config": self.config_path,
                "timestamp": datetime.now().isoformat(),
                "num_frames": len(self.results),
                "results": self.results,
            }, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run perception and gesture monitors for weigh_bottles task"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file (default: demo_data/weigh_bottles/video.mp4)"
    )
    parser.add_argument(
        "--gripper-video", "-g",
        type=str,
        default=None,
        help="Path to gripper camera video (360 degree)"
    )
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Use live webcam instead of video file"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera ID for live mode (default: 0)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="Duration for live mode in seconds (default: 60)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (for headless servers)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save output video to this path"
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--save-results", "-s",
        type=str,
        default=None,
        help="Save detection results to JSON file"
    )
    parser.add_argument(
        "--base-monitors",
        action="store_true",
        help="Use base monitors instead of task-specific monitors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/weigh_bottles.yaml)"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = WeighBottlesMonitorRunner(
        config_path=args.config,
        use_task_monitors=not args.base_monitors,
    )
    
    if args.live:
        # Live webcam mode
        await runner.run_live(
            camera_id=args.camera,
            duration=args.duration,
        )
    else:
        # Video file mode
        video_path = args.video
        if video_path is None:
            video_path = str(REPO_ROOT / "demo_data" / "weigh_bottles" / "video.mp4")
        
        await runner.process_video(
            video_path=video_path,
            gripper_video_path=args.gripper_video,
            display=not args.headless,
            output_path=args.output,
            max_frames=args.max_frames,
        )
    
    # Save results if requested
    if args.save_results:
        runner.save_results(args.save_results)


if __name__ == "__main__":
    asyncio.run(main())

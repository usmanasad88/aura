#!/usr/bin/env python3
"""Weigh Bottles Demo Runner.

Processes video files and uses the AURA framework to detect objects,
predict robot actions, and compare predictions to ground truth.

Usage:
    python -m tasks.weigh_bottles.demo.run_weigh_bottles_demo --video demo_data/weigh_bottles/video.mp4
    python -m tasks.weigh_bottles.demo.run_weigh_bottles_demo --video demo_data/weigh_bottles/video.mp4 --headless
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.monitors.perception_module import PerceptionModule
from aura.utils.config import PerceptionConfig
from aura.core import MonitorType

# Try to import gesture monitor
try:
    from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig
    GESTURE_AVAILABLE = True
except (ImportError, RuntimeError):
    GESTURE_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_number: int
    timestamp_sec: float
    ground_truth_event: Optional[Dict[str, Any]] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    gesture: Optional[str] = None
    safety_triggered: bool = False
    process_time_sec: float = 0.0


class WeighBottlesDemoRunner:
    """Main demo runner for weigh bottles task."""
    
    # Objects of interest for this task
    OBJECTS_OF_INTEREST = [
        "bottle",
        "resin bottle", 
        "hardener bottle",
        "scale",
        "weigh scale",
        "person",
        "hand",
        "table",
        "robot",
        "gripper",
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the demo runner.
        
        Args:
            config_path: Path to weigh_bottles.yaml config
        """
        self.config_path = config_path or str(TASK_DIR / "config" / "weigh_bottles.yaml")
        
        # Load task config
        if Path(self.config_path).exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
        else:
            self.config = {}
            logger.warning(f"Config not found: {self.config_path}, using defaults")
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Initialize monitors
        self._init_monitors()
        
        # Results
        self.results: List[FrameResult] = []
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth events."""
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
    
    def _init_monitors(self):
        """Initialize perception and gesture monitors."""
        # Get objects from config or use defaults
        objects = self.OBJECTS_OF_INTEREST
        perception_cfg = self.config.get('perception', {}).get('object_detection', {})
        if 'classes' in perception_cfg:
            objects = perception_cfg['classes']
        
        # Perception with SAM3
        self.perception_config = PerceptionConfig(
            monitor_type=MonitorType.PERCEPTION,
            enabled=True,
            update_rate_hz=2.0,
            use_sam3=True,  # Enable SAM3
            use_gemini_detection=False,
            default_prompts=objects,
            confidence_threshold=0.3,
            max_objects=len(objects),
        )
        
        logger.info(f"Initializing perception with SAM3, objects: {objects}")
        self.perception = PerceptionModule(self.perception_config)
        
        # Gesture monitor (optional)
        self.gesture = None
        if GESTURE_AVAILABLE:
            try:
                gesture_config = GestureMonitorConfig(
                    enabled=True,
                    min_detection_confidence=0.5,
                    gesture_hold_frames=3,
                )
                self.gesture = GestureMonitor(gesture_config)
                logger.info("Gesture monitor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize gesture monitor: {e}")
        else:
            logger.warning("Gesture monitor not available")
    
    def _get_current_event(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get the ground truth event at given timestamp."""
        current_event = None
        for event in self.ground_truth:
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
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_skip: int = 30,
    ) -> List[FrameResult]:
        """Process video through monitors.
        
        Args:
            video_path: Path to main video
            gripper_video_path: Optional 360 gripper camera video
            display: Show preview window
            output_path: Save output video to path
            start_frame: First frame to process
            end_frame: Last frame to process
            frame_skip: Process every Nth frame
            
        Returns:
            List of FrameResult for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if end_frame is None:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.1f}")
        logger.info(f"  Frames: {start_frame} to {end_frame} (skip={frame_skip})")
        
        # Output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        self.results = []
        start_time = datetime.now()
        
        try:
            for frame_num in range(start_frame, end_frame + 1, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = frame_num / fps
                current_event = self._get_current_event(timestamp)
                
                # Process perception
                t0 = datetime.now()
                perception_output = None
                detected_objects = []
                
                try:
                    perception_output = await self.perception.process_frame(frame)
                    if perception_output:
                        for obj in getattr(perception_output, 'objects', []):
                            detected_objects.append({
                                "name": getattr(obj, 'name', getattr(obj, 'label', 'unknown')),
                                "confidence": getattr(obj, 'confidence', 0.0),
                            })
                except Exception as e:
                    logger.error(f"Perception error at frame {frame_num}: {e}")
                
                # Process gesture
                gesture_name = None
                safety_triggered = False
                if self.gesture:
                    try:
                        gesture_output = await self.gesture.update(frame=frame)
                        if gesture_output:
                            gesture_name = gesture_output.dominant_gesture
                            safety_triggered = gesture_output.safety_triggered
                    except Exception as e:
                        logger.debug(f"Gesture error: {e}")
                
                process_time = (datetime.now() - t0).total_seconds()
                
                # Store result
                result = FrameResult(
                    frame_number=frame_num,
                    timestamp_sec=timestamp,
                    ground_truth_event=current_event,
                    detected_objects=detected_objects,
                    gesture=gesture_name,
                    safety_triggered=safety_triggered,
                    process_time_sec=process_time,
                )
                self.results.append(result)
                
                # Visualize
                vis_frame = self._visualize(frame, result, perception_output)
                
                if display:
                    cv2.imshow("Weigh Bottles Demo", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                if out:
                    out.write(vis_frame)
                
                # Progress
                progress = (frame_num - start_frame) / max(1, end_frame - start_frame) * 100
                if frame_num % (frame_skip * 10) == 0:
                    logger.info(f"Progress: {progress:.1f}% | Frame {frame_num} | "
                               f"Time: {timestamp:.1f}s | Objects: {len(detected_objects)}")
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processed {len(self.results)} frames in {elapsed:.1f}s")
        
        return self.results
    
    def _visualize(self, frame: np.ndarray, result: FrameResult, 
                   perception_output) -> np.ndarray:
        """Create visualization frame."""
        # Use perception visualization if available
        if perception_output is not None:
            vis = self.perception.visualize(frame, perception_output)
        else:
            vis = frame.copy()
        
        # Add timestamp and event info
        y_offset = 30
        cv2.putText(vis, f"Time: {result.timestamp_sec:.2f}s | Frame: {result.frame_number}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(vis, f"Objects: {len(result.detected_objects)}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Ground truth event
        if result.ground_truth_event:
            y_offset += 25
            event = result.ground_truth_event
            event_text = f"GT: {event.get('event', 'N/A')}"
            program = event.get('program')
            if program:
                event_text += f" - {program}"
            cv2.putText(vis, event_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Gesture
        if result.gesture:
            y_offset += 25
            color = (0, 0, 255) if result.safety_triggered else (255, 200, 0)
            cv2.putText(vis, f"Gesture: {result.gesture}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        data = {
            "config": self.config_path,
            "timestamp": datetime.now().isoformat(),
            "num_frames": len(self.results),
            "ground_truth_events": len(self.ground_truth),
            "results": [
                {
                    "frame": r.frame_number,
                    "timestamp": r.timestamp_sec,
                    "objects": r.detected_objects,
                    "gesture": r.gesture,
                    "safety_triggered": r.safety_triggered,
                    "ground_truth": r.ground_truth_event,
                    "process_time": r.process_time_sec,
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Total frames processed: {len(self.results)}")
        
        # Object detection stats
        all_objects = {}
        for r in self.results:
            for obj in r.detected_objects:
                name = obj['name']
                if name not in all_objects:
                    all_objects[name] = 0
                all_objects[name] += 1
        
        print(f"\nObject detections:")
        for name, count in sorted(all_objects.items(), key=lambda x: -x[1]):
            print(f"  - {name}: {count}")
        
        # Gesture stats
        gestures = [r.gesture for r in self.results if r.gesture]
        if gestures:
            from collections import Counter
            gesture_counts = Counter(gestures)
            print(f"\nGestures detected:")
            for gesture, count in gesture_counts.most_common():
                print(f"  - {gesture}: {count}")
        
        # Timing
        times = [r.process_time_sec for r in self.results]
        print(f"\nProcessing time:")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Min: {np.min(times):.2f}s")
        print(f"  Max: {np.max(times):.2f}s")


async def main():
    parser = argparse.ArgumentParser(
        description="Weigh Bottles Demo - AURA Proactive Assistance"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file"
    )
    parser.add_argument(
        "--gripper-video", "-g",
        type=str,
        default=None,
        help="Path to 360 gripper camera video"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file"
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
        default=30,
        help="Frame skip (default: 30)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save results to JSON"
    )
    
    args = parser.parse_args()
    
    # Default video path
    if args.video is None:
        args.video = str(AURA_ROOT / "demo_data" / "weigh_bottles" / "video.mp4")
    
    # Initialize runner
    runner = WeighBottlesDemoRunner(config_path=args.config)
    
    # Process video
    await runner.process_video(
        video_path=args.video,
        gripper_video_path=args.gripper_video,
        display=not args.headless,
        output_path=args.output,
        start_frame=args.start,
        end_frame=args.end,
        frame_skip=args.skip,
    )
    
    # Print summary
    runner.print_summary()
    
    # Save results
    if args.save_results:
        runner.save_results(args.save_results)


if __name__ == "__main__":
    asyncio.run(main())

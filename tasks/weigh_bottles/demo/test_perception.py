#!/usr/bin/env python3
"""Test perception monitor on weigh bottles video.

This script tests the perception module's ability to detect and segment
bottle weighing objects using SAM3.
"""

import sys
import asyncio
from pathlib import Path
import cv2
import yaml
import numpy as np
from datetime import datetime
from typing import Optional

# Add paths
TASK_DIR = Path(__file__).parent.parent  # tasks/weigh_bottles/
AURA_ROOT = Path(__file__).parent.parent.parent.parent  # aura/
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.monitors.perception_module import PerceptionModule
from aura.utils.config import PerceptionConfig
from aura.core import MonitorType


class PerceptionTester:
    """Test harness for perception module on weigh bottles video."""
    
    # Objects of interest for bottle weighing task
    # Use natural language for SAM3 text prompts
    OBJECTS_OF_INTEREST = [
        "plastic bottle",
        "bottle",
        "weighing scale",
        "digital scale",
        "person",
        "hand",
        "table",
        "robot arm",
        "gripper",
    ]
    
    def __init__(self, video_path: str, config_path: Optional[str] = None):
        """Initialize tester.
        
        Args:
            video_path: Path to video file
            config_path: Path to weigh_bottles YAML config (optional)
        """
        self.video_path = video_path
        self.config_path = config_path
        
        # Load config if provided
        objects_of_interest = self.OBJECTS_OF_INTEREST
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.task_config = yaml.safe_load(f)
            # Get objects from config if available
            perception_cfg = self.task_config.get('perception', {})
            if 'object_detection' in perception_cfg:
                objects_of_interest = perception_cfg['object_detection'].get(
                    'classes', self.OBJECTS_OF_INTEREST
                )
        else:
            self.task_config = {}
        
        # Create perception config with SAM3 enabled
        self.perception_config = PerceptionConfig(
            monitor_type=MonitorType.PERCEPTION,
            enabled=True,
            update_rate_hz=2.0,
            use_sam3=True,  # Enable SAM3 for segmentation
            use_gemini_detection=False,  # Use predefined prompts
            default_prompts=objects_of_interest,
            confidence_threshold=0.3,  # Lower threshold for small objects
            max_objects=len(objects_of_interest),
        )
        
        # Initialize perception module
        print(f"🔧 Initializing perception module with SAM3")
        print(f"   Objects of interest: {objects_of_interest}")
        self.perception = PerceptionModule(self.perception_config)
        
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
        
        # Output directory
        self.output_dir = TASK_DIR / "demo" / "perception_outputs"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    async def process_frame(self, frame_number: int, display: bool = True) -> dict:
        """Process a single frame.
        
        Args:
            frame_number: Frame number to process (0-indexed)
            display: Whether to display the frame
            
        Returns:
            Dict with results
        """
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return {"error": "Failed to read frame"}
        
        # Process with perception module
        print(f"\n🔍 Processing frame {frame_number}/{self.total_frames}...")
        start_time = datetime.now()
        
        perception_output = await self.perception.process_frame(frame)
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        if perception_output is None:
            print("❌ Perception failed!")
            return {"error": "Perception failed"}
        
        # Print results
        print(f"✅ Detection complete in {process_time:.2f}s")
        objects = getattr(perception_output, 'objects', [])
        print(f"   Found {len(objects)} objects:")
        
        # Group by category
        by_category = {}
        for obj in objects:
            name = getattr(obj, 'name', getattr(obj, 'label', 'unknown'))
            if name not in by_category:
                by_category[name] = []
            by_category[name].append(obj)
        
        for category, objs in by_category.items():
            avg_conf = np.mean([getattr(obj, 'confidence', 0.5) for obj in objs])
            print(f"   - {category}: {len(objs)} instances (avg conf: {avg_conf:.2f})")
            
            # Print details of each instance
            for i, obj in enumerate(objs):
                bbox = getattr(obj, 'bbox', None)
                if bbox:
                    if hasattr(bbox, 'x_max'):
                        w = bbox.x_max - bbox.x_min
                        h = bbox.y_max - bbox.y_min
                        x, y = bbox.x_min, bbox.y_min
                    else:
                        x, y = getattr(bbox, 'x1', 0), getattr(bbox, 'y1', 0)
                        w = getattr(bbox, 'x2', 0) - x
                        h = getattr(bbox, 'y2', 0) - y
                    has_mask = getattr(obj, 'mask', None) is not None
                    conf = getattr(obj, 'confidence', 0.0)
                    print(f"      [{i}] bbox: ({x:.0f}, {y:.0f}, {w:.0f}x{h:.0f}), "
                          f"conf: {conf:.3f}, mask: {has_mask}")
        
        # Visualize
        if display:
            vis_frame = self.perception.visualize(frame, perception_output)
            
            # Add frame info
            timestamp = frame_number / self.fps
            cv2.putText(vis_frame, f"Frame: {frame_number}/{self.total_frames} | Time: {timestamp:.2f}s", 
                       (10, vis_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Process time: {process_time:.2f}s | Objects: {len(objects)}", 
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Weigh Bottles Perception Test", vis_frame)
            
            # Save frame
            output_path = self.output_dir / f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(str(output_path), vis_frame)
            print(f"💾 Saved visualization to {output_path}")
        
        return {
            "frame_number": frame_number,
            "timestamp": frame_number / self.fps,
            "process_time": process_time,
            "objects": objects,
            "by_category": by_category
        }
    
    async def run_continuous(self, 
                            start_frame: int = 0,
                            end_frame: Optional[int] = None,
                            frame_skip: int = 30,
                            display: bool = True,
                            save_video: bool = False,
                            output_video_path: Optional[str] = None):
        """Run perception on multiple frames.
        
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
        
        # Video writer
        out = None
        if save_video:
            if output_video_path is None:
                output_video_path = str(self.output_dir / "perception_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 
                                 self.fps / frame_skip, (self.width, self.height))
            print(f"📹 Saving output to: {output_video_path}")
        
        results = []
        
        print(f"\n{'='*60}")
        print(f"Processing frames {start_frame} to {end_frame} (skip={frame_skip})")
        print(f"{'='*60}")
        
        try:
            for frame_num in range(start_frame, end_frame + 1, frame_skip):
                # Process frame and store the perception output
                result = await self._process_frame_for_video(frame_num, display=display, video_writer=out)
                results.append(result)
                
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[User quit]")
                        break
                    elif key == ord(' '):
                        # Pause on space
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
        successful = [r for r in results if 'error' not in r]
        print(f"Processed {len(successful)}/{len(results)} frames successfully")
        
        if successful:
            avg_time = np.mean([r['process_time'] for r in successful])
            total_objects = sum(len(r.get('objects', [])) for r in successful)
            print(f"Average process time: {avg_time:.2f}s per frame")
            print(f"Total objects detected: {total_objects}")
            
            # Category stats
            all_categories = {}
            for r in successful:
                for cat, objs in r.get('by_category', {}).items():
                    if cat not in all_categories:
                        all_categories[cat] = 0
                    all_categories[cat] += len(objs)
            
            print(f"Objects by category:")
            for cat, count in sorted(all_categories.items(), key=lambda x: -x[1]):
                print(f"  - {cat}: {count}")
        
        return results
    
    async def _process_frame_for_video(self, frame_number: int, display: bool = True, 
                                       video_writer = None) -> dict:
        """Process a single frame with video writing support.
        
        Args:
            frame_number: Frame number to process (0-indexed)
            display: Whether to display the frame
            video_writer: OpenCV VideoWriter object
            
        Returns:
            Dict with results
        """
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return {"error": "Failed to read frame"}
        
        # Process with perception module
        print(f"\n🔍 Processing frame {frame_number}/{self.total_frames}...")
        start_time = datetime.now()
        
        perception_output = await self.perception.process_frame(frame)
        
        process_time = (datetime.now() - start_time).total_seconds()
        
        if perception_output is None:
            print("❌ Perception failed!")
            return {"error": "Perception failed"}
        
        # Print results
        print(f"✅ Detection complete in {process_time:.2f}s")
        objects = getattr(perception_output, 'objects', [])
        print(f"   Found {len(objects)} objects:")
        
        # Group by category
        by_category = {}
        for obj in objects:
            name = getattr(obj, 'name', getattr(obj, 'label', 'unknown'))
            if name not in by_category:
                by_category[name] = []
            by_category[name].append(obj)
        
        for category, objs in by_category.items():
            avg_conf = np.mean([getattr(obj, 'confidence', 0.5) for obj in objs])
            print(f"   - {category}: {len(objs)} instances (avg conf: {avg_conf:.2f})")
            
            # Print details of each instance
            for i, obj in enumerate(objs):
                bbox = getattr(obj, 'bbox', None)
                if bbox:
                    if hasattr(bbox, 'x_max'):
                        w = bbox.x_max - bbox.x_min
                        h = bbox.y_max - bbox.y_min
                        x, y = bbox.x_min, bbox.y_min
                    else:
                        x, y = getattr(bbox, 'x1', 0), getattr(bbox, 'y1', 0)
                        w = getattr(bbox, 'x2', 0) - x
                        h = getattr(bbox, 'y2', 0) - y
                    has_mask = getattr(obj, 'mask', None) is not None
                    conf = getattr(obj, 'confidence', 0.0)
                    print(f"      [{i}] bbox: ({x:.0f}, {y:.0f}, {w:.0f}x{h:.0f}), "
                          f"conf: {conf:.3f}, mask: {has_mask}")
        
        # Visualize using the actual perception output
        vis_frame = self._visualize_safe(frame, perception_output)
        
        # Add frame info
        timestamp = frame_number / self.fps
        cv2.putText(vis_frame, f"Frame: {frame_number}/{self.total_frames} | Time: {timestamp:.2f}s", 
                   (10, vis_frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Process time: {process_time:.2f}s | Objects: {len(objects)}", 
                   (10, vis_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write to video
        if video_writer is not None:
            try:
                # Ensure frame is correct type for video writer
                if vis_frame.dtype != np.uint8:
                    vis_frame = vis_frame.astype(np.uint8)
                video_writer.write(vis_frame)
            except Exception as e:
                print(f"⚠️ Video write error: {e}")
        
        # Display
        if display:
            cv2.imshow("Weigh Bottles Perception Test", vis_frame)
        
        return {
            "frame_number": frame_number,
            "timestamp": frame_number / self.fps,
            "process_time": process_time,
            "objects": objects,
            "by_category": by_category
        }
    
    def _visualize_safe(self, frame: np.ndarray, perception_output) -> np.ndarray:
        """Safe visualization that handles mask edge cases.
        
        Args:
            frame: BGR image
            perception_output: PerceptionOutput object
            
        Returns:
            Visualization frame
        """
        try:
            return self.perception.visualize(frame, perception_output)
        except Exception as e:
            print(f"⚠️ Visualization error: {e}, using simple visualization")
            # Fallback to simple bounding box visualization
            vis_frame = frame.copy()
            objects = getattr(perception_output, 'objects', [])
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                     (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, obj in enumerate(objects):
                color = colors[i % len(colors)]
                bbox = getattr(obj, 'bbox', None)
                if bbox:
                    if hasattr(bbox, 'x_max'):
                        x1, y1 = int(bbox.x_min), int(bbox.y_min)
                        x2, y2 = int(bbox.x_max), int(bbox.y_max)
                    else:
                        x1, y1 = int(getattr(bbox, 'x1', 0)), int(getattr(bbox, 'y1', 0))
                        x2, y2 = int(getattr(bbox, 'x2', 0)), int(getattr(bbox, 'y2', 0))
                    
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    name = getattr(obj, 'name', 'object')
                    conf = getattr(obj, 'confidence', 0.0)
                    cv2.putText(vis_frame, f"{name}: {conf:.2f}", 
                               (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 2)
            
            return vis_frame
    
    def close(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test perception module on weigh bottles video"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file (default: demo_data/weigh_bottles/video.mp4)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (default: tasks/weigh_bottles/config/weigh_bottles.yaml)"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=None,
        help="Process single frame number"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start frame for continuous mode"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End frame for continuous mode"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=30,
        help="Frame skip interval (default: 30 = 1 frame per second at 30fps)"
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
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path"
    )
    
    args = parser.parse_args()
    
    # Default paths
    if args.video is None:
        args.video = str(AURA_ROOT / "demo_data" / "weigh_bottles" / "video.mp4")
    
    if args.config is None:
        args.config = str(TASK_DIR / "config" / "weigh_bottles.yaml")
    
    # Initialize tester
    tester = PerceptionTester(args.video, args.config)
    
    try:
        if args.frame is not None:
            # Single frame mode
            result = await tester.process_frame(args.frame, display=not args.headless)
            if not args.headless:
                print("\nPress any key to exit...")
                cv2.waitKey(0)
        else:
            # Continuous mode
            await tester.run_continuous(
                start_frame=args.start,
                end_frame=args.end,
                frame_skip=args.skip,
                display=not args.headless,
                save_video=args.save_video,
                output_video_path=args.output,
            )
    finally:
        tester.close()


if __name__ == "__main__":
    asyncio.run(main())

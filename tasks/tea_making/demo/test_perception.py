#!/usr/bin/env python3
"""Test perception monitor on tea making video.

This script tests the perception module's ability to detect and segment
tea-making objects from the 360 video frame by frame.
"""

import sys
import asyncio
from pathlib import Path
import cv2
import yaml
import numpy as np
from datetime import datetime

# Add paths
TASK_DIR = Path(__file__).parent.parent  # tasks/tea_making/
AURA_ROOT = Path(__file__).parent.parent.parent.parent  # aura/
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.monitors.perception_module import PerceptionModule
from aura.utils.config import PerceptionConfig
from aura.core import MonitorType


def extract_view_from_360(equirect_frame: np.ndarray, 
                         yaw: float = 0, 
                         pitch: float = 0, 
                         fov: float = 90,
                         output_size: tuple = (640, 480)) -> np.ndarray:
    """Extract perspective view from 360 equirectangular frame.
    
    Args:
        equirect_frame: Equirectangular image (2:1 ratio)
        yaw: Horizontal rotation in degrees (-180 to 180)
        pitch: Vertical rotation in degrees (-90 to 90)
        fov: Field of view in degrees
        output_size: (width, height) of output perspective image
        
    Returns:
        Perspective view as BGR image
    """
    height, width = equirect_frame.shape[:2]
    out_width, out_height = output_size
    
    # Create output meshgrid
    u = np.linspace(-1, 1, out_width)
    v = np.linspace(-1, 1, out_height)
    u, v = np.meshgrid(u, v)
    
    # Convert to spherical coordinates
    fov_rad = np.radians(fov)
    theta = np.arctan2(u, 1 / np.tan(fov_rad / 2))
    phi = np.arctan2(v, 1 / np.tan(fov_rad / 2))
    
    # Apply yaw and pitch rotation
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    
    # Rotate
    theta_rot = theta + yaw_rad
    phi_rot = phi + pitch_rad
    
    # Map to equirectangular coordinates
    x = ((theta_rot + np.pi) / (2 * np.pi)) * width
    y = ((phi_rot + np.pi/2) / np.pi) * height
    
    # Clamp to valid range
    x = np.clip(x, 0, width - 1).astype(np.float32)
    y = np.clip(y, 0, height - 1).astype(np.float32)
    
    # Remap
    perspective = cv2.remap(equirect_frame, x, y, cv2.INTER_LINEAR)
    
    return perspective


class PerceptionTester:
    """Test harness for perception module on tea making video."""
    
    def __init__(self, video_path: str, config_path: str):
        """Initialize tester.
        
        Args:
            video_path: Path to 360 video file
            config_path: Path to tea making YAML config
        """
        self.video_path = video_path
        self.config_path = config_path
        
        # Load config
        with open(config_path) as f:
            self.task_config = yaml.safe_load(f)
        
        # Create perception config from task config
        perception_cfg = self.task_config['monitors']['perception']
        
        # Get objects of interest from config
        objects_of_interest = perception_cfg.get('objects_of_interest', [
            "pot", "cup", "spoon", "hand", "person"
        ])
        
        self.perception_config = PerceptionConfig(
            monitor_type=MonitorType.PERCEPTION,
            enabled=perception_cfg.get('enabled', True),
            update_rate_hz=perception_cfg.get('update_rate_hz', 2.0),
            use_sam3=True,
            use_gemini_detection=False,  # Use config prompts instead
            default_prompts=objects_of_interest,
            confidence_threshold=0.3,  # Lower threshold for tea making objects
            max_objects=len(objects_of_interest),
        )
        
        # Initialize perception module
        print(f"🔧 Initializing perception module with prompts: {objects_of_interest}")
        self.perception = PerceptionModule(self.perception_config)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video loaded: {self.total_frames} frames @ {self.fps:.1f} fps")
        
        # 360 view parameters
        self.view_yaw = 0  # Front view
        self.view_pitch = 0  # Level
        self.view_fov = 90  # degrees
        self.output_size = (960, 720)  # Larger for better detection
    
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
        ret, equirect_frame = self.cap.read()
        
        if not ret:
            return {"error": "Failed to read frame"}
        
        # Extract perspective view
        frame = extract_view_from_360(
            equirect_frame,
            yaw=self.view_yaw,
            pitch=self.view_pitch,
            fov=self.view_fov,
            output_size=self.output_size
        )
        
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
        print(f"   Found {len(perception_output.objects)} objects:")
        
        # Group by category
        by_category = {}
        for obj in perception_output.objects:
            if obj.name not in by_category:
                by_category[obj.name] = []
            by_category[obj.name].append(obj)
        
        for category, objs in by_category.items():
            avg_conf = np.mean([obj.confidence for obj in objs])
            print(f"   - {category}: {len(objs)} instances (avg conf: {avg_conf:.2f})")
            
            # Print details of each instance
            for i, obj in enumerate(objs):
                bbox = obj.bbox
                w = bbox.x_max - bbox.x_min
                h = bbox.y_max - bbox.y_min
                has_mask = obj.mask is not None
                print(f"      [{i}] bbox: ({bbox.x_min:.0f}, {bbox.y_min:.0f}, {w:.0f}x{h:.0f}), "
                      f"conf: {obj.confidence:.3f}, mask: {has_mask}")
        
        # Visualize
        if display:
            vis_frame = self.perception.visualize(frame, perception_output)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {frame_number}/{self.total_frames}", 
                       (10, vis_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Process time: {process_time:.2f}s", 
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Perception Test", vis_frame)
            
            # Save frame
            output_dir = TASK_DIR / "demo" / "perception_outputs"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(str(output_path), vis_frame)
            print(f"💾 Saved visualization to {output_path}")
        
        return {
            "frame_number": frame_number,
            "process_time": process_time,
            "objects": perception_output.objects,
            "by_category": by_category
        }
    
    async def create_output_video(self, output_path: str, 
                                 start_frame: int = 0,
                                 end_frame: int = None,
                                 frame_skip: int = 1,
                                 preview: bool = False):
        """Create output video with perception overlays.
        
        Args:
            output_path: Path to save output video
            start_frame: First frame to process
            end_frame: Last frame to process (None = all)
            frame_skip: Process every Nth frame (1 = all frames)
            preview: Show preview window while processing
        """
        if end_frame is None:
            end_frame = self.total_frames - 1
        
        end_frame = min(end_frame, self.total_frames - 1)
        
        num_frames = (end_frame - start_frame + 1) // frame_skip
        print(f"\n{'='*60}")
        print(f"Creating output video with perception overlays")
        print(f"{'='*60}")
        print(f"Input video: {self.video_path}")
        print(f"Output video: {output_path}")
        print(f"Frames: {start_frame} to {end_frame} (skip={frame_skip}, total={num_frames})")
        print(f"Resolution: {self.output_size}")
        print(f"{'='*60}\n")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = self.fps / frame_skip
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, self.output_size)
        
        if not writer.isOpened():
            print(f"❌ Failed to create video writer at {output_path}")
            return
        
        # Process frames
        processed_count = 0
        start_time = datetime.now()
        
        for frame_idx in range(start_frame, end_frame + 1, frame_skip):
            # Process frame
            result = await self.process_frame(frame_idx, display=False)
            
            if 'error' in result:
                print(f"⚠️  Skipping frame {frame_idx}: {result['error']}")
                continue
            
            # Get visualization
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, equirect_frame = self.cap.read()
            if not ret:
                continue
            
            frame = extract_view_from_360(
                equirect_frame,
                yaw=self.view_yaw,
                pitch=self.view_pitch,
                fov=self.view_fov,
                output_size=self.output_size
            )
            
            # Recreate perception output
            from aura.core import PerceptionOutput
            perception_output = PerceptionOutput(
                timestamp=datetime.now(),
                objects=result['objects'],
                scene_description=f"Frame {frame_idx}"
            )
            
            vis_frame = self.perception.visualize(frame, perception_output)
            
            # Add timestamp and progress
            elapsed = (datetime.now() - start_time).total_seconds()
            time_per_frame = elapsed / (processed_count + 1) if processed_count > 0 else 0
            remaining_frames = (end_frame - frame_idx) // frame_skip
            eta_sec = remaining_frames * time_per_frame
            
            cv2.putText(vis_frame, f"Frame: {frame_idx}/{end_frame}", 
                       (10, vis_frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Time: {frame_idx/self.fps:.1f}s", 
                       (10, vis_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"ETA: {eta_sec:.0f}s", 
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            writer.write(vis_frame)
            processed_count += 1
            
            # Preview
            if preview:
                cv2.imshow("Video Creation Progress", vis_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("\n⚠️  Video creation cancelled by user")
                    break
            
            # Progress update
            if processed_count % 10 == 0:
                progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                print(f"  Progress: {progress:.1f}% ({processed_count}/{num_frames} frames, "
                      f"{time_per_frame:.2f}s/frame, ETA: {eta_sec:.0f}s)")
        
        # Cleanup
        writer.release()
        if preview:
            cv2.destroyAllWindows()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print("✅ Video creation complete!")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Frames processed: {processed_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average: {total_time/processed_count:.2f}s per frame")
        print(f"Output FPS: {output_fps:.2f}")
        print(f"{'='*60}\n")

    async def test_sample_frames(self, num_frames: int = 5, display: bool = True):
        """Test perception on sample frames throughout the video.
        
        Args:
            num_frames: Number of frames to sample
            display: Whether to display frames
        """
        print(f"\n{'='*60}")
        print(f"Testing perception on {num_frames} sample frames")
        print(f"{'='*60}\n")
        
        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, self.total_frames - 1, num_frames, dtype=int)
        
        results = []
        for idx in frame_indices:
            result = await self.process_frame(idx, display=display)
            results.append(result)
            
            if display:
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("Stopping early (q pressed)")
                    break
                elif key == ord('s'):
                    print("Skipping to next frame")
        
        if display:
            cv2.destroyAllWindows()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}\n")
        
        avg_time = np.mean([r.get('process_time', 0) for r in results if 'process_time' in r])
        print(f"Average processing time: {avg_time:.2f}s per frame")
        
        # Object detection statistics
        all_categories = set()
        for r in results:
            if 'by_category' in r:
                all_categories.update(r['by_category'].keys())
        
        print(f"\nDetected object categories: {sorted(all_categories)}")
        
        for category in sorted(all_categories):
            detections = [len(r.get('by_category', {}).get(category, [])) 
                         for r in results if 'by_category' in r]
            avg_detections = np.mean(detections) if detections else 0
            print(f"  {category}: avg {avg_detections:.1f} per frame")
        
        return results
    
    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test perception on tea making video")
    parser.add_argument("--video", type=str, 
                       default=str(AURA_ROOT / "demo_data" / "002.360"),
                       help="Path to 360 video file")
    parser.add_argument("--config", type=str,
                       default=str(TASK_DIR / "config" / "tea_making.yaml"),
                       help="Path to task config file")
    parser.add_argument("--frames", type=int, default=5,
                       help="Number of frames to test (for sample mode)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display frames (just save)")
    
    # Video output mode
    parser.add_argument("--create-video", action="store_true",
                       help="Create output video with perception overlays")
    parser.add_argument("--output", type=str,
                       default=str(TASK_DIR / "demo" / "perception_outputs" / "perception_output.mp4"),
                       help="Output video path")
    parser.add_argument("--start", type=int, default=0,
                       help="Start frame for video creation")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame for video creation (None = all)")
    parser.add_argument("--skip", type=int, default=15,
                       help="Process every Nth frame (default: 15 = 2fps from 30fps)")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview window during video creation")
    
    args = parser.parse_args()
    
    # Create tester
    tester = PerceptionTester(args.video, args.config)
    
    try:
        if args.create_video:
            # Create output video
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            await tester.create_output_video(
                output_path=str(output_path),
                start_frame=args.start,
                end_frame=args.end,
                frame_skip=args.skip,
                preview=args.preview
            )
        else:
            # Test on sample frames
            await tester.test_sample_frames(
                num_frames=args.frames,
                display=not args.no_display
            )
    finally:
        tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

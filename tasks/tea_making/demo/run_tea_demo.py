#!/usr/bin/env python3
"""Tea Making Demo Runner.

Processes a video file (including 360 video) and uses the AURA framework
to predict when robot actions should be executed. Compares predictions
to ground truth if available.

Usage:
    python -m tasks.tea_making.demo.run_tea_demo --video demo_data/002.360
    python -m tasks.tea_making.demo.run_tea_demo --video demo_data/002.LRV --headless
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
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from aura.core.scene_graph import (
    SemanticSceneGraph, GraphReasoner,
    ObjectNode, AgentNode, RegionNode,
    SSGEdge, SpatialRelation, SemanticRelation,
)
from aura.core.scene_graph.nodes import ObjectState, AgentState, Affordance
from aura.brain import DecisionEngine, SkillRegistry
from aura.brain.decision_engine import DecisionEngineConfig, ActionPrediction


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """A video frame with metadata."""
    frame: np.ndarray
    timestamp_sec: float
    frame_number: int


class VideoProcessor:
    """Processes video files including 360 video."""
    
    def __init__(self, video_path: str, target_fps: float = 2.0):
        """Initialize video processor.
        
        Args:
            video_path: Path to video file
            target_fps: Target frame rate for processing
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        self.total_frames = 0
        self.original_fps = 0
        self.duration_sec = 0
        self.frame_interval = 1
        
    def open(self) -> bool:
        """Open video file. Returns True if successful."""
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration_sec = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        
        # Calculate frame interval to achieve target FPS
        self.frame_interval = max(1, int(self.original_fps / self.target_fps))
        
        logger.info(f"Opened video: {self.video_path}")
        logger.info(f"  Duration: {self.duration_sec:.1f}s, FPS: {self.original_fps:.1f}")
        logger.info(f"  Total frames: {self.total_frames}, Processing interval: {self.frame_interval}")
        
        return True
    
    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self, frame_number: int = None) -> Optional[VideoFrame]:
        """Get a specific frame or next frame."""
        if not self.cap:
            return None
        
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = current_frame / self.original_fps if self.original_fps > 0 else 0
        
        return VideoFrame(
            frame=frame,
            timestamp_sec=timestamp,
            frame_number=current_frame
        )
    
    def extract_view_from_360(self, frame: np.ndarray, 
                               yaw: float = 0, pitch: float = 0,
                               fov: float = 90, output_size: tuple = (640, 480)) -> np.ndarray:
        """Extract a perspective view from equirectangular 360 video.
        
        Args:
            frame: Equirectangular frame
            yaw: Horizontal angle in degrees (-180 to 180)
            pitch: Vertical angle in degrees (-90 to 90)
            fov: Field of view in degrees
            output_size: (width, height) of output
        
        Returns:
            Perspective view as numpy array
        """
        h, w = frame.shape[:2]
        out_w, out_h = output_size
        
        # Create output pixel coordinates
        x = np.linspace(-1, 1, out_w)
        y = np.linspace(-1, 1, out_h)
        xx, yy = np.meshgrid(x, y)
        
        # Convert FOV to radians
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        # Calculate 3D coordinates on unit sphere
        zz = np.ones_like(xx) * f
        
        # Normalize
        norm = np.sqrt(xx**2 + yy**2 + zz**2)
        xx = xx / norm
        yy = yy / norm
        zz = zz / norm
        
        # Rotate by pitch (around x-axis)
        pitch_rad = np.radians(pitch)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        yy_rot = yy * cos_p - zz * sin_p
        zz_rot = yy * sin_p + zz * cos_p
        yy = yy_rot
        zz = zz_rot
        
        # Rotate by yaw (around y-axis)
        yaw_rad = np.radians(yaw)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        xx_rot = xx * cos_y + zz * sin_y
        zz_rot = -xx * sin_y + zz * cos_y
        xx = xx_rot
        zz = zz_rot
        
        # Convert to spherical coordinates
        theta = np.arctan2(xx, zz)  # longitude
        phi = np.arcsin(np.clip(yy, -1, 1))  # latitude
        
        # Map to equirectangular image coordinates
        u = (theta / np.pi + 1) / 2 * w
        v = (phi / (np.pi / 2) + 1) / 2 * h
        
        # Clamp to valid range
        u = np.clip(u, 0, w - 1).astype(np.float32)
        v = np.clip(v, 0, h - 1).astype(np.float32)
        
        # Remap
        output = cv2.remap(frame, u, v, cv2.INTER_LINEAR)
        
        return output
    
    def frames(self, max_frames: int = None, 
               extract_360_view: bool = True) -> "VideoProcessor":
        """Generator for video frames.
        
        Args:
            max_frames: Maximum number of frames to yield
            extract_360_view: If True, extract front view from 360 video
        """
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        while True:
            if max_frames and frame_count >= max_frames:
                break
            
            # Skip frames to achieve target FPS
            for _ in range(self.frame_interval - 1):
                self.cap.grab()
            
            video_frame = self.get_frame()
            if video_frame is None:
                break
            
            # Check if this is 360 video (2:1 aspect ratio)
            h, w = video_frame.frame.shape[:2]
            is_360 = abs(w / h - 2.0) < 0.1
            
            if is_360 and extract_360_view:
                # Extract front-facing view
                video_frame.frame = self.extract_view_from_360(
                    video_frame.frame,
                    yaw=0, pitch=-10,  # Slightly down to see table
                    fov=100,
                    output_size=(640, 480)
                )
            
            yield video_frame
            frame_count += 1


class TeaMakingDemo:
    """Demo for tea making proactive assistance."""
    
    def __init__(self, config_path: str = None, headless: bool = False):
        """Initialize demo.
        
        Args:
            config_path: Path to task config YAML
            headless: Run without display
        """
        self.headless = headless
        self.config = self._load_config(config_path)
        
        # Initialize decision engine with config
        engine_config = DecisionEngineConfig(
            gemini_model=self.config.get("llm", {}).get("model", "gemini-2.5-pro-preview-06-05"),
            enable_llm_reasoning=self.config.get("decision_engine", {}).get("enable_llm_reasoning", True),
            max_reasoning_time_sec=self.config.get("llm", {}).get("timeout_sec", 10.0),
            decision_interval_sec=self.config.get("decision_engine", {}).get("decision_interval_sec", 1.0),
            proactive_threshold=self.config.get("decision_engine", {}).get("proactive_threshold", 0.7),
        )
        
        self.engine = DecisionEngine(engine_config)
        
        # Load task files
        config_dir = TASK_DIR / "config"
        self.engine.load_task(
            dag_path=str(config_dir / "tea_making_dag.json"),
            state_path=str(config_dir / "tea_making_state.json"),
            skills_path=str(config_dir / "robot_skills.json"),
            initial_scene_path=str(config_dir / "initial_scene.json"),
        )
        
        # Load ground truth if available
        gt_path = config_dir / "ground_truth.json"
        if gt_path.exists():
            self.engine.load_ground_truth(str(gt_path))
        
        # Results tracking
        self.predictions: List[ActionPrediction] = []
        self.frame_log: List[Dict[str, Any]] = []
        
        logger.info("TeaMakingDemo initialized")
        logger.info(f"  LLM Model: {engine_config.gemini_model}")
        logger.info(f"  SSG Nodes: {self.engine.graph.node_count}")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load task configuration."""
        if config_path is None:
            config_path = TASK_DIR / "config" / "tea_making.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def process_frame(self, video_frame: VideoFrame) -> Optional[ActionPrediction]:
        """Process a single video frame.
        
        Args:
            video_frame: Frame to process
        
        Returns:
            ActionPrediction if robot should act, None otherwise
        """
        # Update current time
        self.engine.current_video_time_sec = video_frame.timestamp_sec
        
        # Simulate perception update (in real system, would use perception monitor)
        # For demo, we'll use simple heuristics based on time
        self._simulate_state_updates(video_frame)
        
        # Decide action
        prediction = await self.engine.decide_action(video_frame.timestamp_sec)
        
        if prediction:
            self.predictions.append(prediction)
            logger.info(f"[{video_frame.timestamp_sec:.1f}s] PREDICTION: {prediction.action_id} "
                       f"-> {prediction.target_id} (conf: {prediction.confidence:.2f})")
        
        return prediction
    
    def _simulate_state_updates(self, video_frame: VideoFrame) -> None:
        """Simulate state updates based on video timing.
        
        In a real system, this would come from perception/intent monitors.
        For demo, we simulate based on expected task timeline.
        """
        t = video_frame.timestamp_sec
        
        # Simple timeline-based state simulation
        if t > 30:
            self.engine.graph.set_task_state("workspace_setup", True)
        if t > 60:
            self.engine.graph.set_task_state("water_in_pot", True)
        if t > 70:
            self.engine.graph.set_task_state("pot_on_stove", True)
            self.engine.graph.set_task_state("stove_on", True)
        if t > 120:
            self.engine.graph.set_task_state("water_boiling", True)
        if t > 140:
            self.engine.graph.set_task_state("chai_added", True)
        if t > 170:
            self.engine.graph.set_task_state("tea_poured", True)
            self.engine.graph.set_task_state("cup_has_tea", True)
    
    async def run(self, video_path: str, max_frames: int = None,
                  decision_interval_sec: float = 2.0) -> Dict[str, Any]:
        """Run the demo on a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            decision_interval_sec: How often to make decisions
        
        Returns:
            Results dictionary with predictions and evaluation
        """
        processor = VideoProcessor(video_path, target_fps=1.0 / decision_interval_sec)
        
        if not processor.open():
            return {"error": f"Could not open video: {video_path}"}
        
        self.engine.start_task()
        last_decision_time = -decision_interval_sec
        
        try:
            for video_frame in processor.frames(max_frames=max_frames):
                # Only make decisions at intervals
                if video_frame.timestamp_sec - last_decision_time >= decision_interval_sec:
                    prediction = await self.process_frame(video_frame)
                    last_decision_time = video_frame.timestamp_sec
                    
                    # Log frame info
                    self.frame_log.append({
                        "time_sec": video_frame.timestamp_sec,
                        "frame_number": video_frame.frame_number,
                        "prediction": prediction.action_id if prediction else None,
                        "target": prediction.target_id if prediction else None,
                        "confidence": prediction.confidence if prediction else None,
                    })
                
                # Display frame if not headless
                if not self.headless:
                    display_frame = cv2.resize(video_frame.frame, (800, 600))
                    
                    # Add overlay
                    overlay_text = f"Time: {video_frame.timestamp_sec:.1f}s"
                    cv2.putText(display_frame, overlay_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.predictions:
                        last_pred = self.predictions[-1]
                        pred_text = f"Last: {last_pred.action_id} -> {last_pred.target_id}"
                        cv2.putText(display_frame, pred_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow("Tea Making Demo", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted")
                        break
        
        finally:
            processor.close()
            if not self.headless:
                cv2.destroyAllWindows()
        
        # Stop task and get summary
        summary = self.engine.stop_task()
        
        # Add predictions to summary
        summary["predictions"] = [
            {
                "time_sec": p.predicted_time_sec,
                "action_id": p.action_id,
                "target_id": p.target_id,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            for p in self.predictions
        ]
        summary["frame_log"] = self.frame_log
        
        return summary
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print results summary."""
        print("\n" + "=" * 60)
        print("TEA MAKING DEMO RESULTS")
        print("=" * 60)
        
        print(f"\nDuration: {results.get('duration_sec', 0):.1f} seconds")
        print(f"Decisions Made: {results.get('decisions_made', 0)}")
        print(f"Predictions: {len(results.get('predictions', []))}")
        
        if results.get("predictions"):
            print("\nPredictions:")
            for pred in results["predictions"]:
                print(f"  [{pred['time_sec']:.1f}s] {pred['action_id']} -> {pred['target_id']} "
                      f"(conf: {pred['confidence']:.2f})")
        
        if "evaluation" in results:
            eval_data = results["evaluation"]
            print("\nEvaluation vs Ground Truth:")
            print(f"  Accuracy: {eval_data.get('accuracy', 0) * 100:.1f}%")
            print(f"  Precision: {eval_data.get('precision', 0) * 100:.1f}%")
            print(f"  Recall: {eval_data.get('recall', 0) * 100:.1f}%")
            if eval_data.get("avg_timing_error"):
                print(f"  Avg Timing Error: {eval_data['avg_timing_error']:.2f}s")
        
        print("\nDecision Report:")
        print(self.engine.get_decision_report())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tea Making Demo")
    parser.add_argument("--video", type=str, default=str(AURA_ROOT / "demo_data" / "002.360"),
                       help="Path to video file")
    parser.add_argument("--headless", action="store_true",
                       help="Run without display")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Decision interval in seconds")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to task config YAML")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Check if GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set - LLM reasoning will be disabled")
    
    # Create and run demo
    demo = TeaMakingDemo(config_path=args.config, headless=args.headless)
    
    results = asyncio.run(demo.run(
        video_path=args.video,
        max_frames=args.max_frames,
        decision_interval_sec=args.interval,
    ))
    
    # Print results
    demo.print_results(results)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    return 0 if "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Combined test for IntentMonitor and MotionPredictor.

Tests both monitors together on a live webcam feed:
- IntentMonitor: What is the person doing? (Gemini-based, ~3s intervals)
- MotionPredictor: Where will their hands go? (MediaPipe-based, real-time)
"""

import asyncio
import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from aura.monitors.intent_monitor import IntentMonitor, IntentPrediction, visualize_intent
from aura.monitors.motion_predictor import MotionPredictor, visualize_motion_prediction
from aura.utils.config import IntentMonitorConfig, MotionPredictorConfig


async def run_combined_test(duration: float = 60.0):
    """Run both monitors on webcam feed.
    
    Args:
        duration: How long to run in seconds
    """
    print("=" * 60)
    print("Combined Intent + Motion Monitor Test")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print("Press 'q' to quit early")
    print()
    
    # Initialize IntentMonitor
    intent_config = IntentMonitorConfig(
        enabled=True,
        fps=2.0,
        capture_duration=2.0,
        prediction_interval=3.0,
        model="gemini-2.0-flash-exp"
    )
    intent_monitor = IntentMonitor(intent_config)
    print(f"✓ IntentMonitor initialized (Gemini: {intent_monitor.client is not None})")
    
    # Initialize MotionPredictor
    motion_config = MotionPredictorConfig(
        enabled=True,
        fps=15.0,
        window_duration=1.0,
        prediction_horizon=0.5
    )
    motion_predictor = MotionPredictor(motion_config)
    print(f"✓ MotionPredictor initialized (MediaPipe: {motion_predictor.hands is not None})")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✓ Webcam opened (640x480)")
    print()
    print("Starting...")
    print("-" * 60)
    
    start_time = time.time()
    frame_count = 0
    intent_count = 0
    motion_count = 0
    
    last_intent_prediction: IntentPrediction = None
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # --- Motion Prediction (fast, every frame) ---
            motion_output = await motion_predictor.update(frame=frame)
            if motion_output and motion_output.predictions:
                motion_count += 1
                vis_frame = visualize_motion_prediction(vis_frame, motion_output)
            
            # --- Intent Recognition (slow, periodic) ---
            intent_output = await intent_monitor.update(frame=frame)
            if intent_output and intent_output.intent:
                intent_count += 1
                # Get the raw prediction for visualization
                if intent_monitor.last_prediction:
                    last_intent_prediction = intent_monitor.last_prediction
                    print(f"\n[Intent #{intent_count}] {elapsed:.1f}s")
                    print(f"  Current: {last_intent_prediction.current_action} ({last_intent_prediction.current_action_confidence:.1%})")
                    print(f"  Next: {last_intent_prediction.predicted_next_action} ({last_intent_prediction.predicted_next_confidence:.1%})")
                    print(f"  Reasoning: {last_intent_prediction.reasoning[:80]}...")
            
            # Overlay intent on frame
            if last_intent_prediction:
                vis_frame = draw_intent_overlay(vis_frame, last_intent_prediction)
            
            # Draw status bar
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            status_text = f"Time: {elapsed:.1f}s | FPS: {fps_actual:.1f} | Intents: {intent_count} | Motions: {motion_count}"
            cv2.rectangle(vis_frame, (0, 0), (640, 30), (0, 0, 0), -1)
            cv2.putText(vis_frame, status_text, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Intent + Motion Monitor", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User quit]")
                break
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(motion_predictor, 'cleanup'):
            motion_predictor.cleanup()
    
    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Intent predictions: {intent_count}")
    print(f"  Motion predictions: {motion_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Avg FPS: {frame_count/elapsed:.1f}")
    
    return True


def draw_intent_overlay(frame: np.ndarray, prediction: IntentPrediction) -> np.ndarray:
    """Draw intent information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Current action
    current_text = f"Action: {prediction.current_action}"
    conf_text = f"({prediction.current_action_confidence:.0%})"
    cv2.putText(frame, current_text, (10, h - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, conf_text, (10 + len(current_text) * 15, h - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    
    # Predicted next action
    next_text = f"Next: {prediction.predicted_next_action}"
    next_conf = f"({prediction.predicted_next_confidence:.0%})"
    cv2.putText(frame, next_text, (10, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, next_conf, (10 + len(next_text) * 12, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    
    # Reasoning (truncated)
    reasoning = prediction.reasoning[:60] + "..." if len(prediction.reasoning) > 60 else prediction.reasoning
    cv2.putText(frame, reasoning, (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Test IntentMonitor + MotionPredictor together")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    args = parser.parse_args()
    
    asyncio.run(run_combined_test(args.duration))


if __name__ == "__main__":
    main()

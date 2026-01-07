#!/usr/bin/env python3
"""Enhanced combined test for IntentMonitor and MotionPredictor with gesture display.

Tests both monitors together on a live webcam feed:
- IntentMonitor: What is the person doing? (Gemini-based, ~3s intervals)
- MotionPredictor: Where will their hands go? + What gesture? (MediaPipe-based, real-time)
"""

import asyncio
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


async def run_combined_test_with_gesture(duration: float = 60.0):
    """Run both monitors on webcam feed with gesture display.
    
    Args:
        duration: How long to run in seconds
    """
    print("=" * 60)
    print("Combined Intent + Motion + Gesture Monitor Test")
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
    
    # Initialize MotionPredictor (with gesture recognition)
    motion_config = MotionPredictorConfig(
        enabled=True,
        fps=15.0,
        window_duration=1.0,
        prediction_interval=0.1
    )
    motion_predictor = MotionPredictor(motion_config)
    print(f"✓ MotionPredictor initialized (Hands: {motion_predictor.hands is not None}, Gestures: {motion_predictor.gesture_recognizer is not None})")
    print()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open camera")
        return False
    
    frame_count = 0
    intent_count = 0
    motion_count = 0
    gesture_history = {}
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Resize for processing
            frame = cv2.resize(frame, (640, 480))
            vis_frame = frame.copy()
            
            # Get intent prediction
            intent_output = await intent_monitor.update(frame=frame)
            if intent_output and intent_output.intent:
                intent_count += 1
                pred = IntentPrediction(
                    current_action=intent_output.intent.type.value,
                    current_action_confidence=intent_output.intent.confidence,
                    predicted_next_action=intent_output.alternatives[0].type.value if intent_output.alternatives else "Idle",
                    predicted_next_confidence=intent_output.alternatives[0].confidence if intent_output.alternatives else 0.0,
                    reasoning=intent_output.intent.reasoning,
                    task_state={}
                )
                vis_frame = visualize_intent(vis_frame, pred)
            
            # Get motion prediction with gesture
            motion_output = await motion_predictor.update(frame=frame)
            if motion_output and motion_output.predictions:
                motion_count += 1
                vis_frame = visualize_motion_prediction(vis_frame, motion_output)
                
                # Extract and display gesture if available
                pred = motion_output.predictions[0]
                
                # Try to get gesture from hand tracking result
                if hasattr(motion_predictor, 'last_hand_pose') and motion_predictor.last_hand_pose:
                    # Access the gesture from the pose's source data
                    gesture = None
                    if hasattr(motion_predictor, 'hand_history') and motion_predictor.hand_history:
                        last_track = list(motion_predictor.hand_history)[-1] if motion_predictor.hand_history else None
                        if last_track and hasattr(last_track, 'gesture'):
                            gesture = last_track.gesture
                            
                            if gesture and gesture != "None":
                                gesture_history[gesture] = gesture_history.get(gesture, 0) + 1
                                
                                # Draw gesture on frame
                                text_color = (0, 200, 100)
                                cv2.rectangle(vis_frame, (10, 10), (200, 60), (0, 0, 0), -1)
                                cv2.putText(vis_frame, f"Gesture: {gesture}", (20, 35),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Draw FPS
            cv2.putText(vis_frame, f"FPS: {frame_count / elapsed:.1f} | Frame: {frame_count}",
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Intent + Motion + Gesture Monitor", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User quit]")
                break
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)
            
            if elapsed > duration:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Intent predictions: {intent_count}")
    print(f"  Motion predictions: {motion_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Avg FPS: {frame_count/elapsed:.1f}")
    
    if gesture_history:
        print()
        print("Detected Gestures:")
        for gesture, count in sorted(gesture_history.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {gesture}: {count} times")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test IntentMonitor + MotionPredictor + Gestures together")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    args = parser.parse_args()
    
    try:
        success = asyncio.run(run_combined_test_with_gesture(args.duration))
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

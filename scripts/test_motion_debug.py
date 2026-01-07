#!/usr/bin/env python3
"""Debug script for MotionPredictor hand detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import time
from aura.monitors.motion_predictor import MotionPredictor
from aura.utils.config import MotionPredictorConfig

def test_motion_predictor_detection():
    """Test hand detection directly."""
    print("Testing MotionPredictor Hand Detection")
    print("=" * 60)
    
    # Initialize config
    config = MotionPredictorConfig(
        enabled=True,
        fps=15.0,
        window_duration=1.0,
        prediction_interval=0.2
    )
    
    # Initialize predictor
    predictor = MotionPredictor(config)
    print(f"✓ MotionPredictor initialized")
    print(f"  - Hands object: {predictor.hands is not None}")
    print(f"  - FPS: {predictor.fps}")
    print()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return
    
    print("Starting hand detection (Press 'q' to quit)")
    print("-" * 60)
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test hand tracking directly
            result = predictor._track_hands(frame, time.time())
            
            if result:
                detection_count += 1
                print(f"[Frame {frame_count}] ✓ Hand detected!")
                print(f"  - Handedness: {result.handedness}")
                print(f"  - Confidence: {result.confidence:.2f}")
                print(f"  - Wrist position: ({result.wrist_position[0]:.2f}, {result.wrist_position[1]:.2f})")
                print(f"  - Landmarks: {len(result.landmarks)}")
            else:
                if frame_count % 10 == 0:
                    print(f"[Frame {frame_count}] - No hands detected")
            
            # Display frame
            cv2.imshow("Hand Detection Debug", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Limit to 10 seconds
            if time.time() - start_time > 10:
                print("\n(10 second timeout reached)")
                break
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    print("-" * 60)
    elapsed = time.time() - start_time
    print(f"Test Results:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Hands detected: {detection_count}")
    print(f"  - Detection rate: {(detection_count/frame_count*100) if frame_count > 0 else 0:.1f}%")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Avg FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    test_motion_predictor_detection()

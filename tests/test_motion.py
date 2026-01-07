#!/usr/bin/env python3
"""Test script for MotionPredictor (MediaPipe hand tracking).

Tests the lightweight hand tracking and trajectory prediction.
"""

import asyncio
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from aura.monitors.motion_predictor import (
    MotionPredictor, 
    HandTrackingResult,
    visualize_motion_prediction,
    MEDIAPIPE_AVAILABLE
)
from aura.utils.config import MotionPredictorConfig


def test_motion_predictor_init():
    """Test MotionPredictor initialization."""
    print("\n=== Test 1: MotionPredictor Initialization ===")
    
    config = MotionPredictorConfig(
        enabled=True,
        fps=15.0,
        window_duration=1.0,
        prediction_horizon=0.5
    )
    
    predictor = MotionPredictor(config)
    print(f"✓ MotionPredictor created")
    print(f"  Monitor type: {predictor.monitor_type}")
    print(f"  FPS: {predictor.fps}")
    print(f"  Window duration: {predictor.window_duration}s")
    print(f"  Max frames: {predictor.max_frames}")
    print(f"  MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    print(f"  Hands tracker: {predictor.hands is not None}")
    return True


def test_hand_tracking_result():
    """Test HandTrackingResult dataclass."""
    print("\n=== Test 2: HandTrackingResult Dataclass ===")
    
    result = HandTrackingResult(
        timestamp=time.time(),
        landmarks=[(0.5, 0.5, 0.0)] * 21,  # 21 hand landmarks
        handedness="Right",
        confidence=0.95,
        wrist_position=(0.5, 0.5)
    )
    
    print(f"✓ HandTrackingResult created")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Landmarks: {len(result.landmarks)} points")
    print(f"  Handedness: {result.handedness}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Wrist position: {result.wrist_position}")
    return True


async def test_trajectory_prediction():
    """Test trajectory prediction with synthetic data."""
    print("\n=== Test 3: Trajectory Prediction ===")
    
    config = MotionPredictorConfig(
        enabled=True,
        fps=10.0,
        window_duration=0.5,
        prediction_horizon=0.3
    )
    
    predictor = MotionPredictor(config)
    
    # Add synthetic hand history (simulating right hand moving right)
    for i in range(5):
        result = HandTrackingResult(
            timestamp=time.time() - (4-i) * 0.1,
            landmarks=[(0.3 + i*0.05, 0.5, 0.0)] * 21,
            handedness="Right",
            confidence=0.9,
            wrist_position=(0.3 + i*0.05, 0.5)
        )
        predictor.hand_history.append(result)
    
    # Predict trajectory
    prediction = predictor._predict_trajectory()
    
    if prediction:
        print(f"✓ Trajectory prediction successful")
        print(f"  Trajectory points: {len(prediction.predicted_trajectory.poses) if prediction.predicted_trajectory else 0}")
        print(f"  Confidence: {prediction.confidence}")
    else:
        print("⚠ No prediction (may need more history)")
    
    return True


async def test_live_tracking(duration: float = 10.0):
    """Test live hand tracking with webcam.
    
    Args:
        duration: How long to run the test in seconds
    """
    print(f"\n=== Test 4: Live Hand Tracking ({duration}s) ===")
    print("Press 'q' to quit early")
    
    if not MEDIAPIPE_AVAILABLE:
        print("✗ MediaPipe not available, skipping live test")
        return False
    
    config = MotionPredictorConfig(
        enabled=True,
        fps=15.0,
        window_duration=1.0,
        prediction_horizon=0.5
    )
    
    predictor = MotionPredictor(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    frame_count = 0
    tracking_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Update predictor
            output = await predictor.update(frame=frame)
            
            if output and output.current_pose:
                tracking_count += 1
            
            # Visualize
            if output:
                frame = visualize_motion_prediction(frame, output)
            
            # Add status
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"Motion Tracker - {elapsed:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_actual:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands tracked: {tracking_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Motion Predictor Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.001)  # Small delay for responsiveness
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n✓ Motion tracking test complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Frames with hands: {tracking_count}")
    print(f"  Detection rate: {tracking_count/frame_count*100:.1f}%")
    return True


def test_visualization():
    """Test motion visualization without webcam."""
    print("\n=== Test 5: Motion Visualization ===")
    
    from aura.core import MotionOutput, PredictedMotion, Trajectory, Pose3D
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create mock trajectory
    trajectory = Trajectory(
        poses=[
            Pose3D(x=0.5, y=0.5, z=0),
            Pose3D(x=0.55, y=0.45, z=0),
            Pose3D(x=0.6, y=0.4, z=0),
            Pose3D(x=0.65, y=0.35, z=0),
            Pose3D(x=0.7, y=0.3, z=0),
        ],
        timestamps=[0.0, 0.1, 0.2, 0.3, 0.4]
    )
    
    predicted_motion = PredictedMotion(
        entity_id="human_hand",
        predicted_trajectory=trajectory,
        confidence=0.85,
        prediction_horizon_sec=0.5
    )
    
    output = MotionOutput(
        timestamp=datetime.now(),
        predictions=[predicted_motion],
        collision_risk=0.0
    )
    
    result = visualize_motion_prediction(frame, output)
    
    print(f"✓ Visualization created: {result.shape}")
    
    # Save visualization
    cv2.imwrite("/tmp/motion_visualization_test.png", result)
    print(f"  Saved to /tmp/motion_visualization_test.png")
    return True


def run_unit_tests():
    """Run unit tests without webcam."""
    print("=" * 50)
    print("MotionPredictor Unit Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("Initialization", test_motion_predictor_init()))
    results.append(("HandTrackingResult", test_hand_tracking_result()))
    results.append(("Trajectory Prediction", asyncio.run(test_trajectory_prediction())))
    results.append(("Visualization", test_visualization()))
    
    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    return all(r[1] for r in results)


def main():
    parser = argparse.ArgumentParser(description="Test MotionPredictor (MediaPipe hand tracking)")
    parser.add_argument("--live", action="store_true", help="Run live webcam test")
    parser.add_argument("--duration", type=float, default=10.0, help="Live test duration (seconds)")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    args = parser.parse_args()
    
    if args.unit:
        success = run_unit_tests()
    elif args.live:
        success = asyncio.run(test_live_tracking(args.duration))
    else:
        # Default: run unit tests
        success = run_unit_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

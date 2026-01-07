#!/usr/bin/env python
"""Test script for motion prediction monitor."""

import asyncio
import argparse
import logging
import time
import cv2
import numpy as np

from aura.monitors.motion_predictor import (
    MotionPredictor, visualize_motion_prediction
)
from aura.utils.config import MotionPredictorConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_webcam(
    duration: float = 30.0,
    display: bool = True,
    use_hand_tracking: bool = True
):
    """Test motion prediction with webcam.
    
    Args:
        duration: How long to run (seconds)
        display: Show visualization window
        use_hand_tracking: Use MediaPipe hand tracking
    """
    # Create config
    config = MotionPredictorConfig(
        enabled=True,
        fps=2.0,
        window_duration=2.0,
        prediction_horizon=2.0,
        use_hand_tracking=use_hand_tracking,
        model="gemini-2.0-flash-exp"
    )
    
    # Create monitor
    predictor = MotionPredictor(config)
    logger.info("Motion predictor initialized")
    logger.info(f"Capturing at {config.fps} fps, {config.window_duration}s window")
    logger.info(f"Hand tracking: {'enabled' if use_hand_tracking else 'disabled'}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Webcam opened successfully")
    logger.info(f"Running for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    prediction_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame
            output = await predictor.update(frame=frame)
            
            if output:
                prediction_count += 1
                logger.info(f"Prediction {prediction_count}:")
                logger.info(f"  Intent: {output.intent.type.name} ({output.intent.confidence:.2f})")
                logger.info(f"  Description: {output.intent.description}")
                
                if output.predicted_motion:
                    logger.info(f"  Trajectory points: {len(output.predicted_motion.trajectory_points)}")
                    logger.info(f"  Time horizon: {output.predicted_motion.time_horizon:.1f}s")
                
                if output.current_pose:
                    logger.info(f"  Hand joints detected: {len(output.current_pose.joints)}")
            
            # Visualize
            if display and output:
                vis_frame = visualize_motion_prediction(
                    frame,
                    output.intent,
                    output.predicted_motion,
                    output.current_pose
                )
                cv2.imshow("Motion Prediction", vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"motion_prediction_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    logger.info(f"Saved screenshot to {filename}")
            elif not output and display:
                # Show frame while buffering
                cv2.putText(frame, "Buffering frames...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Motion Prediction", frame)
                cv2.waitKey(1)
    
    finally:
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 50)
        logger.info("Test Summary:")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Frames processed: {frame_count}")
        logger.info(f"  Predictions made: {prediction_count}")
        logger.info(f"  FPS: {frame_count / elapsed:.1f}")
        logger.info("=" * 50)
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
        predictor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Test motion prediction monitor")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Test duration in seconds")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable visualization window")
    parser.add_argument("--no-hand-tracking", action="store_true",
                       help="Disable MediaPipe hand tracking")
    
    args = parser.parse_args()
    
    asyncio.run(test_webcam(
        duration=args.duration,
        display=not args.no_display,
        use_hand_tracking=not args.no_hand_tracking
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test script for gesture recognition monitor.

This script demonstrates how to use the GestureMonitor for:
- Real-time gesture detection
- Safety control signals
- Intent mapping from gestures

Usage:
    python scripts/test_gesture_monitor.py [--camera_id 0] [--show_viz]
"""

import asyncio
import argparse
import logging
import sys
import cv2

from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig
from aura.core import IntentType


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main test loop."""
    parser = argparse.ArgumentParser(description='Test gesture recognition monitor')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--show_viz', action='store_true',
                       help='Show visualization window')
    parser.add_argument('--use_realsense', action='store_true',
                       help='Use Intel RealSense camera')
    parser.add_argument('--fps', type=int, default=10,
                       help='Processing FPS (default: 10)')
    parser.add_argument('--num_hands', type=int, default=2,
                       help='Maximum number of hands to detect (default: 2)')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to gesture_recognizer.task model')
    
    args = parser.parse_args()
    
    # Initialize camera
    if args.use_realsense:
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            logger.info("Intel RealSense initialized")
            
            def get_frame():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                import numpy as np
                return np.asanyarray(color_frame.get_data())
        except ImportError:
            logger.error("pyrealsense2 not installed. Install with: pip install pyrealsense2")
            return
        except Exception as e:
            logger.error(f"Failed to initialize RealSense: {e}")
            logger.info("Falling back to standard webcam...")
            args.use_realsense = False
    
    if not args.use_realsense:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {args.camera_id}")
            return
        logger.info(f"Standard webcam {args.camera_id} initialized")
        
        def get_frame():
            ret, frame = cap.read()
            return frame if ret else None
    
    # Initialize gesture monitor
    config = GestureMonitorConfig(
        enabled=True,
        update_rate_hz=args.fps,
        timeout_sec=5.0,
        model_path=args.model_path,
        num_hands=args.num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        gesture_hold_frames=3,
        stop_gestures={'Open_Palm', 'Pointing_Up'},
        resume_gestures={'Thumb_Up', 'Victory'},
        enable_intent_mapping=True,
    )
    
    monitor = GestureMonitor(config)
    logger.info("GestureMonitor initialized")
    logger.info("Press 'q' to quit, 'r' to reset safety state")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            frame = get_frame()
            if frame is None:
                logger.warning("Failed to capture frame")
                await asyncio.sleep(0.1)
                continue
            
            # Process with gesture monitor
            output = await monitor.update(frame=frame)
            
            if not output.is_valid:
                logger.error(f"Monitor error: {output.error}")
                continue
            
            # Log results every 30 frames (~3 seconds at 10 fps)
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info("-" * 60)
                logger.info(f"Frame {frame_count}")
                
                if output.gestures:
                    for gesture in output.gestures:
                        logger.info(
                            f"  Detected: {gesture.gesture_name} "
                            f"({gesture.confidence:.2f}) - {gesture.handedness} hand"
                        )
                else:
                    logger.info("  No gestures detected")
                
                if output.dominant_gesture:
                    logger.info(f"  Dominant gesture: {output.dominant_gesture}")
                
                logger.info(f"  Safety status: {'STOP' if output.safety_triggered else 'SAFE'}")
                
                if output.intent:
                    logger.info(f"  Mapped intent: {output.intent.type.name}")
                
                # Show statistics
                stats = monitor.get_gesture_statistics()
                if stats:
                    logger.info(f"  Recent gestures: {stats.get('recent_gestures', [])}")
            
            # Visualization
            if args.show_viz:
                viz_frame = monitor.get_visualization_frame(frame, output)
                cv2.imshow("Gesture Monitor Test", viz_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('r'):
                    monitor.reset_safety()
                    logger.info("Safety state reset")
            
            # Control loop rate
            await asyncio.sleep(1.0 / args.fps)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        if args.use_realsense:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics")
        logger.info("=" * 60)
        stats = monitor.get_gesture_statistics()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python
"""Test script for perception module."""

import argparse
import asyncio
import time
import logging
import sys
import cv2
import numpy as np

from aura.monitors.perception_module import PerceptionModule, PerceptionConfig
from aura.utils.config import load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_webcam(
    duration: float = 10.0, 
    auto_detect: bool = True,
    display: bool = True,
    max_frames: int = 100
):
    """Test perception with webcam input.
    
    Args:
        duration: How long to run (seconds)
        auto_detect: Use Gemini to auto-detect objects
        display: Show visualization window
        max_frames: Maximum frames to process
    """
    # Load config
    config = load_config()
    
    # Override settings for test
    config.monitors.perception.use_gemini_detection = auto_detect
    config.monitors.perception.use_sam3 = True
    
    # Create perception module
    logger.info("Initializing perception module...")
    perception = PerceptionModule(config.monitors.perception)
    
    # Open webcam
    logger.info("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Setup video writer if not displaying
    out = None
    if not display:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('perception_test_output.mp4', fourcc, 10.0, (640, 480))
        logger.info(f"Saving to perception_test_output.mp4 (max {max_frames} frames)")
    else:
        logger.info("Press 'q' to quit")
    
    start_time = time.time()
    frame_count = 0
    process_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            t0 = time.time()
            output = await perception.process_frame(frame)
            process_time = time.time() - t0
            process_times.append(process_time)
            
            if output is None:
                logger.warning("Processing returned None")
                continue
            
            # Visualize
            vis_frame = perception.visualize(frame, output)
            
            # Add FPS info
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, vis_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display or save
            if display:
                cv2.imshow("Perception Test", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                out.write(vis_frame)
            
            frame_count += 1
            
            # Log progress
            if frame_count % 10 == 0:
                logger.info(f"Frame {frame_count} | {fps:.1f} FPS | "
                           f"{len(output.objects)} objects | "
                           f"Process time: {process_time:.3f}s")
                
                for obj in output.objects:
                    logger.info(f"  - {obj.name}: conf={obj.confidence:.3f}, "
                               f"bbox=({obj.bbox.x_min:.0f}, {obj.bbox.y_min:.0f}, "
                               f"{obj.bbox.x_max:.0f}, {obj.bbox.y_max:.0f})")
            
            # Check termination conditions
            if elapsed >= duration or frame_count >= max_frames:
                logger.info(f"Reached limit (duration={elapsed:.1f}s, frames={frame_count})")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        elif out is not None:
            out.release()
            logger.info("Output saved to perception_test_output.mp4")
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary:")
    logger.info(f"  Total frames: {frame_count}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Average FPS: {frame_count/elapsed:.1f}")
    if process_times:
        logger.info(f"  Avg process time: {np.mean(process_times):.3f}s")
        logger.info(f"  Min process time: {np.min(process_times):.3f}s")
        logger.info(f"  Max process time: {np.max(process_times):.3f}s")
    logger.info(f"{'='*60}")


async def test_image(image_path: str, prompts: list = None):
    """Test perception on a single image.
    
    Args:
        image_path: Path to input image
        prompts: List of object prompts to detect
    """
    # Load config
    config = load_config()
    
    # Override prompts if provided
    if prompts:
        config.monitors.perception.default_prompts = prompts
        config.monitors.perception.use_gemini_detection = False
    else:
        config.monitors.perception.use_gemini_detection = True
    
    # Create perception module
    logger.info("Initializing perception module...")
    perception = PerceptionModule(config.monitors.perception)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Process
    logger.info("Processing...")
    output = await perception.process_frame(frame)
    
    if output is None:
        logger.error("Processing failed")
        return
    
    # Visualize
    vis_frame = perception.visualize(frame, output)
    
    # Save output
    output_path = image_path.replace('.', '_output.')
    cv2.imwrite(output_path, vis_frame)
    logger.info(f"Output saved to: {output_path}")
    
    # Print results
    logger.info(f"\nDetected {len(output.objects)} objects:")
    for obj in output.objects:
        logger.info(f"  - {obj.name}: conf={obj.confidence:.3f}, "
                   f"bbox=({obj.bbox.x_min:.0f}, {obj.bbox.y_min:.0f}, "
                   f"{obj.bbox.x_max:.0f}, {obj.bbox.y_max:.0f})")
        if obj.mask is not None:
            mask_area = obj.mask.sum()
            logger.info(f"    mask area: {mask_area} pixels")


def main():
    parser = argparse.ArgumentParser(description="Test perception module")
    parser.add_argument("--input", default="webcam", 
                       help="Input source: 'webcam' or path to image")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Duration for webcam test (seconds)")
    parser.add_argument("--prompts", nargs="+", default=None,
                       help="Object prompts to detect (for image mode)")
    parser.add_argument("--no-auto-detect", action="store_true",
                       help="Disable Gemini auto-detection")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't show visualization window")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Run appropriate test
    if args.input == "webcam":
        asyncio.run(test_webcam(
            duration=args.duration,
            auto_detect=not args.no_auto_detect,
            display=not args.no_display,
            max_frames=args.max_frames
        ))
    else:
        asyncio.run(test_image(args.input, args.prompts))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple example demonstrating gesture-based robot safety control.

This example shows how to use the GestureMonitor for basic safety control.
Run this to see gestures in action!

Usage:
    python examples/simple_gesture_safety.py
"""

import asyncio
import cv2
from aura.monitors import GestureMonitor, GestureMonitorConfig


async def main():
    """Simple gesture safety control demo."""
    print("=" * 60)
    print("AURA Gesture Safety Control Example")
    print("=" * 60)
    print()
    print("Recognized Safety Gestures:")
    print("  🛑 STOP:   Open Palm, Pointing Up")
    print("  ✅ RESUME: Thumbs Up, Victory")
    print()
    print("Press 'q' to quit, 'r' to reset safety state")
    print("=" * 60)
    print()
    
    # Configure gesture monitor
    config = GestureMonitorConfig(
        stop_gestures={'Open_Palm', 'Pointing_Up'},
        resume_gestures={'Thumb_Up', 'Victory'},
        num_hands=2,
        gesture_hold_frames=3,
        enable_intent_mapping=True,
    )
    
    # Create monitor
    monitor = GestureMonitor(config)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    print("✅ Camera opened successfully")
    print("✅ Gesture monitor initialized")
    print()
    
    # Simulated robot state
    robot_running = True
    robot_position = 0.0
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to capture frame")
                await asyncio.sleep(0.1)
                continue
            
            # Process gestures
            output = await monitor.update(frame=frame)
            
            if not output.is_valid:
                print(f"❌ Error: {output.error}")
                continue
            
            # Simulate robot behavior
            if output.safety_triggered:
                robot_running = False
            else:
                robot_running = True
                robot_position += 0.1  # Simulated movement
            
            # Log status every 30 frames (~3 seconds at 10 fps)
            frame_count += 1
            if frame_count % 30 == 0:
                status = "🤖 RUNNING" if robot_running else "🛑 STOPPED"
                print(f"\nFrame {frame_count}")
                print(f"  Robot Status: {status}")
                print(f"  Robot Position: {robot_position:.1f}")
                
                if output.gestures:
                    for gesture in output.gestures:
                        print(f"  Gesture: {gesture.gesture_name} "
                              f"({gesture.confidence:.2f}) - {gesture.handedness}")
                else:
                    print("  No gestures detected")
                
                if output.dominant_gesture:
                    print(f"  Stable Gesture: {output.dominant_gesture}")
                
                if output.intent:
                    print(f"  Intent: {output.intent.type.name}")
            
            # Visualize
            viz_frame = monitor.get_visualization_frame(frame, output)
            
            # Add robot status overlay
            status_color = (0, 255, 0) if robot_running else (0, 0, 255)
            status_text = f"Robot: {'RUNNING' if robot_running else 'STOPPED'}"
            cv2.putText(viz_frame, status_text, (10, viz_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.putText(viz_frame, f"Position: {robot_position:.1f}", 
                       (10, viz_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Gesture Safety Control", viz_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quit requested")
                break
            elif key == ord('r'):
                monitor.reset_safety()
                robot_running = True
                print("\n🔄 Safety state reset - robot resumed")
            
            # Control loop rate (10 Hz)
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        stats = monitor.get_gesture_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

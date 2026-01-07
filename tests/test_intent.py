#!/usr/bin/env python3
"""Test script for IntentMonitor.

Tests the Gemini-based intent recognition with configurable task graphs.
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
from aura.utils.config import IntentMonitorConfig


def test_intent_monitor_init():
    """Test IntentMonitor initialization."""
    print("\n=== Test 1: IntentMonitor Initialization ===")
    
    config = IntentMonitorConfig(
        enabled=True,
        fps=2.0,
        capture_duration=2.0,
        model="gemini-2.0-flash-exp"
    )
    
    monitor = IntentMonitor(config)
    print(f"✓ IntentMonitor created")
    print(f"  Monitor type: {monitor.monitor_type}")
    print(f"  FPS: {monitor.fps}")
    print(f"  Window duration: {monitor.window_duration}s")
    print(f"  Max frames: {monitor.max_frames}")
    print(f"  Gemini available: {monitor.client is not None}")
    return True


def test_custom_task_graph():
    """Test loading custom task graph."""
    print("\n=== Test 2: Custom Task Graph Loading ===")
    
    # Create temporary task graph file
    import tempfile
    import json
    
    custom_dag = {
        "name": "Test Task",
        "nodes": {
            "start": {
                "description": "Starting position",
                "next_possible": ["action1"]
            },
            "action1": {
                "description": "First action",
                "next_possible": ["end"]
            },
            "end": {
                "description": "Task complete",
                "next_possible": []
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_dag, f)
        dag_path = f.name
    
    try:
        config = IntentMonitorConfig(
            enabled=True,
            dag_file=dag_path,
            task_name="Test Task"
        )
        
        monitor = IntentMonitor(config)
        print(f"✓ Custom task graph loaded")
        print(f"  DAG nodes: {list(custom_dag['nodes'].keys())}")
        return True
    finally:
        os.unlink(dag_path)


async def test_frame_buffering():
    """Test frame buffering."""
    print("\n=== Test 3: Frame Buffering ===")
    
    config = IntentMonitorConfig(
        enabled=True,
        fps=5.0,
        capture_duration=1.0
    )
    
    monitor = IntentMonitor(config)
    
    # Simulate frame captures via update
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Call update which adds to buffer internally
        await monitor.update(frame=frame)
        await asyncio.sleep(0.15)  # Slightly longer than 1/5fps to ensure frames are captured
    
    print(f"✓ Frames buffered: {len(monitor.frame_buffer)}/{monitor.max_frames}")
    return True


async def test_live_intent(duration: float = 10.0):
    """Test live intent recognition with webcam.
    
    Args:
        duration: How long to run the test in seconds
    """
    print(f"\n=== Test 4: Live Intent Recognition ({duration}s) ===")
    print("Press 'q' to quit early")
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠ GEMINI_API_KEY not set, using mock predictions")
    
    config = IntentMonitorConfig(
        enabled=True,
        fps=2.0,
        capture_duration=2.0,
        prediction_interval=3.0,
        model="gemini-2.0-flash-exp"
    )
    
    monitor = IntentMonitor(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    last_prediction: IntentPrediction = None
    prediction_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Update monitor
            output = await monitor.update(frame=frame)
            
            if output and output.intent:
                last_prediction = IntentPrediction(
                    current_action=output.intent.action_type,
                    confidence=output.confidence,
                    predicted_next_action=getattr(output, 'predicted_action', 'unknown'),
                    prediction_confidence=getattr(output, 'prediction_confidence', 0.5),
                    reasoning=getattr(output, 'reasoning', ''),
                    timestamp=time.time()
                )
                prediction_count += 1
                print(f"\n[Prediction #{prediction_count}]")
                print(f"  Current: {last_prediction.current_action} ({last_prediction.confidence:.1%})")
                print(f"  Next: {last_prediction.predicted_next_action}")
            
            # Visualize
            if last_prediction:
                frame = visualize_intent(frame, last_prediction)
            
            # Add status
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Intent Test - {elapsed:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Predictions: {prediction_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Intent Monitor Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.01)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n✓ Intent test complete: {prediction_count} predictions in {duration:.1f}s")
    return True


def test_visualization():
    """Test intent visualization."""
    print("\n=== Test 5: Visualization ===")
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    prediction = IntentPrediction(
        current_action="reach",
        current_action_confidence=0.85,
        predicted_next_action="grasp",
        predicted_next_confidence=0.72,
        reasoning="Hand moving toward object on table",
        task_state={"target_object": "coffee cup", "hand_used": "right"}
    )
    
    result = visualize_intent(frame, prediction)
    
    print(f"✓ Visualization created: {result.shape}")
    
    # Save visualization
    cv2.imwrite("/tmp/intent_visualization_test.png", result)
    print(f"  Saved to /tmp/intent_visualization_test.png")
    return True


def run_unit_tests():
    """Run unit tests without webcam."""
    print("=" * 50)
    print("IntentMonitor Unit Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("Initialization", test_intent_monitor_init()))
    results.append(("Task Graph Loading", test_custom_task_graph()))
    results.append(("Frame Buffering", asyncio.run(test_frame_buffering())))
    results.append(("Visualization", test_visualization()))
    
    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    return all(r[1] for r in results)


def main():
    parser = argparse.ArgumentParser(description="Test IntentMonitor")
    parser.add_argument("--live", action="store_true", help="Run live webcam test")
    parser.add_argument("--duration", type=float, default=30.0, help="Live test duration (seconds)")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    args = parser.parse_args()
    
    if args.unit:
        success = run_unit_tests()
    elif args.live:
        success = asyncio.run(test_live_intent(args.duration))
    else:
        # Default: run unit tests
        success = run_unit_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

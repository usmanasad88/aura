#!/usr/bin/env python3
"""AURA Task Assistant — Generic realtime video + intent + DAG-driven robot control.

Runs RCWPS intent analysis leveraging a generic Task config profile, and uses 
the task DAG and rules to decide when the robot should move objects and issue voice announcements.

Usage:
    uv run python scripts/run_aura_assistant.py \
        --task hand_layup \
        --video demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4 \
        --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from aura.assistant.intent_monitor import AURAIntentMonitor
from aura.assistant.decision_engine import AURADecisionEngine
from aura.monitors.gesture_monitor import GestureMonitor, GestureMonitorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _print_intent(result, frame_count: int) -> None:
    phase = result.current_phase
    action = result.current_action
    predicted = result.predicted_next_action
    conf = result.prediction_confidence
    completed = len(result.steps_completed)
    in_progress = result.steps_in_progress
    gen_time = result.generation_time_sec

    # Print dynamically captured state variables cleanly
    state_vals = " | ".join(f"{k}: {v}" for k, v in result.state.items() if k not in [
        "current_phase", "current_action", "predicted_next_action", "prediction_confidence",
        "steps_completed", "steps_in_progress", "steps_pending", "reasoning", "human_state"
    ])

    print(f"\n{'─' * 60}\n"
          f"  Frame {result.frame_num} | {result.timestamp:.1f}s | predict #{frame_count} ({gen_time:.1f}s)\n"
          f"  Phase: {phase}  |  Action: {action}  |  In progress: {in_progress}\n"
          f"  Completed: {completed} steps  |  State: {state_vals}\n"
          f"  Predicted next: {predicted} ({conf:.0%})\n"
          f"  Reasoning: {result.reasoning}\n"
          f"{'─' * 60}")

def _print_actions(actions) -> None:
    for a in actions:
        status = "OK" if a.success else "PEND" if not a.executed else "FAIL"
        print(f"  >> ROBOT [{status}] {a.action_type} {a.object_name} (trigger: {a.trigger_step})")

async def run_live(
    task_name: str,
    video_path: str | None,
    robot_url: str,
    speed: float,
    predict_interval: float,
    model: str,
    enable_voice: bool,
    dry_run: bool,
    input_device: str | None,
    output_device: str | None,
    sample_rate: int,
    voice_name: str,
    webcam: int | str | None = None
):
    config_dir = _project_root / "tasks" / task_name / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory {config_dir} not found for task '{task_name}'.")

    # Load video
    if webcam is not None:
        from aura.sources.webcam import WebcamSource
        source = WebcamSource(device=webcam)
        source.open()
        print(f"Webcam: {webcam} @ {source.fps:.1f} fps")
    else:
        from aura.sources.realtime_video import RealtimeVideoSource
        source = RealtimeVideoSource(path=video_path, speed=speed)
        source.open()
        print(f"Video: {video_path} @ {source.fps:.1f} fps | {source.duration:.1f}s | speed={speed}x")

    intent_monitor = AURAIntentMonitor(
        config_dir=str(config_dir),
        model=model,
        realtime=True,
        enable_logging=True,
    )
    print(f"Intent monitor: task={task_name}, model={model}, realtime=True")

    robot_client = None
    if not dry_run:
        from aura.interfaces.robot_control_client import RobotControlClient
        robot_client = RobotControlClient(robot_url)
        if robot_client.is_available():
            print(f"Robot API: connected at {robot_url}")
        else:
            print(f"Robot API at {robot_url} not reachable.")

    sound_monitor = None
    def voice_callback(text: str): print(f"  VOICE: {text}")

    if enable_voice:
        try:
            from aura.interfaces.robot_control_client import RobotControlClient as _RC
            from aura.interfaces.voice_action_bridge import VoiceActionBridge
            from aura.monitors.sound_monitor import SoundMonitor

            bridge = VoiceActionBridge(robot_client or _RC(robot_url))
            config = bridge.build_sound_config(
                system_instruction=intent_monitor.system_instruction,
                voice_name=voice_name,
                enable_speech_output=True,
                input_device_name=input_device,
                output_device_name=output_device,
                input_sample_rate=sample_rate,
            )
            sound_monitor = SoundMonitor(config=config, tool_handlers=bridge.tool_handlers)
            await sound_monitor.start_listening()
            
            _loop = asyncio.get_event_loop()
            async def _voice_cb(text: str): await sound_monitor.send_text(text)
            def _live_voice(text: str):
                print(f"  VOICE: {text}")
                asyncio.run_coroutine_threadsafe(_voice_cb(text), _loop)
            voice_callback = _live_voice
        except Exception as e:
            print(f"Voice disabled ({e})")

    engine = AURADecisionEngine(
        config_dir=str(config_dir),
        robot_client=robot_client,
        on_voice=voice_callback,
        dry_run=dry_run,
    )

    intent_log_dir = intent_monitor.prompt_logger.get_session_dir()
    if intent_log_dir:
        engine.set_log_dir(intent_log_dir)

    print("\n" + "=" * 60)
    print(f"  AURA Assistant [{task_name}] — running")
    print("  Waiting for 'Thumb_Up' gesture to trigger intent prediction...")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    print("Initializing Gesture Monitor...")
    gesture_config = GestureMonitorConfig(
        resume_gestures={'Thumb_Up'},
        gesture_hold_frames=3
    )
    gesture_monitor = GestureMonitor(gesture_config)

    frame_buffer = []
    last_predict_time = 0.0
    predict_count = 0

    try:
        while True:
            frame = source.read()
            if frame is None: break
            frame_buffer.append(frame.image)
            if len(frame_buffer) > 10: frame_buffer = frame_buffer[-10:]

            # Check for gestures
            gesture_output = await gesture_monitor.update(frame=frame.image)
            is_thumbs_up = gesture_output and gesture_output.dominant_gesture == 'Thumb_Up'

            wall_now = time.monotonic()
            if is_thumbs_up and (wall_now - last_predict_time >= predict_interval) and len(frame_buffer) >= 1:
                print(f"\n[Gesture] 'Thumb_Up' detected! Triggering intent prediction (debounce cooldown: {predict_interval}s)...")
                last_predict_time = wall_now
                predict_count += 1
                result = intent_monitor.predict(frames=frame_buffer[-5:], timestamp=frame.timestamp, frame_num=frame.frame_number)
                _print_intent(result, predict_count)
                actions = engine.update(result)
                if actions: _print_actions(actions)
            
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        source.close()
        if sound_monitor: await sound_monitor.stop_listening()

    print(engine.get_summary())
    engine.save_summary()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Name of the task directory in tasks/")
    parser.add_argument("--video", default=None)
    parser.add_argument("--webcam", default=None, nargs="?", const=0)
    parser.add_argument("--robot-url", default="http://localhost:5050")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--predict-interval", type=float, default=3.0)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--voice", dest="voice", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", dest="dry_run", action="store_false")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--voice-name", default="Zephyr")
    args = parser.parse_args()

    webcam_dev = None
    if args.webcam is not None:
        try: webcam_dev = int(args.webcam)
        except ValueError: webcam_dev = args.webcam

    asyncio.run(run_live(
        task_name=args.task, video_path=args.video, robot_url=args.robot_url, speed=args.speed,
        predict_interval=args.predict_interval, model=args.model, enable_voice=args.voice,
        dry_run=args.dry_run, input_device=args.input_device, output_device=args.output_device,
        sample_rate=args.rate, voice_name=args.voice_name, webcam=webcam_dev
    ))

if __name__ == "__main__":
    main()

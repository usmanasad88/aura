#!/usr/bin/env python3
"""Hand Layup Robot Assistant — realtime video + intent + DAG-driven robot control.

Reads a video (or webcam) at real-time pace, runs RCWPS intent analysis every
few seconds, and uses the task DAG to decide when the robot should move objects
and issue voice announcements.

Replay mode (--replay-logs):
    Skip Gemini entirely and replay a saved intent-monitor session.  This lets
    you test the decision engine and voice pipeline without API costs.

    uv run python scripts/run_hand_layup_assistant.py \\
        --replay-logs tasks/hand_layup/logs/intent_monitor/session_20260221_124316 \\
        --dry-run --no-voice

Two-terminal workflow (with robot):
  Terminal 1 (robot API):
    cd ~/ur5-robotiq-ros2-control
    ./run_external_api.sh --no-ros

  Terminal 2 (this script):
    cd ~/Repos/aura
    uv run python scripts/run_hand_layup_assistant.py \\
        --video demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4

Live intent analysis (no replay):
    uv run python scripts/run_hand_layup_assistant.py \\
        --video demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4 \\
        --dry-run --no-voice
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so task imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _print_intent(result, frame_count: int, replayed: bool = False) -> None:
    """Pretty-print an IntentResult to the console."""
    phase = result.current_phase
    action = result.current_action
    predicted = result.predicted_next_action
    conf = result.prediction_confidence
    completed = len(result.steps_completed)
    in_progress = result.steps_in_progress
    gen_time = result.generation_time_sec
    tag = " [REPLAY]" if replayed else ""

    print(
        f"\n{'─' * 60}\n"
        f"  Frame {result.frame_num} | {result.timestamp:.1f}s | "
        f"predict #{frame_count} ({gen_time:.1f}s){tag}\n"
        f"  Phase: {phase}  |  Action: {action}\n"
        f"  In progress: {in_progress}\n"
        f"  Completed: {completed} steps  |  "
        f"Layers: {result.layers_placed}p/{result.layers_resined}r\n"
        f"  Mixed: {result.mixture_mixed}  |  "
        f"Consolidated: {result.consolidated}  |  "
        f"Gloves: {result.human_wearing_gloves}\n"
        f"  Predicted next: {predicted} ({conf:.0%})\n"
        f"  Reasoning: {result.reasoning}\n"
        f"{'─' * 60}"
    )


def _print_actions(actions) -> None:
    """Print robot actions that were triggered."""
    for a in actions:
        status = "OK" if a.success else "PEND" if not a.executed else "FAIL"
        print(
            f"  >> ROBOT [{status}] {a.action_type} "
            f"{a.object_name} (trigger: {a.trigger_step})"
        )


# ── Replay loop ──────────────────────────────────────────────────────────

async def run_replay(
    replay_logs: str,
    robot_url: str,
    enable_voice: bool,
    dry_run: bool,
    input_device: str | None,
    output_device: str | None,
    sample_rate: int,
    voice_name: str,
):
    """Replay saved intent logs through the decision engine."""
    from tasks.hand_layup.monitors.intent_monitor import IntentLogReplayer
    from tasks.hand_layup.decision_engine import HandLayupDecisionEngine

    replayer = IntentLogReplayer(replay_logs)
    print(
        f"Replay mode: {replay_logs}\n"
        f"  {replayer.total} saved predictions to replay\n"
        f"  No Gemini API calls will be made."
    )

    # ── Robot client (optional) ───────────────────────────────────────
    robot_client = None
    if not dry_run:
        from aura.interfaces.robot_control_client import RobotControlClient

        robot_client = RobotControlClient(robot_url)
        if robot_client.is_available():
            print(f"Robot API: connected at {robot_url}")
            print(robot_client.get_commands_summary())
        else:
            print(
                f"Robot API at {robot_url} not reachable — "
                "commands will fail at dispatch time."
            )

    # ── Voice output (optional) ───────────────────────────────────────
    sound_monitor = None
    voice_callback = None

    if enable_voice:
        try:
            from aura.interfaces.robot_control_client import RobotControlClient as _RC
            from aura.interfaces.voice_action_bridge import VoiceActionBridge
            from aura.monitors.sound_monitor import SoundMonitor

            vc = robot_client
            if vc is None:
                vc = _RC(robot_url)

            bridge = VoiceActionBridge(vc)
            config = bridge.build_sound_config(
                system_instruction=(
                    "You are a robot assistant helping with a fiberglass hand layup task. "
                    "You receive status updates about the task and announce them to the human. "
                    "Keep announcements short and clear (1-2 sentences)."
                ),
                voice_name=voice_name,
                enable_speech_output=True,
                input_device_name=input_device,
                output_device_name=output_device,
                input_sample_rate=sample_rate,
            )
            sound_monitor = SoundMonitor(
                config=config,
                tool_handlers=bridge.tool_handlers,
            )
            await sound_monitor.start_listening()
            print("Voice output: enabled")

            async def _voice_cb(text: str):
                await sound_monitor.send_text(text)

            _loop = asyncio.get_event_loop()

            def voice_callback(text: str):
                print(f"  VOICE: {text}")
                asyncio.run_coroutine_threadsafe(_voice_cb(text), _loop)

        except Exception as e:
            logger.warning("Could not start voice output: %s", e)
            print(f"Voice output: disabled ({e})")
            sound_monitor = None

    if voice_callback is None:
        def voice_callback(text: str):
            print(f"  VOICE: {text}")

    # ── Decision engine ───────────────────────────────────────────────
    engine = HandLayupDecisionEngine(
        robot_client=robot_client,
        on_voice=voice_callback,
        dry_run=dry_run,
    )
    print(f"Decision engine: dry_run={dry_run}")
    print(f"  Programs discovered: {len(engine._available_programs)}")

    # ── Replay loop ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Hand Layup Assistant — REPLAY MODE")
    print(f"  Replaying {replayer.total} saved predictions")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    predict_count = 0

    try:
        for result in replayer:
            predict_count += 1
            _print_intent(result, predict_count, replayed=True)

            actions = engine.update(result)
            if actions:
                _print_actions(actions)

            # Brief pause between replayed results for readability
            await asyncio.sleep(0.3)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if sound_monitor is not None:
            await sound_monitor.stop_listening()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nReplayed {predict_count}/{replayer.total} predictions.")
    print(engine.get_summary())


# ── Live video loop ──────────────────────────────────────────────────────

async def run_live(
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
    webcam: int | str | None = None,
):
    """Run live intent analysis on a video or webcam with the decision engine."""
    from tasks.hand_layup.monitors.intent_monitor import HandLayupIntentMonitor
    from tasks.hand_layup.decision_engine import HandLayupDecisionEngine

    # ── Video source ──────────────────────────────────────────────────
    if webcam is not None:
        from aura.sources.webcam import WebcamSource
        source = WebcamSource(device=webcam)
        source.open()
        w, h = source.resolution
        print(f"Webcam: device={webcam}  {w}x{h} @ {source.fps:.1f} fps")
    else:
        from aura.sources.realtime_video import RealtimeVideoSource
        source = RealtimeVideoSource(path=video_path, speed=speed)
        source.open()
        print(
            f"Video: {video_path}\n"
            f"  {source.resolution[0]}x{source.resolution[1]} @ "
            f"{source.fps:.1f} fps | {source.duration:.1f}s | speed={speed}x"
        )

    # ── Intent monitor ────────────────────────────────────────────────
    intent_monitor = HandLayupIntentMonitor(
        model=model,
        realtime=True,
        enable_logging=True,
    )
    print(f"Intent monitor: model={model}, realtime=True")

    # ── Robot client (optional) ───────────────────────────────────────
    robot_client = None
    if not dry_run:
        from aura.interfaces.robot_control_client import RobotControlClient

        robot_client = RobotControlClient(robot_url)
        if robot_client.is_available():
            print(f"Robot API: connected at {robot_url}")
            print(robot_client.get_commands_summary())
        else:
            print(
                f"Robot API at {robot_url} not reachable — "
                "commands will fail at dispatch time."
            )

    # ── Voice output (optional) ───────────────────────────────────────
    sound_monitor = None
    voice_callback = None

    if enable_voice:
        try:
            from aura.interfaces.robot_control_client import RobotControlClient as _RC
            from aura.interfaces.voice_action_bridge import VoiceActionBridge
            from aura.monitors.sound_monitor import SoundMonitor

            vc = robot_client
            if vc is None:
                vc = _RC(robot_url)

            bridge = VoiceActionBridge(vc)
            config = bridge.build_sound_config(
                system_instruction=(
                    "You are a robot assistant helping with a fiberglass hand layup task. "
                    "You receive status updates about the task and announce them to the human. "
                    "Keep announcements short and clear (1-2 sentences)."
                ),
                voice_name=voice_name,
                enable_speech_output=True,
                input_device_name=input_device,
                output_device_name=output_device,
                input_sample_rate=sample_rate,
            )
            sound_monitor = SoundMonitor(
                config=config,
                tool_handlers=bridge.tool_handlers,
            )
            await sound_monitor.start_listening()
            print("Voice output: enabled")

            async def _voice_cb(text: str):
                await sound_monitor.send_text(text)

            _loop = asyncio.get_event_loop()

            def voice_callback(text: str):
                print(f"  VOICE: {text}")
                asyncio.run_coroutine_threadsafe(_voice_cb(text), _loop)

        except Exception as e:
            logger.warning("Could not start voice output: %s", e)
            print(f"Voice output: disabled ({e})")
            sound_monitor = None

    if voice_callback is None:
        def voice_callback(text: str):
            print(f"  VOICE: {text}")

    # ── Decision engine ───────────────────────────────────────────────
    engine = HandLayupDecisionEngine(
        robot_client=robot_client,
        on_voice=voice_callback,
        dry_run=dry_run,
    )
    print(f"Decision engine: dry_run={dry_run}")
    print(f"  Programs discovered: {len(engine._available_programs)}")

    # ── Main loop ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Hand Layup Assistant — running")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    frame_buffer = []
    last_predict_time = 0.0
    predict_count = 0
    frames_read = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                print("\nSource ended.")
                break

            frames_read += 1
            frame_buffer.append(frame.image)
            # Keep a rolling window of recent frames
            if len(frame_buffer) > 10:
                frame_buffer = frame_buffer[-10:]

            # Run intent prediction at the configured interval
            wall_now = time.monotonic()
            if wall_now - last_predict_time >= predict_interval and len(frame_buffer) >= 1:
                last_predict_time = wall_now
                predict_count += 1

                # Take the most recent frames for prediction
                predict_frames = frame_buffer[-5:]
                result = intent_monitor.predict(
                    frames=predict_frames,
                    timestamp=frame.timestamp,
                    frame_num=frame.frame_number,
                )

                _print_intent(result, predict_count)

                # Decision engine processes the intent result
                actions = engine.update(result)
                if actions:
                    _print_actions(actions)

            # Small sleep to avoid busy-waiting between frames
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        source.close()
        if sound_monitor is not None:
            await sound_monitor.stop_listening()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nProcessed {frames_read} frames, {predict_count} predictions.")
    print(engine.get_summary())

    log_dir = intent_monitor.get_log_dir()
    if log_dir:
        print(f"\nIntent logs saved to: {log_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hand Layup Robot Assistant — realtime video + intent + DAG-driven robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", default=None,
        help="Path to video file (required for live mode, ignored in replay mode)",
    )
    parser.add_argument(
        "--webcam", default=None, nargs="?", const=0, metavar="DEVICE",
        help="Use webcam as video source. Optionally specify device index or "
             "path (default: 0). Example: --webcam or --webcam 2 or --webcam /dev/video0",
    )
    parser.add_argument(
        "--replay-logs", default=None, metavar="SESSION_DIR",
        help="Path to a saved intent-monitor session directory. "
             "Replays logged predictions without calling Gemini.",
    )
    parser.add_argument(
        "--robot-url", default="http://localhost:5050",
        help="URL of the UR5 External Control API (default: http://localhost:5050)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Video playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--predict-interval", type=float, default=3.0,
        help="Seconds between intent predictions (default: 3.0)",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model for intent analysis (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--voice", dest="voice", action="store_true", default=False,
        help="Enable voice output via Gemini Live",
    )
    parser.add_argument(
        "--no-voice", dest="voice", action="store_false",
        help="Disable voice output (default)",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True,
        help="Log robot commands without executing (default)",
    )
    parser.add_argument(
        "--live", dest="dry_run", action="store_false",
        help="Actually execute robot commands",
    )
    parser.add_argument("--input-device", default=None, help="Audio input device substring")
    parser.add_argument("--output-device", default=None, help="Audio output device substring")
    parser.add_argument("--rate", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--voice-name", default="Zephyr", help="Gemini voice (default: Zephyr)")

    args = parser.parse_args()

    # Parse webcam device — could be an int or a path string
    webcam_device = None
    if args.webcam is not None:
        try:
            webcam_device = int(args.webcam)
        except (ValueError, TypeError):
            webcam_device = args.webcam  # keep as string (e.g. /dev/video0)

    if args.replay_logs:
        # Replay mode — no video or Gemini needed
        asyncio.run(run_replay(
            replay_logs=args.replay_logs,
            robot_url=args.robot_url,
            enable_voice=args.voice,
            dry_run=args.dry_run,
            input_device=args.input_device,
            output_device=args.output_device,
            sample_rate=args.rate,
            voice_name=args.voice_name,
        ))
    elif args.video or webcam_device is not None:
        # Live mode — video file or webcam + Gemini intent analysis
        asyncio.run(run_live(
            video_path=args.video,
            robot_url=args.robot_url,
            speed=args.speed,
            predict_interval=args.predict_interval,
            model=args.model,
            enable_voice=args.voice,
            dry_run=args.dry_run,
            input_device=args.input_device,
            output_device=args.output_device,
            sample_rate=args.rate,
            voice_name=args.voice_name,
            webcam=webcam_device,
        ))
    else:
        parser.error("One of --video, --webcam, or --replay-logs is required.")


if __name__ == "__main__":
    main()

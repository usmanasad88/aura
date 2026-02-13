#!/usr/bin/env python3
"""Voice-controlled robot interface.

Connects to the UR5 External Control REST API (external_control_api.py)
and lets you control the robot with voice commands via Gemini Live.

Two-terminal workflow:
  Terminal 1 (robot API):
    cd ~/ur5-robotiq-ros2-control
    ./run_external_api.sh --no-ros          # dry-run, or without --no-ros for real robot

  Terminal 2 (this script):
    cd ~/Repos/aura
    uv run python scripts/run_voice_robot_control.py
    uv run python scripts/run_voice_robot_control.py --robot-url http://192.168.1.10:5050
    uv run python scripts/run_voice_robot_control.py --input-device USB --output-device Analog --rate 48000
"""

import argparse
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_voice_control(
    robot_url: str,
    input_device: str | None,
    output_device: str | None,
    sample_rate: int,
    voice_name: str,
    system_instruction: str,
):
    from aura.interfaces.robot_control_client import RobotControlClient
    from aura.interfaces.voice_action_bridge import VoiceActionBridge
    from aura.monitors.sound_monitor import SoundMonitor

    # ── Connect to robot API ──────────────────────────────────────────
    print(f"Connecting to robot API at {robot_url} …")
    client = RobotControlClient(robot_url)

    if client.is_available():
        print("✓ Robot API is reachable")
        summary = client.get_commands_summary()
        print(summary)
    else:
        print("⚠  Robot API is not reachable — commands will fail at dispatch time.")
        print("   Make sure external_control_api.py is running in another terminal.\n")

    # ── Build voice bridge ────────────────────────────────────────────
    def on_action(entry):
        status = "✓" if entry.success else "✗"
        print(f"\n  [{status}] {entry.function_name}({entry.args}) → {entry.response.get('message', '')}")

    bridge = VoiceActionBridge(client, on_action=on_action)

    config = bridge.build_sound_config(
        system_instruction=system_instruction,
        voice_name=voice_name,
        enable_speech_output=True,
        input_device_name=input_device,
        output_device_name=output_device,
        input_sample_rate=sample_rate,
    )

    def on_response(text: str):
        print(f"\n[Gemini]: {text}")

    monitor = SoundMonitor(
        config=config,
        on_response=on_response,
        tool_handlers=bridge.tool_handlers,
    )

    # ── Run ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Voice Robot Control — speak to command the robot")
    print("  Type 'q' to quit, or type a text command.")
    print("=" * 55 + "\n")

    try:
        await monitor.start_listening()

        while True:
            text = await asyncio.to_thread(input, "voice > ")
            if text.strip().lower() in ("q", "quit", "exit"):
                break
            if text.strip():
                await monitor.send_text(text)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await monitor.stop_listening()
        print("\nAction log:")
        for entry in bridge.get_action_log():
            s = "✓" if entry["success"] else "✗"
            print(f"  {entry['timestamp']}  [{s}] {entry['function']}({entry['args']})")


DEFAULT_SYSTEM_INSTRUCTION = """\
You are a voice interface for a UR5 robot arm with a Robotiq 2F-85 gripper.
You can move the robot to named positions, run saved programs, open/close
the gripper, move in relative directions (left/right/up/down/forward/back),
save positions, and pause/resume/stop execution.

RULES:
- When the human gives a robot command, you MUST call execute_robot_command.
  Do NOT just say you are doing it — actually call the tool. Speaking about
  an action without calling the tool means the robot does NOT move.
- Execute clear commands IMMEDIATELY. Do not ask for confirmation.
- If you hear background noise, silence, or unclear sounds, stay SILENT.
  Only respond to clear human speech directed at you.
- Never repeat the same response. If you already acknowledged, stay quiet.
- Keep spoken responses to 1-2 short sentences. No markdown or formatting."""


def main():
    parser = argparse.ArgumentParser(
        description="Voice-controlled UR5 robot interface (Gemini Live + External Control API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot-url", default="http://localhost:5050",
        help="URL of the UR5 External Control API (default: http://localhost:5050)",
    )
    parser.add_argument("--input-device", default=None, help="Substring of input audio device name")
    parser.add_argument("--output-device", default=None, help="Substring of output audio device name")
    parser.add_argument("--rate", type=int, default=16000, help="Input sample rate (default: 16000)")
    parser.add_argument("--voice", default="Zephyr", help="Gemini voice name (default: Zephyr)")
    parser.add_argument(
        "--system-instruction", default=DEFAULT_SYSTEM_INSTRUCTION,
        help="System instruction for Gemini (default: built-in robot assistant prompt)",
    )
    args = parser.parse_args()

    asyncio.run(run_voice_control(
        robot_url=args.robot_url,
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.rate,
        voice_name=args.voice,
        system_instruction=args.system_instruction,
    ))


if __name__ == "__main__":
    main()

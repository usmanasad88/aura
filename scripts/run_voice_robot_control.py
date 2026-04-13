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
    task: str | None = None,
):
    from aura.monitors.sound_monitor import SoundMonitor
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent

    # ── Decide which bridge to use ────────────────────────────────────
    # If --task is given, use SkillActionBridge (generic, reads robot_skills.json)
    # Otherwise fall back to VoiceActionBridge (UR5-specific, discovers from API)
    if task:
        from aura.brain.skill_registry import SkillRegistry
        from aura.interfaces.skill_action_bridge import SkillActionBridge

        config_dir = _project_root / "tasks" / task / "config"
        skills = SkillRegistry()
        skills_path = config_dir / "robot_skills.json"
        if skills_path.exists():
            skills.load_from_file(str(skills_path))
            print(f"Loaded {len(skills.list_skills())} skills from {skills_path.name}")
        else:
            print(f"Warning: {skills_path} not found — no skills loaded")

        def on_action(entry):
            status = "OK" if entry.success else "FAIL"
            print(f"\n  [{status}] {entry.function_name}({entry.args}) -> {entry.response.get('message', '')}")

        bridge = SkillActionBridge(
            skills=skills,
            robot_url=robot_url,
            dry_run=False,
            on_action=on_action,
        )
    else:
        from aura.interfaces.robot_control_client import RobotControlClient
        from aura.interfaces.voice_action_bridge import VoiceActionBridge

        # ── Connect to robot API ──────────────────────────────────
        print(f"Connecting to robot API at {robot_url} ...")
        client = RobotControlClient(robot_url)

        if client.is_available():
            print("Robot API is reachable")
            summary = client.get_commands_summary()
            print(summary)
        else:
            print("Robot API is not reachable -- commands will fail at dispatch time.")
            print("Make sure external_control_api.py is running in another terminal.\n")

        def on_action(entry):
            status = "OK" if entry.success else "FAIL"
            print(f"\n  [{status}] {entry.function_name}({entry.args}) -> {entry.response.get('message', '')}")

        bridge = VoiceActionBridge(client, on_action=on_action)

    # ── Build sound config ────────────────────────────────────────────
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
    mode = f"task={task}" if task else "UR5 direct"
    print(f"\n{'=' * 55}")
    print(f"  Voice Robot Control ({mode})")
    print("  Speak to command the robot, type 'q' to quit.")
    print(f"{'=' * 55}\n")

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
            s = "OK" if entry["success"] else "FAIL"
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
        description="Voice-controlled robot interface (Gemini Live + skill execution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", default=None,
        help="Task name (directory under tasks/) — uses SkillActionBridge with robot_skills.json. "
             "If omitted, falls back to UR5-specific VoiceActionBridge via the live API.",
    )
    parser.add_argument(
        "--robot-url", default="http://localhost:5050",
        help="URL of the robot External Control API (default: http://localhost:5050)",
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
        task=args.task,
    ))


if __name__ == "__main__":
    main()

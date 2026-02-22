#!/usr/bin/env python3
"""Run the voice-controlled robot assistant for the hand layup task.

This script:
1. Connects to the UR5 External Control API to discover available commands
2. Sets up the affordance monitor (what the robot CAN do)
3. Starts the Gemini Live sound monitor with function-calling tools
4. The human operator speaks commands, Gemini selects and dispatches them

Usage:
    # With real robot API running
    python -m tasks.hand_layup.demo.run_voice_control

    # Custom API URL
    python -m tasks.hand_layup.demo.run_voice_control --api-url http://192.168.1.10:5050

    # Dry-run (no robot, just list commands)
    python -m tasks.hand_layup.demo.run_voice_control --dry-run

    # Text-only mode (no microphone, type commands)
    python -m tasks.hand_layup.demo.run_voice_control --text-mode
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Ensure AURA paths
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def on_action(entry):
    """Callback for every dispatched robot action."""
    icon = "✅" if entry.success else "❌"
    msg = entry.response.get("message", "")
    print(f"\n🔧 {icon} {entry.function_name}({entry.args}) → {msg}\n")


def on_response(text: str):
    """Callback for Gemini spoken/text responses."""
    print(f"\n🤖 Robot: {text}\n")


async def run_voice_control(args):
    """Main async entry point."""
    from aura.interfaces.robot_control_client import RobotControlClient
    from aura.interfaces.voice_action_bridge import VoiceActionBridge
    from tasks.hand_layup.monitors.affordance_monitor import HandLayupAffordanceMonitor

    # ── 1. Connect to robot API ──────────────────────────────────────────
    api_url = args.api_url
    client = RobotControlClient(api_url)

    if not args.dry_run:
        if client.is_available():
            logger.info(f"✅ Connected to robot API at {api_url}")
        else:
            logger.warning(f"⚠️  Robot API at {api_url} is not reachable")
            if not args.text_mode:
                logger.warning("   Start it with: cd ~/Repos/ur_ws && ./run_external_api.sh")

    # ── 2. Discover commands ─────────────────────────────────────────────
    commands = client.get_commands(force_refresh=True)
    summary = client.get_commands_summary()
    print("\n" + "=" * 60)
    print("  Available Robot Commands")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")

    if args.dry_run:
        print("Dry-run complete. Exiting.")
        return

    # ── 3. Set up affordance monitor ─────────────────────────────────────
    affordance = HandLayupAffordanceMonitor(robot_client=client)
    logger.info(f"Affordance monitor: {len(affordance.programs)} skills loaded")
    if affordance.api_connected:
        logger.info("  ↳ Live API connection active")

    # ── 4. Set up voice → action bridge ──────────────────────────────────
    bridge = VoiceActionBridge(
        client=client,
        on_action=on_action,
    )

    # Load YAML config for system instruction
    config_path = TASK_DIR / "config" / "hand_layup.yaml"
    system_instruction = ""
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        va_cfg = cfg.get("voice_action", {})
        system_instruction = va_cfg.get("system_instruction", "")

    # ── 5. Text mode (no microphone) ─────────────────────────────────────
    if args.text_mode:
        print("\n📝 Text mode — type commands to control the robot.")
        print("   Type 'commands' to list available commands.")
        print("   Type 'status' to check robot status.")
        print("   Type 'quit' to exit.\n")

        # Initialise Gemini once
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            print("❌ google-genai not installed — install with: pip install google-genai")
            return

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Set GEMINI_API_KEY or GOOGLE_API_KEY to use text mode.")
            return

        gclient = genai.Client(api_key=api_key)
        tool_decls = bridge.tool_declarations
        gen_config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=tool_decls)],
            temperature=0.3,
        )

        # Seed the conversation with system context so Gemini knows the
        # available commands and behavioural rules from the first turn.
        system_ctx = (
            f"{system_instruction}\n\n"
            f"Available robot commands:\n{summary}\n\n"
            "Rules:\n"
            "• If the user gives a clear robot command, execute it IMMEDIATELY "
            "by calling the tool — do NOT ask for confirmation.\n"
            "• If the request is ambiguous or could match multiple commands, "
            "ask the user to clarify before executing.\n"
            "• For non-robot questions, reply conversationally.\n"
            "• When saving positions, spaces in names are converted to "
            "underscores (e.g. 'position 1' → 'position_1'). Use the "
            "underscore form when referring to saved positions."
        )
        conversation: list = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=system_ctx),
            ]),
            types.Content(role="model", parts=[
                types.Part.from_text(text=(
                    "Ready. I'll execute clear commands immediately and ask "
                    "for clarification only when the request is ambiguous. "
                    "What would you like me to do?"
                )),
            ]),
        ]

        def _generate(conv):
            """Call Gemini, process tool calls, feed results back."""
            resp = gclient.models.generate_content(
                model="gemini-2.5-flash",
                contents=conv,
                config=gen_config,
            )
            if not (resp.candidates and resp.candidates[0].content):
                return

            content = resp.candidates[0].content
            conv.append(content)

            func_results = []  # collect function-response Parts
            for part in content.parts:
                if part.function_call:
                    fc = part.function_call
                    fn, fa = fc.name, dict(fc.args) if fc.args else {}
                    print(f"🔧 Calling: {fn}({fa})")
                    if fn in bridge.tool_handlers:
                        result = bridge.tool_handlers[fn](**fa)
                        ok = result.get("success", False)
                        print(f"   {'✅' if ok else '❌'} {result.get('message', json.dumps(result))}")
                    else:
                        result = {"success": False, "message": f"Unknown tool: {fn}"}
                        print(f"   ❌ Unknown tool: {fn}")
                    func_results.append(
                        types.Part.from_function_response(name=fn, response=result)
                    )
                elif part.text:
                    print(f"🤖 {part.text}")

            # If we got function calls, feed results back for a follow-up.
            if func_results:
                conv.append(types.Content(role="user", parts=func_results))
                _generate(conv)

        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "commands":
                print(summary)
                continue
            if user_input.lower() == "status":
                st = client.get_status()
                print(json.dumps(st, indent=2))
                continue

            conversation.append(
                types.Content(role="user", parts=[
                    types.Part.from_text(text=user_input),
                ])
            )

            try:
                _generate(conversation)
            except Exception as e:
                print(f"❌ Error: {e}")
                conversation.pop()  # drop failed user turn

        # Print action log
        if bridge.action_log:
            print("\n" + "=" * 60)
            print("  Action Log")
            print("=" * 60)
            for entry in bridge.get_action_log():
                st_icon = "✅" if entry["success"] else "❌"
                print(f"  {st_icon} {entry['timestamp']}: {entry['function']}({entry['args']})")
        return

    # ── 6. Live voice mode (microphone + speaker) ────────────────────────
    try:
        from aura.monitors.sound_monitor import SoundMonitor
    except ImportError as e:
        logger.error(f"Cannot import SoundMonitor: {e}")
        logger.error("Install dependencies: pip install pyaudio google-genai")
        return

    sound_config = bridge.build_sound_config(
        system_instruction=system_instruction,
        enable_speech_output=True,
        input_device_name="0x46d",
        input_sample_rate=16000,
        output_device_name="ALC887",
    )

    sound = SoundMonitor(
        config=sound_config,
        on_response=on_response,
        tool_handlers=bridge.tool_handlers,
    )

    print("\n" + "=" * 60)
    print("  🎤 Voice Control Active")
    print("  Speak commands to control the robot.")
    print("  Say 'list commands' to hear available commands.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        await sound.start_listening()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n🛑 Shutting down...")
    finally:
        try:
            await sound.stop_listening()
        except Exception:
            pass

        # Print action log
        if bridge.action_log:
            print("\n" + "=" * 60)
            print("  Action Log")
            print("=" * 60)
            for entry in bridge.get_action_log():
                st_icon = "✅" if entry["success"] else "❌"
                print(f"  {st_icon} {entry['timestamp']}: {entry['function']}({entry['args']})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Voice-controlled UR5 assistant for hand layup")
    parser.add_argument("--api-url", default="http://localhost:5050",
                        help="UR5 External Control API URL (default: http://localhost:5050)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list available commands and exit")
    parser.add_argument("--text-mode", action="store_true",
                        help="Type commands instead of using microphone")
    args = parser.parse_args()

    try:
        asyncio.run(run_voice_control(args))
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C


if __name__ == "__main__":
    main()

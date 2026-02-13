"""Voice-to-Action Bridge for AURA framework.

Connects the SoundMonitor (Gemini Live with function calling) to the
RobotControlClient (UR5 External Control REST API).

Workflow:
1. On init, fetches available robot commands from the external API
2. Converts them into Gemini function-calling tool declarations
3. Registers tool handlers that dispatch commands to the API
4. When a human says e.g. "move the robot to Home", Gemini calls the
   matching tool, which triggers the HTTP request to the robot

Usage::

    from aura.interfaces.robot_control_client import RobotControlClient
    from aura.interfaces.voice_action_bridge import VoiceActionBridge
    from aura.monitors.sound_monitor import SoundMonitor, SoundConfig

    client = RobotControlClient("http://localhost:5050")
    bridge = VoiceActionBridge(client)

    config = bridge.build_sound_config(
        system_instruction="You are a robot assistant helping with hand layup...",
    )
    sound = SoundMonitor(config=config, tool_handlers=bridge.tool_handlers)
    await sound.start_listening()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionLogEntry:
    """Record of an action dispatched by voice command."""
    timestamp: str
    function_name: str
    args: Dict[str, Any]
    response: Dict[str, Any]
    success: bool


class VoiceActionBridge:
    """Bridges Gemini Live function-calling to the UR5 External Control API.

    Attributes:
        client: RobotControlClient instance
        tool_declarations: List of Gemini function declaration dicts
        tool_handlers: Dict mapping function names → callables (for SoundMonitor)
        action_log: History of dispatched actions
    """

    def __init__(
        self,
        client,  # RobotControlClient — imported lazily to avoid circular deps
        categories: Optional[List[str]] = None,
        on_action: Optional[Callable[[ActionLogEntry], None]] = None,
    ):
        """
        Args:
            client: RobotControlClient connected to the UR5 external API
            categories: Which command categories to expose. None = all.
                        Possible: ["program", "named_position_joint",
                                   "named_position_pose", "gripper", "control"]
            on_action: Callback fired after every dispatched action
        """
        self.client = client
        self.categories = categories
        self.on_action = on_action
        self.action_log: List[ActionLogEntry] = []

        # Build declarations + handlers from the API
        self.tool_declarations: List[Dict[str, Any]] = []
        self.tool_handlers: Dict[str, Callable] = {}
        self._refresh_tools()

    # ── tool building ────────────────────────────────────────────────────

    def _refresh_tools(self):
        """(Re-)fetch commands from API and rebuild tools + handlers."""
        try:
            commands = self.client.get_robot_commands(force_refresh=True)
        except Exception as e:
            logger.warning(f"Could not fetch robot commands: {e}")
            commands = []

        if self.categories:
            commands = [c for c in commands if c.category in self.categories]

        self.tool_declarations = []
        self.tool_handlers = {}

        # ── Add a generic "execute_robot_command" tool that Gemini can use ──
        # This is simpler and more robust than one tool per command.
        self.tool_declarations.append({
            "name": "execute_robot_command",
            "description": (
                "Execute a robot command. Use this when the human asks the robot "
                "to do something (move, pick, place, open/close gripper, run a program, "
                "go to a named position, pause, stop, move left/right/up/down, "
                "save a position, etc.). "
                "Call get_available_commands first to see what's available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command_type": {
                        "type": "string",
                        "enum": ["program", "named_position", "gripper",
                                 "pause", "resume", "stop",
                                 "move_relative", "save_position"],
                        "description": "Type of command to execute",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Name of the program or position. "
                            "For programs: e.g. 'move_to_home.prog'. "
                            "For named positions: e.g. 'Home', 'TableCenter'. "
                            "For gripper: 'open', 'close', or a number 0.0-1.0. "
                            "For save_position: the name to assign to the saved position. "
                            "For move_relative: not used (set direction instead)."
                        ),
                    },
                    "duration": {
                        "type": "number",
                        "description": "Movement duration in seconds (optional, default 3.0)",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "forward", "back", "up", "down"],
                        "description": (
                            "Direction for move_relative commands. "
                            "left/right = Y axis, forward/back = X axis, "
                            "up/down = Z axis (global gravity)"
                        ),
                    },
                    "distance": {
                        "type": "number",
                        "description": "Distance in metres for move_relative (default 0.05 = 5cm)",
                    },
                    "position_type": {
                        "type": "string",
                        "enum": ["joint", "pose"],
                        "description": "For save_position: 'joint' (raw angles, default) or 'pose' (Cartesian)",
                    },
                },
                "required": ["command_type"],
            },
        })

        self.tool_declarations.append({
            "name": "get_available_commands",
            "description": (
                "Get the list of all available robot commands. Call this to see "
                "what programs, named positions, and gripper actions the robot "
                "can perform before executing a command."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        })

        # Register handlers
        self.tool_handlers["execute_robot_command"] = self._handle_execute
        self.tool_handlers["get_available_commands"] = self._handle_get_commands

        # NOTE: We intentionally do NOT register per-command tools.
        # Having dozens of overlapping tools confuses the model — it may
        # describe the action in speech instead of calling a tool.
        # Two generic tools (execute + list) are cleaner and more reliable.

        logger.info(
            f"VoiceActionBridge: {len(self.tool_declarations)} tools, "
            f"{len(commands)} robot commands"
        )

    def _make_handler(self, cmd):
        """Create a closure handler for a specific RobotCommand."""
        def handler(**kwargs):
            return self._dispatch(cmd.api_endpoint, dict(cmd.payload_template, **kwargs))
        return handler

    # ── handlers ─────────────────────────────────────────────────────────

    def _handle_get_commands(self, **kwargs) -> Dict[str, Any]:
        """Return a summary of available commands."""
        summary = self.client.get_commands_summary()
        return {"status": "ok", "commands": summary}

    def _handle_execute(self, command_type: str = "", name: str = "", duration: float = 3.0,
                         direction: str = "", distance: float = 0.05,
                         position_type: str = "joint", **kwargs) -> Dict[str, Any]:
        """Dispatch a high-level command to the appropriate API endpoint."""
        try:
            if command_type == "program":
                resp = self.client.execute_program(name)
            elif command_type == "named_position":
                # Normalise name: spaces → underscores (matches save_position behaviour)
                normalised_name = name.strip().replace(' ', '_') if name else name
                resp = self.client.move_to_named(normalised_name, duration=duration)
            elif command_type == "gripper":
                if name in ("open", "close"):
                    resp = self.client._post("/api/gripper", {"action": name})
                else:
                    try:
                        pos = float(name)
                        resp = self.client.gripper_set_position(pos)
                    except ValueError:
                        resp = {"success": False, "message": f"Invalid gripper value: {name}"}
            elif command_type == "pause":
                resp = self.client.pause()
            elif command_type == "resume":
                resp = self.client.resume()
            elif command_type == "stop":
                resp = self.client.stop()
            elif command_type == "move_relative":
                if not direction:
                    resp = {"success": False, "message": "Missing 'direction' for move_relative"}
                else:
                    resp = self.client.move_relative(direction, distance=distance)
            elif command_type == "save_position":
                if not name:
                    resp = {"success": False, "message": "Missing 'name' for save_position"}
                else:
                    resp = self.client.save_position(name, pos_type=position_type)
            else:
                resp = {"success": False, "message": f"Unknown command type: {command_type}"}

            self._log_action("execute_robot_command", {"command_type": command_type, "name": name}, resp)
            return resp

        except Exception as e:
            err = {"success": False, "message": str(e)}
            self._log_action("execute_robot_command", {"command_type": command_type, "name": name}, err)
            return err

    def _dispatch(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Low-level dispatch to the API."""
        try:
            resp = self.client._post(endpoint, payload)
            self._log_action(endpoint, payload, resp)
            return resp
        except Exception as e:
            err = {"success": False, "message": str(e)}
            self._log_action(endpoint, payload, err)
            return err

    def _log_action(self, func_name: str, args: Dict, response: Dict):
        entry = ActionLogEntry(
            timestamp=datetime.now().isoformat(),
            function_name=func_name,
            args=args,
            response=response,
            success=response.get("success", False),
        )
        self.action_log.append(entry)
        if self.on_action:
            self.on_action(entry)
        logger.info(f"[VoiceAction] {func_name}({args}) → success={entry.success}")

    # ── config builder ───────────────────────────────────────────────────

    def build_sound_config(
        self,
        system_instruction: str = "",
        voice_name: str = "Zephyr",
        enable_speech_output: bool = True,
        **extra_kwargs,
    ):
        """Build a SoundConfig pre-wired with robot tool declarations.

        Returns a SoundConfig that can be passed to SoundMonitor.
        """
        from aura.monitors.sound_monitor import SoundConfig

        # Build system instruction that includes available commands
        commands_summary = self.client.get_commands_summary()
        full_instruction = (
            f"{system_instruction}\n\n"
            "CRITICAL RULES — follow these without exception:\n\n"
            "1. ALWAYS USE TOOL CALLS for robot actions. When the human asks the\n"
            "   robot to do something (move, go home, open gripper, run a program,\n"
            "   etc.), you MUST call execute_robot_command. NEVER just say 'I'm\n"
            "   moving to home' without also issuing the tool call. Talking about\n"
            "   an action is NOT the same as doing it.\n\n"
            "2. Do NOT ask for confirmation. Execute clear commands immediately.\n"
            "   Only ask for clarification when the request is genuinely ambiguous\n"
            "   (e.g. it could match multiple different commands).\n\n"
            "3. IGNORE background noise. If you hear sounds that are not clearly\n"
            "   human speech directed at you (coughs, bumps, music, murmuring,\n"
            "   unintelligible noise), do NOT respond. Stay silent. Only respond\n"
            "   when you hear a clear, directed utterance.\n\n"
            "4. Do NOT repeat yourself. If you already confirmed an action or\n"
            "   answered a question, do not say the same thing again. If the\n"
            "   human hasn't said anything new, stay silent.\n\n"
            "5. Keep responses to 1-2 SHORT spoken sentences. Never output\n"
            "   markdown, headers, bullet points, or reasoning text. Speak\n"
            "   naturally and concisely.\n\n"
            "6. When saving positions, spaces in names become underscores\n"
            "   (e.g. 'position 1' → 'position_1'). Use the underscore form\n"
            "   when moving to a saved position.\n\n"
            "7. If you're unsure which command to use, call get_available_commands\n"
            "   first to see what's available, then call execute_robot_command.\n\n"
            f"Available commands:\n{commands_summary}"
        )

        return SoundConfig(
            system_instruction=full_instruction,
            voice_name=voice_name,
            enable_speech_output=enable_speech_output,
            tools=self.tool_declarations,
            keywords_of_interest=[
                "robot", "move", "bring", "remove", "gripper", "open", "close",
                "home", "stop", "pause", "continue", "execute", "run", "program",
                "left", "right", "forward", "back", "up", "down", "save", "position",
            ],
            **extra_kwargs,
        )

    # ── convenience ──────────────────────────────────────────────────────

    def get_action_log(self) -> List[Dict[str, Any]]:
        """Return the action log as serialisable dicts."""
        return [
            {
                "timestamp": e.timestamp,
                "function": e.function_name,
                "args": e.args,
                "response": e.response,
                "success": e.success,
            }
            for e in self.action_log
        ]

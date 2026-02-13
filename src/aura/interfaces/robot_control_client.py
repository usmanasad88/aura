"""Robot Control API Client for AURA framework.

Communicates with the UR5 External Control REST API
(``external_control_api.py`` in the ur_ws workspace) to:
- Fetch available commands (programs, named positions, gripper actions)
- Send execution commands (run program, move, gripper, pause/resume/stop)
- Poll robot status (joint state, executor running)

This client is intentionally transport-agnostic at the public API level so
that callers don't need to know about HTTP details.

Usage::

    from aura.interfaces.robot_control_client import RobotControlClient

    client = RobotControlClient("http://localhost:5050")

    # Discover what the robot can do
    commands = client.get_commands()

    # Execute a program
    client.execute_program("pick_and_place_object.prog")

    # Move to a named position
    client.move_to_named("Home", duration=3.0)

    # Gripper
    client.gripper_open()
    client.gripper_close()
    client.gripper_set_position(0.5)

    # Execution control
    client.pause()
    client.resume()
    client.stop()
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RobotCommand:
    """A single command the robot can accept, discovered from the API."""
    category: str          # "program" | "named_position_joint" | "named_position_pose" | "gripper" | "control"
    name: str              # Human-readable name
    api_endpoint: str      # e.g. "/api/program/execute"
    api_method: str        # "POST"
    payload_template: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_tool_description(self) -> Dict[str, Any]:
        """Return a Gemini-compatible function declaration dict."""
        props: Dict[str, Any] = {}
        required: List[str] = []

        if self.category == "program":
            props["program"] = {"type": "string", "description": f"Program file name (e.g. '{self.name}')"}
            required.append("program")
        elif self.category in ("named_position_joint", "named_position_pose"):
            props["name"] = {"type": "string", "description": f"Named position (e.g. '{self.name}')"}
            required.append("name")
            if self.category == "named_position_joint":
                props["duration"] = {"type": "number", "description": "Movement duration in seconds (default 3.0)"}
        elif self.category == "gripper":
            props["action"] = {"type": "string", "enum": ["open", "close", "position"],
                               "description": "Gripper action"}
            required.append("action")
            props["position"] = {"type": "number", "description": "Gripper position 0.0-1.0 (only for action=position)"}
        elif self.category == "relative_move":
            props["direction"] = {"type": "string", "enum": ["left", "right", "forward", "back", "up", "down"],
                                  "description": f"Direction to move (e.g. '{self.name}')"}
            required.append("direction")
            props["distance"] = {"type": "number", "description": "Distance in metres (default 0.05 = 5cm)"}
        elif self.category == "save_position":
            props["name"] = {"type": "string", "description": "Name to assign to the saved position"}
            required.append("name")
            props["type"] = {"type": "string", "enum": ["joint", "pose"],
                             "description": "Position type: 'joint' (raw angles) or 'pose' (Cartesian)"}

        # Sanitise name for function-calling (alphanumeric + underscores)
        func_name = f"robot_{self.category}__{self.name}".replace(".", "_").replace(" ", "_").replace("-", "_")

        return {
            "name": func_name,
            "description": self.description or self.name,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


class RobotControlClient:
    """HTTP client for the UR5 External Control REST API."""

    def __init__(self, base_url: str = "http://localhost:5050", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._commands_cache: Optional[Dict[str, Any]] = None
        self._robot_commands: List[RobotCommand] = []

    # ── low-level HTTP ───────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:
                return {"success": False, "message": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _post(self, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        return self._request("POST", path, body)

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path)

    # ── discovery ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the external control API is reachable."""
        try:
            resp = self._get("/api/status")
            return "executor_running" in resp
        except Exception:
            return False

    def get_commands(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch the full command catalogue from the API.

        Returns the raw JSON dict with keys:
        ``programs``, ``named_positions``, ``gripper_actions``, ``execution_control``
        """
        if self._commands_cache and not force_refresh:
            return self._commands_cache
        self._commands_cache = self._get("/api/commands")
        self._build_robot_commands()
        return self._commands_cache

    def _build_robot_commands(self):
        """Parse the raw commands JSON into RobotCommand objects."""
        cmds = self._commands_cache or {}
        self._robot_commands = []

        # Programs
        for prog in cmds.get("programs", []):
            self._robot_commands.append(RobotCommand(
                category="program",
                name=prog["name"],
                api_endpoint="/api/program/execute",
                api_method="POST",
                payload_template={"program": prog["name"]},
                description=prog.get("description", f"Execute program {prog['name']}"),
            ))

        # Named positions — joint
        for pos in cmds.get("named_positions", {}).get("joint", []):
            self._robot_commands.append(RobotCommand(
                category="named_position_joint",
                name=pos["name"],
                api_endpoint="/api/move/named",
                api_method="POST",
                payload_template={"name": pos["name"]},
                description=pos.get("description", f"Move to joint position {pos['name']}"),
                extra={"joint_positions": pos.get("joint_positions", [])},
            ))

        # Named positions — pose (Cartesian via cuRobo)
        for pos in cmds.get("named_positions", {}).get("pose", []):
            self._robot_commands.append(RobotCommand(
                category="named_position_pose",
                name=pos["name"],
                api_endpoint="/api/move/named",
                api_method="POST",
                payload_template={"name": pos["name"]},
                description=pos.get("description", f"Move to Cartesian pose {pos['name']}"),
                extra={"position": pos.get("position", []), "quaternion": pos.get("quaternion", [])},
            ))

        # Gripper actions
        for action in cmds.get("gripper_actions", []):
            self._robot_commands.append(RobotCommand(
                category="gripper",
                name=action,
                api_endpoint="/api/gripper",
                api_method="POST",
                payload_template={"action": action},
                description=f"Gripper: {action}",
            ))

        # Execution control
        for ctrl in cmds.get("execution_control", []):
            self._robot_commands.append(RobotCommand(
                category="control",
                name=ctrl,
                api_endpoint=f"/api/program/{ctrl}",
                api_method="POST",
                payload_template={},
                description=f"Execution control: {ctrl}",
            ))

        # Relative motion directions
        for direction in cmds.get("relative_directions", []):
            self._robot_commands.append(RobotCommand(
                category="relative_move",
                name=direction,
                api_endpoint="/api/move/relative",
                api_method="POST",
                payload_template={"direction": direction},
                description=f"Move end-effector {direction} by a specified distance",
            ))

        # Save position (virtual command — always available)
        self._robot_commands.append(RobotCommand(
            category="save_position",
            name="save_position",
            api_endpoint="/api/position/save",
            api_method="POST",
            payload_template={},
            description="Save the current robot position as a named position",
        ))

    def get_robot_commands(self, force_refresh: bool = False) -> List[RobotCommand]:
        """Return parsed RobotCommand objects."""
        if not self._robot_commands or force_refresh:
            self.get_commands(force_refresh=True)
        return list(self._robot_commands)

    def get_tool_declarations(self, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get Gemini function-calling tool declarations for available commands.

        Args:
            categories: Filter to specific categories, e.g. ["program", "named_position_joint"].
                        None means all.

        Returns:
            List of function declaration dicts suitable for ``google.genai.types.Tool``.
        """
        cmds = self.get_robot_commands()
        if categories:
            cmds = [c for c in cmds if c.category in categories]
        return [c.to_tool_description() for c in cmds]

    def get_commands_summary(self) -> str:
        """Return a human-readable summary of all commands for prompt injection."""
        cmds = self.get_robot_commands()
        lines = ["Available robot commands:"]
        by_cat: Dict[str, List[RobotCommand]] = {}
        for c in cmds:
            by_cat.setdefault(c.category, []).append(c)
        for cat, items in by_cat.items():
            lines.append(f"\n  [{cat}]")
            for item in items:
                lines.append(f"    - {item.name}: {item.description}")
        return "\n".join(lines)

    # ── status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get robot / executor status."""
        return self._get("/api/status")

    # ── program execution ────────────────────────────────────────────────

    def execute_program(self, program: str) -> Dict[str, Any]:
        return self._post("/api/program/execute", {"program": program})

    def load_program(self, program: str) -> Dict[str, Any]:
        return self._post("/api/program/load", {"program": program})

    def pause(self) -> Dict[str, Any]:
        return self._post("/api/program/pause")

    def resume(self) -> Dict[str, Any]:
        return self._post("/api/program/resume")

    def stop(self) -> Dict[str, Any]:
        return self._post("/api/program/stop")

    # ── gripper ──────────────────────────────────────────────────────────

    def gripper_open(self) -> Dict[str, Any]:
        return self._post("/api/gripper", {"action": "open"})

    def gripper_close(self) -> Dict[str, Any]:
        return self._post("/api/gripper", {"action": "close"})

    def gripper_set_position(self, position: float) -> Dict[str, Any]:
        return self._post("/api/gripper", {"action": "position", "position": position})

    # ── movement ─────────────────────────────────────────────────────────

    def move_to_named(self, name: str, duration: float = 3.0) -> Dict[str, Any]:
        return self._post("/api/move/named", {"name": name, "duration": duration})

    def move_to_joints(self, positions: List[float], duration: float = 3.0) -> Dict[str, Any]:
        return self._post("/api/move/joints", {"positions": positions, "duration": duration})

    def move_to_pose(self, position: List[float], quaternion: List[float]) -> Dict[str, Any]:
        return self._post("/api/move/pose", {"position": position, "quaternion": quaternion})

    def move_relative(self, direction: str, distance: float = 0.05) -> Dict[str, Any]:
        """Move end-effector by a Cartesian offset.

        Args:
            direction: One of 'left', 'right', 'forward', 'back', 'up', 'down'.
                       up/down align with global gravity (Z axis).
            distance: Distance in metres (default 0.05 = 5 cm).
        """
        return self._post("/api/move/relative", {"direction": direction, "distance": distance})

    def save_position(self, name: str, pos_type: str = "joint") -> Dict[str, Any]:
        """Save the current robot position as a named position.

        Args:
            name: Name to assign to the position.
            pos_type: 'joint' (default) or 'pose'.
        """
        return self._post("/api/position/save", {"name": name, "type": pos_type})

    # ── generic dispatch (used by voice-to-action bridge) ────────────────

    def dispatch_command(self, command: RobotCommand, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a RobotCommand, optionally overriding payload fields.

        Args:
            command: The RobotCommand to execute
            overrides: Dict of payload fields to override (e.g. {"duration": 5.0})

        Returns:
            API response dict
        """
        payload = dict(command.payload_template)
        if overrides:
            payload.update(overrides)
        return self._post(command.api_endpoint, payload)

    def dispatch_by_function_name(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command by its Gemini function-calling name.

        This is the bridge between the sound monitor's tool calls and the API.

        Args:
            function_name: e.g. "robot_program__move_to_home_prog"
            args: Arguments from Gemini function call

        Returns:
            API response dict with success/message
        """
        cmds = self.get_robot_commands()

        for cmd in cmds:
            expected_name = f"robot_{cmd.category}__{cmd.name}".replace(".", "_").replace(" ", "_").replace("-", "_")
            if expected_name == function_name:
                return self.dispatch_command(cmd, overrides=args)

        # Fallback: try to match by partial name
        for cmd in cmds:
            if cmd.name.replace(".", "_").replace(" ", "_") in function_name:
                return self.dispatch_command(cmd, overrides=args)

        return {"success": False, "message": f"No command matching function '{function_name}'"}

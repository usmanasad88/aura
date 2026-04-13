"""Generic Skill-to-Action Bridge for AURA framework.

Replaces the UR5-specific ``VoiceActionBridge`` with a generic bridge
that works with any task's ``robot_skills.json`` via the ``SkillRegistry``.

The bridge:
1. Reads skills from a ``SkillRegistry`` (loaded from robot_skills.json)
2. Converts them into Gemini function-calling tool declarations
3. Registers tool handlers that either:
   - Dispatch robot commands via the ``RobotControlClient`` (live mode)
   - Log actions and update the SSG (dry-run mode)
   - Update scene state from human speech (e.g. object locations)
   - Relay context to/from the decision engine

Also provides SSG read/write tools so the human can tell the system
about the scene ("the resin is on the left table") and ask about task
status ("what step are we on?").

Usage::

    from aura.brain.skill_registry import SkillRegistry
    from aura.interfaces.skill_action_bridge import SkillActionBridge
    from aura.monitors.sound_monitor import SoundMonitor

    registry = SkillRegistry()
    registry.load_from_file("tasks/hand_layup/config/robot_skills.json")

    bridge = SkillActionBridge(registry, robot_url="http://localhost:5050")
    config = bridge.build_sound_config(system_instruction="...")
    monitor = SoundMonitor(config=config, tool_handlers=bridge.tool_handlers)
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


class SkillActionBridge:
    """Generic bridge from Gemini Live tool calls to robot skill execution.

    Unlike ``VoiceActionBridge`` (which is UR5-specific and discovers
    commands from the live API), this bridge reads skill definitions
    from a ``SkillRegistry`` and generates tool declarations accordingly.
    It also provides tools for SSG state queries and updates.

    Attributes:
        skills: SkillRegistry with loaded skills
        tool_declarations: Gemini function declaration dicts
        tool_handlers: Dict mapping function names to callables
        action_log: History of dispatched actions
    """

    def __init__(
        self,
        skills: "SkillRegistry",
        robot_url: str = "http://localhost:5050",
        dry_run: bool = True,
        ssg: Optional[Any] = None,
        on_action: Optional[Callable[[ActionLogEntry], None]] = None,
        on_ssg_update: Optional[Callable[[str, Any], None]] = None,
        on_context_message: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            skills: Loaded SkillRegistry (from robot_skills.json)
            robot_url: Base URL for robot HTTP API
            dry_run: If True, log actions without executing
            ssg: Live SemanticSceneGraph instance (shared with workflow)
            on_action: Callback fired after every dispatched action
            on_ssg_update: Callback when human speech updates scene state
            on_context_message: Callback when human provides context
                that should be relayed to the decision engine
        """
        from aura.brain.skill_registry import SkillRegistry

        self.skills: SkillRegistry = skills
        self.robot_url = robot_url
        self.dry_run = dry_run
        self.ssg = ssg
        self.on_action = on_action
        self.on_ssg_update = on_ssg_update
        self.on_context_message = on_context_message
        self.action_log: List[ActionLogEntry] = []

        # Robot client (lazy — only created in live mode)
        self._robot_client = None

        # Build tool declarations + handlers
        self.tool_declarations: List[Dict[str, Any]] = []
        self.tool_handlers: Dict[str, Callable] = {}
        self._build_tools()

    # ── Robot client (lazy) ─────────────────────────────────────────────

    @property
    def robot_client(self):
        if self._robot_client is None and not self.dry_run:
            from aura.interfaces.robot_control_client import RobotControlClient
            self._robot_client = RobotControlClient(self.robot_url)
        return self._robot_client

    # ── Tool building ───────────────────────────────────────────────────

    def _build_tools(self):
        """Build Gemini tool declarations and handlers from the registry."""
        self.tool_declarations = []
        self.tool_handlers = {}

        # 1. execute_skill — run a robot skill by ID
        self._add_execute_skill_tool()

        # 2. get_available_skills — list what the robot can do
        self._add_list_skills_tool()

        # 3. update_scene_state — human tells the system about the scene
        self._add_update_scene_tool()

        # 4. get_task_status — human asks about current task progress
        self._add_get_status_tool()

        # 5. relay_context — human says something relevant to the task
        self._add_relay_context_tool()

        logger.info(
            "SkillActionBridge: %d tools, %d skills",
            len(self.tool_declarations), len(self.skills.list_skills()),
        )

    def _add_execute_skill_tool(self):
        """Add the main skill execution tool."""
        skill_ids = self.skills.list_skill_ids()
        skill_descriptions = []
        for s in self.skills.list_skills():
            desc = f"  - {s.id}: {s.description}"
            if s.parameters:
                params = ", ".join(
                    f"{p.name}({p.type})" for p in s.parameters
                )
                desc += f" [params: {params}]"
            skill_descriptions.append(desc)

        self.tool_declarations.append({
            "name": "execute_skill",
            "description": (
                "Execute a robot skill. Use this when the human asks "
                "the robot to do something (move an object, consolidate, "
                "clean, move to a position, open/close gripper, etc.). "
                "Call get_available_skills first if unsure which skill to use.\n\n"
                "Available skills:\n" + "\n".join(skill_descriptions)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "enum": skill_ids,
                        "description": "ID of the skill to execute",
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Skill-specific parameters as key-value pairs. "
                            "Check the skill definition for required parameters."
                        ),
                    },
                },
                "required": ["skill_id"],
            },
        })
        self.tool_handlers["execute_skill"] = self._handle_execute_skill

    def _add_list_skills_tool(self):
        """Add skill listing tool."""
        self.tool_declarations.append({
            "name": "get_available_skills",
            "description": (
                "Get the list of all available robot skills with their "
                "descriptions, parameters, and preconditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        })
        self.tool_handlers["get_available_skills"] = self._handle_list_skills

    def _add_update_scene_tool(self):
        """Add scene state update tool (human tells robot about the scene)."""
        self.tool_declarations.append({
            "name": "update_scene_state",
            "description": (
                "Update the scene state based on information from the human. "
                "Use this when the human tells you about the location of an "
                "object, the state of something, or corrects the system's "
                "understanding. Examples: 'the resin is on the left table', "
                "'I already mixed the hardener', 'the roller is broken'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "State variable to update (e.g. 'resin_bottle.location', "
                            "'mixture_mixed', 'roller.state')"
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "New value for the state variable",
                    },
                },
                "required": ["key", "value"],
            },
        })
        self.tool_handlers["update_scene_state"] = self._handle_update_scene

    def _add_get_status_tool(self):
        """Add task status query tool."""
        self.tool_declarations.append({
            "name": "get_task_status",
            "description": (
                "Get the current task status, including completed steps, "
                "current phase, object locations, and scene state. "
                "Use this when the human asks about progress, what step "
                "we're on, where something is, or what's happening."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        })
        self.tool_handlers["get_task_status"] = self._handle_get_status

    def _add_relay_context_tool(self):
        """Add context relay tool for decision engine communication."""
        self.tool_declarations.append({
            "name": "relay_context",
            "description": (
                "Relay important context from human speech to the decision "
                "engine. Use this when the human says something relevant to "
                "the task that should influence future robot decisions, but "
                "is NOT a direct command or scene update. Examples: "
                "'I'm going to take a break', 'we need to hurry up', "
                "'skip the second layer', 'I'll do the consolidation myself'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The relevant context to relay",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How urgently this should influence decisions",
                    },
                },
                "required": ["context"],
            },
        })
        self.tool_handlers["relay_context"] = self._handle_relay_context

    # ── Handlers ────────────────────────────────────────────────────────

    def _handle_execute_skill(
        self, skill_id: str = "", parameters: Dict[str, Any] | None = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute a robot skill."""
        parameters = parameters or {}
        skill = self.skills.get(skill_id)
        if not skill:
            resp = {"success": False, "message": f"Unknown skill: {skill_id}"}
            self._log_action("execute_skill", {"skill_id": skill_id}, resp)
            return resp

        # Validate parameters
        is_valid, error = skill.validate_parameters(parameters)
        if not is_valid:
            resp = {"success": False, "message": error}
            self._log_action("execute_skill", {"skill_id": skill_id, **parameters}, resp)
            return resp

        api_call = skill.metadata.get("api_call")

        if self.dry_run:
            resp = {
                "success": True,
                "message": f"[DRY-RUN] Would execute: {skill.name}",
                "skill_id": skill_id,
                "mode": "dry_run",
            }
            logger.info("[DRY-RUN] Skill %s (%s) — parameters: %s", skill_id, skill.name, parameters)
        elif api_call and self.robot_client:
            try:
                endpoint = api_call.get("endpoint", "")
                body = dict(api_call.get("body", {}))
                # Substitute parameters into body template
                for k, v in parameters.items():
                    if f"<{k}>" in str(body.get(k, "")):
                        body[k] = v
                    elif k not in body:
                        body[k] = v
                resp = self.robot_client._post(endpoint, body)
            except Exception as e:
                resp = {"success": False, "message": str(e)}
        elif self.robot_client:
            # No api_call defined — try generic dispatch
            resp = {"success": False, "message": f"Skill '{skill_id}' has no api_call defined"}
        else:
            resp = {"success": False, "message": "Robot not available"}

        # Apply effects to SSG on success
        if resp.get("success") and self.ssg:
            for effect_key, effect_val in skill.effects.items():
                parts = effect_key.split(".")
                if len(parts) == 2 and parts[1] == "location":
                    try:
                        self.ssg.set_location(parts[0], effect_val)
                    except Exception:
                        pass
                self.ssg.set_task_state(effect_key, effect_val)
            if self.on_ssg_update:
                self.on_ssg_update("skill_effects", skill.effects)

        self._log_action("execute_skill", {"skill_id": skill_id, **parameters}, resp)
        return resp

    def _handle_list_skills(self, **kwargs) -> Dict[str, Any]:
        """Return skill summaries."""
        skills = []
        for s in self.skills.list_skills():
            skills.append({
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "can_interrupt": s.can_interrupt,
            })
        return {"success": True, "skills": skills}

    def _handle_update_scene(
        self, key: str = "", value: str = "", **kwargs
    ) -> Dict[str, Any]:
        """Update scene state from human speech."""
        if not key:
            return {"success": False, "message": "Missing state key"}

        logger.info("Scene state update from human: %s = %s", key, value)

        if self.ssg:
            # Handle dotted keys like "resin_bottle.location"
            parts = key.split(".")
            if len(parts) == 2 and parts[1] == "location":
                try:
                    self.ssg.set_location(parts[0], value)
                except Exception:
                    pass
            self.ssg.set_task_state(key, value)

        if self.on_ssg_update:
            self.on_ssg_update(key, value)

        return {"success": True, "message": f"Updated {key} = {value}"}

    def _handle_get_status(self, **kwargs) -> Dict[str, Any]:
        """Return current task status from SSG."""
        if not self.ssg:
            return {"success": True, "status": "No scene graph available"}

        summary = self.ssg.get_state_summary_for_llm()
        task_state = dict(self.ssg.task_state)
        return {
            "success": True,
            "summary": summary,
            "task_state": task_state,
        }

    def _handle_relay_context(
        self, context: str = "", urgency: str = "medium", **kwargs
    ) -> Dict[str, Any]:
        """Relay context to the decision engine."""
        if not context:
            return {"success": False, "message": "Missing context"}

        logger.info("Human context (urgency=%s): %s", urgency, context)

        # Store in SSG as a context message for the decision engine
        if self.ssg:
            existing = list(self.ssg.task_state.get("human_context_messages", []))
            existing.append({
                "text": context,
                "urgency": urgency,
                "timestamp": datetime.now().isoformat(),
            })
            # Keep last 10 messages
            self.ssg.set_task_state("human_context_messages", existing[-10:])

        if self.on_context_message:
            self.on_context_message(context)

        return {"success": True, "message": "Context relayed to decision engine"}

    # ── Logging ─────────────────────────────────────────────────────────

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
        logger.info("[SkillAction] %s(%s) -> success=%s", func_name, args, entry.success)

    # ── Config builder ──────────────────────────────────────────────────

    def build_sound_config(
        self,
        system_instruction: str = "",
        voice_name: str = "Zephyr",
        enable_speech_output: bool = True,
        **extra_kwargs,
    ):
        """Build a SoundConfig pre-wired with skill tool declarations.

        Returns a SoundConfig that can be passed to SoundMonitor.
        """
        from aura.monitors.sound_monitor import SoundConfig

        skills_summary = self.skills.get_skills_for_llm()
        full_instruction = (
            f"{system_instruction}\n\n"
            "CRITICAL RULES -- follow these without exception:\n\n"
            "1. ALWAYS USE TOOL CALLS for robot actions. When the human asks the\n"
            "   robot to do something, you MUST call execute_skill. NEVER just say\n"
            "   you are doing it -- actually call the tool.\n\n"
            "2. Do NOT ask for confirmation. Execute clear commands immediately.\n"
            "   Only ask for clarification when the request is genuinely ambiguous.\n\n"
            "3. IGNORE background noise. If you hear sounds that are not clearly\n"
            "   human speech directed at you, do NOT respond. Stay silent.\n\n"
            "4. Do NOT repeat yourself.\n\n"
            "5. Keep responses to 1-2 SHORT spoken sentences.\n\n"
            "6. When the human tells you about the scene (object locations, states),\n"
            "   call update_scene_state to record it.\n\n"
            "7. When the human says something relevant to the task that should\n"
            "   influence robot decisions (but is not a command), call relay_context.\n\n"
            "8. When the human asks about task progress or the scene, call\n"
            "   get_task_status and summarise the result briefly.\n\n"
            "9. If unsure which skill to use, call get_available_skills first.\n\n"
            f"{skills_summary}"
        )

        return SoundConfig(
            system_instruction=full_instruction,
            voice_name=voice_name,
            enable_speech_output=enable_speech_output,
            tools=self.tool_declarations,
            keywords_of_interest=[
                "robot", "move", "bring", "remove", "fetch", "get", "take",
                "gripper", "open", "close", "roller", "resin", "hardener",
                "home", "stop", "pause", "clean", "consolidate",
                "where", "what", "status", "step", "phase", "done",
                "left", "right", "forward", "back", "up", "down",
            ],
            **extra_kwargs,
        )

    # ── Convenience ─────────────────────────────────────────────────────

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

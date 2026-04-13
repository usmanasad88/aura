"""AURA interfaces — bridges to external systems (robot, game, etc.)."""

from .robot_control_client import RobotControlClient, RobotCommand
from .voice_action_bridge import VoiceActionBridge
from .voice_action_bridge import ActionLogEntry as _VoiceActionLogEntry
from .skill_action_bridge import SkillActionBridge, ActionLogEntry
from .audio_workflow_bridge import AudioWorkflowBridge, VoiceEvent

__all__ = [
    "RobotControlClient",
    "RobotCommand",
    "VoiceActionBridge",      # kept for backward compat
    "SkillActionBridge",
    "ActionLogEntry",
    "AudioWorkflowBridge",
    "VoiceEvent",
]

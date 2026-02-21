"""Hand Layup Task Package for AURA framework.

This is a task-specific implementation that can be deleted
without affecting the base AURA framework.

Task: Human performs fiberglass hand layup on a mold.
Robot assists by placing objects on the table when required
and performing housekeeping (moving objects from the workplace
to storage when no longer needed). The system monitors the
process for safety (gloves), correct resin/hardener ratio,
shelf life, and layup quality.

Monitors:
- HandLayupAffordanceMonitor: Manages robot object-moving skills
- HandLayupPerformanceMonitor: Detects process issues and safety violations
"""

from .monitors import (
    HandLayupAffordanceMonitor,
    RobotProgram,
    ProgramStatus,
    HandLayupPerformanceMonitor,
    PerformanceStatus,
    FailureType,
    HandLayupIntentMonitor,
    IntentResult,
    PromptLogger,
)
from .decision_engine import HandLayupDecisionEngine, RobotAction, VoiceMessage

__all__ = [
    "HandLayupAffordanceMonitor",
    "RobotProgram",
    "ProgramStatus",
    "HandLayupPerformanceMonitor",
    "PerformanceStatus",
    "FailureType",
    "HandLayupIntentMonitor",
    "IntentResult",
    "PromptLogger",
    "HandLayupDecisionEngine",
    "RobotAction",
    "VoiceMessage",
]

"""Monitors for hand layup task.

This module provides task-specific monitors that inherit from
the base AURA framework monitors.
"""

from .affordance_monitor import (
    HandLayupAffordanceMonitor,
    AffordanceMonitorConfig,
    RobotProgram,
    ProgramStatus,
)
from .performance_monitor import (
    HandLayupPerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceCheckResult,
    PerformanceStatus,
    FailureType,
)
from .intent_monitor import (
    HandLayupIntentMonitor,
    IntentResult,
    PromptLogger,
)

__all__ = [
    # Affordance monitoring
    "HandLayupAffordanceMonitor",
    "AffordanceMonitorConfig",
    "RobotProgram",
    "ProgramStatus",
    # Performance monitoring
    "HandLayupPerformanceMonitor",
    "PerformanceMonitorConfig",
    "PerformanceCheckResult",
    "PerformanceStatus",
    "FailureType",
    # Intent monitoring (RCWPS)
    "HandLayupIntentMonitor",
    "IntentResult",
    "PromptLogger",
]

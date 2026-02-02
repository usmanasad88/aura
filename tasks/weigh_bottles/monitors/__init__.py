"""Monitors for weigh bottles task.

This module provides task-specific monitors that inherit from
the base AURA framework monitors.
"""

from .affordance_monitor import (
    WeighBottlesAffordanceMonitor,
    AffordanceMonitorConfig,
    RobotProgram,
    ProgramStatus,
)
from .performance_monitor import (
    WeighBottlesPerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceCheckResult,
    PerformanceStatus,
    FailureType,
)

__all__ = [
    # Affordance monitoring
    "WeighBottlesAffordanceMonitor",
    "AffordanceMonitorConfig",
    "RobotProgram",
    "ProgramStatus",
    # Performance monitoring
    "WeighBottlesPerformanceMonitor",
    "PerformanceMonitorConfig",
    "PerformanceCheckResult", 
    "PerformanceStatus",
    "FailureType",
]

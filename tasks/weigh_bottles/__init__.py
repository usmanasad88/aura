"""Weigh Bottles Task Package for AURA framework.

This is a task-specific implementation that can be deleted
without affecting the base AURA framework.

Task: Robot picks resin and hardener bottles from storage table,
delivers to human at weighing station, human weighs, robot returns bottles.

Monitors:
- WeighBottlesAffordanceMonitor: Manages robot program execution
- WeighBottlesPerformanceMonitor: Detects task execution failures
"""

from .monitors import (
    WeighBottlesAffordanceMonitor,
    RobotProgram,
    ProgramStatus,
    WeighBottlesPerformanceMonitor,
    PerformanceStatus,
    FailureType,
)

__all__ = [
    "WeighBottlesAffordanceMonitor",
    "RobotProgram",
    "ProgramStatus",
    "WeighBottlesPerformanceMonitor",
    "PerformanceStatus",
    "FailureType",
]

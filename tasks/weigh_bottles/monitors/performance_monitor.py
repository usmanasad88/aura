"""Performance Monitor for Weigh Bottles Task.

This task-specific monitor inherits from the base PerformanceMonitor
and adds bottle weighing specific failure detection prompts and logic.

Detected failure types:
- Dropped bottle
- Failed pick (gripper missed the bottle)
- Collision with objects or human
- Robot stuck/not moving when it should be
- Incorrect trajectory (going to wrong location)
- Human intervention required
"""

import logging
from typing import Optional

from aura.monitors.performance_monitor import (
    PerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceCheckResult,
    PerformanceStatus,
    FailureType,
)


logger = logging.getLogger(__name__)


class WeighBottlesPerformanceMonitor(PerformanceMonitor):
    """Performance monitor for the weigh bottles task.
    
    Uses Gemini to analyze video frames and determine if the robot
    is correctly executing bottle pick/return operations. Detects failures
    like dropped bottles, failed grasps, collisions, etc.
    
    This is a task-specific subclass that provides specialized prompts
    for the bottle weighing scenario.
    
    Usage:
        monitor = WeighBottlesPerformanceMonitor(config)
        
        # Set current instruction being executed
        monitor.set_current_instruction("Pick hardener bottle from storage")
        
        # Add frames for analysis
        monitor.add_frame(frame)
        
        # Check performance (async)
        result = await monitor.check_performance()
        if not result.is_ok:
            print(f"Failure detected: {result.failure_type.name}")
    """
    
    # Bottle weighing specific system prompt
    BOTTLE_SYSTEM_PROMPT = """You are an AI safety monitor for a collaborative robot task.
You analyze video frames to determine if the robot is correctly executing its assigned task.
Your job is to detect failures and safety issues.

The robot is performing a bottle weighing assistance task:
- Picking resin and hardener bottles from a storage table
- Delivering bottles to a human operator at a weighing station
- Returning bottles to storage after weighing

The storage table is typically on one side, the human with the scale on the other.
Watch for the robot gripper state (open/closed), bottle positions, and human actions.

You must detect issues such as:
- Dropped bottle (bottle falling from gripper)
- Failed grasp (gripper closed but missed the bottle)
- Collision with objects, table, or human
- Robot stuck or not moving when it should
- Wrong trajectory (moving to wrong location)
- Human intervention or blocking

Be conservative - if unsure, report WARNING rather than ERROR.
Only report CRITICAL for clear dangerous situations like imminent collision with human."""

    # Bottle weighing specific failure detection prompt
    BOTTLE_FAILURE_PROMPT = """Analyze these video frames to check if the robot is correctly performing the bottle task.

CURRENT INSTRUCTION: {instruction}

Look for these specific issues in the bottle weighing context:
1. DROPPED_OBJECT: Bottle falling or slipping from gripper during pick/move/handover
2. FAILED_GRASP: Gripper closed but bottle not in gripper (common at storage table)
3. COLLISION: Robot contacting table, bottles, scale, or human unexpectedly
4. WRONG_TRAJECTORY: Robot moving to wrong table or wrong bottle
5. ROBOT_STUCK: No motion when robot should be moving toward target
6. HUMAN_INTERVENTION: Human blocking robot path or taking over the task
7. UNEXPECTED_STATE: Bottle not where expected, wrong bottle picked, etc.

Consider the frames in sequence - they show approximately {window_sec} seconds of motion.

Respond in JSON format:
{{
    "status": "OK" | "WARNING" | "ERROR" | "CRITICAL",
    "failure_type": "NONE" | "DROPPED_OBJECT" | "FAILED_GRASP" | "COLLISION" | "WRONG_TRAJECTORY" | "ROBOT_STUCK" | "HUMAN_INTERVENTION" | "UNEXPECTED_STATE" | "UNKNOWN",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of what you observed in the bottle task",
    "details": {{
        "robot_moving": true/false,
        "gripper_state": "open" | "closed" | "unknown",
        "bottle_visible": true/false,
        "bottle_in_gripper": true/false,
        "human_in_frame": true/false,
        "near_storage_table": true/false,
        "near_weighing_station": true/false
    }}
}}"""

    def __init__(self, config: Optional[PerformanceMonitorConfig] = None):
        """Initialize weigh bottles performance monitor.
        
        Args:
            config: Monitor configuration
        """
        super().__init__(config)
        logger.info("WeighBottlesPerformanceMonitor initialized")
    
    def _get_system_prompt(self) -> str:
        """Get bottle weighing specific system prompt."""
        return self.BOTTLE_SYSTEM_PROMPT
    
    def _get_failure_detection_prompt(self, instruction: str, window_sec: float) -> str:
        """Get bottle weighing specific failure detection prompt."""
        return self.BOTTLE_FAILURE_PROMPT.format(
            instruction=instruction,
            window_sec=f"{window_sec:.1f}"
        )


# Re-export base classes for convenience
__all__ = [
    "WeighBottlesPerformanceMonitor",
    "PerformanceMonitorConfig",
    "PerformanceCheckResult",
    "PerformanceStatus",
    "FailureType",
]

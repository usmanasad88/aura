"""Performance Monitor for Hand Layup Task.

This task-specific monitor inherits from the base PerformanceMonitor
and adds hand layup specific failure detection prompts and logic.

Monitored conditions:
- Gloves worn during resin handling
- Resin/hardener ratio within acceptable range
- Mixture shelf life not exceeded
- Layer coverage adequate (no dry spots)
- Correct layup sequence followed
- No wrinkles or air bubbles in layup
"""

import logging
import time
from typing import Optional

from aura.monitors.performance_monitor import (
    PerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceCheckResult,
    PerformanceStatus,
    FailureType,
)


logger = logging.getLogger(__name__)


class HandLayupPerformanceMonitor(PerformanceMonitor):
    """Performance monitor for the hand layup task.
    
    Uses Gemini to analyze video frames and determine if the human
    is correctly performing the hand layup process. Monitors safety
    (gloves), quality (ratio, coverage, wrinkles), and timing
    (shelf life).
    
    Usage:
        monitor = HandLayupPerformanceMonitor(config)
        
        # Set current task step
        monitor.set_current_instruction("Apply resin to layer 2")
        
        # Add frames
        monitor.add_frame(frame)
        
        # Check performance
        result = await monitor.check_performance()
        if not result.is_ok:
            print(f"Issue: {result.failure_type.name} - {result.reasoning}")
    """
    
    # Hand layup specific system prompt
    LAYUP_SYSTEM_PROMPT = """You are an AI quality and safety monitor for a fiberglass hand layup process.
You analyze video frames to determine if the human operator is correctly performing the layup task
and to detect safety or quality issues.

The process involves:
1. Measuring resin and hardener into a cup on a scale
2. Mixing resin and hardener together
3. Placing 4 layers of pre-cut fiberglass on a metal mold
4. Applying mixed resin to each layer with a brush
5. Using a roller to consolidate all layers at the end

Objects of interest: fiberglass sheets, metal mold, resin bottle, hardener bottle, 
mixing cup with stick, weigh scale, small brush, medium brush, roller.

You must monitor for:
- SAFETY: Human must wear gloves when handling resin, hardener, or wet fiberglass
- QUALITY: Each layer should be fully saturated with resin before the next layer
- QUALITY: No visible wrinkles, folds, or air bubbles in the layup
- QUALITY: Resin/hardener ratio should be correct (visible check of pouring amounts)
- TIMING: Mixed resin has a limited shelf life (pot life)
- PROCESS: Steps should follow the correct sequence

Be conservative - if unsure, report WARNING rather than ERROR.
Only report CRITICAL for clear dangerous situations like no gloves during resin handling."""

    LAYUP_FAILURE_PROMPT = """Analyze these video frames to check the hand layup process.

CURRENT STEP: {instruction}

Look for these specific issues:
1. NO_GLOVES: Human handling resin, hardener, or wet fiberglass without gloves
2. POOR_COVERAGE: Dry spots visible on fiberglass layer (not fully saturated with resin)
3. WRINKLES: Visible wrinkles, folds, or bridging in the fiberglass layer
4. AIR_BUBBLES: Trapped air visible under fiberglass layer
5. WRONG_SEQUENCE: Human performing step out of order (e.g., placing layer before applying resin to previous)
6. SHELF_LIFE: Signs that mixture is becoming too viscous or gelling
7. SPILL: Resin or hardener spilled outside the work area
8. INCORRECT_RATIO: Visibly wrong proportions during resin/hardener measurement

Consider the frames in sequence - they show approximately {window_sec} seconds of the process.

Respond in JSON format:
{{
    "status": "OK" | "WARNING" | "ERROR" | "CRITICAL",
    "failure_type": "NONE" | "NO_GLOVES" | "POOR_COVERAGE" | "WRINKLES" | "AIR_BUBBLES" | "WRONG_SEQUENCE" | "SHELF_LIFE" | "SPILL" | "INCORRECT_RATIO" | "UNKNOWN",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of what you observed",
    "details": {{
        "human_wearing_gloves": true/false/unknown,
        "current_step_identified": "description of what human is doing",
        "resin_applied": true/false/unknown,
        "layer_quality": "good" | "acceptable" | "poor" | "unknown",
        "objects_visible": ["list of recognized objects"],
        "safety_concern": true/false
    }}
}}"""

    def __init__(self, config: Optional[PerformanceMonitorConfig] = None):
        """Initialize hand layup performance monitor.
        
        Args:
            config: Monitor configuration
        """
        super().__init__(config)
        
        # Shelf life tracking
        self._shelf_life_start: Optional[float] = None
        self._shelf_life_max_seconds: float = 1800.0  # 30 minutes default
        self._shelf_life_warning_seconds: float = 1200.0  # 20 minutes
        
        logger.info("HandLayupPerformanceMonitor initialized")
    
    def start_shelf_life_timer(self) -> None:
        """Start the mixture shelf life timer.
        
        Call this when resin/hardener mixing is complete.
        """
        self._shelf_life_start = time.time()
        logger.info("Shelf life timer started")
    
    def get_shelf_life_remaining(self) -> Optional[float]:
        """Get remaining shelf life in seconds.
        
        Returns:
            Remaining seconds, or None if timer not started
        """
        if self._shelf_life_start is None:
            return None
        elapsed = time.time() - self._shelf_life_start
        remaining = self._shelf_life_max_seconds - elapsed
        return max(0.0, remaining)
    
    def is_shelf_life_warning(self) -> bool:
        """Check if shelf life is approaching limit.
        
        Returns:
            True if within warning window
        """
        remaining = self.get_shelf_life_remaining()
        if remaining is None:
            return False
        return remaining <= (self._shelf_life_max_seconds - self._shelf_life_warning_seconds)
    
    def is_shelf_life_exceeded(self) -> bool:
        """Check if shelf life has been exceeded.
        
        Returns:
            True if shelf life is exceeded
        """
        remaining = self.get_shelf_life_remaining()
        if remaining is None:
            return False
        return remaining <= 0.0
    
    def _get_system_prompt(self) -> str:
        """Get hand layup specific system prompt."""
        return self.LAYUP_SYSTEM_PROMPT
    
    def _get_failure_detection_prompt(self, instruction: str, window_sec: float) -> str:
        """Get hand layup specific failure detection prompt."""
        return self.LAYUP_FAILURE_PROMPT.format(
            instruction=instruction,
            window_sec=f"{window_sec:.1f}"
        )


# Re-export base classes for convenience
__all__ = [
    "HandLayupPerformanceMonitor",
    "PerformanceMonitorConfig",
    "PerformanceCheckResult",
    "PerformanceStatus",
    "FailureType",
]

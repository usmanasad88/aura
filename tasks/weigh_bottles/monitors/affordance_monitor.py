"""Affordance Monitor for Weigh Bottles Task.

This task-specific monitor inherits from the base AffordanceMonitor
and adds bottle weighing specific programs and logic.

Available programs:
- pick_hardener_bottle.prog: Pick hardener from storage, deliver to human
- pick_resin_bottle.prog: Pick resin from storage, deliver to human  
- return_hardener_bottle.prog: Return hardener to storage
- return_resin_bottle.prog: Return resin to storage
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

from aura.monitors.affordance_monitor import (
    AffordanceMonitor,
    AffordanceMonitorConfig,
    RobotProgram,
    ProgramStatus,
)


logger = logging.getLogger(__name__)


class WeighBottlesAffordanceMonitor(AffordanceMonitor):
    """Affordance monitor for the weigh bottles task.
    
    Manages robot programs for picking and returning bottles
    in the bottle weighing assistance scenario.
    
    Programs run sequentially - a new program can only start
    after the previous one completes and the robot stops moving.
    
    Usage:
        monitor = WeighBottlesAffordanceMonitor()
        
        # Get available programs
        affordances = await monitor.update()
        
        # Start a program
        success = await monitor.start_program("pick_hardener_bottle.prog")
        
        # Mark program complete (call when robot stops moving)
        monitor.mark_program_complete("pick_hardener_bottle.prog")
    """
    
    # Default programs for bottle weighing task
    DEFAULT_PROGRAMS = [
        RobotProgram(
            id="pick_hardener_bottle.prog",
            name="Pick Hardener Bottle",
            description="Pick hardener bottle from storage and deliver to weighing station",
            prerequisites=[],
            duration_seconds=43.2,
        ),
        RobotProgram(
            id="pick_resin_bottle.prog", 
            name="Pick Resin Bottle",
            description="Pick resin bottle from storage and deliver to weighing station",
            prerequisites=["pick_hardener_bottle.prog"],
            duration_seconds=52.1,
        ),
        RobotProgram(
            id="return_hardener_bottle.prog",
            name="Return Hardener Bottle",
            description="Return hardener bottle from human to storage table",
            prerequisites=["pick_resin_bottle.prog"],
            duration_seconds=46.0,
        ),
        RobotProgram(
            id="return_resin_bottle.prog",
            name="Return Resin Bottle", 
            description="Return resin bottle from human to storage table",
            prerequisites=["return_hardener_bottle.prog"],
            duration_seconds=56.1,
        ),
    ]
    
    def __init__(
        self,
        config: Optional[AffordanceMonitorConfig] = None,
        programs_config_path: Optional[str] = None,
        on_program_start: Optional[Callable[[str], None]] = None,
        on_program_complete: Optional[Callable[[str], None]] = None,
    ):
        """Initialize affordance monitor.
        
        Args:
            config: Affordance configuration
            programs_config_path: Path to JSON file with program definitions
            on_program_start: Callback when a program starts
            on_program_complete: Callback when a program completes
        """
        self._programs_config_path = programs_config_path
        super().__init__(
            config=config,
            on_program_start=on_program_start,
            on_program_complete=on_program_complete,
        )
    
    def _load_programs(self) -> None:
        """Load bottle weighing programs from config or defaults."""
        if self._programs_config_path and Path(self._programs_config_path).exists():
            try:
                with open(self._programs_config_path) as f:
                    data = json.load(f)
                
                # Parse programs from JSON
                programs_data = data.get("robot_programs", {})
                for prog_id, prog_info in programs_data.items():
                    self.programs[prog_id] = RobotProgram(
                        id=prog_id,
                        name=prog_info.get("name", prog_id),
                        description=prog_info.get("description", ""),
                        duration_seconds=prog_info.get("duration_seconds", 0.0),
                        prerequisites=prog_info.get("prerequisites", []),
                    )
                
                logger.info(f"Loaded {len(self.programs)} programs from {self._programs_config_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load programs from {self._programs_config_path}: {e}")
        
        # Use defaults
        for prog in self.DEFAULT_PROGRAMS:
            self.programs[prog.id] = RobotProgram(
                id=prog.id,
                name=prog.name,
                description=prog.description,
                prerequisites=prog.prerequisites.copy(),
                duration_seconds=prog.duration_seconds,
            )
        
        logger.info(f"Loaded {len(self.programs)} default bottle weighing programs")


# Re-export base classes for convenience
__all__ = [
    "WeighBottlesAffordanceMonitor",
    "AffordanceMonitorConfig",
    "RobotProgram",
    "ProgramStatus",
]

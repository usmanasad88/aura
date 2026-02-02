"""Base Affordance Monitor for AURA framework.

Provides the abstract base class for affordance monitoring - tracking
what actions the robot can perform given the current scene and state.

Subclasses should implement task-specific affordance logic.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable

from aura.core import (
    MonitorType,
    AffordanceOutput,
    Affordance,
    RobotActionType,
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig


logger = logging.getLogger(__name__)


class ProgramStatus(Enum):
    """Status of a robot program/skill."""
    AVAILABLE = auto()      # Can be executed
    RUNNING = auto()        # Currently executing
    COMPLETED = auto()      # Finished successfully
    FAILED = auto()         # Execution failed
    BLOCKED = auto()        # Prerequisites not met


@dataclass
class RobotProgram:
    """Represents a saved robot program or skill.
    
    Attributes:
        id: Unique identifier for the program
        name: Human-readable name
        description: What the program does
        status: Current execution status
        prerequisites: List of program IDs that must complete first
        duration_seconds: Expected execution time
        started_at: When execution started
        completed_at: When execution completed
    """
    id: str
    name: str
    description: str
    status: ProgramStatus = ProgramStatus.AVAILABLE
    prerequisites: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def is_available(self) -> bool:
        """Check if program can be executed."""
        return self.status == ProgramStatus.AVAILABLE
    
    @property
    def is_running(self) -> bool:
        """Check if program is currently running."""
        return self.status == ProgramStatus.RUNNING
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since program started."""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


@dataclass 
class AffordanceMonitorConfig(MonitorConfig):
    """Configuration for affordance monitor."""
    check_physical_constraints: bool = True
    use_llm_reasoning: bool = False
    llm_model: str = "gemini-2.0-flash"


class AffordanceMonitor(BaseMonitor):
    """Base class for affordance monitoring.
    
    Tracks what actions the robot can perform given current scene and state.
    Manages program execution sequencing and prerequisites.
    
    Subclasses should override:
    - _load_programs(): Load task-specific programs
    - _get_scene_affordances(): Compute affordances from perception
    
    Usage:
        monitor = MyTaskAffordanceMonitor(config)
        
        # Get available actions
        output = await monitor.update()
        for affordance in output.affordances:
            print(f"Can do: {affordance.action_type}")
        
        # Start a program
        success = await monitor.start_program("pick_object.prog")
        
        # Mark complete when robot stops
        monitor.mark_program_complete("pick_object.prog")
    """
    
    def __init__(
        self,
        config: Optional[AffordanceMonitorConfig] = None,
        on_program_start: Optional[Callable[[str], None]] = None,
        on_program_complete: Optional[Callable[[str], None]] = None,
    ):
        """Initialize affordance monitor.
        
        Args:
            config: Monitor configuration
            on_program_start: Callback when a program starts
            on_program_complete: Callback when a program completes
        """
        super().__init__(config or AffordanceMonitorConfig())
        
        # Programs registry
        self.programs: Dict[str, RobotProgram] = {}
        self._load_programs()
        
        # Execution state
        self.current_program: Optional[str] = None
        self.program_queue: List[str] = []
        self.completed_programs: List[str] = []
        
        # Callbacks
        self.on_program_start = on_program_start
        self.on_program_complete = on_program_complete
        
        # Robot state (set externally)
        self._robot_is_moving = False
        self._robot_at_home = True
        
        logger.info(f"AffordanceMonitor initialized with {len(self.programs)} programs")
    
    @abstractmethod
    def _load_programs(self) -> None:
        """Load task-specific programs. Override in subclass."""
        pass
    
    def _get_scene_affordances(self, perception_output: Optional[Any] = None) -> List[Affordance]:
        """Compute affordances from perception data.
        
        Override in subclass for scene-based affordance detection.
        
        Args:
            perception_output: Output from perception monitor
            
        Returns:
            List of scene-based affordances
        """
        return []
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.AFFORDANCE
    
    def set_robot_state(self, is_moving: bool, at_home: bool = False) -> None:
        """Update robot state from external source.
        
        Args:
            is_moving: Whether the robot is currently moving
            at_home: Whether the robot is at home position
        """
        was_moving = self._robot_is_moving
        self._robot_is_moving = is_moving
        self._robot_at_home = at_home
        
        # Auto-complete current program if robot stopped moving
        if was_moving and not is_moving and self.current_program:
            self.mark_program_complete(self.current_program)
    
    def _check_prerequisites(self, program_id: str) -> bool:
        """Check if a program's prerequisites are met."""
        program = self.programs.get(program_id)
        if not program:
            return False
        
        for prereq_id in program.prerequisites:
            if prereq_id not in self.completed_programs:
                return False
        
        return True
    
    def _update_program_availability(self) -> None:
        """Update availability status of all programs."""
        for prog_id, program in self.programs.items():
            if program.status in (ProgramStatus.RUNNING, ProgramStatus.COMPLETED):
                continue
            
            # Check prerequisites
            if self._check_prerequisites(prog_id):
                # Available if no program is running and robot is at home
                if self.current_program is None and self._robot_at_home:
                    program.status = ProgramStatus.AVAILABLE
                else:
                    program.status = ProgramStatus.BLOCKED
            else:
                program.status = ProgramStatus.BLOCKED
    
    async def start_program(self, program_id: str) -> bool:
        """Request to start a robot program.
        
        Args:
            program_id: ID of the program to start
            
        Returns:
            True if program was started, False otherwise
        """
        program = self.programs.get(program_id)
        if not program:
            logger.error(f"Unknown program: {program_id}")
            return False
        
        # Check if another program is running
        if self.current_program is not None:
            logger.warning(f"Cannot start {program_id}: {self.current_program} is running")
            return False
        
        # Check prerequisites
        if not self._check_prerequisites(program_id):
            unmet = [p for p in program.prerequisites if p not in self.completed_programs]
            logger.warning(f"Cannot start {program_id}: prerequisites not met: {unmet}")
            return False
        
        # Check robot state
        if self._robot_is_moving:
            logger.warning(f"Cannot start {program_id}: robot is moving")
            return False
        
        # Start the program
        program.status = ProgramStatus.RUNNING
        program.started_at = datetime.now()
        program.completed_at = None
        self.current_program = program_id
        self._robot_is_moving = True
        self._robot_at_home = False
        
        logger.info(f"Started program: {program_id}")
        
        if self.on_program_start:
            self.on_program_start(program_id)
        
        self._update_program_availability()
        return True
    
    def mark_program_complete(self, program_id: str, success: bool = True) -> None:
        """Mark a program as completed.
        
        Args:
            program_id: ID of the completed program
            success: Whether the program succeeded
        """
        program = self.programs.get(program_id)
        if not program:
            logger.warning(f"Unknown program: {program_id}")
            return
        
        if program.status != ProgramStatus.RUNNING:
            logger.warning(f"Program {program_id} was not running")
            return
        
        program.completed_at = datetime.now()
        program.status = ProgramStatus.COMPLETED if success else ProgramStatus.FAILED
        
        if success:
            self.completed_programs.append(program_id)
        
        self.current_program = None
        self._robot_is_moving = False
        self._robot_at_home = True  # Assume programs return to home
        
        elapsed = program.elapsed_time
        logger.info(f"Program {program_id} completed in {elapsed:.1f}s")
        
        if self.on_program_complete:
            self.on_program_complete(program_id)
        
        self._update_program_availability()
    
    def get_available_programs(self) -> List[RobotProgram]:
        """Get list of currently available programs."""
        self._update_program_availability()
        return [p for p in self.programs.values() if p.status == ProgramStatus.AVAILABLE]
    
    def is_program_available(self, program_id: str) -> bool:
        """Check if a specific program is available to run.
        
        Args:
            program_id: ID of the program to check
            
        Returns:
            True if program can be executed now
        """
        self._update_program_availability()
        program = self.programs.get(program_id)
        return program is not None and program.status == ProgramStatus.AVAILABLE
    
    def get_next_program(self) -> Optional[RobotProgram]:
        """Get the next program in the sequence (if available)."""
        available = self.get_available_programs()
        return available[0] if available else None
    
    def get_current_program(self) -> Optional[RobotProgram]:
        """Get the currently running program."""
        if self.current_program:
            return self.programs.get(self.current_program)
        return None
    
    def get_program_progress(self) -> Dict[str, Any]:
        """Get progress summary of all programs."""
        return {
            "total_programs": len(self.programs),
            "completed": len(self.completed_programs),
            "current": self.current_program,
            "remaining": len(self.programs) - len(self.completed_programs) - (1 if self.current_program else 0),
            "programs": {
                prog_id: {
                    "name": prog.name,
                    "status": prog.status.name,
                    "elapsed_time": prog.elapsed_time if prog.started_at else 0,
                }
                for prog_id, prog in self.programs.items()
            }
        }
    
    def reset(self) -> None:
        """Reset all programs to initial state."""
        self.current_program = None
        self.completed_programs = []
        self._robot_is_moving = False
        self._robot_at_home = True
        
        for program in self.programs.values():
            program.status = ProgramStatus.AVAILABLE
            program.started_at = None
            program.completed_at = None
        
        self._update_program_availability()
        logger.info("Affordance monitor reset")
    
    async def _process(self, **inputs) -> AffordanceOutput:
        """Process and return available affordances.
        
        Args:
            robot_moving: Optional bool, whether robot is moving
            robot_at_home: Optional bool, whether robot is at home
            perception_output: Optional perception data for scene affordances
            
        Returns:
            AffordanceOutput with available actions
        """
        # Update robot state if provided
        if "robot_moving" in inputs:
            self._robot_is_moving = inputs["robot_moving"]
        if "robot_at_home" in inputs:
            self._robot_at_home = inputs["robot_at_home"]
        
        # Update program availability
        self._update_program_availability()
        
        # Get program-based affordances
        affordances = []
        for program in self.get_available_programs():
            affordances.append(Affordance(
                action_type=RobotActionType.FOLLOW_TRAJECTORY,
                target_object=program.id.split("_")[1] if "_" in program.id else None,
                parameters={
                    "program_id": program.id,
                    "program_name": program.name,
                    "expected_duration": program.duration_seconds,
                },
                feasibility=1.0,
                reasoning=program.description,
            ))
        
        # Get scene-based affordances
        perception_output = inputs.get("perception_output")
        scene_affordances = self._get_scene_affordances(perception_output)
        affordances.extend(scene_affordances)
        
        return AffordanceOutput(
            monitor_type=MonitorType.AFFORDANCE,
            timestamp=datetime.now(),
            is_valid=True,
            affordances=affordances,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize monitor state to dictionary."""
        return {
            "current_program": self.current_program,
            "completed_programs": self.completed_programs,
            "robot_moving": self._robot_is_moving,
            "robot_at_home": self._robot_at_home,
            "programs": {
                prog_id: {
                    "name": prog.name,
                    "status": prog.status.name,
                    "started_at": prog.started_at.isoformat() if prog.started_at else None,
                    "completed_at": prog.completed_at.isoformat() if prog.completed_at else None,
                }
                for prog_id, prog in self.programs.items()
            }
        }

"""Affordance Monitor for Hand Layup Task.

This task-specific monitor inherits from the base AffordanceMonitor
and defines object-moving skills for the hand layup process.

The robot can:
- Move objects from storage to workplace (deliver)
- Move objects from workplace to storage (housekeeping)
- Execute named programs from the UR5 program executor
- Move to named positions (joint or Cartesian)
- Control the gripper

Objects that can be moved:
- weigh_scale, resin_bottle, hardener_bottle, cup,
  brush_small, brush_medium, roller

Objects that must NOT be moved:
- mold (stays on workplace permanently)
- fiberglass_sheet (handled by human only)

When ``robot_api_url`` is provided, the monitor fetches real-time
available commands from the UR5 External Control REST API.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

from aura.monitors.affordance_monitor import (
    AffordanceMonitor,
    AffordanceMonitorConfig,
    RobotProgram,
    ProgramStatus,
)


logger = logging.getLogger(__name__)


class HandLayupAffordanceMonitor(AffordanceMonitor):
    """Affordance monitor for the hand layup task.
    
    Manages robot skills for moving objects between storage and
    workplace tables. Tracks which objects are where, and determines
    which move operations are currently available based on:
    - Object location (storage vs workplace)
    - Current task phase (what's needed / no longer needed)
    - Whether human is currently using the object
    
    Usage:
        monitor = HandLayupAffordanceMonitor()
        
        # Get available skills
        affordances = monitor.get_available_skills()
        
        # Start a skill
        success = await monitor.start_program("move_to_storage__weigh_scale")
        
        # Mark skill complete
        monitor.mark_program_complete("move_to_storage__weigh_scale")
    """
    
    # Objects the robot can move
    MOVABLE_OBJECTS = [
        "weigh_scale",
        "resin_bottle",
        "hardener_bottle",
        "cup",
        "brush_small",
        "brush_medium",
        "roller",
    ]
    
    # Objects that must not be moved by robot
    FIXED_OBJECTS = ["mold", "fiberglass_sheet"]
    
    # When to suggest housekeeping (object -> after which DAG node)
    HOUSEKEEPING_TRIGGERS = {
        "weigh_scale": "place_layer_1",
        "resin_bottle": "apply_resin_layer_4",
        "hardener_bottle": "apply_resin_layer_4",
        "cup": "apply_resin_layer_4",
        "brush_small": "apply_resin_layer_4",
        "brush_medium": "consolidate_with_roller",
        "roller": "cleanup",
    }
    
    def __init__(
        self,
        config: Optional[AffordanceMonitorConfig] = None,
        programs_config_path: Optional[str] = None,
        robot_api_url: Optional[str] = None,
        robot_client=None,
        on_program_start: Optional[Callable[[str], None]] = None,
        on_program_complete: Optional[Callable[[str], None]] = None,
    ):
        """Initialize affordance monitor.
        
        Args:
            config: Affordance configuration
            programs_config_path: Path to JSON file with program definitions
            robot_api_url: URL of the UR5 External Control API (e.g. "http://localhost:5050").
                           When set, available programs/positions are fetched from the live API.
            robot_client: Pre-built RobotControlClient instance (takes precedence over robot_api_url)
            on_program_start: Callback when a skill starts
            on_program_complete: Callback when a skill completes
        """
        self._programs_config_path = programs_config_path
        self._robot_api_url = robot_api_url
        
        # Build or store robot control client
        self._robot_client = robot_client
        if self._robot_client is None and self._robot_api_url:
            try:
                from aura.interfaces.robot_control_client import RobotControlClient
                self._robot_client = RobotControlClient(self._robot_api_url)
            except Exception as e:
                logger.warning(f"Could not create RobotControlClient: {e}")
        
        # Track API-fetched commands (programs, named positions)
        self._api_commands: Optional[Dict[str, Any]] = None
        
        # Track object locations
        self.object_locations: Dict[str, str] = {
            obj: "workplace" for obj in self.MOVABLE_OBJECTS
        }
        
        # Track current task node for housekeeping decisions
        self.current_task_node: Optional[str] = None
        self.completed_task_nodes: List[str] = []
        
        super().__init__(
            config=config,
            on_program_start=on_program_start,
            on_program_complete=on_program_complete,
        )
    
    def _load_programs(self) -> None:
        """Load hand layup robot skills as programs.
        
        Sources (in priority order):
        1. Live robot API (if robot_api_url / robot_client is set)
        2. JSON config file (programs_config_path)
        3. Default hardcoded object-moving skills
        
        Creates two programs per movable object:
        - move_to_workplace__{object}: Deliver from storage to workplace
        - move_to_storage__{object}: Housekeeping from workplace to storage
        
        Plus any programs and named positions fetched from the API.
        """
        # --- Source 1: Live robot API ---
        if self._robot_client is not None:
            try:
                self._api_commands = self._robot_client.get_commands(force_refresh=True)
                self._load_from_api(self._api_commands)
                logger.info(f"Loaded {len(self.programs)} skills from robot API")
            except Exception as e:
                logger.warning(f"Could not fetch from robot API: {e}")
                self._api_commands = None
        
        # --- Source 2 / 3: JSON config or defaults (object-moving skills) ---
        # Always add object-moving skills on top of API-fetched ones
        self._load_object_moving_skills()
    
    def _load_from_api(self, commands: Dict[str, Any]) -> None:
        """Populate programs from the external control API response."""
        # Programs (.prog files on the robot)
        for prog in commands.get("programs", []):
            prog_id = f"api_program__{prog['name']}"
            self.programs[prog_id] = RobotProgram(
                id=prog_id,
                name=prog["name"],
                description=prog.get("description", f"Robot program: {prog['name']}"),
                prerequisites=[],
                duration_seconds=10.0,
            )
        
        # Named positions — joint
        for pos in commands.get("named_positions", {}).get("joint", []):
            pos_id = f"api_joint__{pos['name']}"
            self.programs[pos_id] = RobotProgram(
                id=pos_id,
                name=f"Move to {pos['name']} (joint)",
                description=pos.get("description", f"Move to joint position {pos['name']}"),
                prerequisites=[],
                duration_seconds=5.0,
            )
        
        # Named positions — Cartesian pose
        for pos in commands.get("named_positions", {}).get("pose", []):
            pos_id = f"api_pose__{pos['name']}"
            self.programs[pos_id] = RobotProgram(
                id=pos_id,
                name=f"Move to {pos['name']} (pose)",
                description=pos.get("description", f"Move to Cartesian pose {pos['name']}"),
                prerequisites=[],
                duration_seconds=8.0,
            )
    
    def _load_object_moving_skills(self) -> None:
        """Load the hand-layup-specific object-moving skills."""
        if self._programs_config_path and Path(self._programs_config_path).exists():
            try:
                with open(self._programs_config_path) as f:
                    data = json.load(f)
                
                skills = data.get("robot_skills", {})
                for skill_id, skill_info in skills.items():
                    for obj in self.MOVABLE_OBJECTS:
                        prog_id = f"{skill_id}__{obj}"
                        if prog_id not in self.programs:
                            self.programs[prog_id] = RobotProgram(
                                id=prog_id,
                                name=f"{skill_info.get('description', skill_id)} - {obj}",
                                description=f"{skill_info.get('description', '')} ({obj})",
                                duration_seconds=skill_info.get("duration_seconds", 15.0),
                                prerequisites=[],
                            )
                
                logger.info(f"Loaded object-moving skills from {self._programs_config_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load skills from {self._programs_config_path}: {e}")
        
        # Build default programs
        for obj in self.MOVABLE_OBJECTS:
            # Deliver to workplace
            deliver_id = f"move_to_workplace__{obj}"
            if deliver_id not in self.programs:
                self.programs[deliver_id] = RobotProgram(
                    id=deliver_id,
                    name=f"Deliver {obj} to Workplace",
                    description=f"Pick {obj} from storage table and place on workplace table",
                    prerequisites=[],
                    duration_seconds=15.0,
                )
            
            # Housekeeping to storage
            remove_id = f"move_to_storage__{obj}"
            if remove_id not in self.programs:
                self.programs[remove_id] = RobotProgram(
                    id=remove_id,
                    name=f"Remove {obj} to Storage",
                    description=f"Pick {obj} from workplace table and place on storage table (housekeeping)",
                    prerequisites=[],
                    duration_seconds=15.0,
                )
        
        logger.info(f"Total skills loaded: {len(self.programs)}")
    
    def get_available_skills(self, current_node: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently available robot skills based on object locations.
        
        Args:
            current_node: Current task DAG node (for housekeeping suggestions)
            
        Returns:
            List of available skill descriptions with priority info
        """
        if current_node:
            self.current_task_node = current_node
        
        available = []
        
        for obj in self.MOVABLE_OBJECTS:
            loc = self.object_locations.get(obj, "workplace")
            
            if loc == "workplace":
                # Can remove to storage
                skill_id = f"move_to_storage__{obj}"
                is_housekeeping = self._should_housekeep(obj)
                available.append({
                    "skill_id": skill_id,
                    "object": obj,
                    "action": "remove_to_storage",
                    "from": "workplace",
                    "to": "storage",
                    "is_housekeeping_suggestion": is_housekeeping,
                    "priority": "high" if is_housekeeping else "low",
                })
            elif loc == "storage":
                # Can deliver to workplace
                skill_id = f"move_to_workplace__{obj}"
                available.append({
                    "skill_id": skill_id,
                    "object": obj,
                    "action": "deliver_to_workplace",
                    "from": "storage",
                    "to": "workplace",
                    "is_housekeeping_suggestion": False,
                    "priority": "medium",
                })
        
        return available
    
    def _should_housekeep(self, obj: str) -> bool:
        """Check if an object should be moved to storage based on task progress.
        
        Args:
            obj: Object identifier
            
        Returns:
            True if the object is no longer needed and should be stored
        """
        trigger_node = self.HOUSEKEEPING_TRIGGERS.get(obj)
        if trigger_node and trigger_node in self.completed_task_nodes:
            return True
        return False
    
    def update_object_location(self, obj: str, location: str) -> None:
        """Update the tracked location of an object.
        
        Args:
            obj: Object identifier
            location: New location ('workplace', 'storage', 'robot_gripper')
        """
        if obj in self.object_locations:
            self.object_locations[obj] = location
            logger.info(f"Object {obj} moved to {location}")
    
    def mark_task_node_complete(self, node: str) -> None:
        """Record that a task DAG node has been completed.
        
        This is used to trigger housekeeping suggestions.
        
        Args:
            node: Completed DAG node identifier
        """
        if node not in self.completed_task_nodes:
            self.completed_task_nodes.append(node)
            logger.info(f"Task node completed: {node}")
    
    def get_housekeeping_suggestions(self) -> List[Dict[str, str]]:
        """Get list of objects that should be removed from workplace.
        
        Returns:
            List of housekeeping suggestions with object and reason
        """
        suggestions = []
        for obj in self.MOVABLE_OBJECTS:
            if (self.object_locations.get(obj) == "workplace" and
                    self._should_housekeep(obj)):
                trigger = self.HOUSEKEEPING_TRIGGERS.get(obj, "unknown")
                suggestions.append({
                    "object": obj,
                    "skill_id": f"move_to_storage__{obj}",
                    "reason": f"No longer needed after {trigger}",
                })
        return suggestions

    # ── Robot API integration ────────────────────────────────────────────

    @property
    def robot_client(self):
        """Access the underlying RobotControlClient (may be None)."""
        return self._robot_client

    @property
    def api_connected(self) -> bool:
        """Check if the external control API is reachable."""
        if self._robot_client is None:
            return False
        try:
            return self._robot_client.is_available()
        except Exception:
            return False

    def refresh_api_commands(self) -> bool:
        """Re-fetch commands from the external control API.
        
        Returns True if new commands were successfully loaded.
        """
        if self._robot_client is None:
            return False
        try:
            self._api_commands = self._robot_client.get_commands(force_refresh=True)
            self._load_from_api(self._api_commands)
            logger.info(f"Refreshed API commands: {len(self.programs)} total skills")
            return True
        except Exception as e:
            logger.warning(f"Failed to refresh API commands: {e}")
            return False

    def execute_via_api(self, program_id: str) -> Dict[str, Any]:
        """Execute a program/skill via the external control API.
        
        Dispatches the right API call based on the program_id prefix:
        - ``api_program__*``  → POST /api/program/execute
        - ``api_joint__*``    → POST /api/move/named
        - ``api_pose__*``     → POST /api/move/named
        - ``move_to_*__obj``  → POST /api/move/named (if mapped)
        
        Args:
            program_id: The affordance/program ID to execute
            
        Returns:
            API response dict with success/message
        """
        if self._robot_client is None:
            return {"success": False, "message": "No robot API client configured"}
        
        if program_id.startswith("api_program__"):
            prog_name = program_id[len("api_program__"):]
            return self._robot_client.execute_program(prog_name)
        
        elif program_id.startswith("api_joint__"):
            pos_name = program_id[len("api_joint__"):]
            return self._robot_client.move_to_named(pos_name)
        
        elif program_id.startswith("api_pose__"):
            pos_name = program_id[len("api_pose__"):]
            return self._robot_client.move_to_named(pos_name)
        
        else:
            # Not an API-backed program — return info only
            return {
                "success": False,
                "message": f"Program '{program_id}' is not API-backed (object-moving skill — requires real robot mapping)",
            }

    def get_api_commands_summary(self) -> str:
        """Return a human-readable summary of API-fetched commands."""
        if self._robot_client is None:
            return "No robot API configured"
        try:
            return self._robot_client.get_commands_summary()
        except Exception as e:
            return f"Error fetching commands: {e}"


# Re-export base classes for convenience
__all__ = [
    "HandLayupAffordanceMonitor",
    "AffordanceMonitorConfig",
    "RobotProgram",
    "ProgramStatus",
]

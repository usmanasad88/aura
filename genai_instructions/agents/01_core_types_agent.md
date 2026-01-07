# Agent 01: Core Types Agent

## Task: Define Core Data Types and Enums

### Objective
Create the foundational data types that all other components will use. This ensures consistent interfaces across the entire AURA framework.

### Prerequisites
- Sprint 0 complete (setup agent finished)
- `uv sync` successful

### Files to Create

#### 1. `src/aura/core/enums.py`

```python
"""Core enumerations for AURA framework."""

from enum import Enum, auto


class ActionStatus(Enum):
    """Status of an action being executed."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class IntentType(Enum):
    """Types of detected human intent."""
    IDLE = auto()
    REACHING = auto()
    GRASPING = auto()
    MOVING = auto()
    PLACING = auto()
    GESTURING = auto()
    SPEAKING = auto()
    UNKNOWN = auto()


class RobotActionType(Enum):
    """Types of robot actions."""
    MOVE_TO_POSE = auto()
    FOLLOW_TRAJECTORY = auto()
    OPEN_GRIPPER = auto()
    CLOSE_GRIPPER = auto()
    WAIT = auto()
    SPEAK = auto()
    DISPLAY_MESSAGE = auto()


class MonitorType(Enum):
    """Types of monitors in the system."""
    INTENT = auto()
    MOTION = auto()
    PERCEPTION = auto()
    SOUND = auto()
    AFFORDANCE = auto()
    PERFORMANCE = auto()


class TaskState(Enum):
    """Overall task state."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    WAITING_FOR_HUMAN = auto()
    WAITING_FOR_ROBOT = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
```

#### 2. `src/aura/core/types.py`

```python
"""Core data types for AURA framework."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .enums import (
    ActionStatus, IntentType, RobotActionType, 
    MonitorType, TaskState
)


# ============================================================================
# Geometry Types
# ============================================================================

@dataclass
class Pose2D:
    """2D position and orientation."""
    x: float
    y: float
    theta: float = 0.0  # radians


@dataclass
class Pose3D:
    """3D position and orientation (quaternion)."""
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    
    @classmethod
    def from_position(cls, x: float, y: float, z: float) -> "Pose3D":
        """Create pose from position only (identity rotation)."""
        return cls(x=x, y=y, z=z)


@dataclass
class BoundingBox:
    """2D bounding box in image coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of bounding box."""
        return ((self.x_min + self.x_max) // 2, 
                (self.y_min + self.y_max) // 2)
    
    @property
    def area(self) -> int:
        """Area of bounding box."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass  
class Trajectory:
    """A sequence of poses over time."""
    poses: List[Pose3D]
    timestamps: List[float]  # seconds from start
    
    @property
    def duration(self) -> float:
        """Total duration of trajectory."""
        return self.timestamps[-1] if self.timestamps else 0.0


# ============================================================================
# Object Types
# ============================================================================

@dataclass
class TrackedObject:
    """An object being tracked in the scene."""
    id: str
    name: str
    category: str
    pose: Optional[Pose3D] = None
    bbox: Optional[BoundingBox] = None
    mask: Optional[np.ndarray] = None  # Binary mask
    confidence: float = 1.0
    last_seen: Optional[datetime] = None
    velocity: Optional[Tuple[float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Semantic scene representation."""
    objects: List[TrackedObject]
    relationships: Dict[str, List[Tuple[str, str]]]  # e.g., "on": [("cup", "table")]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Human Types
# ============================================================================

@dataclass
class JointPosition:
    """A single joint position."""
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0


@dataclass
class HumanPose:
    """Human pose estimation result."""
    joints: Dict[str, JointPosition]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def left_hand(self) -> Optional[JointPosition]:
        return self.joints.get("left_wrist") or self.joints.get("left_hand")
    
    @property
    def right_hand(self) -> Optional[JointPosition]:
        return self.joints.get("right_wrist") or self.joints.get("right_hand")


@dataclass
class Intent:
    """Detected human intent."""
    type: IntentType
    target_object: Optional[str] = None  # Object ID
    target_location: Optional[Pose3D] = None
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictedMotion:
    """Predicted future motion of human or object."""
    entity_id: str  # "human" or object ID
    predicted_trajectory: Trajectory
    confidence: float = 0.0
    prediction_horizon_sec: float = 2.0


# ============================================================================
# Communication Types
# ============================================================================

@dataclass
class Utterance:
    """A spoken or text communication."""
    text: str
    speaker: str  # "human" or "robot"
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None  # Extracted intent from speech
    is_command: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Action Types
# ============================================================================

@dataclass
class Action:
    """An action the robot can take."""
    id: str
    type: RobotActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def mark_started(self):
        self.status = ActionStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def mark_completed(self):
        self.status = ActionStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: str):
        self.status = ActionStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error


@dataclass
class Affordance:
    """An action possibility given current scene."""
    action_type: RobotActionType
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    feasibility: float = 1.0  # 0-1, how feasible is this action
    reasoning: str = ""


# ============================================================================
# Task Types
# ============================================================================

@dataclass
class TaskNode:
    """A node in the task graph."""
    id: str
    name: str
    description: str
    preconditions: List[str] = field(default_factory=list)  # Node IDs
    postconditions: List[str] = field(default_factory=list)
    assignee: str = "any"  # "human", "robot", or "any"
    is_completed: bool = False
    is_active: bool = False


@dataclass
class TaskGraph:
    """Task graph representing the SOP."""
    id: str
    name: str
    description: str
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    
    def get_available_nodes(self) -> List[TaskNode]:
        """Get nodes whose preconditions are met."""
        available = []
        completed_ids = {n.id for n in self.nodes.values() if n.is_completed}
        for node in self.nodes.values():
            if node.is_completed:
                continue
            if all(pre in completed_ids for pre in node.preconditions):
                available.append(node)
        return available


# ============================================================================
# State Types
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    actions_attempted: int = 0
    actions_succeeded: int = 0
    actions_failed: int = 0
    average_action_time_sec: float = 0.0
    last_error: Optional[str] = None


@dataclass
class AuraState:
    """Complete system state."""
    # Scene understanding
    objects: List[TrackedObject] = field(default_factory=list)
    scene_graph: Optional[SceneGraph] = None
    
    # Human tracking
    human_pose: Optional[HumanPose] = None
    human_intent: Optional[Intent] = None
    predicted_motion: Optional[PredictedMotion] = None
    
    # Task state
    task_graph: Optional[TaskGraph] = None
    task_state: TaskState = TaskState.NOT_STARTED
    
    # Communication
    verbal_history: List[Utterance] = field(default_factory=list)
    pending_clarifications: List[str] = field(default_factory=list)
    
    # Robot capabilities
    available_affordances: List[Affordance] = field(default_factory=list)
    current_action: Optional[Action] = None
    action_queue: List[Action] = field(default_factory=list)
    
    # Performance
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Timing
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_timestamp(self):
        self.last_updated = datetime.now()


# ============================================================================
# Monitor Output Types
# ============================================================================

@dataclass
class MonitorOutput:
    """Base class for monitor outputs."""
    monitor_type: MonitorType
    timestamp: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class PerceptionOutput(MonitorOutput):
    """Output from perception module."""
    objects: List[TrackedObject] = field(default_factory=list)
    scene_description: Optional[str] = None
    
    def __post_init__(self):
        self.monitor_type = MonitorType.PERCEPTION


@dataclass
class IntentOutput(MonitorOutput):
    """Output from intent monitor."""
    intent: Optional[Intent] = None
    alternatives: List[Intent] = field(default_factory=list)
    
    def __post_init__(self):
        self.monitor_type = MonitorType.INTENT


@dataclass
class MotionOutput(MonitorOutput):
    """Output from motion predictor."""
    predictions: List[PredictedMotion] = field(default_factory=list)
    collision_risk: float = 0.0
    
    def __post_init__(self):
        self.monitor_type = MonitorType.MOTION


@dataclass
class SoundOutput(MonitorOutput):
    """Output from sound monitor."""
    utterances: List[Utterance] = field(default_factory=list)
    is_relevant: bool = True
    
    def __post_init__(self):
        self.monitor_type = MonitorType.SOUND


@dataclass
class AffordanceOutput(MonitorOutput):
    """Output from affordance monitor."""
    affordances: List[Affordance] = field(default_factory=list)
    
    def __post_init__(self):
        self.monitor_type = MonitorType.AFFORDANCE


@dataclass
class PerformanceOutput(MonitorOutput):
    """Output from performance monitor."""
    action_status: Optional[ActionStatus] = None
    progress: float = 0.0  # 0-1
    should_abort: bool = False
    
    def __post_init__(self):
        self.monitor_type = MonitorType.PERFORMANCE
```

#### 3. `src/aura/core/__init__.py`

```python
"""Core types and enums for AURA framework."""

from .enums import (
    ActionStatus,
    IntentType,
    RobotActionType,
    MonitorType,
    TaskState,
)

from .types import (
    # Geometry
    Pose2D,
    Pose3D,
    BoundingBox,
    Trajectory,
    # Objects
    TrackedObject,
    SceneGraph,
    # Human
    JointPosition,
    HumanPose,
    Intent,
    PredictedMotion,
    # Communication
    Utterance,
    # Actions
    Action,
    Affordance,
    # Tasks
    TaskNode,
    TaskGraph,
    # State
    PerformanceMetrics,
    AuraState,
    # Monitor outputs
    MonitorOutput,
    PerceptionOutput,
    IntentOutput,
    MotionOutput,
    SoundOutput,
    AffordanceOutput,
    PerformanceOutput,
)

__all__ = [
    # Enums
    "ActionStatus",
    "IntentType", 
    "RobotActionType",
    "MonitorType",
    "TaskState",
    # Types
    "Pose2D",
    "Pose3D",
    "BoundingBox",
    "Trajectory",
    "TrackedObject",
    "SceneGraph",
    "JointPosition",
    "HumanPose",
    "Intent",
    "PredictedMotion",
    "Utterance",
    "Action",
    "Affordance",
    "TaskNode",
    "TaskGraph",
    "PerformanceMetrics",
    "AuraState",
    "MonitorOutput",
    "PerceptionOutput",
    "IntentOutput",
    "MotionOutput",
    "SoundOutput",
    "AffordanceOutput",
    "PerformanceOutput",
]
```

#### 4. `tests/test_core/test_types.py`

```python
"""Tests for core types."""

import pytest
from datetime import datetime
import numpy as np

from aura.core import (
    Pose3D, BoundingBox, Trajectory,
    TrackedObject, Intent, IntentType,
    Action, RobotActionType, ActionStatus,
    TaskNode, TaskGraph, AuraState,
)


class TestPose3D:
    def test_from_position(self):
        pose = Pose3D.from_position(1.0, 2.0, 3.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0  # Identity quaternion


class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        assert bbox.center == (50, 50)
    
    def test_area(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        assert bbox.area == 5000


class TestTrajectory:
    def test_duration(self):
        poses = [Pose3D.from_position(0, 0, 0), Pose3D.from_position(1, 1, 1)]
        traj = Trajectory(poses=poses, timestamps=[0.0, 2.5])
        assert traj.duration == 2.5
    
    def test_empty_duration(self):
        traj = Trajectory(poses=[], timestamps=[])
        assert traj.duration == 0.0


class TestTrackedObject:
    def test_creation(self):
        obj = TrackedObject(
            id="obj_1",
            name="cup",
            category="container",
            confidence=0.95
        )
        assert obj.id == "obj_1"
        assert obj.confidence == 0.95


class TestAction:
    def test_lifecycle(self):
        action = Action(
            id="action_1",
            type=RobotActionType.MOVE_TO_POSE,
            parameters={"x": 1.0, "y": 2.0}
        )
        assert action.status == ActionStatus.PENDING
        
        action.mark_started()
        assert action.status == ActionStatus.IN_PROGRESS
        assert action.started_at is not None
        
        action.mark_completed()
        assert action.status == ActionStatus.COMPLETED
        assert action.completed_at is not None
    
    def test_failure(self):
        action = Action(id="a1", type=RobotActionType.CLOSE_GRIPPER)
        action.mark_started()
        action.mark_failed("Object slipped")
        assert action.status == ActionStatus.FAILED
        assert action.error_message == "Object slipped"


class TestTaskGraph:
    def test_available_nodes(self):
        graph = TaskGraph(
            id="sorting",
            name="Sorting Task",
            description="Sort objects",
            nodes={
                "start": TaskNode(id="start", name="Start", description="Begin", is_completed=True),
                "pick": TaskNode(id="pick", name="Pick", description="Pick object", preconditions=["start"]),
                "place": TaskNode(id="place", name="Place", description="Place object", preconditions=["pick"]),
            }
        )
        
        available = graph.get_available_nodes()
        assert len(available) == 1
        assert available[0].id == "pick"


class TestAuraState:
    def test_update_timestamp(self):
        state = AuraState()
        old_time = state.last_updated
        import time
        time.sleep(0.01)
        state.update_timestamp()
        assert state.last_updated > old_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Validation

Run tests:
```bash
cd /home/mani/Repos/aura
uv run pytest tests/test_core/test_types.py -v
```

Expected output: All tests pass.

### Handoff Notes

After completing this task, create `genai_instructions/handoff/01_core_types.md`:

```markdown
# Core Types Implementation Complete

## Files Created
- src/aura/core/enums.py
- src/aura/core/types.py  
- src/aura/core/__init__.py
- tests/test_core/test_types.py

## Test Results
All X tests passed.

## Notes for Next Agent
- All types use dataclasses for easy serialization
- MonitorOutput types have `__post_init__` to set monitor_type
- AuraState is the main state container - add fields as needed
- Trajectory uses list of poses, not numpy arrays, for JSON serialization
```

### Dependencies for Next Tasks
- Task 1.2 (Base Monitor) depends on these types
- Task 1.3 (Config) is independent
- All Sprint 2 tasks depend on these types

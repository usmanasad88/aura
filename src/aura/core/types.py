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
    monitor_type: MonitorType = field(default=MonitorType.PERCEPTION)
    objects: List[TrackedObject] = field(default_factory=list)
    scene_description: Optional[str] = None


@dataclass
class IntentOutput(MonitorOutput):
    """Output from intent monitor."""
    monitor_type: MonitorType = field(default=MonitorType.INTENT)
    intent: Optional[Intent] = None
    alternatives: List[Intent] = field(default_factory=list)


@dataclass
class MotionOutput(MonitorOutput):
    """Output from motion predictor."""
    monitor_type: MonitorType = field(default=MonitorType.MOTION)
    predictions: List[PredictedMotion] = field(default_factory=list)
    collision_risk: float = 0.0


@dataclass
class SoundOutput(MonitorOutput):
    """Output from sound monitor."""
    monitor_type: MonitorType = field(default=MonitorType.SOUND)
    utterances: List[Utterance] = field(default_factory=list)
    is_relevant: bool = True


@dataclass
class AffordanceOutput(MonitorOutput):
    """Output from affordance monitor."""
    monitor_type: MonitorType = field(default=MonitorType.AFFORDANCE)
    affordances: List[Affordance] = field(default_factory=list)


@dataclass
class PerformanceOutput(MonitorOutput):
    """Output from performance monitor."""
    monitor_type: MonitorType = field(default=MonitorType.PERFORMANCE)
    action_status: Optional[ActionStatus] = None
    progress: float = 0.0  # 0-1
    should_abort: bool = False

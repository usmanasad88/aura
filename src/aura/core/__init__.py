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

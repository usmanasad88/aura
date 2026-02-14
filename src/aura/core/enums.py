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
    POSE_TRACKING = auto()


class TaskState(Enum):
    """Overall task state."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    WAITING_FOR_HUMAN = auto()
    WAITING_FOR_ROBOT = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()

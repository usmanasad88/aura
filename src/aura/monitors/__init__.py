"""Monitor components for AURA framework."""

from .base_monitor import BaseMonitor, MonitorConfig
from .monitor_bus import MonitorEventBus, MonitorEvent
from .perception_module import PerceptionModule, PerceptionConfig
from .intent_monitor import IntentMonitor, IntentPrediction, visualize_intent
from .motion_predictor import (
    MotionPredictor, 
    HandTrackingResult, 
    visualize_motion_prediction
)
from .gesture_monitor import (
    GestureMonitor,
    GestureMonitorConfig,
    GestureOutput,
    GestureRecognitionResult,
)
from .affordance_monitor import (
    AffordanceMonitor,
    AffordanceMonitorConfig,
    RobotProgram,
    ProgramStatus,
)
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceCheckResult,
    PerformanceStatus,
    FailureType,
)
from .pose_tracking_monitor import PoseTrackingMonitor
from .local_vlm_monitor import (
    LocalVLMMonitor,
    LocalVLMConfig,
    LocalVLMOutput,
    VLMPerception,
    DetectedObject,
)
from .body_pose_monitor import BodyPoseMonitor, BodyPoseMonitorConfig

__all__ = [
    # Base
    "BaseMonitor",
    "MonitorConfig", 
    "MonitorEventBus",
    "MonitorEvent",
    # Perception
    "PerceptionModule",
    "PerceptionConfig",
    # Intent
    "IntentMonitor",
    "IntentPrediction",
    "visualize_intent",
    # Motion
    "MotionPredictor",
    "HandTrackingResult",
    "visualize_motion_prediction",
    # Gesture
    "GestureMonitor",
    "GestureMonitorConfig",
    "GestureOutput",
    "GestureRecognitionResult",
    # Affordance
    "AffordanceMonitor",
    "AffordanceMonitorConfig",
    "RobotProgram",
    "ProgramStatus",
    # Performance
    "PerformanceMonitor",
    "PerformanceMonitorConfig",
    "PerformanceCheckResult",
    "PerformanceStatus",
    "FailureType",
    # Pose Tracking
    "PoseTrackingMonitor",
    # Local VLM
    "LocalVLMMonitor",
    "LocalVLMConfig",
    "LocalVLMOutput",
    "VLMPerception",
    "DetectedObject",
    # Body Pose (SAM-3D-Body via ZMQ)
    "BodyPoseMonitor",
    "BodyPoseMonitorConfig",
]

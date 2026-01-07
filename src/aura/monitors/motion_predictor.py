"""Motion prediction monitor for hand tracking and trajectory prediction.

Uses MediaPipe for real-time hand landmark tracking and predicts future
hand positions based on recent motion history.

This is a lightweight, fast monitor that works in conjunction with
IntentMonitor for complete human intention understanding.
"""

import os
import time
import asyncio
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import math

import cv2
import numpy as np

# Optional MediaPipe for hand tracking
try:
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.framework.formats import landmark_pb2
    import mediapipe as mp
    from mediapipe import Image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from aura.core import (
    MonitorType, MotionOutput, Intent, IntentType,
    PredictedMotion, HumanPose, JointPosition, Pose3D, Trajectory
)
from aura.monitors.base_monitor import BaseMonitor
from aura.utils.config import MotionPredictorConfig


logger = logging.getLogger(__name__)


@dataclass
class HandTrackingResult:
    """Result from hand tracking."""
    timestamp: float
    landmarks: List[Tuple[float, float, float]]  # (x, y, z) normalized
    handedness: str  # "Left" or "Right"
    confidence: float
    wrist_position: Tuple[float, float]  # (x, y) normalized
    gesture: Optional[str] = None  # Recognized gesture (e.g., "Open_Palm", "Pointing_Up")


class MotionPredictor(BaseMonitor):
    """Real-time motion prediction using MediaPipe hand tracking.
    
    Tracks hand landmarks over time and predicts future hand positions
    based on motion history. This is a fast, local computation that
    doesn't require API calls.
    
    For intent/action recognition, use IntentMonitor which uses Gemini.
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self, config: Optional[MotionPredictorConfig] = None):
        """Initialize motion predictor.
        
        Args:
            config: Configuration for motion prediction
        """
        super().__init__(config)
        
        self.fps = config.fps if config else 15.0  # Higher FPS for motion tracking
        self.window_duration = config.window_duration if config else 1.0
        self.max_frames = int(self.fps * self.window_duration)
        
        self.prediction_horizon = config.prediction_horizon if config else 1.0
        self.num_prediction_points = getattr(config, 'num_prediction_points', 5) if config else 5
        
        # Hand tracking history: store recent hand positions
        self.hand_history: deque = deque(maxlen=self.max_frames)
        self.last_capture_time = 0
        self.capture_interval = 1.0 / self.fps
        
        # Prediction state
        self.last_prediction_time = 0
        self.prediction_interval = getattr(config, 'prediction_interval', 0.1) if config else 0.1
        
        # MediaPipe hand tracking and gesture recognition (v0.10+ API)
        self.hands = None
        self.gesture_recognizer = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        
        if MEDIAPIPE_AVAILABLE:
            try:
                logger.info("Initializing MediaPipe hand tracking and gesture recognition...")
                
                # Initialize HandLandmarker
                self.hands = self._init_hand_landmarker()
                if self.hands:
                    logger.info("✓ MediaPipe HandLandmarker initialized")
                
                # Initialize GestureRecognizer for gesture classification
                self.gesture_recognizer = self._init_gesture_recognizer()
                if self.gesture_recognizer:
                    logger.info("✓ MediaPipe GestureRecognizer initialized")
                
                # Setup drawing utilities
                self.mp_drawing = mp.solutions.hands
                self.mp_drawing_styles = mp.solutions.drawing_utils
                
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")
                self.hands = None
                self.gesture_recognizer = None
        else:
            logger.warning("MediaPipe not available, hand tracking disabled")
        
        # Last tracking result
        self.last_hand_pose: Optional[HumanPose] = None
        self.last_predicted_motion: Optional[PredictedMotion] = None
    
    def _init_hand_landmarker(self) -> Optional[vision.HandLandmarker]:
        """Initialize MediaPipe HandLandmarker with model download if needed."""
        try:
            # Create model directory
            model_dir = os.path.expanduser("~/.mediapipe/models")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "hand_landmarker.task")
            
            # Download model if it doesn't exist
            if not os.path.exists(model_path):
                logger.info("Downloading MediaPipe hand landmarker model...")
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(model_url, model_path)
                logger.info(f"Hand landmarker model downloaded")
            
            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            logger.warning(f"Failed to initialize HandLandmarker: {e}")
            return None
    
    def _init_gesture_recognizer(self) -> Optional[vision.GestureRecognizer]:
        """Initialize MediaPipe GestureRecognizer with model download if needed."""
        try:
            # Create model directory
            model_dir = os.path.expanduser("~/.mediapipe/models")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "gesture_recognizer.task")
            
            # Download model if it doesn't exist
            if not os.path.exists(model_path):
                logger.info("Downloading MediaPipe gesture recognizer model...")
                model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
                urllib.request.urlretrieve(model_url, model_path)
                logger.info(f"Gesture recognizer model downloaded")
            
            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return vision.GestureRecognizer.create_from_options(options)
        except Exception as e:
            logger.warning(f"Failed to initialize GestureRecognizer: {e}")
            return None
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.MOTION
    
    async def _process(self, frame: Optional[np.ndarray] = None) -> Optional[MotionOutput]:
        """Process frame and predict motion.
        
        Args:
            frame: Current video frame (BGR)
        
        Returns:
            MotionOutput with hand pose and predicted trajectory
        """
        if frame is None or self.hands is None:
            return None
        
        current_time = time.time()
        
        # Track hands at specified FPS
        if current_time - self.last_capture_time >= self.capture_interval:
            hand_result = self._track_hands(frame, current_time)
            if hand_result:
                self.hand_history.append(hand_result)
            self.last_capture_time = current_time
        
        # Predict trajectory at prediction interval
        if current_time - self.last_prediction_time >= self.prediction_interval:
            self.last_prediction_time = current_time
            
            # Get current hand pose
            hand_pose = self._get_current_pose()
            
            # Predict future trajectory
            predicted_motion = self._predict_trajectory()
            
            self.last_hand_pose = hand_pose
            self.last_predicted_motion = predicted_motion
            
            # Build predictions list
            predictions = [predicted_motion] if predicted_motion else []
            
            return MotionOutput(
                timestamp=datetime.now(),
                predictions=predictions,
                collision_risk=0.0
            )
        
        # Return last result if available
        if self.last_predicted_motion:
            return MotionOutput(
                timestamp=datetime.now(),
                predictions=[self.last_predicted_motion],
                collision_risk=0.0
            )
        
        return None
    
    def _track_hands(self, frame: np.ndarray, timestamp: float) -> Optional[HandTrackingResult]:
        """Track hands in frame using MediaPipe and recognize gestures.
        
        Args:
            frame: Current video frame (BGR)
            timestamp: Current timestamp
        
        Returns:
            HandTrackingResult or None if no hands detected
        """
        if self.hands is None:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from numpy array
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands using HandLandmarker
        detection_result = self.hands.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return None
        
        # Get first hand's landmarks and handedness
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Extract landmarks (normalized coordinates)
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        
        # Get wrist position
        wrist = landmarks[self.WRIST]
        
        # Get handedness label and confidence
        hand_label = "right"
        hand_confidence = 0.9
        if detection_result.handedness and len(detection_result.handedness) > 0:
            handedness_list = detection_result.handedness[0]
            if handedness_list and len(handedness_list) > 0:
                category = handedness_list[0]
                hand_label = category.category_name.lower() if hasattr(category, 'category_name') else category.label.lower()
                hand_confidence = category.score
        
        # Recognize gesture if GestureRecognizer is available
        gesture = None
        if self.gesture_recognizer:
            try:
                gesture_result = self.gesture_recognizer.recognize(mp_image)
                if gesture_result.gestures and len(gesture_result.gestures) > 0:
                    top_gesture = gesture_result.gestures[0][0]
                    gesture = top_gesture.category_name
            except Exception as e:
                logger.debug(f"Gesture recognition error: {e}")
        
        return HandTrackingResult(
            timestamp=timestamp,
            landmarks=landmarks,
            handedness=hand_label,
            confidence=hand_confidence,
            wrist_position=(wrist[0], wrist[1]),
            gesture=gesture
        )
    
    def _get_current_pose(self) -> Optional[HumanPose]:
        """Get current hand pose from latest tracking."""
        if not self.hand_history:
            return None
        
        latest = self.hand_history[-1]
        
        # Convert to JointPosition dict (HumanPose.joints is Dict[str, JointPosition])
        joint_names = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
        
        joints: Dict[str, JointPosition] = {}
        for idx, (x, y, z) in enumerate(latest.landmarks):
            name = f"hand_{latest.handedness.lower()}_{joint_names[idx]}"
            joints[name] = JointPosition(
                name=name,
                x=x,
                y=y,
                z=z,
                confidence=latest.confidence
            )
        
        return HumanPose(
            joints=joints,
            timestamp=datetime.now()
        )
    
    def _predict_trajectory(self) -> Optional[PredictedMotion]:
        """Predict future hand trajectory based on motion history.
        
        Uses linear extrapolation with velocity estimation.
        """
        if len(self.hand_history) < 3:
            return None
        
        # Get recent wrist positions and timestamps
        recent = list(self.hand_history)[-10:]  # Last 10 samples
        positions = [(h.wrist_position[0], h.wrist_position[1], h.timestamp) for h in recent]
        
        # Calculate velocity (simple linear regression)
        n = len(positions)
        if n < 2:
            return None
        
        # Calculate average velocity
        total_dx = 0
        total_dy = 0
        total_dt = 0
        
        for i in range(1, n):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = positions[i][2] - positions[i-1][2]
            
            if dt > 0:
                total_dx += dx
                total_dy += dy
                total_dt += dt
        
        if total_dt == 0:
            return None
        
        vx = total_dx / total_dt
        vy = total_dy / total_dt
        
        # Current position
        current_x, current_y = positions[-1][0], positions[-1][1]
        
        # Generate prediction points
        trajectory_poses = []
        trajectory_timestamps = []
        time_step = self.prediction_horizon / self.num_prediction_points
        
        for i in range(1, self.num_prediction_points + 1):
            t = i * time_step
            
            # Simple linear extrapolation with damping
            damping = 0.8 ** i  # Reduce confidence over time
            pred_x = current_x + vx * t * damping
            pred_y = current_y + vy * t * damping
            
            # Clamp to valid range
            pred_x = max(0.0, min(1.0, pred_x))
            pred_y = max(0.0, min(1.0, pred_y))
            
            pose = Pose3D(
                x=pred_x,
                y=pred_y,
                z=0.0,
                qw=1.0, qx=0.0, qy=0.0, qz=0.0
            )
            trajectory_poses.append(pose)
            trajectory_timestamps.append(t)
        
        # Calculate confidence based on motion consistency
        speed = math.sqrt(vx**2 + vy**2)
        confidence = min(1.0, speed * 2)  # Higher speed = more confident prediction
        
        # Create trajectory object
        trajectory = Trajectory(
            poses=trajectory_poses,
            timestamps=trajectory_timestamps
        )
        
        return PredictedMotion(
            entity_id="human_hand",
            predicted_trajectory=trajectory,
            confidence=confidence,
            prediction_horizon_sec=self.prediction_horizon
        )
    
    def _infer_motion_intent(self) -> Intent:
        """Infer basic intent from motion patterns."""
        if len(self.hand_history) < 3:
            return Intent(
                type=IntentType.IDLE,
                confidence=0.5,
                reasoning="Waiting for motion data"
            )
        
        # Calculate recent motion
        recent = list(self.hand_history)[-5:]
        
        # Calculate displacement
        start = recent[0].wrist_position
        end = recent[-1].wrist_position
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate time span
        dt = recent[-1].timestamp - recent[0].timestamp
        speed = distance / dt if dt > 0 else 0
        
        # Determine intent based on motion
        if speed < 0.01:
            return Intent(
                type=IntentType.IDLE,
                confidence=0.8,
                reasoning="Hand stationary"
            )
        elif speed > 0.1:
            # Fast movement
            if dy < -0.05:  # Moving up
                return Intent(
                    type=IntentType.REACHING,
                    confidence=0.7,
                    reasoning="Reaching upward"
                )
            elif dy > 0.05:  # Moving down
                return Intent(
                    type=IntentType.PLACING,
                    confidence=0.6,
                    reasoning="Lowering hand"
                )
            else:
                return Intent(
                    type=IntentType.MOVING,
                    confidence=0.6,
                    reasoning="Hand in motion"
                )
        else:
            # Slow movement - could be gesturing or fine manipulation
            return Intent(
                type=IntentType.GESTURING,
                confidence=0.5,
                reasoning="Slow hand movement"
            )
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current hand velocity (normalized units/second)."""
        if len(self.hand_history) < 2:
            return (0.0, 0.0)
        
        recent = list(self.hand_history)[-5:]
        if len(recent) < 2:
            return (0.0, 0.0)
        
        start = recent[0]
        end = recent[-1]
        dt = end.timestamp - start.timestamp
        
        if dt == 0:
            return (0.0, 0.0)
        
        vx = (end.wrist_position[0] - start.wrist_position[0]) / dt
        vy = (end.wrist_position[1] - start.wrist_position[1]) / dt
        
        return (vx, vy)
    
    def get_hand_position(self) -> Optional[Tuple[float, float]]:
        """Get current normalized hand (wrist) position."""
        if not self.hand_history:
            return None
        return self.hand_history[-1].wrist_position
    
    def cleanup(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()


def visualize_motion_prediction(
    frame: np.ndarray,
    output: MotionOutput
) -> np.ndarray:
    """Visualize motion prediction on frame.
    
    Args:
        frame: Input frame (BGR)
        output: MotionOutput from MotionPredictor
    
    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Draw prediction info
    if output.predictions:
        prediction = output.predictions[0]
        
        info_text = f"Motion Tracking (confidence: {prediction.confidence:.2f})"
        cv2.putText(vis_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw predicted trajectory
        trajectory = prediction.predicted_trajectory
        if trajectory and trajectory.poses:
            points = []
            for i, pose in enumerate(trajectory.poses):
                x = int(pose.x * w)
                y = int(pose.y * h)
                points.append((x, y))
                
                # Draw prediction point with fading color
                alpha = 1.0 - (i / len(trajectory.poses)) * 0.7
                color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
                cv2.circle(vis_frame, (x, y), 4, color, -1)
            
            # Draw trajectory line
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(vis_frame, points[i], points[i + 1], (200, 100, 200), 2)
            
            # Draw arrow showing direction
            if len(points) >= 2:
                cv2.arrowedLine(vis_frame, points[0], points[-1], (255, 0, 255), 2, tipLength=0.3)
        
        # Add time horizon
        horizon_text = f"Prediction: {prediction.prediction_horizon_sec:.1f}s ahead"
        cv2.putText(vis_frame, horizon_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(vis_frame, "No motion prediction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    return vis_frame


def draw_hand_landmarks(
    frame: np.ndarray,
    hand_pose: HumanPose,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """Draw full hand skeleton on frame.
    
    Args:
        frame: Input frame (BGR)
        hand_pose: Hand pose with joint positions
        color: Color for landmarks and connections
    
    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Define hand connections (MediaPipe format)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Get joint positions as dict
    joints_by_idx = {}
    for i, joint in enumerate(hand_pose.joints):
        x = int(joint.position.x * w)
        y = int(joint.position.y * h)
        joints_by_idx[i] = (x, y)
    
    # Draw connections
    for start, end in connections:
        if start in joints_by_idx and end in joints_by_idx:
            cv2.line(vis_frame, joints_by_idx[start], joints_by_idx[end], color, 2)
    
    # Draw joints
    for idx, (x, y) in joints_by_idx.items():
        cv2.circle(vis_frame, (x, y), 4, color, -1)
    
    return vis_frame

"""Gesture recognition monitor for human gesture detection and safety.

Uses MediaPipe Gesture Recognition to detect hand gestures and can be used for:
- Safety control (stop/resume signals)
- Intent recognition (pointing, grabbing, waving)
- Human-robot interaction commands

Recognized gestures include:
- Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
"""

import os
import time
import asyncio
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from collections import deque

import cv2
import numpy as np

# MediaPipe imports
MEDIAPIPE_AVAILABLE = False
LANDMARK_PB2_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    
    # This import may not exist in newer mediapipe versions
    try:
        from mediapipe.framework.formats import landmark_pb2
        LANDMARK_PB2_AVAILABLE = True
    except ImportError:
        # landmark_pb2 not available, drawing will be limited
        landmark_pb2 = None
        
except ImportError:
    mp = None
    python = None
    vision = None
    landmark_pb2 = None

from aura.core import (
    MonitorType, MonitorOutput, Intent, IntentType
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig


logger = logging.getLogger(__name__)


@dataclass
class GestureRecognitionResult:
    """Result from gesture recognition."""
    gesture_name: str  # Name of recognized gesture
    confidence: float  # Confidence score (0-1)
    handedness: str  # "Left" or "Right"
    hand_landmarks: Optional[List[Any]] = None  # MediaPipe landmarks
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GestureOutput(MonitorOutput):
    """Output from gesture monitor."""
    monitor_type: MonitorType = field(default=MonitorType.INTENT)
    gestures: List[GestureRecognitionResult] = field(default_factory=list)
    dominant_gesture: Optional[str] = None  # Most confident gesture
    safety_triggered: bool = False  # Safety stop signal
    intent: Optional[Intent] = None  # Interpreted intent from gesture


@dataclass
class GestureMonitorConfig(MonitorConfig):
    """Configuration for gesture monitor."""
    model_path: str = ""  # Path to gesture_recognizer.task
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_hands: int = 2
    gesture_hold_frames: int = 3  # Frames to hold gesture before triggering
    
    # Safety gesture sets
    stop_gestures: Set[str] = field(default_factory=lambda: {'Open_Palm', 'Pointing_Up'})
    resume_gestures: Set[str] = field(default_factory=lambda: {'Thumb_Up', 'Victory'})
    
    # Intent mapping
    enable_intent_mapping: bool = True
    

class GestureMonitor(BaseMonitor):
    """Real-time gesture recognition for human-robot interaction.
    
    Detects hand gestures using MediaPipe and can map them to:
    - Safety controls (stop/resume)
    - Intent types (reaching, pointing, etc.)
    - Robot commands
    
    Example usage:
        config = GestureMonitorConfig(
            stop_gestures={'Open_Palm', 'Pointing_Up'},
            resume_gestures={'Thumb_Up'}
        )
        monitor = GestureMonitor(config)
        output = await monitor.update(frame=camera_frame)
        if output.safety_triggered:
            # Stop robot
            pass
    """
    
    # Available gestures from MediaPipe model
    KNOWN_GESTURES = {
        'Closed_Fist', 'Open_Palm', 'Pointing_Up', 
        'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou'
    }
    
    # Default gesture-to-intent mapping
    GESTURE_TO_INTENT = {
        'Pointing_Up': IntentType.GESTURING,
        'Open_Palm': IntentType.GESTURING,
        'Thumb_Up': IntentType.GESTURING,
        'Victory': IntentType.GESTURING,
        'Closed_Fist': IntentType.GRASPING,
        'ILoveYou': IntentType.GESTURING,
    }
    
    def __init__(self, config: Optional[GestureMonitorConfig] = None):
        """Initialize gesture monitor.
        
        Args:
            config: Configuration for gesture recognition
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is required for gesture recognition. "
                "Install with: uv add mediapipe"
            )
        
        super().__init__(config or GestureMonitorConfig())
        self.gesture_config: GestureMonitorConfig = self.config
        
        # Gesture tracking
        self.gesture_counter: Dict[str, int] = {}  # Track consecutive frames
        self.gesture_history: deque = deque(maxlen=30)  # Recent gestures
        self.safety_triggered = False
        self.last_gesture = "None"
        
        # MediaPipe components
        self.recognizer: Optional[vision.GestureRecognizer] = None
        
        # Try to get solutions (may not exist in newer mediapipe versions)
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except AttributeError:
            logger.warning("MediaPipe solutions not available, using Tasks API only")
            self.mp_hands = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
        
        # Initialize recognizer
        self._init_recognizer()
        
        logger.info(f"GestureMonitor initialized with {self.gesture_config.num_hands} hands")
        logger.info(f"Stop gestures: {self.gesture_config.stop_gestures}")
        logger.info(f"Resume gestures: {self.gesture_config.resume_gestures}")
    
    @property
    def monitor_type(self) -> MonitorType:
        """Return monitor type."""
        return MonitorType.INTENT
    
    def _init_recognizer(self):
        """Initialize MediaPipe Gesture Recognizer."""
        model_path = self.gesture_config.model_path
        
        # Find or download the model
        if not model_path:
            model_path = self._find_or_download_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Gesture recognizer model not found at {model_path}. "
                f"Downloading..."
            )
        
        logger.info(f"Loading gesture model from: {model_path}")
        
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=self.gesture_config.num_hands,
                min_hand_detection_confidence=self.gesture_config.min_detection_confidence,
                min_tracking_confidence=self.gesture_config.min_tracking_confidence,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            logger.info("MediaPipe Gesture Recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize gesture recognizer: {e}")
            raise
    
    def _find_or_download_model(self) -> str:
        """Find or download the gesture recognizer model."""
        # Check common locations
        possible_paths = [
            os.path.expanduser('~/.mediapipe/gesture_recognizer.task'),
            os.path.expanduser('~/.cache/mediapipe/gesture_recognizer.task'),
            os.path.join(os.path.dirname(__file__), '../../models/gesture_recognizer.task'),
            '/tmp/gesture_recognizer.task',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Download the model
        return self._download_model()
    
    def _download_model(self) -> str:
        """Download the gesture recognizer model."""
        model_url = (
            'https://storage.googleapis.com/mediapipe-models/'
            'gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
        )
        model_dir = os.path.expanduser('~/.cache/mediapipe')
        model_path = os.path.join(model_dir, 'gesture_recognizer.task')
        
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Downloading gesture recognizer model to {model_path}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download gesture recognizer model: {e}")
        
        return model_path
    
    async def _process(self, frame: np.ndarray, **inputs) -> GestureOutput:
        """Process frame and detect gestures.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            **inputs: Additional inputs (ignored)
            
        Returns:
            GestureOutput with detected gestures and safety status
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return GestureOutput(
                is_valid=False,
                error="Invalid frame input"
            )
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
        
        # Run gesture recognition
        try:
            result = self.recognizer.recognize(mp_image)
        except Exception as e:
            logger.warning(f"Gesture recognition error: {e}")
            return GestureOutput(
                is_valid=False,
                error=str(e)
            )
        
        # Extract gesture results
        gestures = []
        if result.gestures and len(result.gestures) > 0:
            for hand_idx, hand_gestures in enumerate(result.gestures):
                if not hand_gestures:
                    continue
                
                # Get top gesture for this hand
                top_gesture = hand_gestures[0]
                
                # Get handedness
                handedness = "Unknown"
                if result.handedness and len(result.handedness) > hand_idx:
                    handedness = result.handedness[hand_idx][0].category_name
                
                # Get landmarks
                landmarks = None
                if result.hand_landmarks and len(result.hand_landmarks) > hand_idx:
                    landmarks = result.hand_landmarks[hand_idx]
                
                gestures.append(GestureRecognitionResult(
                    gesture_name=top_gesture.category_name,
                    confidence=top_gesture.score,
                    handedness=handedness,
                    hand_landmarks=landmarks,
                ))
        
        # Update gesture tracking
        detected_gesture = "None"
        gesture_score = 0.0
        
        if gestures:
            # Get most confident gesture
            best_gesture = max(gestures, key=lambda g: g.confidence)
            detected_gesture = best_gesture.gesture_name
            gesture_score = best_gesture.confidence
        
        # Update gesture counter for debouncing
        if detected_gesture != "None":
            self.gesture_counter[detected_gesture] = self.gesture_counter.get(detected_gesture, 0) + 1
            # Reset other gesture counters
            for g in list(self.gesture_counter.keys()):
                if g != detected_gesture:
                    self.gesture_counter[g] = 0
        else:
            # Decay all counters
            for g in list(self.gesture_counter.keys()):
                self.gesture_counter[g] = max(0, self.gesture_counter[g] - 1)
        
        # Check for stable gesture (held for enough frames)
        stable_gesture = None
        for gesture, count in self.gesture_counter.items():
            if count >= self.gesture_config.gesture_hold_frames:
                stable_gesture = gesture
                break
        
        # Update safety state based on stable gesture
        prev_state = self.safety_triggered
        
        if stable_gesture in self.gesture_config.stop_gestures:
            self.safety_triggered = True
            if not prev_state:
                logger.warning(f"SAFETY TRIGGERED! Gesture: {stable_gesture}")
        elif stable_gesture in self.gesture_config.resume_gestures:
            self.safety_triggered = False
            if prev_state:
                logger.info(f"SAFETY CLEARED! Gesture: {stable_gesture}")
        
        # Map gesture to intent
        intent = None
        if (self.gesture_config.enable_intent_mapping and 
            stable_gesture and 
            stable_gesture in self.GESTURE_TO_INTENT):
            intent = Intent(
                type=self.GESTURE_TO_INTENT[stable_gesture],
                confidence=gesture_score,
                reasoning=f"Detected gesture: {stable_gesture}"
            )
        
        # Add to history
        self.gesture_history.append({
            'timestamp': datetime.now(),
            'gesture': detected_gesture,
            'confidence': gesture_score,
            'safety_triggered': self.safety_triggered
        })
        
        self.last_gesture = detected_gesture
        
        return GestureOutput(
            gestures=gestures,
            dominant_gesture=stable_gesture,
            safety_triggered=self.safety_triggered,
            intent=intent,
            is_valid=True
        )
    
    def get_visualization_frame(self, frame: np.ndarray, output: GestureOutput) -> np.ndarray:
        """Draw gesture visualization on frame.
        
        Args:
            frame: Input frame (BGR)
            output: GestureOutput from processing
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw hand landmarks
        for gesture_result in output.gestures:
            if gesture_result.hand_landmarks:
                # Convert to proto format for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z
                    ) for landmark in gesture_result.hand_landmarks
                ])
                
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks_proto,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Draw status bar at top
        color = (0, 0, 255) if output.safety_triggered else (0, 255, 0)
        status_text = "STOP" if output.safety_triggered else "SAFE"
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 80), (0, 0, 0), -1)
        
        # Status
        cv2.putText(annotated_frame, f"Status: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Detected gestures
        if output.gestures:
            gestures_text = ", ".join([
                f"{g.gesture_name}({g.confidence:.2f})" 
                for g in output.gestures
            ])
            cv2.putText(annotated_frame, f"Gestures: {gestures_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions at bottom
        h = annotated_frame.shape[0]
        cv2.rectangle(annotated_frame, (0, h-50), (annotated_frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame, 
            f"STOP: {', '.join(self.gesture_config.stop_gestures)} | "
            f"RESUME: {', '.join(self.gesture_config.resume_gestures)}", 
            (10, h-15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        return annotated_frame
    
    def reset_safety(self):
        """Reset safety state (for testing/recovery)."""
        self.safety_triggered = False
        self.gesture_counter.clear()
        logger.info("Safety state reset")
    
    def get_gesture_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent gesture detections.
        
        Returns:
            Dictionary with gesture counts, frequencies, etc.
        """
        if not self.gesture_history:
            return {}
        
        gesture_counts = {}
        for entry in self.gesture_history:
            gesture = entry['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        return {
            'total_detections': len(self.gesture_history),
            'gesture_counts': gesture_counts,
            'current_gesture': self.last_gesture,
            'safety_triggered': self.safety_triggered,
            'recent_gestures': [
                f"{entry['gesture']} ({entry['confidence']:.2f})"
                for entry in list(self.gesture_history)[-5:]
            ]
        }

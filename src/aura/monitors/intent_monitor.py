"""Intent Monitor for human action recognition and prediction.

Uses Gemini to analyze video frames and predict current/future actions
based on a configurable task graph (DAG) and state schema.

Inspired by the realtime_test implementation from hcdt project.
"""

import os
import json
import time
import asyncio
import logging
import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from io import BytesIO
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from aura.core import (
    MonitorType, IntentOutput, Intent, IntentType
)
from aura.monitors.base_monitor import BaseMonitor
from aura.utils.config import IntentMonitorConfig


logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, max_dimension: int = 640) -> str:
    """Convert PIL Image to base64 string, resizing if needed."""
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_json_file(file_path: str) -> Optional[Any]:
    """Load JSON file from path."""
    if not os.path.exists(file_path):
        logger.warning(f"JSON file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


@dataclass
class IntentPrediction:
    """Result of intent prediction."""
    current_action: str
    current_action_confidence: float
    predicted_next_action: str
    predicted_next_confidence: float
    reasoning: str
    task_state: Dict[str, Any]
    raw_response: Optional[str] = None


class IntentMonitor(BaseMonitor):
    """Real-time intent monitoring using Gemini and configurable task graphs.
    
    Captures webcam frames over a time window, then queries Gemini to:
    - Identify current human action from task graph
    - Track state variables
    - Predict next likely action
    
    Task graphs and state schemas are loaded from JSON files, allowing
    customization for different tasks/scenarios.
    """
    
    def __init__(self, config: Optional[IntentMonitorConfig] = None):
        """Initialize intent monitor.
        
        Args:
            config: Configuration for intent monitoring
        """
        super().__init__(config)
        
        self.fps = config.fps if config else 2.0
        self.window_duration = config.capture_duration if config else 2.0
        self.max_frames = int(self.fps * self.window_duration)
        self.max_image_dimension = config.max_image_dimension if config else 640
        
        # Frame buffer: store recent frames with timestamps
        self.frame_buffer: deque = deque(maxlen=self.max_frames)
        self.last_capture_time = 0
        self.capture_interval = 1.0 / self.fps
        
        # Prediction state
        self.last_prediction_time = 0
        self.prediction_interval = config.prediction_interval if config else 2.0
        
        # Load task graph and state schema
        self.task_graph = self._load_task_graph(config)
        self.state_schema = self._load_state_schema(config)
        self.task_state: Dict[str, Any] = {}
        
        # Prompt configuration
        self.task_name = config.task_name if config and config.task_name else "General Activity"
        self.system_prompt = config.system_prompt if config and config.system_prompt else self._default_system_prompt()
        self.task_context = config.task_context if config and config.task_context else self._default_task_context()
        self.analysis_instructions = config.analysis_instructions if config and config.analysis_instructions else ""
        self.output_format = config.output_format if config and config.output_format else ""
        
        # Gemini client
        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(
                    http_options={"api_version": "v1beta"},
                    api_key=api_key
                )
                self.model = config.model if config else "gemini-2.0-flash-exp"
            else:
                logger.warning("GEMINI_API_KEY not set, intent recognition disabled")
                self.client = None
        else:
            logger.warning("google-genai not installed, intent recognition disabled")
            self.client = None
        
        # Last prediction result
        self.last_prediction: Optional[IntentPrediction] = None
    
    def _load_task_graph(self, config: Optional[IntentMonitorConfig]) -> List[Dict]:
        """Load task graph from config or use default."""
        if config and config.dag_file:
            dag = load_json_file(config.dag_file)
            if dag:
                return dag
        return self._default_task_graph()
    
    def _load_state_schema(self, config: Optional[IntentMonitorConfig]) -> List[Dict]:
        """Load state schema from config or use default."""
        if config and config.state_file:
            schema = load_json_file(config.state_file)
            if schema:
                return schema
        return self._default_state_schema()
    
    def _default_task_graph(self) -> List[Dict]:
        """Default task graph for general activity recognition."""
        return [
            {"id": 1, "action": "Idle", "description": "Not actively doing anything", "dependencies": []},
            {"id": 2, "action": "Reaching", "description": "Extending hand/arm toward something", "dependencies": []},
            {"id": 3, "action": "Grasping", "description": "Closing hand to grab an object", "dependencies": [2]},
            {"id": 4, "action": "Moving object", "description": "Moving an object from one place to another", "dependencies": [3]},
            {"id": 5, "action": "Placing", "description": "Putting down or releasing an object", "dependencies": [4]},
            {"id": 6, "action": "Gesturing", "description": "Making communicative hand gestures", "dependencies": []},
            {"id": 7, "action": "Pointing", "description": "Pointing at something", "dependencies": []},
            {"id": 8, "action": "Waving", "description": "Waving hand", "dependencies": []},
            {"id": 9, "action": "Typing", "description": "Typing on keyboard", "dependencies": []},
            {"id": 10, "action": "Using tool", "description": "Using a tool or instrument", "dependencies": [3]},
        ]
    
    def _default_state_schema(self) -> List[Dict]:
        """Default state schema for general activity tracking."""
        return [
            {"name": "is_hand_visible", "type": "Boolean", "description": "True if at least one hand is visible in frame"},
            {"name": "holding_object", "type": "String", "description": "Description of object being held, or null"},
            {"name": "gaze_target", "type": "String", "description": "What the person appears to be looking at"},
            {"name": "body_posture", "type": "String", "description": "Current posture: sitting, standing, leaning"},
            {"name": "hand_position", "type": "String", "description": "General hand position: raised, lowered, extended"},
        ]
    
    def _default_system_prompt(self) -> str:
        """Default system prompt."""
        return """You are an AI assistant specialized in real-time human action recognition and prediction from video frames.
You analyze video frames to identify what a person is currently doing and predict their next action."""
    
    def _default_task_context(self) -> str:
        """Default task context."""
        return """You are analyzing a sequence of video frames showing a person performing various activities.
Identify their current action and predict what they will do next based on the task graph."""
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.INTENT
    
    async def _process(self, frame: Optional[np.ndarray] = None) -> Optional[IntentOutput]:
        """Process frame and predict intent.
        
        Args:
            frame: Current video frame (BGR)
        
        Returns:
            IntentOutput with current and predicted intent, or None if not ready
        """
        if frame is None:
            return None
        
        current_time = time.time()
        
        # Add frame to buffer at specified FPS
        if current_time - self.last_capture_time >= self.capture_interval:
            self.frame_buffer.append((current_time, frame.copy()))
            self.last_capture_time = current_time
        
        # Run prediction if enough frames and interval elapsed
        if len(self.frame_buffer) < self.max_frames:
            return None
        
        if current_time - self.last_prediction_time < self.prediction_interval:
            # Return last prediction if available
            if self.last_prediction:
                return self._prediction_to_output(self.last_prediction)
            return None
        
        self.last_prediction_time = current_time
        
        # Predict intent using Gemini
        if self.client:
            prediction = await self._predict_with_gemini()
            if prediction:
                self.last_prediction = prediction
                self.task_state = prediction.task_state
                return self._prediction_to_output(prediction)
        
        # Fallback if no client
        return IntentOutput(
            timestamp=datetime.now(),
            intent=Intent(
                type=IntentType.UNKNOWN,
                confidence=0.0,
                reasoning="No prediction available"
            ),
            alternatives=[]
        )
    
    def _prediction_to_output(self, prediction: IntentPrediction) -> IntentOutput:
        """Convert IntentPrediction to IntentOutput."""
        # Map action string to IntentType
        current_type = self._action_to_intent_type(prediction.current_action)
        predicted_type = self._action_to_intent_type(prediction.predicted_next_action)
        
        current_intent = Intent(
            type=current_type,
            confidence=prediction.current_action_confidence,
            reasoning=f"{prediction.current_action}: {prediction.reasoning}",
            target_object=prediction.task_state.get("holding_object")
        )
        
        predicted_intent = Intent(
            type=predicted_type,
            confidence=prediction.predicted_next_confidence,
            reasoning=prediction.predicted_next_action
        )
        
        return IntentOutput(
            timestamp=datetime.now(),
            intent=current_intent,
            alternatives=[predicted_intent] if predicted_intent.confidence > 0 else []
        )
    
    def _action_to_intent_type(self, action: str) -> IntentType:
        """Map action string to IntentType enum."""
        action_lower = action.lower()
        
        if "idle" in action_lower or "waiting" in action_lower:
            return IntentType.IDLE
        elif "reach" in action_lower or "extend" in action_lower:
            return IntentType.REACHING
        elif "grasp" in action_lower or "grab" in action_lower or "pick" in action_lower:
            return IntentType.GRASPING
        elif "mov" in action_lower or "carry" in action_lower:
            return IntentType.MOVING
        elif "plac" in action_lower or "put" in action_lower or "release" in action_lower:
            return IntentType.PLACING
        elif "gestur" in action_lower or "point" in action_lower or "wave" in action_lower:
            return IntentType.GESTURING
        elif "speak" in action_lower or "talk" in action_lower:
            return IntentType.SPEAKING
        else:
            return IntentType.UNKNOWN
    
    async def _predict_with_gemini(self) -> Optional[IntentPrediction]:
        """Predict intent using Gemini."""
        if not self.client or len(self.frame_buffer) < 2:
            return None
        
        try:
            # Prepare frames for Gemini
            frames_for_gemini = []
            for timestamp, frame in list(self.frame_buffer):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize to reduce token usage
                if max(pil_image.size) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(pil_image.size)
                    new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                frames_for_gemini.append(pil_image)
            
            # Build prompt
            prompt = self._build_prompt()
            
            # Query Gemini
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=[prompt] + frames_for_gemini,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    response_mime_type="application/json",
                    response_schema=self._get_response_schema()
                )
            )
            
            # Parse response
            result = json.loads(response.text)
            
            return IntentPrediction(
                current_action=result.get("current_action", "Unknown"),
                current_action_confidence=result.get("current_action_confidence", 0.5),
                predicted_next_action=result.get("predicted_next_action", "Unknown"),
                predicted_next_confidence=result.get("predicted_next_confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                task_state=result.get("state", {}),
                raw_response=response.text
            )
            
        except Exception as e:
            logger.error(f"Error predicting intent with Gemini: {e}")
            return None
    
    def _get_response_schema(self) -> Dict:
        """Get JSON schema for Gemini response."""
        # Build state properties from schema
        state_properties = {}
        for var in self.state_schema:
            var_type = var.get("type", "String").lower()
            if var_type == "boolean":
                state_properties[var["name"]] = {"type": "boolean"}
            elif var_type == "list":
                state_properties[var["name"]] = {"type": "array", "items": {"type": "string"}}
            else:
                state_properties[var["name"]] = {"type": "string"}
        
        return {
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "properties": state_properties
                },
                "current_action": {"type": "string"},
                "current_action_confidence": {"type": "number"},
                "predicted_next_action": {"type": "string"},
                "predicted_next_confidence": {"type": "number"},
                "reasoning": {"type": "string"}
            }
        }
    
    def _build_prompt(self) -> str:
        """Build the full prompt for Gemini."""
        task_graph_str = json.dumps(self.task_graph, indent=2)
        state_schema_str = json.dumps(self.state_schema, indent=2)
        
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(self.system_prompt)
        prompt_parts.append("")
        
        # Task context
        prompt_parts.append(f"## Task: {self.task_name}")
        prompt_parts.append(self.task_context)
        prompt_parts.append("")
        
        # Task graph
        prompt_parts.append("## Task Graph (Available Actions)")
        prompt_parts.append("These are the possible actions the person can perform:")
        prompt_parts.append(f"```json\n{task_graph_str}\n```")
        prompt_parts.append("")
        
        # State schema
        prompt_parts.append("## State Variables to Track")
        prompt_parts.append(f"```json\n{state_schema_str}\n```")
        prompt_parts.append("")
        
        # Analysis instructions
        if self.analysis_instructions:
            prompt_parts.append("## Analysis Instructions")
            prompt_parts.append(self.analysis_instructions)
        else:
            prompt_parts.append("## Analysis Instructions")
            prompt_parts.append(f"""Analyze the provided sequence of {len(self.frame_buffer)} frames and:

1. **Update STATE variables**: Based on visual observations, update each state variable.
2. **Identify CURRENT action**: What is the person doing right now? Use actions from the Task Graph.
3. **Predict NEXT action**: Based on motion patterns and task dependencies, what will they do next?
4. **Provide confidence scores**: Rate your confidence (0.0 to 1.0) for both predictions.
5. **Explain your reasoning**: Briefly describe what visual cues led to your predictions.""")
        prompt_parts.append("")
        
        # Output format
        if self.output_format:
            prompt_parts.append("## Output Format")
            prompt_parts.append(self.output_format)
        else:
            prompt_parts.append("## Output Format")
            prompt_parts.append("Respond with a JSON object matching the schema provided.")
        
        return "\n".join(prompt_parts)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current task state."""
        return self.task_state.copy()
    
    def get_task_graph(self) -> List[Dict]:
        """Get the current task graph."""
        return self.task_graph.copy()


def visualize_intent(
    frame: np.ndarray,
    prediction: IntentPrediction
) -> np.ndarray:
    """Visualize intent prediction on frame.
    
    Args:
        frame: Input frame (BGR)
        prediction: Intent prediction result
    
    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()
    
    # Draw current action
    current_text = f"Current: {prediction.current_action} ({prediction.current_action_confidence:.2f})"
    cv2.putText(vis_frame, current_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw predicted next action
    predicted_text = f"Next: {prediction.predicted_next_action} ({prediction.predicted_next_confidence:.2f})"
    cv2.putText(vis_frame, predicted_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw reasoning (truncated)
    if prediction.reasoning:
        reasoning = prediction.reasoning[:80] + "..." if len(prediction.reasoning) > 80 else prediction.reasoning
        cv2.putText(vis_frame, reasoning, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw task state if available
    if prediction.task_state:
        y_pos = 120
        for key, value in list(prediction.task_state.items())[:3]:
            state_text = f"  {key}: {value}"
            cv2.putText(vis_frame, state_text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
    
    return vis_frame

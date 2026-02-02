"""Base Performance Monitor for AURA framework.

Provides the abstract base class for performance monitoring - tracking
whether the robot is correctly executing assigned tasks and detecting failures.

Uses Gemini to analyze video frames and determine task execution status.

Subclasses should implement task-specific failure detection prompts.
"""

import os
import json
import time
import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from io import BytesIO
from collections import deque
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image

# Try to import Gemini
genai = None
types = None
GEMINI_AVAILABLE = False
try:
    from google import genai as _genai
    from google.genai import types as _types
    genai = _genai
    types = _types
    GEMINI_AVAILABLE = True
except ImportError:
    pass

from aura.core import (
    MonitorType,
    PerformanceOutput,
    ActionStatus,
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig


logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of task execution failures."""
    NONE = auto()                 # No failure detected
    DROPPED_OBJECT = auto()       # Robot dropped an object
    FAILED_GRASP = auto()         # Failed to grasp the object
    COLLISION = auto()            # Collision detected
    WRONG_TRAJECTORY = auto()     # Moving in wrong direction
    ROBOT_STUCK = auto()          # Robot not moving when should be
    HUMAN_INTERVENTION = auto()   # Human is blocking or intervening
    UNEXPECTED_STATE = auto()     # Scene doesn't match expected state
    TIMEOUT = auto()              # Action taking too long
    UNKNOWN = auto()              # Unknown failure


class PerformanceStatus(Enum):
    """Overall performance status."""
    OK = auto()                   # Everything proceeding normally
    WARNING = auto()              # Potential issue, continue monitoring
    ERROR = auto()                # Clear failure detected
    CRITICAL = auto()             # Stop immediately


@dataclass
class PerformanceMonitorConfig(MonitorConfig):
    """Configuration for performance monitor."""
    fps: float = 2.0                      # Frame capture rate
    window_duration: float = 2.0          # How many seconds of frames to analyze
    check_interval: float = 2.0           # How often to check performance
    max_image_dimension: int = 640        # Max image size for Gemini
    model: str = "gemini-2.0-flash"       # Gemini model for analysis
    timeout_sec: float = 30.0             # API timeout
    confidence_threshold: float = 0.7     # Threshold to report failure
    enable_abort_on_critical: bool = True # Auto-abort on critical failures


@dataclass
class PerformanceCheckResult:
    """Result of a performance check."""
    timestamp: datetime
    current_instruction: str
    status: PerformanceStatus
    failure_type: FailureType
    confidence: float
    reasoning: str
    raw_response: Optional[str] = None
    frame_count: int = 0
    check_duration_sec: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_ok(self) -> bool:
        return self.status == PerformanceStatus.OK
    
    @property
    def should_abort(self) -> bool:
        return self.status == PerformanceStatus.CRITICAL


class PerformanceMonitor(BaseMonitor):
    """Base class for performance monitoring.
    
    Uses Gemini to analyze video frames and determine if the robot
    is correctly executing the assigned task. Detects failures and
    provides reasoning for detected issues.
    
    Subclasses should override:
    - _get_system_prompt(): Task-specific system prompt
    - _get_failure_detection_prompt(): Task-specific detection prompt
    - _get_failure_types(): Task-specific failure types to detect
    
    Usage:
        monitor = MyTaskPerformanceMonitor(config)
        
        # Set current instruction
        monitor.set_current_instruction("Pick object from table")
        
        # Add frames
        monitor.add_frame(frame)
        
        # Check performance
        result = await monitor.check_performance()
        if not result.is_ok:
            print(f"Failure: {result.failure_type.name}")
    """
    
    # Default prompts - override in subclass
    DEFAULT_SYSTEM_PROMPT = """You are an AI safety monitor for a collaborative robot task.
You analyze video frames to determine if the robot is correctly executing its assigned task.
Your job is to detect failures and safety issues.

Be conservative - if unsure, report WARNING rather than ERROR.
Only report CRITICAL for clear dangerous situations."""

    DEFAULT_FAILURE_PROMPT = """Analyze these video frames to check if the robot is correctly performing the task.

CURRENT INSTRUCTION: {instruction}

Look for these issues:
1. DROPPED_OBJECT: Object falling from gripper
2. FAILED_GRASP: Gripper closed but object not in gripper
3. COLLISION: Robot contacting objects unexpectedly
4. WRONG_TRAJECTORY: Robot moving in wrong direction
5. ROBOT_STUCK: No motion when robot should be moving
6. HUMAN_INTERVENTION: Human blocking robot path
7. UNEXPECTED_STATE: Scene doesn't match expected state

Consider the frames in sequence - they show approximately {window_sec} seconds of motion.

Respond in JSON format:
{{
    "status": "OK" | "WARNING" | "ERROR" | "CRITICAL",
    "failure_type": "NONE" | "DROPPED_OBJECT" | "FAILED_GRASP" | "COLLISION" | "WRONG_TRAJECTORY" | "ROBOT_STUCK" | "HUMAN_INTERVENTION" | "UNEXPECTED_STATE" | "UNKNOWN",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of what you observed",
    "details": {{}}
}}"""

    def __init__(self, config: Optional[PerformanceMonitorConfig] = None):
        """Initialize performance monitor.
        
        Args:
            config: Monitor configuration
        """
        self.perf_config = config or PerformanceMonitorConfig()
        super().__init__(self.perf_config)
        
        self.fps = self.perf_config.fps
        self.window_duration = self.perf_config.window_duration
        self.max_frames = int(self.fps * self.window_duration)
        self.max_image_dimension = self.perf_config.max_image_dimension
        self.check_interval = self.perf_config.check_interval
        
        # Frame buffer
        self.frame_buffer: deque = deque(maxlen=self.max_frames)
        self.last_capture_time = 0.0
        self.capture_interval = 1.0 / self.fps
        
        # Check state
        self.last_check_time = 0.0
        self.current_instruction = ""
        self.current_program_id: Optional[str] = None
        
        # Results history
        self.check_history: List[PerformanceCheckResult] = []
        self.last_result: Optional[PerformanceCheckResult] = None
        
        # Gemini client
        self.client = None
        self.model = self.perf_config.model
        if GEMINI_AVAILABLE and genai is not None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(
                    http_options={"api_version": "v1beta"},
                    api_key=api_key
                )
                logger.info(f"PerformanceMonitor using Gemini model: {self.model}")
            else:
                logger.warning("GEMINI_API_KEY not set, performance monitoring disabled")
        else:
            logger.warning("google-genai not installed, performance monitoring disabled")
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERFORMANCE
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Gemini. Override in subclass."""
        return self.DEFAULT_SYSTEM_PROMPT
    
    def _get_failure_detection_prompt(self, instruction: str, window_sec: float) -> str:
        """Get failure detection prompt. Override in subclass."""
        return self.DEFAULT_FAILURE_PROMPT.format(
            instruction=instruction,
            window_sec=f"{window_sec:.1f}"
        )
    
    def set_current_instruction(self, instruction: str, program_id: Optional[str] = None) -> None:
        """Set the current instruction being executed.
        
        Args:
            instruction: Human-readable description of current task
            program_id: Optional robot program ID
        """
        self.current_instruction = instruction
        self.current_program_id = program_id
        logger.debug(f"Current instruction: {instruction}")
    
    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add a frame to the buffer for analysis.
        
        Args:
            frame: Video frame (BGR numpy array)
            timestamp: Optional timestamp (uses current time if not provided)
        """
        current_time = timestamp or time.time()
        
        # Respect capture interval
        if current_time - self.last_capture_time >= self.capture_interval:
            self.frame_buffer.append((current_time, frame.copy()))
            self.last_capture_time = current_time
    
    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.last_capture_time = 0.0
    
    def _prepare_frames_for_gemini(self) -> List[Image.Image]:
        """Prepare buffered frames as PIL Images for Gemini API."""
        pil_images = []
        
        for timestamp, frame in self.frame_buffer:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize if needed
            if max(pil_image.size) > self.max_image_dimension:
                scale = self.max_image_dimension / max(pil_image.size)
                new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            pil_images.append(pil_image)
        
        return pil_images
    
    async def check_performance(self, force: bool = False) -> Optional[PerformanceCheckResult]:
        """Check if the robot is correctly executing the task.
        
        Args:
            force: Force check even if interval hasn't elapsed
            
        Returns:
            PerformanceCheckResult or None if not ready
        """
        current_time = time.time()
        
        # Check interval
        if not force and current_time - self.last_check_time < self.check_interval:
            return self.last_result
        
        # Need enough frames
        if len(self.frame_buffer) < 2:
            logger.debug(f"Not enough frames: {len(self.frame_buffer)}/{self.max_frames}")
            return None
        
        # Need Gemini client
        if not self.client:
            logger.warning("Gemini client not available")
            return PerformanceCheckResult(
                timestamp=datetime.now(),
                current_instruction=self.current_instruction,
                status=PerformanceStatus.WARNING,
                failure_type=FailureType.UNKNOWN,
                confidence=0.0,
                reasoning="Gemini client not available",
                frame_count=len(self.frame_buffer),
            )
        
        # Need an instruction
        if not self.current_instruction:
            logger.warning("No current instruction set")
            return None
        
        self.last_check_time = current_time
        start_time = time.time()
        
        try:
            # Prepare prompt
            window_sec = len(self.frame_buffer) / self.fps
            system_prompt = self._get_system_prompt()
            detection_prompt = self._get_failure_detection_prompt(
                self.current_instruction, window_sec
            )
            
            # Prepare images
            pil_images = self._prepare_frames_for_gemini()
            
            # Build contents: prompt first, then images
            contents = [system_prompt + "\n\n" + detection_prompt] + pil_images
            
            # Build config if types available
            config = None
            if types is not None:
                config = types.GenerateContentConfig(
                    temperature=0.2,  # Low temp for consistent results
                    max_output_tokens=500,
                )
            
            # Call Gemini (PIL Images are accepted as contents)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,  # type: ignore[arg-type]
                config=config
            )
            
            # Parse response
            response_text = response.text if response.text else ""
            result = self._parse_response(response_text, len(pil_images))
            result.check_duration_sec = time.time() - start_time
            
            self.last_result = result
            self.check_history.append(result)
            
            # Log result
            if result.is_ok:
                logger.debug(f"Performance OK: {result.reasoning}")
            else:
                logger.warning(f"Performance {result.status.name}: {result.failure_type.name} - {result.reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            result = PerformanceCheckResult(
                timestamp=datetime.now(),
                current_instruction=self.current_instruction,
                status=PerformanceStatus.WARNING,
                failure_type=FailureType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Error during check: {str(e)}",
                frame_count=len(self.frame_buffer),
                check_duration_sec=time.time() - start_time,
            )
            self.last_result = result
            return result
    
    def _parse_response(self, response_text: str, frame_count: int) -> PerformanceCheckResult:
        """Parse Gemini response into PerformanceCheckResult."""
        try:
            # Extract JSON from response
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            data = json.loads(text)
            
            # Parse status
            status_str = data.get("status", "WARNING").upper()
            status = PerformanceStatus[status_str] if status_str in PerformanceStatus.__members__ else PerformanceStatus.WARNING
            
            # Parse failure type
            failure_str = data.get("failure_type", "NONE").upper()
            failure_type = FailureType[failure_str] if failure_str in FailureType.__members__ else FailureType.UNKNOWN
            
            return PerformanceCheckResult(
                timestamp=datetime.now(),
                current_instruction=self.current_instruction,
                status=status,
                failure_type=failure_type,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                raw_response=response_text,
                frame_count=frame_count,
                details=data.get("details", {}),
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract key info from text
            status = PerformanceStatus.OK if "OK" in response_text.upper() else PerformanceStatus.WARNING
            return PerformanceCheckResult(
                timestamp=datetime.now(),
                current_instruction=self.current_instruction,
                status=status,
                failure_type=FailureType.UNKNOWN,
                confidence=0.5,
                reasoning=response_text[:200],
                raw_response=response_text,
                frame_count=frame_count,
            )
    
    async def _process(self, frame: Optional[np.ndarray] = None, **inputs) -> PerformanceOutput:
        """Process frame and check performance.
        
        Args:
            frame: Current video frame (BGR)
            instruction: Optional instruction to set
            program_id: Optional program ID to set
            
        Returns:
            PerformanceOutput with status
        """
        # Update instruction if provided
        if "instruction" in inputs:
            self.current_instruction = inputs["instruction"]
        if "program_id" in inputs:
            self.current_program_id = inputs["program_id"]
        
        # Add frame to buffer
        if frame is not None:
            self.add_frame(frame)
        
        # Check performance
        result = await self.check_performance()
        
        if result is None:
            return PerformanceOutput(
                monitor_type=MonitorType.PERFORMANCE,
                timestamp=datetime.now(),
                is_valid=False,
                action_status=None,
                progress=0.0,
                should_abort=False,
            )
        
        # Convert result to output
        action_status = ActionStatus.IN_PROGRESS
        if result.status in (PerformanceStatus.ERROR, PerformanceStatus.CRITICAL):
            action_status = ActionStatus.FAILED
        
        return PerformanceOutput(
            monitor_type=MonitorType.PERFORMANCE,
            timestamp=datetime.now(),
            is_valid=True,
            action_status=action_status,
            progress=0.5,  # We don't track progress, just status
            should_abort=result.should_abort,
            error=result.reasoning if not result.is_ok else None,
        )
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of detected failures."""
        failures = [r for r in self.check_history if not r.is_ok]
        return {
            "total_checks": len(self.check_history),
            "failure_count": len(failures),
            "failure_types": {
                ft.name: sum(1 for f in failures if f.failure_type == ft)
                for ft in FailureType
                if sum(1 for f in failures if f.failure_type == ft) > 0
            },
            "last_failure": failures[-1].reasoning if failures else None,
        }
    
    def reset(self) -> None:
        """Reset monitor state."""
        self.frame_buffer.clear()
        self.last_capture_time = 0.0
        self.last_check_time = 0.0
        self.current_instruction = ""
        self.current_program_id = None
        self.check_history = []
        self.last_result = None
        logger.info("Performance monitor reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize monitor state to dictionary."""
        return {
            "current_instruction": self.current_instruction,
            "current_program_id": self.current_program_id,
            "buffer_frames": len(self.frame_buffer),
            "total_checks": len(self.check_history),
            "last_result": {
                "status": self.last_result.status.name,
                "failure_type": self.last_result.failure_type.name,
                "confidence": self.last_result.confidence,
                "reasoning": self.last_result.reasoning,
            } if self.last_result else None,
            "failure_summary": self.get_failure_summary(),
        }

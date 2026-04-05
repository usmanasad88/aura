"""Intent Monitor — RCWPS-based human action recognition and prediction.

Implements a Rolling Context Window with Previous State (RCWPS) approach.
It loads a task-specific DAG, state schema, and task profile to formulate
prompts, then queries Gemini to predict current/future actions and track
state variables.

Can be used in two ways:
1. As a standalone ``AURAIntentMonitor`` (config-dir based, synchronous predict).
2. As a ``BaseMonitor`` subclass via ``IntentMonitor`` (async, frame-buffer based).
"""

import os
import json
import time
import asyncio
import logging
import base64
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque

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


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class IntentResult:
    """Result of a single RCWPS intent prediction."""
    timestamp: float
    frame_num: int

    # State variables returned by Gemini
    state: Dict[str, Any] = field(default_factory=dict)

    # Convenience fields for common AURA state
    current_phase: str = "initialization"
    current_action: str = "idle"
    human_state: str = "idle"

    # Action tracking
    steps_completed: List[str] = field(default_factory=list)
    steps_in_progress: List[str] = field(default_factory=list)
    steps_pending: List[str] = field(default_factory=list)

    # Prediction
    predicted_next_action: str = "unknown"
    prediction_confidence: float = 0.0
    reasoning: str = ""

    # Meta
    raw_response: str = ""
    generation_time_sec: float = 0.0
    prompt_tokens_approx: int = 0


@dataclass
class IntentPrediction:
    """Result of intent prediction (used by BaseMonitor subclass)."""
    current_action: str
    current_action_confidence: float
    predicted_next_action: str
    predicted_next_confidence: float
    reasoning: str
    task_state: Dict[str, Any]
    raw_response: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Logger
# ═══════════════════════════════════════════════════════════════════════════


class PromptLogger:
    """Logs every Gemini prompt/response exchange to disk."""

    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.session_dir = None
            return

        base = Path(log_dir) if log_dir else Path("logs/intent_monitor")
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = base / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.call_counter = 0
        logger.info(f"Prompt logger session: {self.session_dir}")

    def log_call(
        self,
        prompt_text: str,
        response_text: str,
        parsed_response: Optional[Dict[str, Any]],
        frame_images: Optional[List[Image.Image]],
        model: str,
        generation_time: float,
        frame_num: int,
        timestamp: float,
        previous_state: Optional[Dict[str, Any]] = None,
    ):
        if not self.enabled or self.session_dir is None:
            return

        self.call_counter += 1
        call_dir = self.session_dir / f"call_{self.call_counter:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

        if parsed_response is not None:
            with open(call_dir / "response_parsed.json", "w") as f:
                json.dump(parsed_response, f, indent=2, default=str)

        if previous_state is not None:
            with open(call_dir / "previous_state.json", "w") as f:
                json.dump(previous_state, f, indent=2, default=str)

        if frame_images:
            frames_dir = call_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, img in enumerate(frame_images):
                img.save(frames_dir / f"frame_{i}.jpg", "JPEG", quality=80)

        meta = {
            "call_number": self.call_counter,
            "model": model,
            "generation_time_sec": round(generation_time, 3),
            "frame_num": frame_num,
            "timestamp_sec": round(timestamp, 3),
            "num_frames_attached": len(frame_images) if frame_images else 0,
            "response_length_chars": len(response_text),
            "logged_at": datetime.now().isoformat(),
        }
        with open(call_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def get_session_dir(self) -> Optional[Path]:
        return self.session_dir


# ═══════════════════════════════════════════════════════════════════════════
#  AURAIntentMonitor — standalone RCWPS monitor (used by workflow nodes)
# ═══════════════════════════════════════════════════════════════════════════


class AURAIntentMonitor:
    """RCWPS intent monitor driven by a task config directory.

    Loads DAG, state schema, and task profile from ``config_dir``,
    builds structured prompts with previous-state context, and returns
    rich ``IntentResult`` objects.

    Supports multiple LLM backends via ``llm_backend``:

    * ``"gemini"`` (default) — Google Gemini API.
    * ``"sglang"`` / ``"openai"`` / ``"vllm"`` — any OpenAI-compatible
      server (e.g. SGLang serving Qwen 3.5 4B).
    """

    def __init__(
        self,
        config_dir: str,
        model: str = "gemini-3.1-pro-preview",
        max_frames: int = 5,
        max_image_dimension: int = 1024,
        temperature: float = 0.3,
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        realtime: bool = False,
        llm_backend: str = "gemini",
        sglang_base_url: str = "http://localhost:8100/v1",
    ):
        self.realtime = realtime
        self.llm_backend = llm_backend

        if realtime:
            if llm_backend == "gemini" and model == "gemini-3.1-pro-preview":
                model = "gemini-3.1-flash-lite-preview"
            max_frames = min(max_frames, 3)
            max_image_dimension = min(max_image_dimension, 768)

        self.model = model
        self.max_frames = max_frames
        self.max_image_dimension = max_image_dimension
        self.temperature = temperature
        self.config_dir = Path(config_dir)
        self.sglang_base_url = sglang_base_url

        # Load task profile, DAG, and state schema
        dag_path = self.config_dir / "dag.json"
        state_path = self.config_dir / "state_schema.json"
        profile_path = self.config_dir / "task_profile.json"

        self.task_graph_string = dag_path.read_text(encoding="utf-8") if dag_path.exists() else "{}"
        self.state_schema_string = state_path.read_text(encoding="utf-8") if state_path.exists() else "{}"

        self.task_profile = {}
        if profile_path.exists():
            self.task_profile = json.loads(profile_path.read_text(encoding="utf-8"))

        self.system_instruction = self.task_profile.get(
            "system_instruction",
            "You are an AI assistant analyzing video frames of a task.",
        )
        self.state_format_string = json.dumps(self._build_output_format(), indent=2)

        self.previous_state: Optional[Dict[str, Any]] = None
        self.history: List[IntentResult] = []

        # Unified LLM client (supports Gemini, SGLang, vLLM, etc.)
        self._llm_client = None
        try:
            from aura.utils.llm_client import create_llm_client
            self._llm_client = create_llm_client(
                llm_backend,
                model=model,
                base_url=sglang_base_url,
            )
        except Exception as e:
            logger.warning("Failed to create LLM client (%s): %s", llm_backend, e)

        # Legacy Gemini client kept for backward compat with code that
        # accesses self.client directly (e.g. IntentMonitor subclass).
        self.client = None
        if llm_backend == "gemini" and GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)

        self.prompt_logger = PromptLogger(
            log_dir=log_dir,
            enabled=enable_logging,
        )

        logger.info(
            "AURAIntentMonitor — model: %s, backend: %s", model, llm_backend,
        )

    def _build_output_format(self) -> Dict[str, Any]:
        """Construct the expected JSON format for the prompt."""
        fmt = {
            "current_phase": "<phase>",
            "current_action": "<action>",
            "human_state": "<state>",
            "steps_completed": ["<step>", "..."],
            "steps_in_progress": ["<step>", "..."],
            "steps_pending": ["<step>", "..."],
            "predicted_next_action": "<step>",
            "prediction_confidence": 0.0,
            "reasoning": "<one line>",
        }

        # Pull extra state variables from schema
        if self.state_schema_string:
            try:
                schema = json.loads(self.state_schema_string)
                for var, desc in schema.get("state_variables", {}).items():
                    if var not in fmt:
                        t = desc.get("type", "string")
                        if t == "boolean":
                            fmt[var] = "<bool or 'Unknown'>"
                        elif t == "integer":
                            fmt[var] = "<int>"
                        elif t == "number":
                            fmt[var] = "<float>"
                        else:
                            fmt[var] = "<string>"
            except Exception:
                pass
        return fmt

    def _build_prompt(
        self,
        num_frames: int,
        previous_state: Optional[Dict[str, Any]],
        timestamp: float,
        frame_num: int,
        window_duration_sec: float = 0.0,
    ) -> str:
        prev_state_str = json.dumps(previous_state, indent=2) if previous_state else "{}"

        if window_duration_sec > 0:
            window_desc = (
                f"You will be provided with the {num_frames} most recent frames "
                f"spanning the past {window_duration_sec:.1f} seconds of the task video."
            )
        else:
            window_desc = (
                f"You will be provided with a rolling window of the {num_frames} most "
                f"recent frames from the task video."
            )

        prompt = f"""{self.system_instruction}
Your goal is to update the state variables based on the provided task graph, state schema, and the visual information from the images.

## Task Graph Definition
```json
{self.task_graph_string}
```

## State Variables Schema
```json
{self.state_schema_string}
```

## Instructions
{window_desc}
Your task is to update the state variables based on the images and the schemas above.

The state of the system at the start of this window is:
```json
{prev_state_str}
```

For each state variable, decide its current value. Boolean variables can be False, True, or "Unknown".

Additionally, classify every step in the task graph into one of three categories:
- **steps_completed**: Steps that are clearly finished.
- **steps_in_progress**: The step currently being performed (usually at most one).
- **steps_pending**: Steps not yet started.

Also provide:
- **predicted_next_action**: The most likely next step the human will perform.
- **prediction_confidence**: Your confidence (0.0–1.0).
- **reasoning**: A one-line summary of your analysis.

## Current context
- Frame number: {frame_num}
- Timestamp: {timestamp:.2f} seconds

## Output Format
Respond ONLY with a JSON object matching this structure (no markdown fences):
{self.state_format_string}

Here are the frames:
"""
        return prompt

    def _prepare_frames(self, frames: List[np.ndarray]) -> List[Image.Image]:
        pil_images = []
        for frame in frames[-self.max_frames:]:
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame
            img = Image.fromarray(rgb)
            if max(img.size) > self.max_image_dimension:
                scale = self.max_image_dimension / max(img.size)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            pil_images.append(img)
        return pil_images

    def predict(
        self,
        frames: List[np.ndarray],
        timestamp: float = 0.0,
        frame_num: int = 0,
        window_duration_sec: float = 0.0,
    ) -> IntentResult:
        """Run a synchronous RCWPS prediction on the given frames."""
        pil_frames = self._prepare_frames(frames)
        prompt_text = self._build_prompt(
            num_frames=len(pil_frames),
            previous_state=self.previous_state,
            timestamp=timestamp,
            frame_num=frame_num,
            window_duration_sec=window_duration_sec,
        )

        result = IntentResult(timestamp=timestamp, frame_num=frame_num)
        if not self._llm_client:
            logger.warning("No LLM client – returning default IntentResult")
            result.reasoning = "LLM client not available"
            return result

        t0 = time.time()
        try:
            retries = 3
            response_text = ""
            for attempt in range(retries):
                try:
                    response_text = self._llm_client.generate(
                        prompt_text,
                        images=pil_frames,
                        temperature=self.temperature,
                        json_mode=True,
                        max_tokens=4096,
                    )
                    break
                except Exception as e:
                    logger.warning(f"LLM call attempt {attempt+1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise
            generation_time = time.time() - t0
        except Exception as e:
            generation_time = time.time() - t0
            logger.error(f"LLM prediction failed: {e}")
            result.reasoning = f"LLM error: {e}"
            result.generation_time_sec = generation_time
            self.prompt_logger.log_call(
                prompt_text=prompt_text, response_text=str(e), parsed_response=None,
                frame_images=pil_frames, model=self.model, generation_time=generation_time,
                frame_num=frame_num, timestamp=timestamp, previous_state=self.previous_state,
            )
            return result

        parsed = self._parse_response(response_text)
        result.raw_response = response_text
        result.generation_time_sec = generation_time

        if parsed:
            result.state = parsed
            result.current_phase = parsed.get("current_phase", "initialization")
            result.current_action = parsed.get("current_action", "idle")
            result.human_state = parsed.get("human_state", "idle")
            result.steps_completed = parsed.get("steps_completed", [])
            result.steps_in_progress = parsed.get("steps_in_progress", [])
            result.steps_pending = parsed.get("steps_pending", [])
            result.predicted_next_action = parsed.get("predicted_next_action", "unknown")
            result.prediction_confidence = float(parsed.get("prediction_confidence", 0.0))
            result.reasoning = parsed.get("reasoning", "")

            self.previous_state = parsed.copy()
        else:
            result.reasoning = "Failed to parse Gemini response"

        self.prompt_logger.log_call(
            prompt_text=prompt_text, response_text=response_text, parsed_response=parsed,
            frame_images=pil_frames, model=self.model, generation_time=generation_time,
            frame_num=frame_num, timestamp=timestamp, previous_state=self.previous_state,
        )
        self.history.append(result)
        return result

    @staticmethod
    def _parse_response(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        def _unwrap(obj):
            """If the LLM returned a JSON array, extract the first dict."""
            if isinstance(obj, list):
                return obj[0] if obj else None
            return obj

        try:
            return _unwrap(json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return _unwrap(json.loads(cleaned[start:end + 1]))
            except json.JSONDecodeError:
                pass
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  IntentMonitor — BaseMonitor subclass (async, frame-buffer based)
# ═══════════════════════════════════════════════════════════════════════════


def _load_json_file(file_path: str) -> Optional[Any]:
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


class IntentMonitor(BaseMonitor):
    """Async intent monitor using Gemini and configurable task graphs.

    Captures frames over a time window, then queries Gemini to identify
    the current human action, track state variables, and predict the
    next likely action.  Task graphs and state schemas are loaded from
    JSON files.
    """

    def __init__(self, config: Optional[IntentMonitorConfig] = None):
        super().__init__(config)

        self.fps = config.fps if config else 2.0
        self.window_duration = config.capture_duration if config else 2.0
        self.max_frames = int(self.fps * self.window_duration)
        self.max_image_dimension = config.max_image_dimension if config else 640

        # Frame buffer
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

        # LLM backend
        self.model = config.model if config else "gemini-3.1-pro-preview"
        llm_backend = getattr(config, "llm_backend", "gemini") if config else "gemini"
        sglang_url = getattr(config, "sglang_base_url", "http://localhost:8100/v1") if config else "http://localhost:8100/v1"

        self._llm_client = None
        self.client = None  # legacy Gemini client for backward compat
        try:
            from aura.utils.llm_client import create_llm_client
            self._llm_client = create_llm_client(
                llm_backend,
                model=self.model,
                base_url=sglang_url,
            )
        except Exception as e:
            logger.warning("Failed to create LLM client (%s): %s — intent recognition disabled", llm_backend, e)

        # Keep legacy Gemini client for any code that accesses self.client
        if llm_backend == "gemini" and GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(
                    http_options={"api_version": "v1beta"},
                    api_key=api_key,
                )

        self.last_prediction: Optional[IntentPrediction] = None

    def _load_task_graph(self, config: Optional[IntentMonitorConfig]) -> List[Dict]:
        if config and config.dag_file:
            dag = _load_json_file(config.dag_file)
            if dag:
                if isinstance(dag, list):
                    return dag
                elif isinstance(dag, dict):
                    if 'nodes' in dag:
                        nodes = dag['nodes']
                        if isinstance(nodes, dict):
                            return [{"id": k, **v} for k, v in nodes.items()]
                        return nodes
                    return [dag]
        return self._default_task_graph()

    def _load_state_schema(self, config: Optional[IntentMonitorConfig]) -> List[Dict]:
        if config and config.state_file:
            schema = _load_json_file(config.state_file)
            if schema:
                if isinstance(schema, list):
                    return schema
                elif isinstance(schema, dict):
                    if 'state_variables' in schema:
                        variables = schema['state_variables']
                        if isinstance(variables, dict):
                            return [{"name": k, **v} for k, v in variables.items()]
                        return variables
                    return [schema]
        return self._default_state_schema()

    def _default_task_graph(self) -> List[Dict]:
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
        return [
            {"name": "is_hand_visible", "type": "Boolean", "description": "True if at least one hand is visible in frame"},
            {"name": "holding_object", "type": "String", "description": "Description of object being held, or null"},
            {"name": "gaze_target", "type": "String", "description": "What the person appears to be looking at"},
            {"name": "body_posture", "type": "String", "description": "Current posture: sitting, standing, leaning"},
            {"name": "hand_position", "type": "String", "description": "General hand position: raised, lowered, extended"},
        ]

    def _default_system_prompt(self) -> str:
        return """You are an AI assistant specialized in real-time human action recognition and prediction from video frames.
You analyze video frames to identify what a person is currently doing and predict their next action."""

    def _default_task_context(self) -> str:
        return """You are analyzing a sequence of video frames showing a person performing various activities.
Identify their current action and predict what they will do next based on the task graph."""

    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.INTENT

    async def _process(self, frame: Optional[np.ndarray] = None) -> Optional[IntentOutput]:
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
            if self.last_prediction:
                return self._prediction_to_output(self.last_prediction)
            return None

        self.last_prediction_time = current_time

        if self.client:
            prediction = await self._predict_with_gemini()
            if prediction:
                self.last_prediction = prediction
                self.task_state = prediction.task_state
                return self._prediction_to_output(prediction)

        return IntentOutput(
            timestamp=datetime.now(),
            intent=Intent(
                type=IntentType.UNKNOWN,
                confidence=0.0,
                reasoning="No prediction available",
            ),
            alternatives=[],
        )

    def _prediction_to_output(self, prediction: IntentPrediction) -> IntentOutput:
        current_type = self._action_to_intent_type(prediction.current_action)
        predicted_type = self._action_to_intent_type(prediction.predicted_next_action)

        current_intent = Intent(
            type=current_type,
            confidence=prediction.current_action_confidence,
            reasoning=f"{prediction.current_action}: {prediction.reasoning}",
            target_object=prediction.task_state.get("holding_object"),
        )

        predicted_intent = Intent(
            type=predicted_type,
            confidence=prediction.predicted_next_confidence,
            reasoning=prediction.predicted_next_action,
        )

        return IntentOutput(
            timestamp=datetime.now(),
            intent=current_intent,
            alternatives=[predicted_intent] if predicted_intent.confidence > 0 else [],
        )

    def _action_to_intent_type(self, action: str) -> IntentType:
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
        if not self._llm_client or len(self.frame_buffer) < 2:
            return None

        try:
            pil_frames = []
            for timestamp, frame in list(self.frame_buffer):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                if max(pil_image.size) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(pil_image.size)
                    new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                pil_frames.append(pil_image)

            prompt = self._build_prompt()

            response_text = await asyncio.to_thread(
                self._llm_client.generate,
                prompt,
                images=pil_frames,
                temperature=0.5,
                json_mode=True,
            )

            result = json.loads(response_text)
            if isinstance(result, list):
                result = result[0] if result else {}

            return IntentPrediction(
                current_action=result.get("current_action", "Unknown"),
                current_action_confidence=result.get("current_action_confidence", 0.5),
                predicted_next_action=result.get("predicted_next_action", "Unknown"),
                predicted_next_confidence=result.get("predicted_next_confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                task_state=result.get("state", {}),
                raw_response=response_text,
            )

        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return None

    def _get_response_schema(self) -> Dict:
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
                    "properties": state_properties,
                },
                "current_action": {"type": "string"},
                "current_action_confidence": {"type": "number"},
                "predicted_next_action": {"type": "string"},
                "predicted_next_confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
        }

    def _build_prompt(self) -> str:
        task_graph_str = json.dumps(self.task_graph, indent=2)
        state_schema_str = json.dumps(self.state_schema, indent=2)

        prompt_parts = [self.system_prompt, ""]
        prompt_parts += [f"## Task: {self.task_name}", self.task_context, ""]
        prompt_parts += [
            "## Task Graph (Available Actions)",
            "These are the possible actions the person can perform:",
            f"```json\n{task_graph_str}\n```",
            "",
        ]
        prompt_parts += [
            "## State Variables to Track",
            f"```json\n{state_schema_str}\n```",
            "",
        ]

        if self.analysis_instructions:
            prompt_parts += ["## Analysis Instructions", self.analysis_instructions]
        else:
            prompt_parts += [
                "## Analysis Instructions",
                f"""Analyze the provided sequence of {len(self.frame_buffer)} frames and:

1. **Update STATE variables**: Based on visual observations, update each state variable.
2. **Identify CURRENT action**: What is the person doing right now? Use actions from the Task Graph.
3. **Predict NEXT action**: Based on motion patterns and task dependencies, what will they do next?
4. **Provide confidence scores**: Rate your confidence (0.0 to 1.0) for both predictions.
5. **Explain your reasoning**: Briefly describe what visual cues led to your predictions.""",
            ]
        prompt_parts.append("")

        if self.output_format:
            prompt_parts += ["## Output Format", self.output_format]
        else:
            prompt_parts += [
                "## Output Format",
                "Respond with a JSON object matching the schema provided.",
            ]

        return "\n".join(prompt_parts)

    def get_current_state(self) -> Dict[str, Any]:
        return self.task_state.copy()

    def get_task_graph(self) -> List[Dict]:
        return self.task_graph.copy()

    async def predict_from_frames(self, frames: List[np.ndarray]) -> Optional[IntentPrediction]:
        """Predict intent from a list of frames directly (convenience method)."""
        if not self._llm_client or len(frames) < 1:
            return None

        try:
            pil_frames = []
            for frame in frames:
                pil_image = Image.fromarray(frame)
                if max(pil_image.size) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(pil_image.size)
                    new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                pil_frames.append(pil_image)

            prompt = self._build_prompt()

            response_text = await asyncio.to_thread(
                self._llm_client.generate,
                prompt,
                images=pil_frames,
                temperature=0.5,
                json_mode=True,
            )

            result = json.loads(response_text)
            if isinstance(result, list):
                result = result[0] if result else {}
            prediction = IntentPrediction(
                current_action=result.get("current_action", "Unknown"),
                current_action_confidence=result.get("current_action_confidence", 0.5),
                predicted_next_action=result.get("predicted_next_action", "Unknown"),
                predicted_next_confidence=result.get("predicted_next_confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                task_state=result.get("state", {}),
                raw_response=response_text,
            )

            self.last_prediction = prediction
            self.task_state = prediction.task_state
            return prediction

        except Exception as e:
            logger.error(f"Error predicting intent from frames: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization helper
# ═══════════════════════════════════════════════════════════════════════════


def visualize_intent(frame: np.ndarray, prediction: IntentPrediction) -> np.ndarray:
    """Visualize intent prediction on frame."""
    vis_frame = frame.copy()

    current_text = f"Current: {prediction.current_action} ({prediction.current_action_confidence:.2f})"
    cv2.putText(vis_frame, current_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    predicted_text = f"Next: {prediction.predicted_next_action} ({prediction.predicted_next_confidence:.2f})"
    cv2.putText(vis_frame, predicted_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if prediction.reasoning:
        reasoning = prediction.reasoning[:80] + "..." if len(prediction.reasoning) > 80 else prediction.reasoning
        cv2.putText(vis_frame, reasoning, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if prediction.task_state:
        y_pos = 120
        for key, value in list(prediction.task_state.items())[:3]:
            state_text = f"  {key}: {value}"
            cv2.putText(vis_frame, state_text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20

    return vis_frame

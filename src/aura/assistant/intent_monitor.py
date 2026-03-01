"""Generic RCWPS-based Intent Monitor for any AURA task.

Implements a Rolling Context Window with Previous State (RCWPS) approach.
It loads a task-specific DAG, state schema, and task profile to formulate prompts.
"""

import os
import json
import time
import logging
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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


class AURAIntentMonitor:
    """Generic Intent monitor.
    Uses schemas and DAG from a specific task config folder.
    """

    def __init__(
        self,
        config_dir: str,
        model: str = "gemini-3-pro-preview",
        max_frames: int = 5,
        max_image_dimension: int = 640,
        temperature: float = 0.3,
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        realtime: bool = False,
    ):
        self.realtime = realtime
        if realtime:
            if model == "gemini-3-pro-preview":
                model = "gemini-2.5-flash"
            max_frames = min(max_frames, 3)
            max_image_dimension = min(max_image_dimension, 480)

        self.model = model
        self.max_frames = max_frames
        self.max_image_dimension = max_image_dimension
        self.temperature = temperature
        self.config_dir = Path(config_dir)

        # Load task profile, DAG, and state schema
        dag_path = self.config_dir / "dag.json"
        state_path = self.config_dir / "state_schema.json"
        profile_path = self.config_dir / "task_profile.json"

        self.task_graph_string = dag_path.read_text(encoding="utf-8") if dag_path.exists() else "{}"
        self.state_schema_string = state_path.read_text(encoding="utf-8") if state_path.exists() else "{}"
        
        self.task_profile = {}
        if profile_path.exists():
            self.task_profile = json.loads(profile_path.read_text(encoding="utf-8"))

        self.system_instruction = self.task_profile.get("system_instruction", "You are an AI assistant analyzing video frames of a task.")
        self.state_format_string = json.dumps(self._build_output_format(), indent=2)

        self.previous_state: Optional[Dict[str, Any]] = None
        self.history: List[IntentResult] = []

        self.client = None
        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                logger.warning("No API key found – Gemini calls disabled")

        self.prompt_logger = PromptLogger(
            log_dir=log_dir,
            enabled=enable_logging,
        )

    def _build_output_format(self) -> Dict[str, Any]:
        """Construct the expected JSON format for prompt"""
        fmt = {
            "current_phase": "<phase>",
            "current_action": "<action>",
            "human_state": "<state>",
            "steps_completed": ["<step>", "..."],
            "steps_in_progress": ["<step>", "..."],
            "steps_pending": ["<step>", "..."],
            "predicted_next_action": "<step>",
            "prediction_confidence": 0.0,
            "reasoning": "<one line>"
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
    ) -> str:
        prev_state_str = json.dumps(previous_state, indent=2) if previous_state else "{}"
        
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
You will be provided with a rolling window of the {num_frames} most recent frames from the task video.
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
    ) -> IntentResult:
        pil_frames = self._prepare_frames(frames)
        prompt_text = self._build_prompt(
            num_frames=len(pil_frames),
            previous_state=self.previous_state,
            timestamp=timestamp,
            frame_num=frame_num,
        )

        result = IntentResult(timestamp=timestamp, frame_num=frame_num)
        if not self.client:
            logger.warning("No Gemini client – returning default IntentResult")
            result.reasoning = "Gemini client not available"
            return result

        t0 = time.time()
        try:
            parts: list = [types.Part.from_text(text=prompt_text)]
            for img in pil_frames:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

            contents = [types.Content(role="user", parts=parts)]
            generate_config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=0.95,
                top_k=30,
                response_mime_type="text/plain",
            )

            retries = 3
            response_text = ""
            for attempt in range(retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=generate_config,
                    )
                    response_text = response.text or ""
                    break
                except Exception as e:
                    logger.warning(f"Gemini call attempt {attempt+1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise
            generation_time = time.time() - t0
        except Exception as e:
            generation_time = time.time() - t0
            logger.error(f"Gemini prediction failed: {e}")
            result.reasoning = f"Gemini error: {e}"
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

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                pass
        return None

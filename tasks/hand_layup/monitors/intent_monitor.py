"""RCWPS-based Intent Monitor for Hand Layup task.

Implements a Rolling Context Window with Previous State (RCWPS) approach,
adapted from hcdt/LLMcalls/run_phase_two.py::runRCWPS.

Key idea:
  - Each Gemini call receives the task DAG, state schema, and previous state.
  - A rolling window of recent video frames is attached.
  - Gemini returns an updated state JSON: current phase, current action,
    completed steps, in-progress steps, pending steps, and predicted next action.
  - All prompts and responses are logged to disk for debugging.

Usage:
    from tasks.hand_layup.monitors.intent_monitor import HandLayupIntentMonitor

    monitor = HandLayupIntentMonitor()
    result = monitor.predict(frames=[frame1, frame2, ...], timestamp=12.5)
    print(result.state)
    print(result.steps_completed)
"""

import os
import json
import time
import base64
import logging
from io import BytesIO
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini imports (optional)
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config paths (defaults relative to task root)
# ---------------------------------------------------------------------------
TASK_DIR = Path(__file__).parent.parent
CONFIG_DIR = TASK_DIR / "config"
DEFAULT_DAG_PATH = CONFIG_DIR / "hand_layup_dag.json"
DEFAULT_STATE_PATH = CONFIG_DIR / "hand_layup_state.json"
DEFAULT_LOG_DIR = TASK_DIR / "logs" / "intent_monitor"


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class IntentResult:
    """Result of a single RCWPS intent prediction."""
    timestamp: float
    frame_num: int

    # State variables returned by Gemini
    state: Dict[str, Any] = field(default_factory=dict)

    # Convenience fields extracted from state
    current_phase: str = "initialization"
    current_action: str = "idle"
    human_state: str = "idle"
    layers_placed: int = 0
    layers_resined: int = 0
    mixture_mixed: bool = False
    consolidated: bool = False
    human_wearing_gloves: bool = False

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


# ============================================================================
# Prompt Logger
# ============================================================================

class PromptLogger:
    """Logs every Gemini prompt/response exchange to disk.

    Directory layout::

        <log_dir>/
          session_<timestamp>/
            call_0001/
              prompt.txt          # full text prompt
              response.txt        # raw Gemini text
              response_parsed.json
              frames/             # attached frame thumbnails (JPEG)
                frame_0.jpg
                frame_1.jpg
              meta.json           # timing, model, token counts
    """

    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.session_dir = None
            return

        base = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
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

        # Save prompt
        (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

        # Save response
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

        # Save parsed response
        if parsed_response is not None:
            with open(call_dir / "response_parsed.json", "w") as f:
                json.dump(parsed_response, f, indent=2, default=str)

        # Save previous state
        if previous_state is not None:
            with open(call_dir / "previous_state.json", "w") as f:
                json.dump(previous_state, f, indent=2, default=str)

        # Save frame thumbnails
        if frame_images:
            frames_dir = call_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, img in enumerate(frame_images):
                img.save(frames_dir / f"frame_{i}.jpg", "JPEG", quality=80)

        # Save meta
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


# ============================================================================
# RCWPS Intent Monitor
# ============================================================================

class HandLayupIntentMonitor:
    """Rolling Context Window with Previous State intent monitor.

    Follows the RCWPS pattern from hcdt: each call to Gemini includes
    the task DAG, state schema, previous state, and a rolling window
    of video frames.  Gemini returns an updated state JSON.
    """

    # Ordered list of all DAG steps for tracking
    ALL_STEPS = [
        "idle",
        "place_cup_on_scale",
        "add_resin_to_cup",
        "add_hardener_to_cup",
        "weigh_mixture",
        "mix_resin_hardener",
        "place_layer_1",
        "apply_resin_layer_1",
        "place_layer_2",
        "apply_resin_layer_2",
        "place_layer_3",
        "apply_resin_layer_3",
        "place_layer_4",
        "apply_resin_layer_4",
        "consolidate_with_roller",
        "cleanup",
        "task_complete",
    ]

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        dag_path: Optional[str] = None,
        state_path: Optional[str] = None,
        max_frames: int = 5,
        max_image_dimension: int = 640,
        temperature: float = 0.3,
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        realtime: bool = False,
    ):
        self.realtime = realtime

        # Override defaults for realtime / low-latency mode
        if realtime:
            if model == "gemini-3-pro-preview":  # not explicitly overridden
                model = "gemini-2.5-flash"
            max_frames = min(max_frames, 3)
            max_image_dimension = min(max_image_dimension, 480)

        self.model = model
        self.max_frames = max_frames
        self.max_image_dimension = max_image_dimension
        self.temperature = temperature

        # Load task artefacts
        dp = Path(dag_path) if dag_path else DEFAULT_DAG_PATH
        sp = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self.task_graph_string = dp.read_text(encoding="utf-8") if dp.exists() else "{}"
        self.state_schema_string = sp.read_text(encoding="utf-8") if sp.exists() else "{}"

        # Running state (rolls forward between calls)
        self.previous_state: Optional[Dict[str, Any]] = None
        self.history: List[IntentResult] = []

        # Gemini client
        self.client = None
        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                logger.warning("No API key found – Gemini calls disabled")
        else:
            logger.warning("google-genai not installed – Gemini calls disabled")

        # Logging
        self.prompt_logger = PromptLogger(
            log_dir=log_dir,
            enabled=enable_logging,
        )

    # ------------------------------------------------------------------
    # Prompt building (RCWPS style)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        num_frames: int,
        previous_state: Optional[Dict[str, Any]],
        timestamp: float,
        frame_num: int,
    ) -> str:
        """Build the RCWPS prompt text.

        Structure mirrors hcdt ``cfg.prompts.RCWPS``:
        1. System instruction
        2. Task Graph
        3. State Schema
        4. Previous state
        5. Frame presentation instruction
        """
        prev_state_str = json.dumps(previous_state, indent=2) if previous_state else "{}"

        prompt = f"""You are an AI assistant analyzing video frames of a person performing a fiberglass hand layup task.
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

For each state variable, decide its current value.  Boolean variables can be False, True, or "Unknown".

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
{{
  "current_phase": "<phase>",
  "current_action": "<action>",
  "human_state": "<state>",
  "layers_placed": <int>,
  "layers_resined": <int>,
  "mixture_mixed": <bool>,
  "consolidated": <bool>,
  "human_wearing_gloves": <bool or "Unknown">,
  "steps_completed": ["<step>", ...],
  "steps_in_progress": ["<step>", ...],
  "steps_pending": ["<step>", ...],
  "predicted_next_action": "<step>",
  "prediction_confidence": <float>,
  "reasoning": "<one line>"
}}

Here are the frames:
"""
        return prompt

    def _build_realtime_prompt(
        self,
        num_frames: int,
        previous_state: Optional[Dict[str, Any]],
        timestamp: float,
        frame_num: int,
    ) -> str:
        """Build a compact prompt optimised for low-latency realtime use.

        Omits the full DAG and state schema to keep token count small,
        while preserving the RCWPS rolling-state pattern.
        """
        prev_state_str = json.dumps(previous_state, indent=2) if previous_state else "{}"

        prompt = f"""\
Monitor a fiberglass hand layup task from {num_frames} live frames. Update task state.

Steps (in order): idle, place_cup_on_scale, add_resin_to_cup, add_hardener_to_cup, \
weigh_mixture, mix_resin_hardener, place_layer_1, apply_resin_layer_1, place_layer_2, \
apply_resin_layer_2, place_layer_3, apply_resin_layer_3, place_layer_4, \
apply_resin_layer_4, consolidate_with_roller, cleanup, task_complete.

Phases: initialization, resin_preparation, mixing, layer_1_placement, layer_1_resin, \
layer_2_placement, layer_2_resin, layer_3_placement, layer_3_resin, layer_4_placement, \
layer_4_resin, consolidation, cleanup, complete.

Human states: idle, preparing_resin, mixing, placing_fiberglass, applying_resin, \
rolling, inspecting, waiting_for_robot, done.

Previous state:
{prev_state_str}

Frame {frame_num} | {timestamp:.1f}s

Respond ONLY with JSON (no fences):
{{"current_phase":"<phase>","current_action":"<action>","human_state":"<state>",\
"layers_placed":<int>,"layers_resined":<int>,"mixture_mixed":<bool>,\
"consolidated":<bool>,"human_wearing_gloves":<bool>,\
"steps_completed":[...],"steps_in_progress":[...],"steps_pending":[...],\
"predicted_next_action":"<step>","prediction_confidence":<float>,\
"reasoning":"<brief>"}}
"""
        return prompt

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _prepare_frames(
        self, frames: List[np.ndarray]
    ) -> List[Image.Image]:
        """Convert numpy frames to resized PIL images."""
        pil_images = []
        for frame in frames[-self.max_frames:]:
            if frame.ndim == 3 and frame.shape[2] == 3:
                # Assume BGR from OpenCV
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

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(
        self,
        frames: List[np.ndarray],
        timestamp: float = 0.0,
        frame_num: int = 0,
    ) -> IntentResult:
        """Run one RCWPS prediction.

        Args:
            frames: List of video frames (BGR numpy arrays).
                    The last ``max_frames`` will be used.
            timestamp: Current video timestamp in seconds.
            frame_num: Current frame number.

        Returns:
            IntentResult with updated state and action tracking.
        """
        pil_frames = self._prepare_frames(frames)
        build_fn = self._build_realtime_prompt if self.realtime else self._build_prompt
        prompt_text = build_fn(
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

        # ----- Call Gemini (RCWPS style: prompt + images) -----
        t0 = time.time()
        try:
            # Build multi-turn contents like hcdt generate()
            parts: list = [types.Part.from_text(text=prompt_text)]
            for img in pil_frames:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                parts.append(
                    types.Part.from_bytes(
                        data=buf.getvalue(),
                        mime_type="image/jpeg",
                    )
                )

            contents = [types.Content(role="user", parts=parts)]

            generate_config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=0.95,
                top_k=30,
                response_mime_type="text/plain",
            )

            # Non-streaming call with retries (compatible with gemini-3-pro)
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
                    break  # success
                except Exception as e:
                    logger.warning(
                        f"Gemini call attempt {attempt+1}/{retries} failed: {e}"
                    )
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
            # Log even on failure
            self.prompt_logger.log_call(
                prompt_text=prompt_text,
                response_text=str(e),
                parsed_response=None,
                frame_images=pil_frames,
                model=self.model,
                generation_time=generation_time,
                frame_num=frame_num,
                timestamp=timestamp,
                previous_state=self.previous_state,
            )
            return result

        # ----- Parse response -----
        parsed = self._parse_response(response_text)
        result.raw_response = response_text
        result.generation_time_sec = generation_time

        if parsed:
            result.state = parsed
            result.current_phase = parsed.get("current_phase", "initialization")
            result.current_action = parsed.get("current_action", "idle")
            result.human_state = parsed.get("human_state", "idle")
            result.layers_placed = int(parsed.get("layers_placed", 0))
            result.layers_resined = int(parsed.get("layers_resined", 0))
            result.mixture_mixed = bool(parsed.get("mixture_mixed", False))
            result.consolidated = bool(parsed.get("consolidated", False))
            gloves = parsed.get("human_wearing_gloves", "Unknown")
            result.human_wearing_gloves = gloves if isinstance(gloves, bool) else False
            result.steps_completed = parsed.get("steps_completed", [])
            result.steps_in_progress = parsed.get("steps_in_progress", [])
            result.steps_pending = parsed.get("steps_pending", [])
            result.predicted_next_action = parsed.get("predicted_next_action", "unknown")
            result.prediction_confidence = float(parsed.get("prediction_confidence", 0.0))
            result.reasoning = parsed.get("reasoning", "")

            # Roll forward state for next call
            self.previous_state = {
                "current_phase": result.current_phase,
                "current_action": result.current_action,
                "human_state": result.human_state,
                "layers_placed": result.layers_placed,
                "layers_resined": result.layers_resined,
                "mixture_mixed": result.mixture_mixed,
                "consolidated": result.consolidated,
                "human_wearing_gloves": result.human_wearing_gloves,
                "steps_completed": result.steps_completed,
                "steps_in_progress": result.steps_in_progress,
            }
        else:
            result.reasoning = "Failed to parse Gemini response"

        # ----- Log -----
        self.prompt_logger.log_call(
            prompt_text=prompt_text,
            response_text=response_text,
            parsed_response=parsed,
            frame_images=pil_frames,
            model=self.model,
            generation_time=generation_time,
            frame_num=frame_num,
            timestamp=timestamp,
            previous_state=self.previous_state,
        )

        self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from Gemini response, handling fences and junk."""
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from response (len={len(text)})")
        return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_previous_state(self) -> Optional[Dict[str, Any]]:
        """Return the most recent state (input to next call)."""
        return self.previous_state

    def get_history(self) -> List[IntentResult]:
        return list(self.history)

    def get_log_dir(self) -> Optional[Path]:
        return self.prompt_logger.get_session_dir()

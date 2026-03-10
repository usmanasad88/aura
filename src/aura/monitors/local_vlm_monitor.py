"""Local VLM Monitor — real-time visual question answering via SGLang or transformers.

Supports two backends:

* **sglang** (default): Connects to a running SGLang server that exposes
  an OpenAI-compatible ``/v1/chat/completions`` API.
  Start the server with ``scripts/start_sglang_server.sh`` (isolated venv,
  no dependency conflicts).  Uses optimised CUDA kernels and continuous
  batching for high throughput.

* **transformers**: Loads the model directly in-process via HuggingFace
  transformers.  Simpler setup, no separate server, but slower.

The monitor produces :class:`LocalVLMOutput` containing a structured
``VLMPerception`` result with the answer, detected objects, confidence,
and optional scene description.

Example (SGLang backend — start server first)::

    # Terminal 1: ./scripts/start_sglang_server.sh
    # Terminal 2:
    monitor = LocalVLMMonitor(LocalVLMConfig(
        backend="sglang",
        model_id="Qwen/Qwen3.5-0.8B",
        question="What is the human holding?",
    ))
    output = await monitor.process_frame(frame_bgr)
    print(output.perception.answer)

Example (transformers backend)::

    monitor = LocalVLMMonitor(LocalVLMConfig(
        backend="transformers",
        model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    ))
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from aura.core import MonitorType, MonitorOutput
from aura.monitors.base_monitor import BaseMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------


@dataclass
class DetectedObject:
    """A single object detected by the VLM."""

    name: str
    description: str = ""
    confidence: str = "medium"  # "high" / "medium" / "low"
    held_by_human: bool = False


@dataclass
class VLMPerception:
    """Structured VLM perception result."""

    answer: str
    objects: List[DetectedObject] = field(default_factory=list)
    scene_description: str = ""
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class LocalVLMOutput(MonitorOutput):
    """Output from the local VLM monitor."""

    monitor_type: MonitorType = field(default=MonitorType.PERCEPTION)
    perception: Optional[VLMPerception] = None
    processing_time_sec: float = 0.0
    frame_index: int = 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LocalVLMConfig:
    """Configuration for the local VLM monitor.

    Attributes:
        backend: ``"sglang"`` (HTTP to SGLang server) or ``"transformers"``
            (in-process HuggingFace model).
        model_id: HuggingFace model ID.
        sglang_base_url: Base URL for the SGLang OpenAI-compatible API.
        question: Default question asked every frame.
        max_new_tokens: Maximum tokens to generate.
        max_image_dimension: Resize images so the longest edge <= this.
        structured_output: If True, ask the model to return JSON.
    """

    enabled: bool = True
    backend: str = "sglang"  # "sglang" or "transformers"
    model_id: str = "Qwen/Qwen3.5-0.8B"
    sglang_base_url: str = "http://localhost:8100/v1"
    question: str = "What is the human holding? Output Nothing if the human isn't holding anything."
    max_new_tokens: int = 256
    max_image_dimension: int = 512
    # transformers-only settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    # Rate limiting
    update_rate_hz: float = 2.0
    timeout_sec: float = 30.0
    # Structured output prompt engineering
    structured_output: bool = True
    # Extra kwargs
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# transformers model singleton (only used in "transformers" backend)
# ---------------------------------------------------------------------------

_vlm_model = None
_vlm_processor = None


def _load_vlm(config: LocalVLMConfig):
    """Lazy-load model via transformers."""
    global _vlm_model, _vlm_processor

    if _vlm_model is not None and _vlm_processor is not None:
        return _vlm_model, _vlm_processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading model: %s ...", config.model_id)
    t0 = time.time()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    _vlm_processor = AutoProcessor.from_pretrained(config.model_id)
    _vlm_model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        _attn_implementation="eager",
    ).to(config.device)

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs (device=%s, dtype=%s)",
                elapsed, config.device, config.dtype)
    return _vlm_model, _vlm_processor


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _prepare_image(frame: np.ndarray, max_dim: int) -> Image.Image:
    """BGR frame -> resized RGB PIL image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if max(pil.size) > max_dim:
        ratio = max_dim / max(pil.size)
        pil = pil.resize(
            (int(pil.size[0] * ratio), int(pil.size[1] * ratio)),
            Image.Resampling.LANCZOS,
        )
    return pil


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to a base64 data-URL for the OpenAI vision API."""
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=85)
    else:
        img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_text(question: str, structured: bool) -> str:
    if structured:
        return (
            f"{question}\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            "{\n"
            '  "answer": "<short direct answer>",\n'
            # '  "objects": [\n'
            # '    {"name": "<object>", "held_by_human": true/false,\n'
            # '     "confidence": "high/medium/low"}\n'
            # "  ],\n"
            # '  "scene_description": "<one sentence>",\n'
            '  "confidence": <0.0-1.0>\n'
            "}"
        )
    return question


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class LocalVLMMonitor(BaseMonitor):
    """Real-time local VLM perception monitor.

    Supports ``"sglang"`` (HTTP) and ``"transformers"`` (in-process)
    backends.  The SGLang backend is recommended for speed -- it uses
    optimised kernels and avoids dependency conflicts.
    """

    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERCEPTION

    def __init__(self, config: Optional[LocalVLMConfig] = None):
        self.vlm_config = config or LocalVLMConfig()
        from aura.monitors.base_monitor import MonitorConfig as _MC

        mc = _MC(
            enabled=self.vlm_config.enabled,
            update_rate_hz=self.vlm_config.update_rate_hz,
            timeout_sec=self.vlm_config.timeout_sec,
        )
        super().__init__(mc)

        self._model_loaded = False
        self._openai_client = None
        self.frame_count = 0
        self._last_output: Optional[LocalVLMOutput] = None

    # -- model / client management ------------------------------------

    def _ensure_backend(self):
        """Initialise the chosen backend."""
        if self._model_loaded:
            return

        if self.vlm_config.backend == "sglang":
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.vlm_config.sglang_base_url,
                api_key="not-needed",
            )
            # Quick health check
            try:
                models = self._openai_client.models.list()
                names = [m.id for m in models.data]
                logger.info("SGLang server OK -- models: %s", names)
            except Exception as e:
                raise ConnectionError(
                    f"Cannot reach SGLang server at {self.vlm_config.sglang_base_url}. "
                    f"Start it with: ./scripts/start_sglang_server.sh\n"
                    f"Error: {e}"
                ) from e
        else:
            _load_vlm(self.vlm_config)

        self._model_loaded = True

    # -- BaseMonitor ABC ----------------------------------------------

    async def _process(self, **kwargs) -> MonitorOutput:
        raise NotImplementedError(
            "Use process_frame(frame) instead of the generic _process()."
        )

    # -- public API ---------------------------------------------------

    async def process_frame(
        self,
        frame: np.ndarray,
        question: Optional[str] = None,
    ) -> LocalVLMOutput:
        """Run the VLM on a single BGR frame.

        Uses whichever backend is configured (sglang or transformers).
        """
        self._ensure_backend()

        q = question or self.vlm_config.question
        pil_image = _prepare_image(frame, self.vlm_config.max_image_dimension)
        user_text = _build_user_text(q, self.vlm_config.structured_output)

        t0 = time.time()

        if self.vlm_config.backend == "sglang":
            raw_text = self._query_sglang(pil_image, user_text)
        else:
            raw_text = self._query_transformers(pil_image, user_text)

        elapsed = time.time() - t0

        perception = self._parse_response(raw_text, q)

        self.frame_count += 1
        output = LocalVLMOutput(
            timestamp=datetime.now(),
            perception=perception,
            processing_time_sec=elapsed,
            frame_index=self.frame_count,
        )
        self._last_output = output
        return output

    # -- SGLang backend (OpenAI-compatible HTTP) ----------------------

    def _query_sglang(self, pil_image: Image.Image, user_text: str) -> str:
        """Send image + text to the SGLang server (OpenAI-compatible API)."""
        # Encode image as base64 data URL (standard OpenAI format)
        data_url = _pil_to_data_url(pil_image)

        response = self._openai_client.chat.completions.create(
            model=self.vlm_config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
            max_tokens=self.vlm_config.max_new_tokens,
            temperature=0.7,  # Default Qwen temperature
        )

        return response.choices[0].message.content or ""

    # -- transformers backend (in-process) ----------------------------

    def _query_transformers(self, pil_image: Image.Image, user_text: str) -> str:
        """Run inference directly via transformers."""
        import torch

        model, processor = _vlm_model, _vlm_processor

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt_text,
            images=[pil_image],
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.vlm_config.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids, skip_special_tokens=True)

    # -- response parsing ---------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str, question: str) -> VLMPerception:
        """Try to extract structured JSON; fall back to free text."""
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                objects = []
                for obj in data.get("objects", []):
                    objects.append(
                        DetectedObject(
                            name=obj.get("name", "unknown"),
                            description=obj.get("description", ""),
                            confidence=str(obj.get("confidence", "medium")),
                            held_by_human=bool(obj.get("held_by_human", False)),
                        )
                    )
                return VLMPerception(
                    answer=data.get("answer", raw_text.strip()),
                    objects=objects,
                    scene_description=data.get("scene_description", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    raw_text=raw_text,
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        return VLMPerception(
            answer=raw_text.strip(),
            objects=[],
            scene_description="",
            confidence=0.3,
            raw_text=raw_text,
        )

    # -- visualisation helper -----------------------------------------

    @staticmethod
    def visualize(
        frame: np.ndarray, output: LocalVLMOutput,
    ) -> np.ndarray:
        """Draw the VLM answer on a frame copy."""
        vis = frame.copy()
        if output.perception is None:
            return vis

        p = output.perception
        y = 30
        cv2.putText(vis, f"Answer: {p.answer}", (10, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

        for obj in p.objects:
            held = " [HELD]" if obj.held_by_human else ""
            txt = f"  {obj.name}{held} ({obj.confidence})"
            cv2.putText(vis, txt, (10, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 22

        cv2.putText(vis, f"{output.processing_time_sec:.2f}s | conf={p.confidence:.2f}",
                     (10, y + 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return vis

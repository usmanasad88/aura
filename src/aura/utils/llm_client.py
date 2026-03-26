"""Unified LLM client abstraction for AURA.

Supports multiple backends behind a single interface:

* **gemini** — Google Gemini API via ``google-genai`` SDK.
* **openai** — Any OpenAI-compatible API (SGLang, vLLM, Ollama, OpenAI).

Both the ``DecisionEngine`` and ``AURAIntentMonitor`` use this
abstraction so that swapping between cloud Gemini and a local VLM
(e.g. Qwen 3.5 4B served by SGLang) is a one-line config change.

Usage::

    from aura.utils.llm_client import create_llm_client

    # Gemini (default)
    client = create_llm_client("gemini", model="gemini-3.1-pro-preview")

    # Local SGLang server
    client = create_llm_client(
        "openai",
        model="Qwen/Qwen3.5-VL-4B-Instruct",
        base_url="http://localhost:8100/v1",
    )

    # Text-only query
    text = client.generate(prompt="Decide what to do", temperature=0.3)

    # Vision query (list of PIL images)
    text = client.generate(prompt="What is happening?", images=[pil_img],
                           temperature=0.3)
"""

from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Backend-agnostic LLM client interface."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The text prompt.
            images: Optional list of PIL images for vision models.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Request JSON-formatted output if the backend
                supports it.

        Returns:
            The generated text string.
        """


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------


class GeminiClient(LLMClient):
    """Google Gemini API backend."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini backend. "
                "Install with: uv add google-genai"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set for the Gemini backend."
            )
        self._client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=key,
        )

    def generate(
        self,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        from google.genai import types

        # Build content parts
        parts: list = [types.Part.from_text(text=prompt)]
        for img in images or []:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            )

        contents = [types.Content(role="user", parts=parts)]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (SGLang / vLLM / Ollama / OpenAI)
# ---------------------------------------------------------------------------


def _pil_to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a base64 data-URL for the OpenAI vision API."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible HTTP backend (SGLang, vLLM, Ollama, OpenAI)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8100/v1",
        api_key: str = "not-needed",
    ):
        super().__init__(model)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI-compatible backend. "
                "Install with: uv add openai"
            ) from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._base_url = base_url

    def generate(
        self,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        content: list = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url(img)},
            })

        messages = [{"role": "user", "content": content}]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Recognised backend aliases
_OPENAI_BACKENDS = frozenset({"openai", "sglang", "vllm", "ollama", "local"})


def create_llm_client(
    backend: str,
    *,
    model: str = "",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client for the given backend.

    Args:
        backend: One of ``"gemini"``, ``"openai"``, ``"sglang"``,
            ``"vllm"``, ``"ollama"``, ``"local"``.
        model: Model name / HuggingFace ID.
        api_key: API key (required for Gemini; optional for local
            servers).
        base_url: Base URL for OpenAI-compatible servers.  Defaults to
            ``http://localhost:8100/v1`` for SGLang.
    """
    backend = backend.lower().strip()

    if backend == "gemini":
        return GeminiClient(model=model, api_key=api_key)

    if backend in _OPENAI_BACKENDS:
        url = base_url or "http://localhost:8100/v1"
        key = api_key or "not-needed"
        return OpenAICompatibleClient(model=model, base_url=url, api_key=key)

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. "
        f"Choose from: gemini, {', '.join(sorted(_OPENAI_BACKENDS))}"
    )

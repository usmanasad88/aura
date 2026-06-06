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

import asyncio
import base64
import logging
import os
import threading
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

        from aura.utils.rate_limiter import throttle

        # Free-tier models (e.g. Gemini 3.1 Flash Lite) share a single
        # process-wide RPM budget across the intent monitor and the
        # decision engine — both reach the API through this method, so
        # throttling here keeps their *combined* rate under the cap.
        # Paid-tier models pass through untouched.
        throttle(self.model)

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

        # Log finish_reason so truncation causes are visible
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr and fr.name != "STOP":
                logger.warning(
                    "Gemini finish_reason=%s (model=%s, prompt_len=%d)",
                    fr.name, self.model, len(prompt),
                )
        else:
            logger.warning("Gemini returned no candidates (model=%s)", self.model)

        return response.text or ""


# ---------------------------------------------------------------------------
# Gemini Live API backend (persistent WebSocket session)
# ---------------------------------------------------------------------------


class GeminiLiveClient(LLMClient):
    """Google Gemini Live API backend for low-latency streaming.

    Maintains a persistent WebSocket session with the Gemini Live API.
    Frames can be streamed continuously via ``send_frame()``, and
    ``generate()`` sends a text prompt (optionally with images) and
    collects the text response.

    The session auto-reconnects on failure or expiry.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini Live backend. "
                "Install with: uv add google-genai"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set for the Gemini Live backend."
            )
        self._genai_client = genai.Client(api_key=key)
        self._types = types

        # Background event loop for the async Live session
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="gemini-live",
        )
        self._thread.start()

        # Session state
        self._session = None
        self._session_mgr = None
        self._connected = threading.Event()
        self._generating: Optional[asyncio.Lock] = None  # created lazily on bg loop

        # Connect on init
        self._ensure_connected()

    # ── Connection management ─────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._connected.is_set():
            return
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=30)

    async def _connect(self) -> None:
        types = self._types
        # gemini-3.1-flash-live-preview only supports AUDIO response modality;
        # we use output_audio_transcription to get text back.
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            media_resolution="MEDIA_RESOLUTION_LOW",
        )
        self._session_mgr = self._genai_client.aio.live.connect(
            model=self.model, config=config,
        )
        self._session = await self._session_mgr.__aenter__()
        self._connected.set()
        logger.info("Gemini Live session connected: %s", self.model)

    async def _disconnect(self) -> None:
        if self._session_mgr:
            try:
                await self._session_mgr.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
        self._session_mgr = None
        self._connected.clear()

    async def _reconnect(self) -> None:
        logger.info("Reconnecting Gemini Live session …")
        await self._disconnect()
        await self._connect()

    # ── Background frame streaming ────────────────────────────────────

    def send_frame(self, image: Image.Image) -> None:
        """Stream a video frame to the live session (non-blocking, ≤1 FPS).

        Call this from the capture loop so the model has continuous
        visual context between ``generate()`` calls.
        """
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()
        asyncio.run_coroutine_threadsafe(
            self._send_frame_async(jpeg_bytes), self._loop,
        )

    async def _send_frame_async(self, jpeg_bytes: bytes) -> None:
        types = self._types
        # Skip sending frames while a generate() call is active to avoid
        # mixing unsolicited model responses into the receive loop.
        if self._generating and self._generating.locked():
            return
        if self._session:
            try:
                await self._session.send_realtime_input(
                    video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg"),
                )
            except Exception as exc:
                logger.warning("Failed to stream frame to Live session: %s", exc)

    # ── generate() — drop-in replacement for GeminiClient ─────────────

    def generate(
        self,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._generate_async(prompt, images), self._loop,
        )
        return future.result(timeout=120)

    async def _generate_async(
        self, prompt: str, images: Optional[List[Image.Image]],
    ) -> str:
        try:
            return await self._do_generate(prompt, images)
        except Exception as exc:
            logger.warning("Live generate failed (%s), reconnecting …", exc)
            await self._reconnect()
            return await self._do_generate(prompt, images)

    async def _do_generate(
        self, prompt: str, images: Optional[List[Image.Image]],
    ) -> str:
        types = self._types

        if not self._generating:
            self._generating = asyncio.Lock()
        async with self._generating:
            # gemini-3.1 Live models only support send_realtime_input
            # (send_client_content returns 1007 invalid argument).
            # Send images as video frames first, then text prompt.
            n_images = len(images) if images else 0
            logger.debug("Live generate: sending %d images + text (%d chars)", n_images, len(prompt))
            for img in images or []:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                await self._session.send_realtime_input(
                    video=types.Blob(data=buf.getvalue(), mime_type="image/jpeg"),
                )

            # Small delay so the model registers the frames before the prompt
            if n_images:
                await asyncio.sleep(0.5)

            await self._session.send_realtime_input(text=prompt)

            # Collect transcribed text from audio response until turn_complete.
            # Use a timeout to avoid hanging indefinitely.
            collected: list[str] = []
            try:
                async with asyncio.timeout(60):
                    async for response in self._session.receive():
                        sc = response.server_content
                        if sc and sc.output_transcription and sc.output_transcription.text:
                            collected.append(sc.output_transcription.text)
                        if sc and getattr(sc, "turn_complete", False):
                            break
            except TimeoutError:
                logger.warning("Live generate timed out after 60s, returning partial response")

        result = "".join(collected)
        logger.debug("Live generate result (%d chars): %.200s", len(result), result)
        return result

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the live session and background event loop."""
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._disconnect(), self._loop,
            )
            try:
                future.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
        # Auto-detect Gemini Live models (e.g. gemini-3.1-flash-live-preview)
        if "live" in model.lower():
            return GeminiLiveClient(model=model, api_key=api_key)
        return GeminiClient(model=model, api_key=api_key)

    if backend in _OPENAI_BACKENDS:
        url = base_url or "http://localhost:8100/v1"
        key = api_key or "not-needed"
        return OpenAICompatibleClient(model=model, base_url=url, api_key=key)

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. "
        f"Choose from: gemini, {', '.join(sorted(_OPENAI_BACKENDS))}"
    )

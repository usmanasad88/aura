"""Kokoro-based text-to-speech for AURA.

Provides a lightweight, non-blocking speech output backed by the
`Kokoro <https://github.com/hexgrad/kokoro>`_ open-weight TTS model.
Used both as a first-class robot *action type* (``announce``) that the
decision engine can emit, and to vocalise robot commands when they are
dispatched in live mode.

Design notes
------------
* The Kokoro pipeline (and its torch weights) are **lazily loaded** on
  first use, so importing this module is cheap and never required.
* Synthesis + playback run on a background daemon thread by default, so
  ``speak()`` returns immediately and never blocks the LangGraph loop.
* If Kokoro (or an audio output device) is unavailable, the module logs
  a single warning and degrades to a no-op — speech is best-effort and
  must never crash the workflow.

Usage::

    from aura.interfaces.tts import speak
    speak("Moving the milk container to the working area.")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Kokoro emits 24 kHz mono float32 audio.
_KOKORO_SAMPLE_RATE = 24000


@dataclass
class TTSConfig:
    """Configuration for the Kokoro TTS engine."""

    # American-English by default. See Kokoro docs for other lang codes
    # ('a' American, 'b' British, 'e' Spanish, 'f' French, etc.).
    lang_code: str = "a"
    voice: str = "af_heart"
    speed: float = 1.0


class KokoroTTS:
    """Thin wrapper around the Kokoro ``KPipeline`` with audio playback.

    The pipeline is constructed lazily and reused across calls. All
    public methods are safe to call even when Kokoro is not installed —
    they become no-ops after logging one warning.
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        self._pipeline = None
        self._lock = threading.Lock()
        self._unavailable = False  # set once load fails → permanent no-op

    # ── Lazy construction ────────────────────────────────────────────

    def _ensure_pipeline(self) -> bool:
        """Build the Kokoro pipeline on first use. Returns availability."""
        if self._pipeline is not None:
            return True
        if self._unavailable:
            return False
        with self._lock:
            if self._pipeline is not None:
                return True
            if self._unavailable:
                return False
            try:
                from kokoro import KPipeline  # type: ignore

                self._pipeline = KPipeline(lang_code=self.config.lang_code)
                logger.info(
                    "Kokoro TTS ready (lang=%s, voice=%s)",
                    self.config.lang_code, self.config.voice,
                )
                return True
            except Exception as exc:  # ImportError or model load failure
                self._unavailable = True
                logger.warning(
                    "Kokoro TTS unavailable — speech disabled (%s). "
                    "Install with `uv add kokoro` (+ system espeak-ng).",
                    exc,
                )
                return False

    # ── Public API ───────────────────────────────────────────────────

    def speak(self, text: str, blocking: bool = False) -> None:
        """Speak *text*. Non-blocking by default (background thread)."""
        text = (text or "").strip()
        if not text:
            return
        if blocking:
            self._synthesize_and_play(text)
        else:
            threading.Thread(
                target=self._synthesize_and_play,
                args=(text,),
                name="kokoro-tts",
                daemon=True,
            ).start()

    # ── Internals ─────────────────────────────────────────────────────

    def _synthesize_and_play(self, text: str) -> None:
        if not self._ensure_pipeline():
            return
        try:
            import numpy as np
            import sounddevice as sd

            chunks = []
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.config.voice, speed=self.config.speed
            ):
                # ``audio`` is a torch tensor or numpy array of float32.
                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                chunks.append(arr.astype(np.float32, copy=False))

            if not chunks:
                return
            waveform = np.concatenate(chunks)
            # Serialise playback so overlapping announcements don't garble.
            with self._lock:
                sd.play(waveform, _KOKORO_SAMPLE_RATE)
                sd.wait()
        except Exception as exc:
            logger.warning("Kokoro TTS playback failed: %s", exc)


# ── Module-level singleton ───────────────────────────────────────────

_engine: Optional[KokoroTTS] = None
_engine_lock = threading.Lock()


def get_tts_engine() -> KokoroTTS:
    """Return the process-wide :class:`KokoroTTS` singleton."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = KokoroTTS()
    return _engine


def speak(text: str, blocking: bool = False) -> None:
    """Convenience: speak *text* via the shared Kokoro engine."""
    get_tts_engine().speak(text, blocking=blocking)

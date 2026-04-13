"""Audio-Workflow Bridge for AURA framework.

Provides thread-safe communication between the ``SoundMonitor``
(a persistent async Gemini Live session running in the background)
and the LangGraph workflow (which runs per-cycle node functions).

Architecture::

    SoundMonitor (background asyncio task)
        |
        | on_response / on_action callbacks
        v
    AudioWorkflowBridge (thread-safe queues)
        ^                    |
        | drain_*()          | push_speech()
        |                    v
    LangGraph nodes     SoundMonitor.send_text()
    (per-cycle)

The bridge holds:
- A queue of utterances (human + robot) for the SSG
- A queue of context messages for the decision engine
- A queue of action results for tracking
- A method to push text from the decision engine to the sound monitor

Usage in run_aura.py::

    bridge = AudioWorkflowBridge()
    # ... set up SoundMonitor with bridge callbacks ...
    # In the graph loop, nodes call:
    #   utterances = bridge.drain_utterances()
    #   bridge.push_speech("The resin is ready")
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aura.monitors.sound_monitor import SoundMonitor

logger = logging.getLogger(__name__)


@dataclass
class VoiceEvent:
    """A single event from the audio channel."""
    type: str          # "utterance" | "action" | "context" | "ssg_update"
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)


class AudioWorkflowBridge:
    """Thread-safe bridge between SoundMonitor and LangGraph nodes.

    The SoundMonitor runs in a background asyncio task and pushes
    events via callbacks. LangGraph nodes (running in the graph's
    event loop) drain events each cycle.

    This class is designed to be safe for concurrent access from
    the sound monitor's asyncio tasks and the LangGraph node
    functions (which may run in a different thread).
    """

    def __init__(self):
        # Thread-safe queues for monitor -> graph communication
        self._utterance_queue: queue.Queue[VoiceEvent] = queue.Queue()
        self._context_queue: queue.Queue[VoiceEvent] = queue.Queue()
        self._ssg_update_queue: queue.Queue[VoiceEvent] = queue.Queue()

        # Reference to the sound monitor (set after creation)
        self._sound_monitor: Optional["SoundMonitor"] = None
        self._sound_monitor_loop: Optional[asyncio.AbstractEventLoop] = None

        # Lock for sound monitor reference
        self._lock = threading.Lock()

    # ── Sound monitor registration ──────────────────────────────────

    def set_sound_monitor(
        self,
        monitor: "SoundMonitor",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Register the sound monitor instance.

        Args:
            monitor: The running SoundMonitor
            loop: The asyncio event loop the monitor runs in
                  (needed to schedule send_text from sync code)
        """
        with self._lock:
            self._sound_monitor = monitor
            self._sound_monitor_loop = loop

    # ── Callbacks (called by SoundMonitor / SkillActionBridge) ──────

    def on_human_utterance(self, text: str) -> None:
        """Called when the human says something."""
        self._utterance_queue.put(VoiceEvent(
            type="utterance",
            timestamp=datetime.now().isoformat(),
            data={"text": text, "speaker": "human"},
        ))

    def on_robot_utterance(self, text: str) -> None:
        """Called when the robot (Gemini) says something."""
        self._utterance_queue.put(VoiceEvent(
            type="utterance",
            timestamp=datetime.now().isoformat(),
            data={"text": text, "speaker": "robot"},
        ))

    def on_ssg_update(self, key: str, value: Any) -> None:
        """Called when human speech updates scene state."""
        self._ssg_update_queue.put(VoiceEvent(
            type="ssg_update",
            timestamp=datetime.now().isoformat(),
            data={"key": key, "value": value},
        ))

    def on_context_message(self, context: str) -> None:
        """Called when human speech provides context for decision engine."""
        self._context_queue.put(VoiceEvent(
            type="context",
            timestamp=datetime.now().isoformat(),
            data={"text": context},
        ))

    # ── Drain methods (called by LangGraph nodes each cycle) ────────

    def drain_utterances(self) -> List[VoiceEvent]:
        """Drain all pending utterances. Non-blocking."""
        events = []
        while True:
            try:
                events.append(self._utterance_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def drain_ssg_updates(self) -> List[VoiceEvent]:
        """Drain all pending SSG updates. Non-blocking."""
        events = []
        while True:
            try:
                events.append(self._ssg_update_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def drain_context_messages(self) -> List[VoiceEvent]:
        """Drain all pending context messages. Non-blocking."""
        events = []
        while True:
            try:
                events.append(self._context_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # ── Push methods (called by LangGraph nodes to talk to human) ───

    def push_speech(self, text: str) -> None:
        """Send text to the sound monitor for the robot to speak.

        This is how the decision engine communicates with the human.
        Safe to call from any thread — schedules the async send_text
        on the sound monitor's event loop.
        """
        with self._lock:
            monitor = self._sound_monitor
            loop = self._sound_monitor_loop

        if monitor is None:
            logger.debug("push_speech: no sound monitor registered, dropping: %s", text[:80])
            return

        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(monitor.send_text(text), loop)
        else:
            logger.debug("push_speech: event loop not running, dropping: %s", text[:80])

    def push_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> None:
        """Send an image to the sound monitor for visual context.

        Useful for giving Gemini Live the current camera frame.
        """
        with self._lock:
            monitor = self._sound_monitor
            loop = self._sound_monitor_loop

        if monitor is None or loop is None or not loop.is_running():
            return

        asyncio.run_coroutine_threadsafe(
            monitor.send_image(image_bytes, mime_type), loop
        )

    # ── State queries ───────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Whether a sound monitor is connected and presumably running."""
        with self._lock:
            return self._sound_monitor is not None

    @property
    def pending_count(self) -> int:
        """Total number of pending events across all queues."""
        return (
            self._utterance_queue.qsize()
            + self._context_queue.qsize()
            + self._ssg_update_queue.qsize()
        )

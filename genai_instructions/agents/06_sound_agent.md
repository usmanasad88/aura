# Agent 06: Sound Monitor Agent

## Task: Implement Sound Monitor with Gemini Live API

### Objective
Create a real-time sound monitor that uses Gemini Live API for continuous speech recognition and intent extraction.

### Prerequisites
- Sprint 1 complete
- GEMINI_API_KEY environment variable set
- PyAudio installed (`pip install pyaudio`)

### Reference Code
- `/home/mani/Repos/aura/gemini_live_test.py` - Working Gemini Live example

### Key Concepts

The Gemini Live API provides:
1. **Bidirectional audio streaming**: Send audio, receive audio/text responses
2. **Real-time transcription**: Speech to text
3. **Multimodal context**: Can process audio + images simultaneously
4. **Natural conversation**: Maintains context across turns

### Files to Create

#### 1. `src/aura/monitors/sound_monitor.py`

```python
"""Sound monitor for speech recognition and intent extraction."""

import os
import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from collections import deque

import pyaudio

from aura.core import (
    MonitorType, SoundOutput, Utterance
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig

# Lazy import for Gemini
_gemini_client = None

logger = logging.getLogger(__name__)


# Audio settings for Gemini Live
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


@dataclass
class SoundConfig(MonitorConfig):
    """Configuration for sound monitor."""
    model: str = "models/gemini-2.5-flash-native-audio-preview-12-2025"
    voice_name: str = "Zephyr"
    system_instruction: str = ""
    context_tokens: int = 25600
    target_tokens: int = 12800
    enable_speech_output: bool = True
    transcription_only: bool = False
    keywords_of_interest: List[str] = field(default_factory=list)
    

def _get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
    return _gemini_client


class SoundMonitor(BaseMonitor):
    """Sound monitor using Gemini Live API.
    
    Provides real-time speech recognition with optional:
    - Intent extraction
    - Command detection
    - Spoken responses
    
    Uses async event loop for non-blocking audio processing.
    """
    
    def __init__(
        self, 
        config: Optional[SoundConfig] = None,
        on_utterance: Optional[Callable[[Utterance], None]] = None,
        on_response: Optional[Callable[[str], None]] = None
    ):
        super().__init__(config or SoundConfig())
        self.config: SoundConfig = self.config
        
        self._pya = pyaudio.PyAudio()
        self._session = None
        self._audio_in_queue: Optional[asyncio.Queue] = None
        self._out_queue: Optional[asyncio.Queue] = None
        self._audio_stream = None
        self._output_stream = None
        
        self._recent_utterances: deque = deque(maxlen=50)
        self._is_listening = False
        self._tasks: List[asyncio.Task] = []
        
        # Callbacks
        self._on_utterance = on_utterance
        self._on_response = on_response
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.SOUND
    
    def _build_config(self):
        """Build Gemini Live config."""
        from google.genai import types
        
        modalities = ["AUDIO"] if self.config.enable_speech_output else ["TEXT"]
        
        return types.LiveConnectConfig(
            response_modalities=modalities,
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice_name
                    )
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=self.config.context_tokens,
                sliding_window=types.SlidingWindow(
                    target_tokens=self.config.target_tokens
                ),
            ),
            system_instruction=self.config.system_instruction or None,
        )
    
    async def _listen_audio(self):
        """Background task to capture microphone audio."""
        mic_info = self._pya.get_default_input_device_info()
        self._audio_stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        kwargs = {"exception_on_overflow": False}
        
        while self._is_listening:
            try:
                data = await asyncio.to_thread(
                    self._audio_stream.read, 
                    CHUNK_SIZE, 
                    **kwargs
                )
                await self._out_queue.put({
                    "data": data, 
                    "mime_type": "audio/pcm"
                })
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_realtime(self):
        """Background task to send audio/data to Gemini."""
        while self._is_listening:
            try:
                msg = await self._out_queue.get()
                await self._session.send(input=msg)
            except Exception as e:
                logger.error(f"Send error: {e}")
    
    async def _receive_audio(self):
        """Background task to receive responses from Gemini."""
        while self._is_listening:
            try:
                turn = self._session.receive()
                async for response in turn:
                    # Handle audio data
                    if data := response.data:
                        self._audio_in_queue.put_nowait(data)
                        continue
                    
                    # Handle text response
                    if text := response.text:
                        logger.debug(f"Received text: {text}")
                        
                        # Create utterance for transcribed speech or response
                        utterance = Utterance(
                            text=text,
                            speaker="gemini",
                            is_command=self._is_command(text)
                        )
                        self._recent_utterances.append(utterance)
                        
                        if self._on_response:
                            self._on_response(text)
                
                # Clear audio queue on interruption
                while not self._audio_in_queue.empty():
                    self._audio_in_queue.get_nowait()
                    
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)
    
    async def _play_audio(self):
        """Background task to play audio responses."""
        if not self.config.enable_speech_output:
            return
            
        self._output_stream = await asyncio.to_thread(
            self._pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while self._is_listening:
            try:
                bytestream = await self._audio_in_queue.get()
                await asyncio.to_thread(self._output_stream.write, bytestream)
            except Exception as e:
                logger.error(f"Playback error: {e}")
    
    def _is_command(self, text: str) -> bool:
        """Check if text contains command keywords."""
        if not self.config.keywords_of_interest:
            return False
        
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.config.keywords_of_interest)
    
    def _filter_relevant(self, utterances: List[Utterance]) -> List[Utterance]:
        """Filter utterances for relevance."""
        if not self.config.keywords_of_interest:
            return utterances
        
        return [u for u in utterances if u.is_command]
    
    async def start_listening(self):
        """Start continuous listening session."""
        if self._is_listening:
            logger.warning("Already listening")
            return
        
        logger.info("Starting sound monitor...")
        
        client = _get_gemini_client()
        config = self._build_config()
        
        self._audio_in_queue = asyncio.Queue()
        self._out_queue = asyncio.Queue(maxsize=5)
        self._is_listening = True
        
        # Start session
        self._session = await client.aio.live.connect(
            model=self.config.model,
            config=config
        ).__aenter__()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._listen_audio()),
            asyncio.create_task(self._send_realtime()),
            asyncio.create_task(self._receive_audio()),
        ]
        
        if self.config.enable_speech_output:
            self._tasks.append(asyncio.create_task(self._play_audio()))
        
        logger.info("Sound monitor started")
    
    async def stop_listening(self):
        """Stop listening session."""
        self._is_listening = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close streams
        if self._audio_stream:
            self._audio_stream.close()
        if self._output_stream:
            self._output_stream.close()
        if self._session:
            await self._session.__aexit__(None, None, None)
        
        logger.info("Sound monitor stopped")
    
    async def send_text(self, text: str, end_of_turn: bool = True):
        """Send text input to Gemini session.
        
        Args:
            text: Text to send
            end_of_turn: Whether this ends the user's turn
        """
        if not self._session:
            logger.warning("Session not started")
            return
        
        await self._session.send(input=text, end_of_turn=end_of_turn)
        
        # Record as utterance
        utterance = Utterance(
            text=text,
            speaker="user",
            is_command=self._is_command(text)
        )
        self._recent_utterances.append(utterance)
    
    async def send_image(self, image_bytes: bytes, mime_type: str = "image/jpeg"):
        """Send image to Gemini session for multimodal context.
        
        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of image
        """
        if not self._session:
            logger.warning("Session not started")
            return
        
        await self._session.send(input={
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode()
        })
    
    async def _process(self, **kwargs) -> SoundOutput:
        """Get recent utterances.
        
        This doesn't process audio - that's done by background tasks.
        This method returns accumulated utterances since last call.
        """
        utterances = list(self._recent_utterances)
        relevant = self._filter_relevant(utterances)
        
        return SoundOutput(
            utterances=utterances,
            is_relevant=len(relevant) > 0
        )
    
    def get_recent_commands(self) -> List[Utterance]:
        """Get recent utterances marked as commands."""
        return [u for u in self._recent_utterances if u.is_command]
    
    def clear_history(self):
        """Clear utterance history."""
        self._recent_utterances.clear()
```

#### 2. `scripts/test_sound_monitor.py`

```python
#!/usr/bin/env python
"""Test script for sound monitor with Gemini Live."""

import asyncio
import argparse
import signal
import sys

from aura.monitors.sound_monitor import SoundMonitor, SoundConfig


async def test_basic_conversation():
    """Test basic speech conversation."""
    print("=" * 50)
    print("Sound Monitor Test - Basic Conversation")
    print("=" * 50)
    print("\nSpeak into your microphone. Type 'q' to quit.")
    print("Type a message to send text to Gemini.\n")
    
    config = SoundConfig(
        enable_speech_output=True,
        keywords_of_interest=["robot", "move", "pick", "place", "stop", "help"]
    )
    
    received_texts = []
    
    def on_response(text: str):
        print(f"\n[Gemini]: {text}")
        received_texts.append(text)
    
    monitor = SoundMonitor(config=config, on_response=on_response)
    
    try:
        await monitor.start_listening()
        
        while True:
            # Non-blocking input check
            user_input = await asyncio.to_thread(
                input,
                "You (type 'q' to quit): "
            )
            
            if user_input.lower() == 'q':
                break
            
            if user_input.strip():
                await monitor.send_text(user_input)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await monitor.stop_listening()
        
    print(f"\nSession ended. Received {len(received_texts)} responses.")


async def test_command_detection():
    """Test command/keyword detection."""
    print("=" * 50)
    print("Sound Monitor Test - Command Detection")
    print("=" * 50)
    print("\nSay commands with keywords: robot, move, pick, place, stop")
    print("Press Ctrl+C to stop.\n")
    
    config = SoundConfig(
        enable_speech_output=False,  # Text only
        keywords_of_interest=["robot", "move", "pick", "place", "stop", "grab", "put"],
        system_instruction="""You are a robot command interpreter. 
When the user speaks, extract any robot commands and respond with:
- COMMAND: <action> <object> <location> if you detect a command
- UNCLEAR: <what's unclear> if the command is ambiguous
- NONE: if no command was given

Examples:
User: "Robot, pick up the red cup"
Response: COMMAND: pick red_cup none

User: "Move to the table"  
Response: COMMAND: move none table

User: "Hello, how are you?"
Response: NONE: greeting only
"""
    )
    
    commands_detected = []
    
    def on_response(text: str):
        print(f"[Interpreter]: {text}")
        if text.startswith("COMMAND:"):
            commands_detected.append(text)
            print(f"  → Command #{len(commands_detected)} detected!")
    
    monitor = SoundMonitor(config=config, on_response=on_response)
    
    try:
        await monitor.start_listening()
        
        # Run for fixed duration or until interrupt
        await asyncio.sleep(60)  # 1 minute test
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await monitor.stop_listening()
    
    print(f"\nDetected {len(commands_detected)} commands:")
    for cmd in commands_detected:
        print(f"  - {cmd}")


async def test_multimodal():
    """Test multimodal (audio + image) input."""
    import cv2
    from io import BytesIO
    from PIL import Image
    
    print("=" * 50)
    print("Sound Monitor Test - Multimodal (Audio + Vision)")
    print("=" * 50)
    print("\nWill capture webcam frames and send to Gemini.")
    print("Speak to describe what you see or give commands.")
    print("Press 'q' in the video window to quit.\n")
    
    config = SoundConfig(
        enable_speech_output=True,
        system_instruction="""You are assisting a user with a robotic task.
You can see images from their workspace. When they speak:
1. Describe what you see if asked
2. Identify objects they mention
3. Suggest actions the robot could take
Keep responses brief and helpful."""
    )
    
    monitor = SoundMonitor(config=config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    try:
        await monitor.start_listening()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send frame every 2 seconds
            frame_count += 1
            if frame_count % 60 == 0:  # Assuming ~30 FPS
                # Convert frame to JPEG bytes
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                buffer = BytesIO()
                pil_img.save(buffer, format='JPEG', quality=70)
                buffer.seek(0)
                
                await monitor.send_image(buffer.getvalue())
                print("[Sent frame to Gemini]")
            
            cv2.imshow("Workspace View", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.01)  # Yield to other tasks
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await monitor.stop_listening()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test sound monitor")
    parser.add_argument("--mode", type=str, default="conversation",
                       choices=["conversation", "commands", "multimodal"],
                       help="Test mode")
    
    args = parser.parse_args()
    
    if args.mode == "conversation":
        asyncio.run(test_basic_conversation())
    elif args.mode == "commands":
        asyncio.run(test_command_detection())
    elif args.mode == "multimodal":
        asyncio.run(test_multimodal())


if __name__ == "__main__":
    main()
```

### Validation

```bash
cd /home/mani/Repos/aura

# Test basic conversation
uv run python scripts/test_sound_monitor.py --mode conversation

# Test command detection
uv run python scripts/test_sound_monitor.py --mode commands

# Test multimodal
uv run python scripts/test_sound_monitor.py --mode multimodal
```

### Expected Behavior

1. Microphone captures audio continuously
2. Audio is streamed to Gemini Live API
3. Gemini responds with text/audio
4. Commands containing keywords are flagged
5. System instruction customizes Gemini's behavior

### Handoff Notes

Create `genai_instructions/handoff/06_sound_monitor.md`:

```markdown
# Sound Monitor Complete

## Files Created
- src/aura/monitors/sound_monitor.py
- scripts/test_sound_monitor.py

## Key Features
1. Real-time bidirectional audio with Gemini Live
2. Keyword-based command detection
3. Multimodal support (audio + images)
4. Customizable system instruction

## Dependencies
- pyaudio
- google-genai (v1beta API)

## Known Limitations
- Requires active internet connection
- Audio playback may have slight latency
- No local fallback for offline use

## Integration with Brain
The Brain should:
1. Start sound monitor at startup
2. Subscribe to SoundOutput events
3. Extract commands and update state
4. Use send_text() to respond to user
```

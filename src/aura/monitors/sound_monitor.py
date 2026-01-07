"""Sound monitor for speech recognition and intent extraction."""

import os
import asyncio
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from collections import deque

import pyaudio
import numpy as np
import scipy.signal

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
    
    # Audio Hardware Config
    input_device_name: Optional[str] = None  # Substring to match (e.g. "C310")
    output_device_name: Optional[str] = None # Substring to match (e.g. "Line Out")
    input_sample_rate: int = 16000 # Rate to capture at (resampled if needed)
    output_sample_rate: int = 48000 # Default output rate (will try to auto-detect)
    
    # Tool/Function Calling
    tools: Optional[List[Dict[str, Any]]] = None # Definitions of tools

    

def _get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=os.environ.get("GEMINI_API_KEY"),
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
        on_response: Optional[Callable[[str], None]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None
    ):
        super().__init__(config or SoundConfig())
        self._recent_utterances: deque = deque(maxlen=50)
        self._session = None
        self._session_ctx = None
        self._tool_handlers = tool_handlers or {}
        
        # Queues for async processing
        self._out_queue = asyncio.Queue()  # Format: {"data": bytes, "mime_type": str}
        self._audio_in_queue = asyncio.Queue()  # Format: bytes (PCM)
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
        
        # Audio interface
        self._pya = pyaudio.PyAudio()
        self._audio_stream_in = None
        self._audio_stream_out = None
        self._on_response = on_response

        # Device indices
        self._input_device_index = None
        self._output_device_index = None

    def _find_device_by_name(self, name_pattern: str, is_input: bool = True) -> Optional[int]:
        """Find audio device index by name pattern."""
        count = self._pya.get_device_count()
        logger.info(f"Scanning {count} audio devices for pattern '{name_pattern}'...")
        
        candidates = []
        for i in range(count):
            try:
                info = self._pya.get_device_info_by_index(i)
                dev_name = info.get('name', '')
                channels = info.get('maxInputChannels') if is_input else info.get('maxOutputChannels')
                
                if channels > 0 and name_pattern.lower() in dev_name.lower():
                    candidates.append((i, dev_name))
            except Exception:
                continue
                
        if not candidates:
            # Fallback for debugging
            logger.warning(f"No device found matching '{name_pattern}'. Available:")
            for i in range(count):
                try: 
                     info = self._pya.get_device_info_by_index(i)
                     logger.warning(f"  {i}: {info.get('name')}")
                except Exception:
                    pass
            return None
            
        # Return first match
        idx, name = candidates[0]
        logger.info(f"Selected device '{name}' (index {idx})")
        return idx

    def _resample(self, data: bytes, input_rate: int, output_rate: int) -> bytes:
        """Resample PCM audio data."""
        if input_rate == output_rate:
            return data
            
        # Convert to numpy array (int16)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate number of output samples
        num_samples = int(len(audio_data) * output_rate / input_rate)
        
        # Resample
        resampled_data = scipy.signal.resample(audio_data, num_samples)
        
        # Convert back to int16 bytes
        return resampled_data.astype(np.int16).tobytes()

    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.SOUND
    
    def _build_config(self):
        """Build Gemini Live config."""
        from google.genai import types
        
        # Speech config (voice)
        speech_config = None
        if self.config.enable_speech_output:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice_name
                    )
                )
            )
            
        # Context window config
        context_config = types.ContextWindowCompressionConfig(
            trigger_tokens=self.config.context_tokens,
            sliding_window=types.SlidingWindow(
                target_tokens=self.config.target_tokens
            ),
        )

        # Build Connect Config
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"] if self.config.enable_speech_output else ["TEXT"],
            system_instruction=types.Content(parts=[types.Part(text=self.config.system_instruction)]) if self.config.system_instruction else None,
            speech_config=speech_config,
            context_window_compression=context_config,
            tools=[types.Tool(function_declarations=self.config.tools)] if self.config.tools else None
        )
    
    async def _listen_audio(self):
        """Background task to capture microphone audio."""
        try:
            # Resolve input device
            input_idx = None
            if self.config.input_device_name:
                input_idx = self._find_device_by_name(self.config.input_device_name, is_input=True)
            
            if input_idx is None:
                # Fallback to default
                mic_info = self._pya.get_default_input_device_info()
                input_idx = mic_info["index"]
            
            self._input_device_index = input_idx
            
            # Use configured sample rate (e.g. 48000 if HW requires, or 16000 default)
            capture_rate = self.config.input_sample_rate
            
            logger.info(f"Opening input stream on device {input_idx} at {capture_rate}Hz")

            self._audio_stream_in = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=capture_rate,
                input=True,
                input_device_index=input_idx,
                frames_per_buffer=CHUNK_SIZE,
            )
            
            kwargs = {"exception_on_overflow": False}
            
            target_rate = SEND_SAMPLE_RATE # 16000
            
            while True:
                data = await asyncio.to_thread(
                    self._audio_stream_in.read, CHUNK_SIZE, **kwargs
                )
                
                # Resample if needed
                if capture_rate != target_rate:
                    data = await asyncio.to_thread(
                        self._resample, data, capture_rate, target_rate
                    )
                
                await self._out_queue.put({"data": data, "mime_type": "audio/pcm"})
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in listen_audio: {e}")
            if self._audio_stream_in:
                self._audio_stream_in.stop_stream()
                self._audio_stream_in.close()
    
    async def _send_realtime(self):
        """Background task to send audio/data to Gemini."""
        try:
            while True:
                msg = await self._out_queue.get()
                if self._session:
                    await self._session.send(input=msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in send_realtime: {e}")
    
    async def _handle_tool_call(self, tool_call):
        """Execute a tool call and return response."""
        logger.info(f"Tool call detected: {tool_call.function_calls}")
        from google.genai import types
        
        tool_responses = []
        
        for fc in tool_call.function_calls:
            name = fc.name
            args = fc.args
            logger.info(f"Executing tool {name} with args {args}")
            
            result = {"status": "error", "message": f"Tool {name} not found"}
            
            if name in self._tool_handlers:
                try:
                    # Execute handler (sync or async?)
                    # If async, we should await it. If sync, just run.
                    handler = self._tool_handlers[name]
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**args)
                    else:
                        result = await asyncio.to_thread(handler, **args)
                except Exception as e:
                    logger.error(f"Error executing tool {name}: {e}")
                    result = {"status": "error", "message": str(e)}
            
            tool_responses.append(
                types.FunctionResponse(
                    name=name,
                    id=fc.id,
                    response=result
                )
            )
        
        # NOTE: Live session expects a LIST of FunctionResponse, or a ToolResponse?
        # The docs say: session.send(input=[types.FunctionResponse(...)])
        return tool_responses

    async def _receive_audio(self):
        """Background task to receive responses from Gemini."""
        try:
            while True:
                async for response in self._session.receive():
                    if response.server_content is None:
                        # Check for tool call
                        if response.tool_call is not None:
                            tool_responses = await self._handle_tool_call(response.tool_call)
                            await self._session.send(input=tool_responses)
                        continue
                        
                    model_turn = response.server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            # Handle Audio
                            if part.inline_data:
                                self._audio_in_queue.put_nowait(part.inline_data.data)
                            
                            # Handle Text
                            if part.text:
                                text_msg = part.text
                                logger.info(f"Received text: {text_msg}")
                                # Create utterance
                                utterance = Utterance(
                                    text=text_msg,
                                    speaker="robot",
                                    timestamp=datetime.now(),
                                    is_command=False
                                )
                                self._recent_utterances.append(utterance)
                                
                                # Callback
                                if self._on_response:
                                    self._on_response(text_msg)
                            
                            # Handle Executable Code (if any)
                            if part.executable_code:
                                pass
                                
                    # Also handle turn_complete to flush audio if needed? 
                    # The example clears queue on interruption, but here we just stream.
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receive_audio: {e}")

    async def _play_audio(self):
        """Background task to play audio responses."""
        try:
            # Resolve output device
            output_idx = None
            if self.config.output_device_name:
                output_idx = self._find_device_by_name(self.config.output_device_name, is_input=False)
            
            # Auto-detect output rate from device info if not explicitly known?
            # Or use config.
            playback_rate = self.config.output_sample_rate
            
            # If we explicitly found a device, try to get its default rate
            if output_idx is not None:
                try:
                    info = self._pya.get_device_info_by_index(output_idx)
                    dev_rate = int(info.get('defaultSampleRate', 48000))
                    playback_rate = dev_rate
                except Exception:
                    pass
            else:
                 # Default device
                 try:
                    info = self._pya.get_default_output_device_info()
                    dev_rate = int(info.get('defaultSampleRate', 48000))
                    playback_rate = dev_rate
                 except Exception:
                    pass

            logger.info(f"Opening output stream on device {output_idx if output_idx is not None else 'Default'} at {playback_rate}Hz")
            
            self._audio_stream_out = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=playback_rate,
                output=True,
                output_device_index=output_idx
            )
            
            gemini_rate = RECEIVE_SAMPLE_RATE # 24000
            
            while True:
                data = await self._audio_in_queue.get()
                
                # Resample if needed (Up-sampling usually)
                if gemini_rate != playback_rate:
                    data = await asyncio.to_thread(
                        self._resample, data, gemini_rate, playback_rate
                    )
                
                await asyncio.to_thread(
                    self._audio_stream_out.write, data
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in play_audio: {e}")
            if self._audio_stream_out:
                self._audio_stream_out.stop_stream()
                self._audio_stream_out.close()
    
    def _is_command(self, text: str) -> bool:
        """Check if text contains command keywords."""
        if not self.config.keywords_of_interest:
            return False
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.config.keywords_of_interest)
    
    def _filter_relevant(self, utterances: List[Utterance]) -> List[Utterance]:
        """Filter utterances for relevance."""
        return [u for u in utterances if u.is_command]
    
    async def start_listening(self):
        """Start continuous listening session."""
        if self._session_ctx:
            await self.stop_listening()
        
        client = _get_gemini_client()
        gemini_config = self._build_config()
        
        # Connect to Gemini Live
        self._session_ctx = client.aio.live.connect(model=self.config.model, config=gemini_config)
        self._session = await self._session_ctx.__aenter__()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._listen_audio()),
            asyncio.create_task(self._send_realtime()),
            asyncio.create_task(self._receive_audio()),
            asyncio.create_task(self._play_audio())
        ]
        
        logger.info("Sound monitor started")
    
    async def stop_listening(self):
        """Stop listening session."""
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to finish (with timeout)
        if self._tasks:
            try:
                await asyncio.wait(self._tasks, timeout=0.1)
            except Exception:
                pass
        
        self._tasks.clear()
        
        # Close session
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
            
        # Clean up streams
        if self._audio_stream_in:
            self._audio_stream_in.stop_stream()
            self._audio_stream_in.close()
            self._audio_stream_in = None
            
        if self._audio_stream_out:
            self._audio_stream_out.stop_stream()
            self._audio_stream_out.close()
            self._audio_stream_out = None
            
        logger.info("Sound monitor stopped")
    
    async def send_text(self, text: str, end_of_turn: bool = True):
        """Send text input to Gemini session."""
        if not self._session:
            logger.warning("Sound Monitor not started")
            return
            
        await self._session.send(input=text, end_of_turn=end_of_turn)
        
        # Record our own utterance
        utterance = Utterance(
            text=text,
            speaker="human",
            timestamp=datetime.now(),
            is_command=self._is_command(text)
        )
        self._recent_utterances.append(utterance)
    
    async def send_image(self, image_bytes: bytes, mime_type: str = "image/jpeg"):
        """Send image to Gemini session for multimodal context.
        
        Args:
            image_bytes: Raw image data
            mime_type: MIME type of image (default: image/jpeg)
        """
        if not self._session:
            logger.warning("Sound Monitor not started")
            return
            
        await self._out_queue.put({
            "data": base64.b64encode(image_bytes).decode(), 
            "mime_type": mime_type
        })
    
    async def _process(self, **kwargs) -> SoundOutput:
        """Get recent utterances."""
        utterances = list(self._recent_utterances)
        
        return SoundOutput(
            monitor_type=MonitorType.SOUND,
            utterances=utterances,
            is_valid=True,
            is_relevant=len(utterances) > 0
        )
    
    def get_recent_commands(self) -> List[Utterance]:
        """Get recent utterances marked as commands."""
        return [u for u in self._recent_utterances if u.is_command]
    
    def clear_history(self):
        """Clear utterance history."""
        self._recent_utterances.clear()

# Sound Monitor Complete

## Files Created
- src/aura/monitors/sound_monitor.py: Enhanced with device selection, resampling, and **Tool Use (Brain Integration)**.
- scripts/test_sound_monitor.py: Added CLI args and a `tools` mode to test the Brain loop.

## Key Features
1. **Real-time Bidirectional Audio**: Gemini Live integration (Audio In -> Gemini -> Audio Out)
2. **Device Selection**: Can select input/output devices by name pattern (e.g. "USB", "C310", "Line Out")
3. **Automatic Resampling**:
   - Capture: Supports 48kHz/44.1kHz hardware, downsamples to 16kHz for Gemini
   - Playback: Upsamples Gemini's 24kHz output to hardware native rate (e.g. 48kHz/44.1kHz)
   - Uses `scipy.signal.resample` for high quality conversion
4. **Brain Integration (Tool Use)**:
   - Supports defining tools in `SoundConfig`.
   - Supports registering tool handlers (check `test_sound_monitor.py` for example).
   - Allows Brain to intercept execution before response is generated.
   - Example flow: User "Stop" -> Gemini Calls `stop_robot` -> Brain Halts Robot -> Brain Returns "Halted" -> Gemini Says "I have stopped the robot".

## Dependencies
- pyaudio
- google-genai
- scipy (for resampling)
- numpy

## Usage
```bash
# Basic Conversation with specific devices
uv run python scripts/test_sound_monitor.py --mode conversation --input-device "USB" --output-device "Analog" --rate 48000

# Brain Integration Test (Mock Brain)
uv run python scripts/test_sound_monitor.py --mode tools --input-device "USB" --output-device "Analog" --rate 48000
```

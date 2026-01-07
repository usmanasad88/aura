# Configuration System Complete

## Files Created
- `src/aura/utils/__init__.py`
- `src/aura/utils/config.py` (270 lines)
- `config/default.yaml` (enhanced with full structure)
- `config/game_demo.yaml` (game-specific settings)
- `tests/test_utils/test_config.py` (10 tests)

## Features
1. ✅ Pydantic-based type-safe configuration
2. ✅ YAML file loading with validation
3. ✅ Environment variable integration (GEMINI_API_KEY)
4. ✅ Nested config overrides (dot notation)
5. ✅ Automatic logging setup
6. ✅ Monitor-specific configs (Perception, Motion, Sound, Affordance)
7. ✅ Brain, Actions, Communication, Interface configs
8. ✅ Game demo specific config

## Usage
```python
from aura.utils.config import load_config

# Load default
config = load_config()

# Load specific file
config = load_config("config/game_demo.yaml")

# With overrides
config = load_config(overrides={"debug_mode": True})

# Access nested values
print(config.monitors.perception.max_objects)
# Output: 10
```

## Test Results
**10/10 tests passing ✅**

```bash
cd /home/mani/Repos/aura
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest tests/test_utils/test_config.py -v
```

## Configuration Structure

### Monitor Configs
Each monitor has:
- `enabled`: bool
- `update_rate_hz`: float
- `timeout_seconds`: float
- Plus monitor-specific settings

### Available Monitors
- **PerceptionConfig**: SAM3, Gemini detection, object tracking
- **MotionPredictorConfig**: Human pose, trajectory prediction
- **SoundMonitorConfig**: Gemini Live, wake words
- **AffordanceConfig**: LLM-based affordance detection

### Brain Config
- Decision model (Gemini)
- Reasoning depth (quick/standard/deep)
- SOP directory
- Update rates

### Other Configs
- **ActionConfig**: Robot execution settings
- **CommunicationConfig**: TTS, speech, digital twin
- **InterfaceConfig**: Game vs robot interface
- **LoggingConfig**: File logging, log rotation

## Integration with Monitors

Monitors can now use type-safe configs:

```python
from aura.utils.config import load_config

# Load global config
config = load_config("config/game_demo.yaml")

# Pass to monitor
from aura.monitors.perception_module import PerceptionModule
perception = PerceptionModule(config=config.monitors.perception)
```

## Next Steps
- ✅ Configuration system complete
- 🔄 Ready for Sprint 2: Perception Module (Task 2.1)
- All monitors can now use typed, validated configs

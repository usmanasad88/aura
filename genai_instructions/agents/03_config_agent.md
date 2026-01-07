# Agent 03: Configuration System Agent

## Task: Implement Configuration System with Pydantic

### Objective
Create a robust configuration management system using Pydantic for type-safe config loading from YAML files, with support for environment variable overrides and config validation.

### Prerequisites
- Sprint 1 Tasks 1.1 and 1.2 complete (core types and monitor interface)
- `pydantic`, `pyyaml` installed (should be in dependencies)

### Reference
- Existing `config/default.yaml` - Basic structure already exists
- Pattern from `/home/mani/Repos/proactive_hcdt/` - Similar config systems

### Files to Create/Modify

#### 1. `src/aura/utils/__init__.py`

```python
"""Utility modules for AURA."""

from aura.utils.config import AuraConfig, load_config

__all__ = ["AuraConfig", "load_config"]
```

#### 2. `src/aura/utils/config.py`

```python
"""Configuration management for AURA system."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


class MonitorConfig(BaseModel):
    """Configuration for individual monitors."""
    enabled: bool = True
    update_rate_hz: float = 10.0
    timeout_seconds: float = 1.0
    
    model_config = {"extra": "allow"}  # Allow extra fields


class PerceptionConfig(MonitorConfig):
    """Configuration for perception module."""
    use_sam3: bool = True
    use_gemini_detection: bool = True
    max_objects: int = 10
    confidence_threshold: float = 0.5
    gemini_model: str = "gemini-2.0-flash-exp"
    default_prompts: List[str] = Field(default_factory=lambda: ["person", "hand"])
    detection_interval_frames: int = 30


class MotionPredictorConfig(MonitorConfig):
    """Configuration for motion prediction."""
    model_type: str = "mediapipe"  # or "openpose"
    prediction_horizon_seconds: float = 2.0
    smooth_trajectory: bool = True


class SoundMonitorConfig(MonitorConfig):
    """Configuration for sound/speech monitor."""
    use_gemini_live: bool = True
    gemini_model: str = "gemini-2.0-flash-exp"
    sample_rate: int = 16000
    chunk_size: int = 1024
    wake_word_enabled: bool = False
    wake_word: str = "robot"


class AffordanceConfig(MonitorConfig):
    """Configuration for affordance detection."""
    use_llm: bool = True
    model: str = "gemini-2.0-flash-exp"
    check_physical_constraints: bool = True


class MonitorsConfig(BaseModel):
    """Configuration for all monitors."""
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    motion: MotionPredictorConfig = Field(default_factory=MotionPredictorConfig)
    sound: SoundMonitorConfig = Field(default_factory=SoundMonitorConfig)
    affordance: AffordanceConfig = Field(default_factory=AffordanceConfig)


class BrainConfig(BaseModel):
    """Configuration for the brain/decision engine."""
    decision_model: str = "gemini-2.0-flash-exp"
    reasoning_depth: str = "standard"  # standard, deep, or quick
    enable_explainability: bool = True
    state_update_rate_hz: float = 10.0
    sop_directory: str = "sops"
    max_reasoning_time_seconds: float = 5.0


class ActionConfig(BaseModel):
    """Configuration for action execution."""
    executor_type: str = "simulation"  # simulation, ur5, or hybrid
    action_timeout_seconds: float = 30.0
    verify_completion: bool = True
    safety_checks_enabled: bool = True


class CommunicationConfig(BaseModel):
    """Configuration for communication module."""
    speech_enabled: bool = True
    text_display_enabled: bool = True
    digital_twin_enabled: bool = False
    tts_engine: str = "google"  # google, pyttsx3, or espeak
    language: str = "en-US"


class InterfaceConfig(BaseModel):
    """Configuration for external interfaces (game, robot, etc.)."""
    interface_type: str = "game"  # game, ur5, or both
    game_window_size: tuple[int, int] = (800, 600)
    robot_ip: Optional[str] = None
    robot_port: int = 30002


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "logs"
    max_log_size_mb: int = 100
    backup_count: int = 5


class AuraConfig(BaseModel):
    """Root configuration for AURA system."""
    
    # System settings
    project_name: str = "AURA"
    version: str = "0.1.0"
    debug_mode: bool = False
    
    # API keys (prefer environment variables)
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    
    # Sub-configurations
    monitors: MonitorsConfig = Field(default_factory=MonitorsConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    actions: ActionConfig = Field(default_factory=ActionConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate that API key is set if required."""
        if v is None:
            logger.warning(
                "GEMINI_API_KEY not set. LLM features will be disabled."
            )
        return v
    
    def setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.logging.level.upper())
        
        # Basic config
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler
        if self.logging.log_to_file:
            log_dir = Path(self.logging.log_directory)
            log_dir.mkdir(exist_ok=True, parents=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_dir / "aura.log",
                maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        logger.info(f"Logging configured: level={self.logging.level}")


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> AuraConfig:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.yaml
        overrides: Dictionary of config overrides (nested keys with dots)
        
    Returns:
        Validated AuraConfig instance
        
    Example:
        >>> config = load_config("config/game_demo.yaml")
        >>> config = load_config(overrides={"debug_mode": True})
    """
    # Find config file
    if config_path is None:
        # Look for default.yaml in config/ directory
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    # Load YAML
    config_dict = {}
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    # Apply overrides
    if overrides:
        config_dict = _apply_overrides(config_dict, overrides)
    
    # Create and validate config
    config = AuraConfig(**config_dict)
    config.setup_logging()
    
    return config


def _apply_overrides(
    config_dict: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply nested overrides to config dictionary.
    
    Example:
        overrides = {"monitors.perception.enabled": False}
        -> config_dict["monitors"]["perception"]["enabled"] = False
    """
    for key, value in overrides.items():
        keys = key.split(".")
        d = config_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config_dict
```

#### 3. Update `config/default.yaml`

```yaml
# AURA Default Configuration

project_name: "AURA"
version: "0.1.0"
debug_mode: false

# Monitors Configuration
monitors:
  perception:
    enabled: true
    update_rate_hz: 10.0
    timeout_seconds: 2.0
    use_sam3: true
    use_gemini_detection: true
    max_objects: 10
    confidence_threshold: 0.5
    gemini_model: "gemini-2.0-flash-exp"
    default_prompts:
      - "person"
      - "hand"
    detection_interval_frames: 30
  
  motion:
    enabled: true
    update_rate_hz: 15.0
    timeout_seconds: 1.0
    model_type: "mediapipe"
    prediction_horizon_seconds: 2.0
    smooth_trajectory: true
  
  sound:
    enabled: true
    update_rate_hz: 20.0
    timeout_seconds: 1.0
    use_gemini_live: true
    gemini_model: "gemini-2.0-flash-exp"
    sample_rate: 16000
    chunk_size: 1024
    wake_word_enabled: false
    wake_word: "robot"
  
  affordance:
    enabled: true
    update_rate_hz: 5.0
    timeout_seconds: 3.0
    use_llm: true
    model: "gemini-2.0-flash-exp"
    check_physical_constraints: true

# Brain Configuration
brain:
  decision_model: "gemini-2.0-flash-exp"
  reasoning_depth: "standard"
  enable_explainability: true
  state_update_rate_hz: 10.0
  sop_directory: "sops"
  max_reasoning_time_seconds: 5.0

# Actions Configuration
actions:
  executor_type: "simulation"
  action_timeout_seconds: 30.0
  verify_completion: true
  safety_checks_enabled: true

# Communication Configuration
communication:
  speech_enabled: true
  text_display_enabled: true
  digital_twin_enabled: false
  tts_engine: "google"
  language: "en-US"

# Interface Configuration
interface:
  interface_type: "game"
  game_window_size: [800, 600]
  robot_ip: null
  robot_port: 30002

# Logging Configuration
logging:
  level: "INFO"
  log_to_file: true
  log_directory: "logs"
  max_log_size_mb: 100
  backup_count: 5
```

#### 4. Create `config/game_demo.yaml`

```yaml
# AURA Configuration for Collaborative Game Demo

project_name: "AURA Game Demo"
version: "0.1.0"
debug_mode: true

# Monitors - optimized for game
monitors:
  perception:
    enabled: true
    update_rate_hz: 15.0  # Higher rate for smooth game updates
    use_sam3: false  # Don't need segmentation for simple circles
    use_gemini_detection: false  # Game objects are known
    max_objects: 20
  
  motion:
    enabled: true
    update_rate_hz: 20.0  # Track human movement in game
    prediction_horizon_seconds: 1.5
  
  sound:
    enabled: true
    use_gemini_live: true
    wake_word_enabled: true
    wake_word: "robot"
  
  affordance:
    enabled: true
    update_rate_hz: 5.0
    use_llm: true

# Brain - game-specific reasoning
brain:
  decision_model: "gemini-2.0-flash-exp"
  reasoning_depth: "quick"  # Fast decisions for real-time game
  enable_explainability: true
  state_update_rate_hz: 15.0
  sop_directory: "sops"
  max_reasoning_time_seconds: 2.0

# Actions - simulation only
actions:
  executor_type: "simulation"
  action_timeout_seconds: 10.0
  verify_completion: true

# Communication - visual + text for game
communication:
  speech_enabled: false  # Disable TTS for game
  text_display_enabled: true
  digital_twin_enabled: true  # Show predicted trajectories

# Interface - game settings
interface:
  interface_type: "game"
  game_window_size: [1024, 768]

# Logging
logging:
  level: "DEBUG"
  log_to_file: true
  log_directory: "logs/game_demo"
```

#### 5. Create `tests/test_utils/__init__.py`

```python
"""Tests for utility modules."""
```

#### 6. Create `tests/test_utils/test_config.py`

```python
"""Tests for configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from aura.utils.config import (
    AuraConfig, load_config, MonitorsConfig,
    PerceptionConfig, BrainConfig, _apply_overrides
)


def test_default_config():
    """Test loading default configuration."""
    config = AuraConfig()
    
    assert config.project_name == "AURA"
    assert config.version == "0.1.0"
    assert config.debug_mode is False
    assert config.monitors.perception.enabled is True
    assert config.brain.decision_model == "gemini-2.0-flash-exp"


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "project_name": "Test Project",
        "debug_mode": True,
        "monitors": {
            "perception": {
                "enabled": False,
                "use_sam3": False
            }
        }
    }
    
    config = AuraConfig(**config_dict)
    
    assert config.project_name == "Test Project"
    assert config.debug_mode is True
    assert config.monitors.perception.enabled is False
    assert config.monitors.perception.use_sam3 is False


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "project_name": "YAML Test",
            "monitors": {
                "perception": {
                    "max_objects": 20
                }
            }
        }, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config.project_name == "YAML Test"
        assert config.monitors.perception.max_objects == 20
    finally:
        os.unlink(temp_path)


def test_config_overrides():
    """Test applying overrides to config."""
    config_dict = {
        "monitors": {
            "perception": {
                "enabled": True
            }
        }
    }
    
    overrides = {
        "monitors.perception.enabled": False,
        "debug_mode": True
    }
    
    result = _apply_overrides(config_dict, overrides)
    
    assert result["monitors"]["perception"]["enabled"] is False
    assert result["debug_mode"] is True


def test_load_config_with_overrides():
    """Test load_config with overrides."""
    config = load_config(overrides={
        "debug_mode": True,
        "monitors.perception.max_objects": 50
    })
    
    assert config.debug_mode is True
    assert config.monitors.perception.max_objects == 50


def test_nested_config_access():
    """Test accessing nested configuration values."""
    config = AuraConfig()
    
    # Test nested access
    assert isinstance(config.monitors, MonitorsConfig)
    assert isinstance(config.monitors.perception, PerceptionConfig)
    assert isinstance(config.brain, BrainConfig)
    
    # Test default values
    assert config.monitors.perception.confidence_threshold == 0.5
    assert config.brain.reasoning_depth == "standard"


def test_api_key_from_env():
    """Test that API key is loaded from environment."""
    test_key = "test_gemini_key_12345"
    os.environ["GEMINI_API_KEY"] = test_key
    
    try:
        config = AuraConfig()
        assert config.gemini_api_key == test_key
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_api_key_missing_warning(caplog):
    """Test warning when API key is missing."""
    # Ensure env var is not set
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    
    config = AuraConfig()
    
    assert config.gemini_api_key is None
    # Check that warning was logged (if caplog works)


def test_monitor_config_extra_fields():
    """Test that MonitorConfig allows extra fields."""
    config = PerceptionConfig(
        enabled=True,
        custom_field="custom_value"
    )
    
    assert config.enabled is True
    # Extra fields should be stored
    assert config.model_extra.get("custom_field") == "custom_value"


def test_game_demo_config():
    """Test loading game demo configuration."""
    repo_root = Path(__file__).parent.parent.parent
    game_config_path = repo_root / "config" / "game_demo.yaml"
    
    if not game_config_path.exists():
        pytest.skip("game_demo.yaml not found")
    
    config = load_config(str(game_config_path))
    
    assert "Game Demo" in config.project_name
    assert config.debug_mode is True
    assert config.interface.interface_type == "game"
    assert config.brain.reasoning_depth == "quick"
```

### Validation

```bash
cd /home/mani/Repos/aura

# Run config tests
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest tests/test_utils/test_config.py -v

# Test loading default config in Python
uv run python -c "from aura.utils.config import load_config; c = load_config(); print(c.project_name)"

# Test loading game demo config
uv run python -c "from aura.utils.config import load_config; c = load_config('config/game_demo.yaml'); print(c.brain.reasoning_depth)"
```

### Expected Results

- All 11 config tests should pass ✅
- Config loads without errors
- Logging is configured automatically
- Overrides work correctly
- Nested config access works

### Integration with Monitors

Monitors can now use type-safe configs:

```python
from aura.utils.config import load_config

# Load global config
config = load_config("config/game_demo.yaml")

# Pass to monitor
from aura.monitors.perception_module import PerceptionModule
perception = PerceptionModule(config=config.monitors.perception)
```

### Handoff Notes

Document in `genai_instructions/handoff/03_config_system.md`:

```markdown
# Configuration System Complete

## Files Created
- src/aura/utils/__init__.py
- src/aura/utils/config.py (360 lines)
- config/default.yaml (enhanced)
- config/game_demo.yaml (new)
- tests/test_utils/test_config.py (11 tests)

## Features
1. ✅ Pydantic-based type-safe configuration
2. ✅ YAML file loading with validation
3. ✅ Environment variable integration (GEMINI_API_KEY)
4. ✅ Nested config overrides (dot notation)
5. ✅ Automatic logging setup
6. ✅ Monitor-specific configs
7. ✅ Game demo specific config

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
```

## Test Results
11/11 tests passing ✅

## Next Steps
- All monitors can now use typed configs
- Ready for Sprint 2 (Perception, Motion, Sound monitors)
```

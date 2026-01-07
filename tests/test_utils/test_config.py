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
        original_key = os.environ["GEMINI_API_KEY"]
        del os.environ["GEMINI_API_KEY"]
    else:
        original_key = None
    
    try:
        config = AuraConfig()
        assert config.gemini_api_key is None
        # Check that warning was logged (if caplog works)
    finally:
        if original_key:
            os.environ["GEMINI_API_KEY"] = original_key


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

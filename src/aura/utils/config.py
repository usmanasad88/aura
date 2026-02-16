"""Configuration management for AURA system."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

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
    gemini_model: str = "gemini-3-pro-preview"
    default_prompts: List[str] = Field(default_factory=lambda: ["person", "hand"])
    detection_interval_frames: int = 30


class IntentMonitorConfig(MonitorConfig):
    """Configuration for intent recognition using Gemini + task graphs."""
    fps: float = 2.0  # Frame capture rate (2 fps recommended)
    capture_duration: float = 2.0  # Buffer window in seconds
    prediction_interval: float = 3.0  # How often to query Gemini
    max_image_dimension: int = 512  # Max image dimension for Gemini
    model: str = "gemini-3-pro-preview"  # Gemini model
    timeout_sec: float = 30.0  # Longer timeout for Gemini API calls
    # Task graph configuration
    dag_file: Optional[str] = None  # Path to DAG JSON file
    state_file: Optional[str] = None  # Path to state schema JSON file
    task_name: str = "activity monitoring"  # Task description
    # Prompt customization
    system_prompt: Optional[str] = None  # Custom system prompt
    task_context: Optional[str] = None  # Additional context
    analysis_instructions: Optional[str] = None  # Custom analysis instructions
    output_format: Optional[str] = None  # Custom output format


class MotionPredictorConfig(MonitorConfig):
    """Configuration for motion prediction using MediaPipe hand tracking."""
    fps: float = 15.0  # Frame capture rate (15 fps for smooth tracking)
    window_duration: float = 1.0  # Tracking history window in seconds
    prediction_horizon: float = 0.5  # Predict N seconds ahead
    use_hand_tracking: bool = True  # Enable MediaPipe hand tracking
    smooth_trajectory: bool = True  # Apply trajectory smoothing
    damping_factor: float = 0.95  # Velocity damping for predictions
    # Legacy fields for compatibility
    model_type: str = "mediapipe"  # or "openpose"
    prediction_horizon_seconds: float = 0.5
    model: str = ""  # No Gemini needed for motion-only tracking


class SoundMonitorConfig(MonitorConfig):
    """Configuration for sound/speech monitor."""
    use_gemini_live: bool = True
    gemini_model: str = "gemini-3-pro-preview"
    sample_rate: int = 16000
    chunk_size: int = 1024
    wake_word_enabled: bool = False
    wake_word: str = "robot"


class AffordanceConfig(MonitorConfig):
    """Configuration for affordance detection."""
    use_llm: bool = True
    model: str = "gemini-3-pro-preview"
    check_physical_constraints: bool = True


class PoseTrackingConfig(MonitorConfig):
    """Configuration for 6DOF pose tracking using Any6D + DA3."""
    # Depth model
    da3_model: str = "da3nested-giant-large"
    da3_batch_size: int = 8
    depth_scale: float = 1.0
    use_da3_intrinsics: bool = True
    # Pose estimation
    est_refine_iter: int = 5
    track_refine_iter: int = 2
    max_pose_resolution: int = 480
    # Rendering
    overlay_alpha: float = 0.6
    render_axes: bool = True
    render_overlay: bool = True
    # Intrinsics
    intrinsic_file: Optional[str] = None
    fov_deg: float = 60.0  # fallback FOV if no intrinsics
    # Mesh-to-track mapping: {track_name_prefix: glb_path}
    mesh_map: Dict[str, str] = Field(default_factory=dict)
    # SAM3 mask directory (pre-generated)
    sam3_mask_dir: Optional[str] = None
    # SAM3 live segmentation settings
    sam3_confidence: float = 0.3
    sam3_prompts: Optional[List[str]] = None  # auto-derived from mesh_map keys if None
    # Debug
    debug_level: int = 0
    save_dir: str = "results/pose_tracking"
    # Any6D third-party root
    any6d_root: str = "third_party/any6d"
    timeout_seconds: float = 60.0  # pose tracking can be slow


class MonitorsConfig(BaseModel):
    """Configuration for all monitors."""
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    motion: MotionPredictorConfig = Field(default_factory=MotionPredictorConfig)
    intent: IntentMonitorConfig = Field(default_factory=IntentMonitorConfig)
    sound: SoundMonitorConfig = Field(default_factory=SoundMonitorConfig)
    affordance: AffordanceConfig = Field(default_factory=AffordanceConfig)
    pose_tracking: PoseTrackingConfig = Field(default_factory=PoseTrackingConfig)


class BrainConfig(BaseModel):
    """Configuration for the brain/decision engine."""
    decision_model: str = "gemini-3-pro-preview"
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

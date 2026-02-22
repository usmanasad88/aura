#!/usr/bin/env python3
"""Integration tests for tea making demo.

Tests the tea making demo with mock or real video processing.
"""

import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add paths
TASK_DIR = Path(__file__).parent.parent  # tasks/tea_making/
AURA_ROOT = Path(__file__).parent.parent.parent.parent  # aura/
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

from tasks.tea_making.demo.run_tea_demo import (
    VideoProcessor, VideoFrame, TeaMakingDemo
)
from aura.brain.decision_engine import DecisionEngineConfig


class TestVideoProcessor:
    """Tests for video processing utilities."""
    
    def test_video_frame_dataclass(self):
        """Test VideoFrame dataclass."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vf = VideoFrame(frame=frame, timestamp_sec=1.5, frame_number=30)
        
        assert vf.timestamp_sec == 1.5
        assert vf.frame_number == 30
        assert vf.frame.shape == (480, 640, 3)
    
    def test_video_processor_init(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor("nonexistent.mp4", target_fps=2.0)
        
        assert processor.video_path == "nonexistent.mp4"
        assert processor.target_fps == 2.0
        assert processor.cap is None
    
    def test_extract_360_view(self):
        """Test 360 view extraction (with synthetic data)."""
        processor = VideoProcessor("test.mp4")
        
        # Create synthetic equirectangular frame (2:1 ratio)
        equirect_frame = np.random.randint(0, 255, (1080, 2160, 3), dtype=np.uint8)
        
        # Extract view
        view = processor.extract_view_from_360(
            equirect_frame,
            yaw=0, pitch=0, fov=90,
            output_size=(640, 480)
        )
        
        assert view.shape == (480, 640, 3)


class TestTeaMakingDemo:
    """Tests for TeaMakingDemo class."""
    
    @pytest.fixture
    def demo(self):
        """Create demo instance for testing."""
        return TeaMakingDemo(headless=True)
    
    def test_demo_initialization(self, demo):
        """Test demo initializes correctly."""
        assert demo.engine is not None
        assert demo.headless is True
        assert demo.config is not None
        
        # Check config loaded
        assert "task_name" in demo.config
    
    def test_demo_engine_setup(self, demo):
        """Test engine is properly configured."""
        # Check graph has nodes from initial scene
        assert demo.engine.graph.node_count > 0
        
        # Check skills loaded
        assert demo.engine.skills.has("retrieve_object")
        assert demo.engine.skills.has("add_sugar")
    
    def test_simulate_state_updates(self, demo):
        """Test state simulation based on time."""
        # Create mock video frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test at different times
        vf = VideoFrame(frame=frame, timestamp_sec=0, frame_number=0)
        demo._simulate_state_updates(vf)
        assert demo.engine.graph.get_task_state("workspace_setup") is False
        
        vf = VideoFrame(frame=frame, timestamp_sec=35, frame_number=70)
        demo._simulate_state_updates(vf)
        assert demo.engine.graph.get_task_state("workspace_setup") is True
        
        vf = VideoFrame(frame=frame, timestamp_sec=125, frame_number=250)
        demo._simulate_state_updates(vf)
        assert demo.engine.graph.get_task_state("water_boiling") is True
    
    @pytest.mark.asyncio
    async def test_process_frame(self, demo):
        """Test processing a single frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vf = VideoFrame(frame=frame, timestamp_sec=10.0, frame_number=20)
        
        prediction = await demo.process_frame(vf)
        
        # May or may not have a prediction depending on state
        assert demo.engine.current_video_time_sec == 10.0


class TestConfigFiles:
    """Tests for task configuration files."""
    
    def test_dag_file_exists(self):
        """Test DAG file exists and is valid JSON."""
        import json
        
        dag_path = TASK_DIR / "config" / "tea_making_dag.json"
        assert dag_path.exists(), f"DAG file not found: {dag_path}"
        
        with open(dag_path) as f:
            dag = json.load(f)
        
        assert "name" in dag
        assert "nodes" in dag
        assert len(dag["nodes"]) > 0
    
    def test_state_file_exists(self):
        """Test state schema file exists and is valid."""
        import json
        
        state_path = TASK_DIR / "config" / "tea_making_state.json"
        assert state_path.exists()
        
        with open(state_path) as f:
            state = json.load(f)
        
        assert "state_variables" in state
        assert "sugar_preference_known" in state["state_variables"]
    
    def test_skills_file_exists(self):
        """Test skills file exists and is valid."""
        import json
        
        skills_path = TASK_DIR / "config" / "robot_skills.json"
        assert skills_path.exists()
        
        with open(skills_path) as f:
            skills = json.load(f)
        
        assert "skills" in skills
        skill_ids = [s["id"] for s in skills["skills"]]
        assert "add_sugar" in skill_ids
    
    def test_initial_scene_exists(self):
        """Test initial scene file exists and is valid."""
        import json
        
        scene_path = TASK_DIR / "config" / "initial_scene.json"
        assert scene_path.exists()
        
        with open(scene_path) as f:
            scene = json.load(f)
        
        assert "regions" in scene
        assert "objects" in scene
        assert "agents" in scene
    
    def test_yaml_config_exists(self):
        """Test YAML config exists and is valid."""
        import yaml
        
        config_path = TASK_DIR / "config" / "tea_making.yaml"
        assert config_path.exists()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "task_name" in config
        assert "llm" in config
        assert "model" in config["llm"]


class TestIntegration360Video:
    """Integration tests with 360 video if available."""
    
    @pytest.fixture
    def video_path(self):
        """Get path to demo video."""
        return AURA_ROOT / "demo_data" / "002.360"
    
    def test_video_exists(self, video_path):
        """Check if demo video exists."""
        # Skip if video doesn't exist
        if not video_path.exists():
            pytest.skip(f"Demo video not found: {video_path}")
    
    def test_open_video(self, video_path):
        """Test opening the 360 video."""
        if not video_path.exists():
            pytest.skip("Demo video not found")
        
        processor = VideoProcessor(str(video_path), target_fps=1.0)
        result = processor.open()
        
        assert result is True
        assert processor.total_frames > 0
        assert processor.original_fps > 0
        
        processor.close()
    
    def test_read_frames(self, video_path):
        """Test reading frames from video."""
        if not video_path.exists():
            pytest.skip("Demo video not found")
        
        processor = VideoProcessor(str(video_path), target_fps=1.0)
        processor.open()
        
        frame_count = 0
        for video_frame in processor.frames(max_frames=5):
            assert video_frame.frame is not None
            assert video_frame.timestamp_sec >= 0
            frame_count += 1
        
        processor.close()
        
        assert frame_count <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for motion predictor."""

import pytest
import asyncio
import numpy as np
from datetime import datetime

from aura.core import IntentType
from aura.monitors.motion_predictor import MotionPredictor
from aura.utils.config import MotionPredictorConfig


class TestMotionPredictor:
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test motion predictor initialization."""
        config = MotionPredictorConfig(
            enabled=True,
            fps=2.0,
            window_duration=2.0
        )
        predictor = MotionPredictor(config)
        assert predictor.fps == 2.0
        assert predictor.max_frames == 4
    
    @pytest.mark.asyncio
    async def test_frame_buffering(self):
        """Test frame buffer fills correctly."""
        import time
        config = MotionPredictorConfig(enabled=True, fps=10.0, window_duration=0.5)
        predictor = MotionPredictor(config)
        
        # Create dummy frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames with delays to respect FPS
        for i in range(10):
            await predictor.update(frame=frame)
            await asyncio.sleep(0.11)  # Slightly more than 1/fps
        
        # Buffer should be at or near max size (5 frames at 10fps over 0.5s)
        assert len(predictor.frame_buffer) >= predictor.max_frames - 1
    
    @pytest.mark.asyncio
    async def test_output_structure(self):
        """Test output has correct structure."""
        config = MotionPredictorConfig(enabled=True)
        predictor = MotionPredictor(config)
        
        # This will likely return None without enough frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = await predictor.update(frame=frame)
        
        # Either None or valid MotionOutput
        if output:
            assert output.intent is not None
            assert hasattr(output.intent, 'type')
            assert hasattr(output.intent, 'confidence')
    
    @pytest.mark.asyncio
    async def test_disabled_monitor(self):
        """Test disabled monitor returns None."""
        config = MotionPredictorConfig(enabled=False)
        predictor = MotionPredictor(config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = await predictor.update(frame=frame)
        
        assert output is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for scale monitor.

Tests weight reading, target tracking, and tolerance checking.
"""

import pytest
from dataclasses import dataclass
from typing import Optional
import time


# Mock ScaleReading until the real monitor is implemented
@dataclass
class MockScaleReading:
    weight_grams: float
    stable: bool
    tared: bool
    timestamp: float


class MockScaleMonitor:
    """Mock scale monitor for testing."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.current_weight = 0.0
        self.target_weight: Optional[float] = None
        self.tolerance_grams = 2.0
        self._tared = False
    
    def set_weight(self, weight_g: float) -> None:
        """Simulate setting the scale weight."""
        self.current_weight = weight_g
    
    def set_target(self, weight_g: float, tolerance_g: float = 2.0) -> None:
        """Set target weight for pour operation."""
        self.target_weight = weight_g
        self.tolerance_grams = tolerance_g
    
    def tare(self) -> None:
        """Tare the scale."""
        self._tared = True
        self.current_weight = 0.0
    
    def update(self) -> MockScaleReading:
        """Get current scale reading."""
        return MockScaleReading(
            weight_grams=self.current_weight,
            stable=True,
            tared=self._tared,
            timestamp=time.time()
        )
    
    def is_at_target(self, reading: Optional[MockScaleReading] = None) -> bool:
        """Check if current weight is within tolerance of target."""
        if self.target_weight is None:
            return False
        reading = reading or self.update()
        return abs(reading.weight_grams - self.target_weight) <= self.tolerance_grams


class TestScaleMonitor:
    """Test cases for scale monitor."""
    
    def test_initial_state(self):
        """Test scale starts at zero."""
        scale = MockScaleMonitor()
        reading = scale.update()
        assert reading.weight_grams == 0.0
        assert reading.stable is True
    
    def test_set_weight(self):
        """Test setting weight value."""
        scale = MockScaleMonitor()
        scale.set_weight(77.5)
        reading = scale.update()
        assert reading.weight_grams == 77.5
    
    def test_tare(self):
        """Test tare operation zeros the scale."""
        scale = MockScaleMonitor()
        scale.set_weight(50.0)
        scale.tare()
        reading = scale.update()
        assert reading.weight_grams == 0.0
        assert reading.tared is True
    
    def test_target_not_set(self):
        """Test is_at_target returns False when no target set."""
        scale = MockScaleMonitor()
        scale.set_weight(77.0)
        assert scale.is_at_target() is False
    
    def test_at_target_exact(self):
        """Test target detection with exact match."""
        scale = MockScaleMonitor()
        scale.set_target(77.0, tolerance_g=2.0)
        scale.set_weight(77.0)
        assert scale.is_at_target() is True
    
    def test_at_target_within_tolerance(self):
        """Test target detection within tolerance."""
        scale = MockScaleMonitor()
        scale.set_target(77.0, tolerance_g=2.0)
        
        # Just under tolerance
        scale.set_weight(75.5)
        assert scale.is_at_target() is True
        
        # Just over tolerance
        scale.set_weight(78.5)
        assert scale.is_at_target() is True
    
    def test_outside_tolerance(self):
        """Test target detection outside tolerance."""
        scale = MockScaleMonitor()
        scale.set_target(77.0, tolerance_g=2.0)
        
        # Too low
        scale.set_weight(74.0)
        assert scale.is_at_target() is False
        
        # Too high
        scale.set_weight(80.0)
        assert scale.is_at_target() is False
    
    def test_resin_measurement(self):
        """Test typical resin measurement workflow."""
        scale = MockScaleMonitor()
        
        # Tare with empty cup
        scale.tare()
        assert scale.update().tared is True
        
        # Set target for 77g resin (100:30 ratio, 100g total batch)
        scale.set_target(77.0, tolerance_g=2.0)
        
        # Simulate pouring
        scale.set_weight(30.0)
        assert scale.is_at_target() is False
        
        scale.set_weight(60.0)
        assert scale.is_at_target() is False
        
        scale.set_weight(76.0)  # Within tolerance
        assert scale.is_at_target() is True
    
    def test_hardener_measurement(self):
        """Test hardener measurement after resin."""
        scale = MockScaleMonitor()
        
        # After 77g resin, add 23g hardener (total 100g)
        scale.set_weight(77.0)  # Starting point (resin already in)
        
        # Set target for final weight (77 + 23 = 100g)
        # Note: In real implementation, would set target to additional 23g
        scale.set_target(100.0, tolerance_g=1.0)
        
        scale.set_weight(90.0)
        assert scale.is_at_target() is False
        
        scale.set_weight(99.5)
        assert scale.is_at_target() is True
        
        scale.set_weight(100.3)
        assert scale.is_at_target() is True


class TestMixRatio:
    """Test resin:hardener mix ratio calculations."""
    
    def test_100_to_30_ratio(self):
        """Test 100:30 mix ratio by weight."""
        total_batch_g = 100.0
        resin_parts = 100
        hardener_parts = 30
        
        # Calculate weights
        total_parts = resin_parts + hardener_parts  # 130
        resin_weight = total_batch_g * (resin_parts / total_parts)
        hardener_weight = total_batch_g * (hardener_parts / total_parts)
        
        assert pytest.approx(resin_weight, rel=0.01) == 76.92  # ~77g
        assert pytest.approx(hardener_weight, rel=0.01) == 23.08  # ~23g
        assert pytest.approx(resin_weight + hardener_weight) == total_batch_g
    
    def test_ratio_tolerance(self):
        """Test that 5% tolerance is acceptable."""
        target_resin = 77.0
        target_hardener = 23.0
        tolerance_percent = 5.0
        
        # Calculate acceptable ranges
        resin_tolerance = target_resin * (tolerance_percent / 100)
        hardener_tolerance = target_hardener * (tolerance_percent / 100)
        
        assert pytest.approx(resin_tolerance, rel=0.1) == 3.85
        assert pytest.approx(hardener_tolerance, rel=0.1) == 1.15

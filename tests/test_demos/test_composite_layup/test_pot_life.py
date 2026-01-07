"""Tests for pot life monitor.

Tests time tracking and warning thresholds for epoxy working life.
"""

import pytest
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class MockPotLifeStatus:
    mix_start_time: float
    elapsed_seconds: float
    remaining_seconds: float
    warning_level: str  # "ok", "warning", "critical", "expired"
    pot_life_total_seconds: float


class MockPotLifeMonitor:
    """Mock pot life monitor for testing."""
    
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.pot_life_seconds = config.get("pot_life_minutes", 45) * 60
        self.warning_threshold = config.get("warning_minutes", 30) * 60
        self.critical_threshold = config.get("critical_minutes", 40) * 60
        self.mix_start_time: Optional[float] = None
    
    def start_timer(self) -> None:
        """Start the pot life timer."""
        self.mix_start_time = time.time()
    
    def set_elapsed(self, elapsed_seconds: float) -> None:
        """For testing: set elapsed time directly."""
        self.mix_start_time = time.time() - elapsed_seconds
    
    def update(self) -> Optional[MockPotLifeStatus]:
        """Get current pot life status."""
        if self.mix_start_time is None:
            return None
        
        elapsed = time.time() - self.mix_start_time
        remaining = max(0, self.pot_life_seconds - elapsed)
        
        if remaining <= 0:
            level = "expired"
        elif elapsed >= self.critical_threshold:
            level = "critical"
        elif elapsed >= self.warning_threshold:
            level = "warning"
        else:
            level = "ok"
        
        return MockPotLifeStatus(
            mix_start_time=self.mix_start_time,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            warning_level=level,
            pot_life_total_seconds=self.pot_life_seconds
        )
    
    def is_expired(self) -> bool:
        """Check if pot life has expired."""
        status = self.update()
        return status is not None and status.warning_level == "expired"


class TestPotLifeMonitor:
    """Test cases for pot life monitoring."""
    
    def test_not_started(self):
        """Test status is None before timer starts."""
        monitor = MockPotLifeMonitor()
        assert monitor.update() is None
    
    def test_start_timer(self):
        """Test timer starts correctly."""
        monitor = MockPotLifeMonitor()
        monitor.start_timer()
        status = monitor.update()
        
        assert status is not None
        assert status.elapsed_seconds < 1.0
        assert status.warning_level == "ok"
    
    def test_ok_level(self):
        """Test ok status within safe period."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(10 * 60)  # 10 minutes
        
        status = monitor.update()
        assert status.warning_level == "ok"
        assert status.remaining_seconds > 30 * 60
    
    def test_warning_level(self):
        """Test warning at 30 minute threshold."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(31 * 60)  # 31 minutes
        
        status = monitor.update()
        assert status.warning_level == "warning"
    
    def test_critical_level(self):
        """Test critical at 40 minute threshold."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(41 * 60)  # 41 minutes
        
        status = monitor.update()
        assert status.warning_level == "critical"
    
    def test_expired_level(self):
        """Test expired after pot life ends."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(46 * 60)  # 46 minutes (past 45)
        
        status = monitor.update()
        assert status.warning_level == "expired"
        assert status.remaining_seconds == 0
    
    def test_is_expired_false(self):
        """Test is_expired returns False before expiration."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(20 * 60)
        assert monitor.is_expired() is False
    
    def test_is_expired_true(self):
        """Test is_expired returns True after expiration."""
        monitor = MockPotLifeMonitor()
        monitor.set_elapsed(50 * 60)
        assert monitor.is_expired() is True
    
    def test_custom_thresholds(self):
        """Test with custom pot life settings."""
        config = {
            "pot_life_minutes": 30,
            "warning_minutes": 20,
            "critical_minutes": 25
        }
        monitor = MockPotLifeMonitor(config)
        
        # Check thresholds are set correctly
        assert monitor.pot_life_seconds == 30 * 60
        assert monitor.warning_threshold == 20 * 60
        assert monitor.critical_threshold == 25 * 60
        
        # Test warning at 21 minutes
        monitor.set_elapsed(21 * 60)
        assert monitor.update().warning_level == "warning"
        
        # Test critical at 26 minutes
        monitor.set_elapsed(26 * 60)
        assert monitor.update().warning_level == "critical"
        
        # Test expired at 31 minutes
        monitor.set_elapsed(31 * 60)
        assert monitor.update().warning_level == "expired"
    
    def test_remaining_time_calculation(self):
        """Test remaining time decreases correctly."""
        monitor = MockPotLifeMonitor()  # 45 min default
        
        monitor.set_elapsed(0)
        status = monitor.update()
        assert pytest.approx(status.remaining_seconds, abs=5) == 45 * 60
        
        monitor.set_elapsed(15 * 60)
        status = monitor.update()
        assert pytest.approx(status.remaining_seconds, abs=5) == 30 * 60
        
        monitor.set_elapsed(30 * 60)
        status = monitor.update()
        assert pytest.approx(status.remaining_seconds, abs=5) == 15 * 60


class TestPotLifeWorkflow:
    """Test pot life in typical layup workflow."""
    
    def test_typical_layup_timing(self):
        """Test pot life during typical 4-ply layup."""
        monitor = MockPotLifeMonitor()
        
        # Start mixing (t=0)
        monitor.start_timer()
        assert monitor.update().warning_level == "ok"
        
        # After mixing (t=3 min)
        monitor.set_elapsed(3 * 60)
        assert monitor.update().warning_level == "ok"
        
        # After ply 1 (t=8 min)
        monitor.set_elapsed(8 * 60)
        assert monitor.update().warning_level == "ok"
        
        # After ply 2 (t=16 min)
        monitor.set_elapsed(16 * 60)
        assert monitor.update().warning_level == "ok"
        
        # After ply 3 (t=24 min)
        monitor.set_elapsed(24 * 60)
        assert monitor.update().warning_level == "ok"
        
        # After ply 4 (t=32 min) - warning threshold passed
        monitor.set_elapsed(32 * 60)
        assert monitor.update().warning_level == "warning"
        
        # During cleanup (t=38 min) - still in warning
        monitor.set_elapsed(38 * 60)
        assert monitor.update().warning_level == "warning"
    
    def test_cleanup_deadline(self):
        """Test cleanup must complete before pot life expires."""
        monitor = MockPotLifeMonitor()
        
        # Finish layup at 34 minutes (leaves 11 min for cleanup)
        monitor.set_elapsed(34 * 60)
        status = monitor.update()
        
        # Must have enough time for cleanup (typically 10 min needed)
        cleanup_time_needed = 10 * 60
        assert status.remaining_seconds >= cleanup_time_needed, \
            f"Not enough time for cleanup: {status.remaining_seconds/60:.1f} min remaining"

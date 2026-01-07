"""Tests for base monitor and event bus."""

import pytest
import asyncio
from datetime import datetime

from aura.core import MonitorType, MonitorOutput, PerceptionOutput, TrackedObject
from aura.monitors import BaseMonitor, MonitorConfig, MonitorEventBus, MonitorEvent


class MockPerceptionMonitor(BaseMonitor):
    """Mock monitor for testing."""
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERCEPTION
    
    async def _process(self, frame=None) -> MonitorOutput:
        # Simulate some processing
        await asyncio.sleep(0.01)
        obj = TrackedObject(id="test_obj", name="test", category="test")
        return PerceptionOutput(
            objects=[obj]
        )


class TestBaseMonitor:
    @pytest.mark.asyncio
    async def test_update(self):
        """Test basic monitor update."""
        monitor = MockPerceptionMonitor()
        output = await monitor.update(frame=None)
        
        assert output is not None
        assert isinstance(output, PerceptionOutput)
        assert len(output.objects) == 1
        assert output.objects[0].name == "test"
    
    @pytest.mark.asyncio
    async def test_disabled_monitor(self):
        """Test that disabled monitors return None."""
        config = MonitorConfig(enabled=False)
        monitor = MockPerceptionMonitor(config)
        output = await monitor.update(frame=None)
        
        assert output is None
    
    @pytest.mark.asyncio
    async def test_last_output(self):
        """Test last output tracking."""
        monitor = MockPerceptionMonitor()
        await monitor.update(frame=None)
        
        assert monitor.last_output is not None
        assert monitor.last_update_time is not None
        assert isinstance(monitor.last_update_time, datetime)
    
    @pytest.mark.asyncio
    async def test_stale_check(self):
        """Test stale output detection."""
        monitor = MockPerceptionMonitor()
        assert monitor.is_stale()  # No output yet
        
        await monitor.update(frame=None)
        assert not monitor.is_stale(max_age_sec=1.0)
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout handling."""
        class SlowMonitor(BaseMonitor):
            @property
            def monitor_type(self) -> MonitorType:
                return MonitorType.PERCEPTION
            
            async def _process(self, **inputs) -> MonitorOutput:
                await asyncio.sleep(10)  # Longer than default timeout
                return PerceptionOutput()
        
        config = MonitorConfig(timeout_sec=0.1)
        monitor = SlowMonitor(config)
        output = await monitor.update()
        
        assert output is None  # Should timeout


class TestMonitorEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test subscribing to and receiving events."""
        bus = MonitorEventBus()
        received_events = []
        
        async def handler(event: MonitorEvent):
            received_events.append(event)
        
        bus.subscribe(MonitorType.PERCEPTION, handler)
        
        output = PerceptionOutput(objects=[])
        await bus.publish(MonitorType.PERCEPTION, output)
        
        assert len(received_events) == 1
        assert received_events[0].monitor_type == MonitorType.PERCEPTION
    
    @pytest.mark.asyncio
    async def test_global_handler(self):
        """Test global handlers receive all events."""
        bus = MonitorEventBus()
        received_events = []
        
        async def handler(event: MonitorEvent):
            received_events.append(event)
        
        bus.subscribe_all(handler)
        
        await bus.publish(MonitorType.PERCEPTION, PerceptionOutput())
        await bus.publish(MonitorType.SOUND, MonitorOutput(monitor_type=MonitorType.SOUND))
        
        assert len(received_events) == 2
    
    @pytest.mark.asyncio
    async def test_monitor_integration(self):
        """Test monitor integration with event bus."""
        bus = MonitorEventBus()
        monitor = MockPerceptionMonitor()
        monitor.set_event_bus(bus)
        
        received_events = []
        
        async def handler(event: MonitorEvent):
            received_events.append(event)
        
        bus.subscribe(MonitorType.PERCEPTION, handler)
        
        await monitor.update()
        
        assert len(received_events) == 1
        assert bus.get_latest(MonitorType.PERCEPTION) is not None
    
    @pytest.mark.asyncio
    async def test_history(self):
        """Test event history."""
        bus = MonitorEventBus(history_size=5)
        
        for i in range(10):
            await bus.publish(MonitorType.PERCEPTION, PerceptionOutput())
        
        history = bus.get_history()
        assert len(history) <= 5  # Limited by history_size
    
    @pytest.mark.asyncio
    async def test_wait_for(self):
        """Test waiting for next event."""
        bus = MonitorEventBus()
        
        async def publish_later():
            await asyncio.sleep(0.1)
            await bus.publish(MonitorType.PERCEPTION, PerceptionOutput())
        
        # Start publishing task
        asyncio.create_task(publish_later())
        
        # Wait for event
        output = await bus.wait_for(MonitorType.PERCEPTION, timeout=1.0)
        assert output is not None
    
    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        """Test wait_for timeout."""
        bus = MonitorEventBus()
        
        output = await bus.wait_for(MonitorType.PERCEPTION, timeout=0.1)
        assert output is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

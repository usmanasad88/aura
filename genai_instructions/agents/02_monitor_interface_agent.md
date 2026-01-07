# Agent 02: Monitor Interface Agent

## Task: Create Base Monitor Interface and Event Bus

### Objective
Create the abstract base class for all monitors and an event bus for inter-monitor communication.

### Prerequisites
- Task 1.1 (Core Types) complete
- Core types importable

### Files to Create

#### 1. `src/aura/monitors/base_monitor.py`

```python
"""Base monitor interface for AURA framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import asyncio
import threading
import logging

from aura.core import MonitorType, MonitorOutput


logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for a monitor."""
    enabled: bool = True
    update_rate_hz: float = 10.0
    timeout_sec: float = 5.0
    async_mode: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseMonitor(ABC):
    """Abstract base class for all AURA monitors.
    
    Monitors are responsible for:
    1. Processing input data (images, audio, etc.)
    2. Producing structured outputs
    3. Publishing to the event bus
    
    Subclasses must implement:
    - _process(): The main processing logic
    - monitor_type: Property returning the MonitorType
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self._is_running = False
        self._last_output: Optional[MonitorOutput] = None
        self._last_update: Optional[datetime] = None
        self._event_bus: Optional["MonitorEventBus"] = None
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        
    @property
    @abstractmethod
    def monitor_type(self) -> MonitorType:
        """Return the type of this monitor."""
        pass
    
    @abstractmethod
    async def _process(self, **inputs) -> MonitorOutput:
        """Process inputs and return monitor output.
        
        Args:
            **inputs: Monitor-specific inputs (e.g., frame, audio_chunk)
            
        Returns:
            MonitorOutput subclass with results
        """
        pass
    
    def set_event_bus(self, bus: "MonitorEventBus"):
        """Register this monitor with an event bus."""
        self._event_bus = bus
        bus.register_monitor(self)
    
    async def update(self, **inputs) -> Optional[MonitorOutput]:
        """Run one update cycle.
        
        Args:
            **inputs: Inputs to pass to _process()
            
        Returns:
            MonitorOutput if successful, None if disabled or error
        """
        if not self.config.enabled:
            return None
            
        try:
            output = await asyncio.wait_for(
                self._process(**inputs),
                timeout=self.config.timeout_sec
            )
            
            with self._lock:
                self._last_output = output
                self._last_update = datetime.now()
            
            # Publish to event bus
            if self._event_bus and output:
                await self._event_bus.publish(self.monitor_type, output)
            
            return output
            
        except asyncio.TimeoutError:
            logger.warning(f"{self.monitor_type.name} monitor timed out")
            return None
        except Exception as e:
            logger.error(f"{self.monitor_type.name} monitor error: {e}")
            return None
    
    def update_sync(self, **inputs) -> Optional[MonitorOutput]:
        """Synchronous wrapper for update()."""
        return asyncio.run(self.update(**inputs))
    
    async def start_continuous(self, input_provider: Callable[[], Dict[str, Any]]):
        """Start continuous monitoring.
        
        Args:
            input_provider: Callable that returns dict of inputs for each cycle
        """
        self._is_running = True
        interval = 1.0 / self.config.update_rate_hz
        
        while self._is_running:
            try:
                inputs = input_provider()
                await self.update(**inputs)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop(self):
        """Stop continuous monitoring."""
        self._is_running = False
        if self._task:
            self._task.cancel()
    
    @property
    def last_output(self) -> Optional[MonitorOutput]:
        """Get the most recent output."""
        with self._lock:
            return self._last_output
    
    @property
    def last_update_time(self) -> Optional[datetime]:
        """Get timestamp of last successful update."""
        with self._lock:
            return self._last_update
    
    def is_stale(self, max_age_sec: float = 1.0) -> bool:
        """Check if the last output is stale."""
        if self._last_update is None:
            return True
        age = (datetime.now() - self._last_update).total_seconds()
        return age > max_age_sec
```

#### 2. `src/aura/monitors/monitor_bus.py`

```python
"""Event bus for monitor communication."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
import asyncio
import logging
from collections import defaultdict

from aura.core import MonitorType, MonitorOutput


logger = logging.getLogger(__name__)


@dataclass
class MonitorEvent:
    """An event published by a monitor."""
    monitor_type: MonitorType
    output: MonitorOutput
    timestamp: datetime = field(default_factory=datetime.now)


# Type alias for event handlers
EventHandler = Callable[[MonitorEvent], None]
AsyncEventHandler = Callable[[MonitorEvent], Any]  # Coroutine


class MonitorEventBus:
    """Central event bus for monitor communication.
    
    Monitors publish their outputs here, and other components
    (including the Brain) subscribe to receive updates.
    
    Features:
    - Async-first design
    - Multiple subscribers per event type
    - Event history for debugging
    - Priority ordering for handlers
    """
    
    def __init__(self, history_size: int = 100):
        self._handlers: Dict[MonitorType, List[AsyncEventHandler]] = defaultdict(list)
        self._global_handlers: List[AsyncEventHandler] = []
        self._monitors: Dict[MonitorType, "BaseMonitor"] = {}
        self._history: List[MonitorEvent] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()
        
    def register_monitor(self, monitor: "BaseMonitor"):
        """Register a monitor with the bus."""
        self._monitors[monitor.monitor_type] = monitor
        logger.info(f"Registered monitor: {monitor.monitor_type.name}")
    
    def subscribe(
        self, 
        monitor_type: MonitorType, 
        handler: AsyncEventHandler
    ):
        """Subscribe to events from a specific monitor type.
        
        Args:
            monitor_type: The monitor type to subscribe to
            handler: Async callable that takes MonitorEvent
        """
        self._handlers[monitor_type].append(handler)
        logger.debug(f"Added subscriber for {monitor_type.name}")
    
    def subscribe_all(self, handler: AsyncEventHandler):
        """Subscribe to events from all monitors.
        
        Args:
            handler: Async callable that takes MonitorEvent
        """
        self._global_handlers.append(handler)
        logger.debug("Added global subscriber")
    
    async def publish(self, monitor_type: MonitorType, output: MonitorOutput):
        """Publish an event from a monitor.
        
        Args:
            monitor_type: Source monitor type
            output: The monitor output
        """
        event = MonitorEvent(
            monitor_type=monitor_type,
            output=output
        )
        
        # Store in history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)
        
        # Notify specific handlers
        for handler in self._handlers[monitor_type]:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler error for {monitor_type.name}: {e}")
        
        # Notify global handlers
        for handler in self._global_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Global handler error: {e}")
    
    def get_latest(self, monitor_type: MonitorType) -> Optional[MonitorOutput]:
        """Get the latest output from a monitor type.
        
        Args:
            monitor_type: The monitor type
            
        Returns:
            Latest MonitorOutput or None
        """
        if monitor_type in self._monitors:
            return self._monitors[monitor_type].last_output
        return None
    
    def get_all_latest(self) -> Dict[MonitorType, Optional[MonitorOutput]]:
        """Get latest outputs from all registered monitors."""
        return {
            mt: m.last_output 
            for mt, m in self._monitors.items()
        }
    
    def get_history(
        self, 
        monitor_type: Optional[MonitorType] = None,
        limit: int = 10
    ) -> List[MonitorEvent]:
        """Get recent event history.
        
        Args:
            monitor_type: Filter by type, or None for all
            limit: Maximum events to return
            
        Returns:
            List of recent events, newest first
        """
        events = self._history[::-1]  # Reverse for newest first
        
        if monitor_type:
            events = [e for e in events if e.monitor_type == monitor_type]
        
        return events[:limit]
    
    async def wait_for(
        self, 
        monitor_type: MonitorType, 
        timeout: float = 5.0
    ) -> Optional[MonitorOutput]:
        """Wait for the next event from a monitor.
        
        Args:
            monitor_type: Monitor type to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            MonitorOutput when received, None on timeout
        """
        result_event = asyncio.Event()
        result_output: List[MonitorOutput] = []
        
        async def capture(event: MonitorEvent):
            result_output.append(event.output)
            result_event.set()
        
        self.subscribe(monitor_type, capture)
        
        try:
            await asyncio.wait_for(result_event.wait(), timeout=timeout)
            return result_output[0] if result_output else None
        except asyncio.TimeoutError:
            return None
        finally:
            # Remove the temporary handler
            self._handlers[monitor_type].remove(capture)
```

#### 3. `src/aura/monitors/__init__.py`

```python
"""Monitor components for AURA framework."""

from .base_monitor import BaseMonitor, MonitorConfig
from .monitor_bus import MonitorEventBus, MonitorEvent

__all__ = [
    "BaseMonitor",
    "MonitorConfig", 
    "MonitorEventBus",
    "MonitorEvent",
]
```

#### 4. `tests/test_monitors/test_base_monitor.py`

```python
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
        return PerceptionOutput(
            objects=[
                TrackedObject(id="obj_1", name="cup", category="container")
            ]
        )


class TestBaseMonitor:
    @pytest.mark.asyncio
    async def test_update(self):
        monitor = MockPerceptionMonitor()
        output = await monitor.update(frame=None)
        
        assert output is not None
        assert isinstance(output, PerceptionOutput)
        assert len(output.objects) == 1
    
    @pytest.mark.asyncio
    async def test_disabled_monitor(self):
        config = MonitorConfig(enabled=False)
        monitor = MockPerceptionMonitor(config=config)
        output = await monitor.update(frame=None)
        
        assert output is None
    
    @pytest.mark.asyncio
    async def test_last_output(self):
        monitor = MockPerceptionMonitor()
        await monitor.update(frame=None)
        
        assert monitor.last_output is not None
        assert monitor.last_update_time is not None
    
    @pytest.mark.asyncio
    async def test_stale_check(self):
        monitor = MockPerceptionMonitor()
        assert monitor.is_stale()  # No updates yet
        
        await monitor.update(frame=None)
        assert not monitor.is_stale(max_age_sec=1.0)


class TestMonitorEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
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
        bus = MonitorEventBus()
        received_events = []
        
        async def handler(event: MonitorEvent):
            received_events.append(event)
        
        bus.subscribe_all(handler)
        
        await bus.publish(MonitorType.PERCEPTION, PerceptionOutput(objects=[]))
        await bus.publish(MonitorType.INTENT, MonitorOutput(monitor_type=MonitorType.INTENT))
        
        assert len(received_events) == 2
    
    @pytest.mark.asyncio
    async def test_monitor_integration(self):
        bus = MonitorEventBus()
        monitor = MockPerceptionMonitor()
        monitor.set_event_bus(bus)
        
        received = []
        async def handler(event):
            received.append(event)
        
        bus.subscribe(MonitorType.PERCEPTION, handler)
        await monitor.update(frame=None)
        
        assert len(received) == 1
        assert bus.get_latest(MonitorType.PERCEPTION) is not None
    
    @pytest.mark.asyncio
    async def test_history(self):
        bus = MonitorEventBus(history_size=5)
        
        for i in range(10):
            await bus.publish(
                MonitorType.PERCEPTION, 
                PerceptionOutput(objects=[])
            )
        
        history = bus.get_history()
        assert len(history) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Validation

```bash
cd /home/mani/Repos/aura
uv run pytest tests/test_monitors/test_base_monitor.py -v
```

### Handoff Notes

Create `genai_instructions/handoff/02_monitor_interface.md`:

```markdown
# Monitor Interface Implementation Complete

## Files Created
- src/aura/monitors/base_monitor.py
- src/aura/monitors/monitor_bus.py
- src/aura/monitors/__init__.py
- tests/test_monitors/test_base_monitor.py

## Key Design Decisions
1. Async-first: All monitor processing is async
2. Thread-safe: Lock protects last_output
3. Event bus: Decouples monitors from consumers
4. Configurable: MonitorConfig for rate, timeout, etc.

## Usage Pattern
```python
monitor = MyMonitor(config)
monitor.set_event_bus(bus)
output = await monitor.update(frame=my_frame)
```

## Next Steps
- Implement concrete monitors (perception, intent, etc.)
- Brain will subscribe to event bus for all monitors
```

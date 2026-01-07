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
                logger.error(f"Handler error for {monitor_type.name}: {e}", exc_info=True)
        
        # Notify global handlers
        for handler in self._global_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Global handler error: {e}", exc_info=True)
    
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
            MonitorOutput from next event, or None if timeout
        """
        future = asyncio.Future()
        
        async def handler(event: MonitorEvent):
            if not future.done():
                future.set_result(event.output)
        
        # Subscribe temporarily
        self.subscribe(monitor_type, handler)
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
        finally:
            # Clean up handler
            if handler in self._handlers[monitor_type]:
                self._handlers[monitor_type].remove(handler)

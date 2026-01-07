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
            # Use timeout_sec or timeout_seconds for compatibility
            timeout = getattr(self.config, 'timeout_sec', None) or getattr(self.config, 'timeout_seconds', 5.0)
            output = await asyncio.wait_for(
                self._process(**inputs),
                timeout=timeout
            )
            
            with self._lock:
                self._last_output = output
                self._last_update = datetime.now()
            
            # Publish to event bus if registered
            if self._event_bus:
                await self._event_bus.publish(self.monitor_type, output)
            
            return output
            
        except asyncio.TimeoutError:
            logger.warning(f"{self.monitor_type.name} monitor timed out")
            return None
        except Exception as e:
            logger.error(f"{self.monitor_type.name} monitor error: {e}", exc_info=True)
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
                logger.error(f"Error in continuous monitoring: {e}", exc_info=True)
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

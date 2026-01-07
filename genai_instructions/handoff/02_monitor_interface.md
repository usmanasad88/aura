# Monitor Interface Implementation Complete (Task 1.2)

## Date Completed
January 7, 2026

## Files Created
- [src/aura/monitors/base_monitor.py](../../src/aura/monitors/base_monitor.py) - Abstract base class for monitors
- [src/aura/monitors/monitor_bus.py](../../src/aura/monitors/monitor_bus.py) - Event bus for inter-monitor communication
- [src/aura/monitors/__init__.py](../../src/aura/monitors/__init__.py) - Package exports
- [tests/test_monitors/test_base_monitor.py](../../tests/test_monitors/test_base_monitor.py) - Comprehensive tests

## Test Results
✅ All 11 tests passed successfully:
- BaseMonitor update cycle
- Disabled monitor returns None
- Last output tracking
- Stale output detection
- Timeout handling
- Event bus subscribe and publish
- Global event handlers
- Monitor-bus integration
- Event history management
- Wait for next event (with timeout)

## Key Components

### MonitorConfig
Configuration dataclass for monitor behavior:
- `enabled: bool = True` - Enable/disable monitor
- `update_rate_hz: float = 10.0` - Target update frequency
- `timeout_sec: float = 5.0` - Processing timeout
- `async_mode: bool = True` - Async processing mode
- `extra: Dict[str, Any]` - Monitor-specific config

### BaseMonitor (Abstract Class)
Core responsibilities:
1. **Processing**: Subclasses implement `_process(**inputs) -> MonitorOutput`
2. **Output tracking**: Stores last output with timestamp
3. **Event publishing**: Automatically publishes to event bus
4. **Error handling**: Timeout and exception handling
5. **Lifecycle**: Start/stop continuous monitoring

Key methods:
- `async update(**inputs) -> Optional[MonitorOutput]` - Single update cycle
- `update_sync(**inputs)` - Synchronous wrapper
- `async start_continuous(input_provider)` - Background monitoring
- `stop()` - Stop monitoring
- `is_stale(max_age_sec)` - Check output freshness
- `set_event_bus(bus)` - Register with event bus

Properties:
- `monitor_type: MonitorType` - Must be implemented by subclass
- `last_output: Optional[MonitorOutput]` - Most recent output (thread-safe)
- `last_update_time: Optional[datetime]` - Last successful update

### MonitorEventBus
Central hub for monitor communication:
- **Subscribers**: Components register handlers for specific monitor types or all events
- **Publishing**: Monitors publish outputs, handlers are called automatically
- **History**: Maintains rolling history of events (configurable size)
- **Async-first**: All handlers are async (or converted to async)

Key methods:
- `register_monitor(monitor)` - Register monitor instance
- `subscribe(monitor_type, handler)` - Subscribe to specific monitor
- `subscribe_all(handler)` - Subscribe to all monitors
- `async publish(monitor_type, output)` - Publish event
- `get_latest(monitor_type)` - Get latest output from monitor
- `get_all_latest()` - Get latest from all monitors
- `get_history(monitor_type, limit)` - Get recent events
- `async wait_for(monitor_type, timeout)` - Wait for next event

## Design Patterns

### 1. Async-First Design
All processing is async by default:
```python
class MyMonitor(BaseMonitor):
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERCEPTION
    
    async def _process(self, frame=None) -> MonitorOutput:
        # Async processing here
        result = await some_async_operation(frame)
        return PerceptionOutput(objects=result)
```

### 2. Event Bus Pattern
Decouples producers (monitors) from consumers (brain, other monitors):
```python
bus = MonitorEventBus()
monitor = MyMonitor()
monitor.set_event_bus(bus)

# Subscribe to events
async def handle_perception(event: MonitorEvent):
    print(f"New perception: {event.output}")

bus.subscribe(MonitorType.PERCEPTION, handle_perception)

# Updates automatically publish
await monitor.update(frame=my_frame)
```

### 3. Thread-Safe State
Last output is protected by lock:
```python
with self._lock:
    self._last_output = output
    self._last_update = datetime.now()
```

### 4. Graceful Error Handling
Monitors never crash the system:
- Timeouts return None
- Exceptions are logged and return None
- Event bus handlers are isolated (one failure doesn't affect others)

## Usage Examples

### Basic Monitor Implementation
```python
from aura.monitors import BaseMonitor, MonitorConfig
from aura.core import MonitorType, PerceptionOutput, TrackedObject

class ObjectDetector(BaseMonitor):
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.PERCEPTION
    
    async def _process(self, frame) -> PerceptionOutput:
        # Detect objects in frame
        objects = await self.detect(frame)
        return PerceptionOutput(objects=objects)
    
    async def detect(self, frame):
        # Your detection logic
        return [TrackedObject(id="obj1", name="cup", category="object")]
```

### Using with Event Bus
```python
from aura.monitors import MonitorEventBus, MonitorEvent

# Create bus
bus = MonitorEventBus(history_size=100)

# Create and register monitors
detector = ObjectDetector()
detector.set_event_bus(bus)

# Subscribe to events
async def on_detection(event: MonitorEvent):
    output = event.output
    print(f"Detected {len(output.objects)} objects")

bus.subscribe(MonitorType.PERCEPTION, on_detection)

# Run monitor
output = await detector.update(frame=my_frame)
# Handler is automatically called
```

### Continuous Monitoring
```python
import cv2

def get_camera_input():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    return {"frame": frame}

# Start background monitoring at 30 Hz
config = MonitorConfig(update_rate_hz=30.0)
monitor = ObjectDetector(config)
monitor.set_event_bus(bus)

await monitor.start_continuous(get_camera_input)

# Later...
monitor.stop()
```

### Waiting for Events
```python
# Wait for next perception event (with timeout)
output = await bus.wait_for(MonitorType.PERCEPTION, timeout=5.0)
if output:
    print(f"Got perception: {output.objects}")
else:
    print("Timeout waiting for perception")
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│              MonitorEventBus                     │
│  - Subscribers (by type and global)             │
│  - Event history                                 │
│  - Async publishing                              │
└────────────┬────────────────────────────────────┘
             │
             │ publish()
             │
    ┌────────┴────────┬──────────────┐
    │                 │              │
┌───▼────┐      ┌────▼─────┐   ┌───▼────┐
│Monitor1│      │ Monitor2 │   │Monitor3│
│        │      │          │   │        │
│_process│      │ _process │   │_process│
└───▲────┘      └────▲─────┘   └───▲────┘
    │                │              │
    │ update()       │              │
    │                │              │
  Input1          Input2         Input3
```

## Known Limitations

1. **Single Event Bus**: Currently supports one bus per monitor - could extend to multiple buses
2. **No Priority**: Event handlers run in registration order (could add priority levels)
3. **No Filtering**: History stores all events (could add filters)
4. **No Backpressure**: Fast publishers can overwhelm slow subscribers
5. **Memory**: History is in-memory only (could add persistence)

## Performance Considerations

- **Lock Contention**: `_lock` only protects last_output access (minimal)
- **Async Overhead**: Using `asyncio.wait_for` adds ~1ms overhead
- **Event Bus**: Async dispatch is fast (<1ms per handler)
- **History**: O(1) append, O(n) retrieval

## Notes for Next Agent

### Ready to Implement:
- ✅ **Perception Monitor** (Task 2.1) - Use BaseMonitor + SAM3
- ✅ **Motion Predictor** (Task 2.2) - Use BaseMonitor + pose tracking
- ✅ **Sound Monitor** (Task 2.3) - Use BaseMonitor + Gemini Live
- ✅ **Intent Monitor** (Task 5.2) - Use BaseMonitor + LLM reasoning
- ✅ **Affordance Monitor** (Task 4.1) - Use BaseMonitor + scene analysis
- ✅ **Performance Monitor** (Task 3.4) - Use BaseMonitor + action tracking

### Pattern to Follow:
```python
class YourMonitor(BaseMonitor):
    def __init__(self, config: Optional[MonitorConfig] = None):
        super().__init__(config)
        # Initialize your models here
        self.model = YourModel()
    
    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.YOUR_TYPE
    
    async def _process(self, **inputs) -> YourOutput:
        # 1. Extract inputs
        data = inputs.get("your_input")
        
        # 2. Process (can be async or sync)
        result = await self.model.process(data)
        
        # 3. Return typed output
        return YourOutput(
            your_field=result,
            is_valid=True
        )
```

## Validation Commands

```bash
# Run tests
cd /home/mani/Repos/aura
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest tests/test_monitors/test_base_monitor.py -v

# Import check
uv run python -c "from aura.monitors import BaseMonitor, MonitorEventBus; print('✓ Monitors OK')"

# Run both test suites
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest tests/test_core tests/test_monitors -v
```

## Next Steps

### Immediate:
1. Implement concrete monitors (Sprint 2)
2. Create Brain that subscribes to event bus (Sprint 3)
3. Add configuration system (Task 1.3)

### Future Enhancements:
- Add monitor health checks
- Implement priority queues for handlers
- Add event replay for debugging
- Create monitor metrics/dashboard
- Add async input providers for sensors

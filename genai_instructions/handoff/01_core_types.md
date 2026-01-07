# Core Types Implementation Complete (Task 1.1)

## Date Completed
January 7, 2026

## Files Created
- [src/aura/core/enums.py](../../src/aura/core/enums.py) - All enumerations
- [src/aura/core/types.py](../../src/aura/core/types.py) - All dataclasses
- [src/aura/core/__init__.py](../../src/aura/core/__init__.py) - Package exports
- [tests/test_core/test_types.py](../../tests/test_core/test_types.py) - Comprehensive tests

## Test Results
✅ All 10 tests passed successfully:
- Pose3D creation and properties
- BoundingBox center and area calculations
- Trajectory duration
- TrackedObject creation and properties
- Action lifecycle (pending → in_progress → completed/failed)
- TaskGraph precondition logic
- AuraState timestamp management

## Key Features Implemented

### Enumerations
- `ActionStatus`: PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
- `IntentType`: IDLE, REACHING, GRASPING, MOVING, PLACING, GESTURING, SPEAKING, UNKNOWN
- `RobotActionType`: MOVE_TO_POSE, FOLLOW_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, WAIT, SPEAK, DISPLAY_MESSAGE
- `MonitorType`: INTENT, MOTION, PERCEPTION, SOUND, AFFORDANCE, PERFORMANCE
- `TaskState`: NOT_STARTED, IN_PROGRESS, WAITING_FOR_HUMAN, WAITING_FOR_ROBOT, PAUSED, COMPLETED, FAILED

### Core Types
- **Geometry**: Pose2D, Pose3D, BoundingBox, Trajectory
- **Objects**: TrackedObject, SceneGraph
- **Human**: JointPosition, HumanPose, Intent, PredictedMotion
- **Communication**: Utterance
- **Actions**: Action, Affordance
- **Tasks**: TaskNode, TaskGraph
- **State**: PerformanceMetrics, AuraState
- **Monitor Outputs**: MonitorOutput (base), PerceptionOutput, IntentOutput, MotionOutput, SoundOutput, AffordanceOutput, PerformanceOutput

## Design Decisions

1. **Dataclasses**: All types use `@dataclass` for:
   - Automatic `__init__`, `__repr__`, `__eq__`
   - Easy serialization to JSON
   - Type annotations for IDE support

2. **Optional Fields**: Many fields are Optional to allow incremental state building

3. **Factory Defaults**: Lists and dicts use `field(default_factory=...)` to avoid mutable default issues

4. **Timestamps**: Automatic timestamp generation using `field(default_factory=datetime.now)`

5. **MonitorOutput Subclasses**: Each monitor type has its own output class with `__post_init__` to set monitor_type (updated to use `field(default=...)` for cleaner code)

6. **Action Lifecycle**: Helper methods (`mark_started()`, `mark_completed()`, `mark_failed()`) for state transitions

7. **TaskGraph Logic**: `get_available_nodes()` method checks preconditions automatically

## Usage Examples

```python
from aura.core import (
    TrackedObject, Pose3D, BoundingBox,
    Action, RobotActionType, ActionStatus,
    TaskGraph, TaskNode, AuraState
)

# Create a tracked object
obj = TrackedObject(
    id="cup_1",
    name="coffee cup",
    category="container",
    pose=Pose3D.from_position(0.5, 0.3, 0.8),
    bbox=BoundingBox(x_min=100, y_min=150, x_max=200, y_max=250),
    confidence=0.95
)

# Create and manage an action
action = Action(
    id="move_1",
    type=RobotActionType.MOVE_TO_POSE,
    parameters={"target_pose": [0.5, 0.3, 0.8]}
)
action.mark_started()
# ... execute ...
action.mark_completed()

# Build task graph
graph = TaskGraph(
    id="layup_task",
    name="Composite Layup",
    description="Lay up composite materials",
    nodes={
        "n1": TaskNode(id="n1", name="Prepare mold", description="..."),
        "n2": TaskNode(id="n2", name="Mix resin", description="...", preconditions=["n1"]),
    }
)
available = graph.get_available_nodes()  # Returns [n1]

# Create system state
state = AuraState()
state.objects.append(obj)
state.current_action = action
state.update_timestamp()
```

## Known Limitations

1. **NumPy Dependency**: TrackedObject.mask uses np.ndarray - may need serialization helpers for JSON export
2. **No Validation**: Dataclasses don't validate field constraints (e.g., confidence in [0,1])
3. **Timestamps**: Using naive datetime (no timezone info) - consider using timezone-aware datetimes for production

## Notes for Next Agent

- ✅ All types are importable from `aura.core`
- ✅ Type hints are complete for IDE support
- ✅ Tests cover basic functionality
- 🔄 Consider adding Pydantic models for validation if needed
- 🔄 May need serialization/deserialization helpers for complex types (numpy arrays, datetimes)

## Dependencies for Next Tasks

### Ready to Start:
- ✅ **Task 1.2 (Base Monitor)** - Can use MonitorType, MonitorOutput
- ✅ **Task 1.3 (Config)** - Independent of core types
- ✅ **Sprint 2 tasks** - All monitors can use these output types

### Blocked:
- None

## Validation Commands

```bash
# Run tests (must unset ROS environment to avoid pytest plugin conflicts)
cd /home/mani/Repos/aura
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest tests/test_core/test_types.py -v

# Import check
uv run python -c "from aura.core import AuraState, TrackedObject; print('✓ Core types OK')"
```

## Test Environment Note

⚠️ **Important**: When running pytest, must unset ROS environment variables to avoid plugin conflicts:
```bash
unset PYTHONPATH && unset ROS_DISTRO && uv run pytest ...
```

This is because the system has ROS 2 Humble installed, which loads incompatible pytest plugins.

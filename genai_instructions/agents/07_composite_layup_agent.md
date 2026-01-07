# Agent 07: Composite Layup Demo Implementation

## Mission

Implement the composite hand layup assistance demo end-to-end. This agent is responsible for integrating the AURA framework with the specific requirements of fiberglass layup on a mold.

## Context

You are building a proactive robotic assistance system where a UR5 robot with Robotiq 2F-85 gripper helps a human operator perform hand layup of fiberglass composite. The robot:
- Measures and mixes resin/hardener in correct proportions
- Hands tools to the operator on request or proactively
- Monitors layup quality for defects
- Tracks pot life and issues warnings

## Required Reading

Before implementation, read and understand:
1. `sops/composite_layup.json` - Task graph and material specifications
2. `config/composite_layup.yaml` - Demo configuration
3. `genai_instructions/demos/composite_layup_demo.md` - Implementation guide
4. `src/aura/core/types.py` - Base types to use

## Implementation Order

Complete these modules in sequence:

### Step 1: Create Demo Package Structure

```bash
mkdir -p src/aura/demos/composite_layup
touch src/aura/demos/__init__.py
touch src/aura/demos/composite_layup/__init__.py
```

### Step 2: Implement Scale Monitor (Priority: HIGH)

**File**: `src/aura/monitors/scale_monitor.py`

This is critical for resin/hardener measurement. The monitor must:
- Read weight values (via vision OCR or serial)
- Track target weight for controlled pouring
- Detect when weight is stable
- Support tare operation

Key methods:
- `update()` → `ScaleReading` with current weight
- `set_target(weight_g, tolerance_g)` → Set pour target
- `is_at_target()` → Check if within tolerance
- `tare()` → Zero the scale

Test with: `pytest tests/test_monitors/test_scale_monitor.py -v`

### Step 3: Implement Pot Life Monitor (Priority: HIGH)

**File**: `src/aura/monitors/pot_life_monitor.py`

Track time-critical resin working life:
- Start timer when mixing begins
- Calculate remaining time
- Emit warnings at thresholds (30 min, 40 min for 45 min pot life)
- Alert when expired

### Step 4: Implement Defect Detector (Priority: MEDIUM)

**File**: `src/aura/monitors/defect_detector.py`

Vision-based quality inspection:
- Detect dry spots (unwetted fiberglass)
- Detect air bubbles
- Detect wrinkles
- Calculate defect area
- Annotate images with findings

Use SAM3 from `third_party/sam3` for segmentation.

### Step 5: Implement Pour Action (Priority: HIGH)

**File**: `src/aura/actions/pour_action.py`

Controlled pouring with weight feedback:
- Pick container
- Move to pour position
- Tilt to pour with angle based on remaining weight
- Real-time feedback loop with scale monitor
- Stop when target reached

### Step 6: Implement Handoff Action (Priority: HIGH)

**File**: `src/aura/actions/handoff_action.py`

Safe tool handoff to human:
- Pick object from tool station
- Announce intention verbally
- Move to handoff pose (track human hand if available)
- Wait for human to grasp (force feedback)
- Release and retract

### Step 7: Create Demo Main Script (Priority: MEDIUM)

**File**: `src/aura/demos/composite_layup/main.py`

Orchestrate all components:
- Load config and SOP
- Initialize all monitors
- Initialize robot interface
- Run brain with task graph
- Handle shutdown gracefully

### Step 8: Create Test Suite (Priority: HIGH)

**Directory**: `tests/test_demos/test_composite_layup/`

Test files:
- `test_scale_monitor.py`
- `test_pot_life.py`
- `test_defect_detector.py`
- `test_pour_action.py`
- `test_handoff.py`
- `test_integration.py`

## Code Patterns to Follow

### Async Monitor Pattern
```python
class MyMonitor(BaseMonitor):
    async def update(self) -> MonitorOutput:
        # Gather data
        # Process
        # Return typed output
        pass
```

### Action Pattern
```python
class MyAction:
    def __init__(self, robot, monitors):
        self.robot = robot
        self.monitors = monitors
    
    async def execute(self, params) -> ActionStatus:
        # Execute steps
        # Check conditions
        # Return status
        pass
```

### Configuration Access
```python
from aura.utils.config import load_config

config = load_config("config/composite_layup.yaml")
scale_config = config["perception"]["scale_reader"]
```

## Interface Contracts

### ScaleMonitor must provide:
```python
@dataclass
class ScaleReading:
    weight_grams: float
    stable: bool
    tared: bool
    timestamp: float
```

### DefectDetector must provide:
```python
@dataclass
class Defect:
    type: DefectType
    bbox: BoundingBox
    confidence: float
    severity: str
    remedy: str
```

### PotLifeMonitor must provide:
```python
@dataclass
class PotLifeStatus:
    remaining_seconds: float
    warning_level: str  # "ok" | "warning" | "critical" | "expired"
```

## Safety Requirements

1. **Speed Limits**: Robot moves at max 0.1 m/s when human is within 0.5m
2. **Collision Avoidance**: Check human position before every motion
3. **Handoff Safety**: Wait for confirmed grasp before releasing
4. **Verbal Announcements**: Announce before starting motions
5. **Emergency Stop**: Immediate halt on "emergency stop" voice command

## Validation Criteria

Your implementation is complete when:

- [ ] `pytest tests/test_demos/test_composite_layup/ -v` passes all tests
- [ ] Scale monitor reads within ±1g accuracy
- [ ] Pot life warnings fire at correct times
- [ ] Defect detection finds >85% of test defects
- [ ] Pour action stops within ±2g of target
- [ ] Handoff completes safely in simulation
- [ ] Main script runs without errors (simulate mode)

## Dependencies

Ensure these are installed:
```bash
uv add pyserial  # For scale serial communication
uv add pytesseract  # For OCR-based scale reading (optional)
```

## Hints

1. Start with simulated/mock scale for development
2. Use `testing.py` as reference for camera/vision setup
3. The task graph in `composite_layup.json` has all step definitions
4. Robot poses are predefined in config - use them
5. Check `proactive_hcdt` for game communication patterns

## Handoff to Next Agent

When complete, document in `SETUP_COMPLETE.md`:
- Which components were implemented
- Any deviations from spec
- Known issues or TODOs
- Test coverage percentage

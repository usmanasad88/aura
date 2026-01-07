# Agent Instructions: Composite Layup Assistance Demo

## Overview

Implement and validate the composite hand layup assistance demo where a UR5 robot with Robotiq 2F-85 gripper assists a human operator with fiberglass layup on a mold.

## Task Summary

**Demo ID**: `composite_layup`
**Robot**: UR5 + Robotiq 2F-85
**Human**: Single operator performing hand layup
**Objective**: Proactive robotic assistance for composite manufacturing

## Prerequisites

- [ ] Core AURA framework (Sprint 1-2 complete)
- [ ] Perception monitor with object detection
- [ ] Sound monitor with voice commands
- [ ] Robot interface with UR5 control
- [ ] Task graph executor

## Key Files

| File | Purpose |
|------|---------|
| `sops/composite_layup.json` | Task graph and SOP definition |
| `config/composite_layup.yaml` | Demo-specific configuration |
| `src/aura/demos/composite_layup/` | Demo implementation |

---

## Implementation Tasks

### Task 1: Scale Reading Monitor

**File**: `src/aura/monitors/scale_monitor.py`

Implement a monitor that reads weight from a digital scale, either via:
1. Vision OCR (reading display with camera)
2. Serial interface (direct connection)

```python
from dataclasses import dataclass
from aura.monitors.base_monitor import BaseMonitor
from aura.core import MonitorOutput

@dataclass
class ScaleReading(MonitorOutput):
    weight_grams: float
    stable: bool
    tared: bool
    timestamp: float

class ScaleMonitor(BaseMonitor):
    """Monitor digital weigh scale for resin/hardener measurement."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_weight: float | None = None
        self.tolerance_grams: float = 2.0
        
    async def update(self) -> ScaleReading:
        # Implementation: read from camera OCR or serial
        pass
    
    def set_target(self, weight_g: float, tolerance_g: float = 2.0):
        """Set target weight for controlled pouring."""
        self.target_weight = weight_g
        self.tolerance_grams = tolerance_g
    
    def is_at_target(self, reading: ScaleReading) -> bool:
        """Check if current weight is within tolerance of target."""
        if self.target_weight is None:
            return False
        return abs(reading.weight_grams - self.target_weight) <= self.tolerance_grams
```

**Validation**:
```bash
pytest tests/test_monitors/test_scale_monitor.py -v
```

---

### Task 2: Defect Detection Module

**File**: `src/aura/monitors/defect_detector.py`

Implement defect detection for composite layup quality control using SAM3.

```python
from enum import Enum
from dataclasses import dataclass
from aura.core import BoundingBox, MonitorOutput

class DefectType(Enum):
    DRY_SPOT = "dry_spot"
    AIR_BUBBLE = "air_bubble"
    WRINKLE = "wrinkle"
    FIBER_MISALIGNMENT = "fiber_misalignment"
    RESIN_RICH = "resin_rich"
    RESIN_STARVED = "resin_starved"

@dataclass
class Defect:
    type: DefectType
    bbox: BoundingBox
    confidence: float
    severity: str  # "low", "medium", "high"
    remedy: str
    area_mm2: float

@dataclass
class DefectDetectionOutput(MonitorOutput):
    defects: list[Defect]
    image_path: str | None
    ply_number: int

class DefectDetector:
    """Detect defects in composite layup using vision."""
    
    def __init__(self, config: dict):
        self.model = None  # Load SAM3 or custom model
        self.defect_definitions = {
            DefectType.DRY_SPOT: {
                "description": "Unwetted fiberglass area",
                "severity": "high",
                "remedy": "Apply additional resin"
            },
            # ... other defects from composite_layup.json
        }
    
    async def detect(self, image, ply_number: int) -> DefectDetectionOutput:
        """Run defect detection on layup image."""
        pass
    
    def annotate_image(self, image, defects: list[Defect]):
        """Draw defect locations on image for visualization."""
        pass
```

**Validation**:
- Test with sample images of layup defects
- Verify detection accuracy > 85% on test set

---

### Task 3: Pot Life Timer

**File**: `src/aura/monitors/pot_life_monitor.py`

Track epoxy pot life and emit warnings.

```python
import time
from dataclasses import dataclass
from aura.monitors.base_monitor import BaseMonitor
from aura.core import MonitorOutput

@dataclass
class PotLifeStatus(MonitorOutput):
    mix_start_time: float
    elapsed_seconds: float
    remaining_seconds: float
    warning_level: str  # "ok", "warning", "critical", "expired"
    pot_life_total_seconds: float

class PotLifeMonitor(BaseMonitor):
    """Track epoxy pot life after mixing."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.pot_life_seconds = config.get("pot_life_minutes", 45) * 60
        self.warning_threshold = config.get("warning_minutes", 30) * 60
        self.critical_threshold = config.get("critical_minutes", 40) * 60
        self.mix_start_time: float | None = None
    
    def start_timer(self):
        """Call when resin/hardener mixing begins."""
        self.mix_start_time = time.time()
    
    async def update(self) -> PotLifeStatus | None:
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
            
        return PotLifeStatus(
            mix_start_time=self.mix_start_time,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            warning_level=level,
            pot_life_total_seconds=self.pot_life_seconds
        )
```

---

### Task 4: Controlled Pour Action

**File**: `src/aura/actions/pour_action.py`

Implement controlled pouring with weight feedback.

```python
from dataclasses import dataclass
from aura.core import Action, ActionStatus, RobotActionType
from aura.monitors.scale_monitor import ScaleMonitor

@dataclass
class PourParameters:
    container_id: str
    target_weight_g: float
    tolerance_g: float
    pour_rate: str  # "slow", "medium", "fast"

class ControlledPourAction:
    """Pour liquid with real-time weight feedback."""
    
    def __init__(self, robot_interface, scale_monitor: ScaleMonitor):
        self.robot = robot_interface
        self.scale = scale_monitor
    
    async def execute(self, params: PourParameters) -> ActionStatus:
        """
        Execute controlled pour:
        1. Pick up container
        2. Move to pour position above scale
        3. Tilt to pour slowly
        4. Monitor weight in real-time
        5. Stop when target reached
        6. Return container to station
        """
        self.scale.set_target(params.target_weight_g, params.tolerance_g)
        
        # Pick container
        await self.robot.pick(params.container_id)
        
        # Move to pour position
        await self.robot.move_to_named_pose("pour_position")
        
        # Pour with feedback loop
        while True:
            reading = await self.scale.update()
            
            if self.scale.is_at_target(reading):
                break
            
            # Adjust pour angle based on how close to target
            remaining = params.target_weight_g - reading.weight_grams
            angle = self._calculate_pour_angle(remaining, params.pour_rate)
            await self.robot.set_pour_angle(angle)
        
        # Upright and return
        await self.robot.set_pour_angle(0)
        await self.robot.place(params.container_id, "prep_station")
        
        return ActionStatus.COMPLETED
    
    def _calculate_pour_angle(self, remaining_g: float, rate: str) -> float:
        """Calculate tilt angle based on remaining amount."""
        base_angles = {"slow": 15, "medium": 25, "fast": 35}
        base = base_angles.get(rate, 20)
        
        # Reduce angle as approaching target
        if remaining_g < 10:
            return base * 0.3
        elif remaining_g < 20:
            return base * 0.6
        return base
```

---

### Task 5: Handoff Coordinator

**File**: `src/aura/actions/handoff_action.py`

Implement safe tool handoff to human.

```python
from dataclasses import dataclass
from aura.core import Action, ActionStatus, Pose3D

@dataclass
class HandoffParameters:
    object_id: str
    handoff_pose: Pose3D | None  # None = use detected human hand
    verbal_announcement: str
    wait_for_grasp: bool

class HandoffAction:
    """Safely hand object to human operator."""
    
    def __init__(self, robot_interface, perception_monitor, sound_monitor):
        self.robot = robot_interface
        self.perception = perception_monitor
        self.sound = sound_monitor
    
    async def execute(self, params: HandoffParameters) -> ActionStatus:
        """
        Execute handoff:
        1. Pick up object
        2. Announce intention
        3. Move to handoff position (or track human hand)
        4. Wait for human to grasp
        5. Release gripper
        6. Retract safely
        """
        # Pick object
        await self.robot.pick(params.object_id)
        
        # Announce
        if params.verbal_announcement:
            await self.sound.speak(params.verbal_announcement)
        
        # Determine handoff pose
        if params.handoff_pose:
            target = params.handoff_pose
        else:
            # Track human hand position
            human = await self.perception.get_human_pose()
            target = self._calculate_handoff_pose(human)
        
        # Move to handoff with slow approach
        await self.robot.move_to_pose(target, velocity_scaling=0.1)
        
        if params.wait_for_grasp:
            # Wait for human to grasp (detect gripper force change)
            await self._wait_for_grasp()
        
        # Release
        await self.robot.open_gripper()
        
        # Retract
        await self.robot.move_relative(z=0.1)  # Move up 10cm
        
        return ActionStatus.COMPLETED
    
    async def _wait_for_grasp(self, timeout_s: float = 10.0):
        """Wait for human to take object from gripper."""
        # Monitor gripper force or external force sensors
        pass
    
    def _calculate_handoff_pose(self, human_pose) -> Pose3D:
        """Calculate comfortable handoff position based on human pose."""
        # Position in front of human, at comfortable height
        pass
```

---

### Task 6: Demo Main Script

**File**: `src/aura/demos/composite_layup/main.py`

Main entry point for the demo.

```python
#!/usr/bin/env python3
"""
Composite Layup Assistance Demo

Run with:
    python -m aura.demos.composite_layup.main --config config/composite_layup.yaml
"""

import argparse
import asyncio
from pathlib import Path

from aura.brain import AuraBrain
from aura.monitors import MonitorEventBus
from aura.interfaces import RobotInterface
from aura.utils.config import load_config


async def main(config_path: str, simulate: bool = False):
    """Run composite layup assistance demo."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize components
    event_bus = MonitorEventBus()
    
    # Initialize monitors
    from aura.monitors.perception_module import PerceptionMonitor
    from aura.monitors.sound_module import SoundMonitor
    from aura.monitors.scale_monitor import ScaleMonitor
    from aura.monitors.pot_life_monitor import PotLifeMonitor
    from aura.monitors.defect_detector import DefectDetector
    
    monitors = {
        "perception": PerceptionMonitor(config["perception"]),
        "sound": SoundMonitor(config["sound"]),
        "scale": ScaleMonitor(config.get("perception", {}).get("scale_reader", {})),
        "pot_life": PotLifeMonitor(config.get("timing_constraints", {})),
        "defect": DefectDetector(config.get("perception", {}).get("defect_detection", {})),
    }
    
    # Register monitors
    for name, monitor in monitors.items():
        event_bus.register(monitor, name)
    
    # Initialize robot interface
    robot = RobotInterface(config["robot"], simulate=simulate)
    await robot.connect()
    
    # Initialize brain with task graph
    sop_path = config["demo"]["sop_file"]
    brain = AuraBrain(config["brain"], event_bus, robot)
    await brain.load_sop(sop_path)
    
    print("=" * 60)
    print("AURA Composite Layup Assistance Demo")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"SOP: {sop_path}")
    print(f"Simulate: {simulate}")
    print("=" * 60)
    
    # Main loop
    try:
        await brain.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await robot.disconnect()
        event_bus.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite Layup Assistance Demo")
    parser.add_argument(
        "--config", 
        default="config/composite_layup.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (no real robot)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(args.config, args.simulate))
```

---

## Test Plan

### Unit Tests

Create `tests/test_demos/test_composite_layup/`:

```python
# test_scale_monitor.py
def test_scale_target_reached():
    """Test weight target detection."""
    pass

def test_scale_tolerance():
    """Test tolerance bounds."""
    pass

# test_defect_detector.py  
def test_dry_spot_detection():
    """Test detection of unwetted fiberglass."""
    pass

def test_bubble_detection():
    """Test air bubble detection."""
    pass

# test_pot_life.py
def test_pot_life_warnings():
    """Test warning levels at correct times."""
    pass

# test_pour_action.py
def test_controlled_pour():
    """Test pour with simulated scale feedback."""
    pass

# test_handoff.py
def test_handoff_sequence():
    """Test tool handoff action sequence."""
    pass
```

### Integration Test

```python
# test_integration.py
async def test_full_layup_sequence():
    """Test complete layup sequence in simulation."""
    config = load_config("config/composite_layup.yaml")
    config["robot"]["use_fake_hardware"] = True
    
    # Run through task graph
    # Verify each node completes
    # Check pot life tracking
    # Verify defect detection triggers on test images
    pass
```

---

## Validation Checklist

- [ ] Scale monitor reads weight accurately (±1g)
- [ ] Pot life timer fires warnings at correct thresholds
- [ ] Defect detection identifies test defects (>85% accuracy)
- [ ] Controlled pour stops within tolerance
- [ ] Handoff waits for human grasp
- [ ] Voice commands recognized
- [ ] Task graph executes in correct order
- [ ] Safety constraints enforced (speed limits near human)
- [ ] Cleanup task initiated before pot life expires

---

## Hardware Setup Notes

### Weigh Scale
- Recommend: Digital scale with serial output (RS-232 or USB)
- Alternative: Camera positioned to read LCD display
- Resolution: 0.1g or better
- Capacity: 500g minimum

### Camera Mounting
- Position: Above mold, angled 30-45° for surface inspection
- Field of view: Cover entire mold surface
- Lighting: Diffuse overhead lighting to reduce glare on resin

### Tool Layout
- Keep frequently used tools (roller, brush) within easy robot reach
- Containers should have graspable sections for 2F-85 gripper
- Mixing cup: Use graduated cup with straight sides

---

## Voice Command Reference

| Command | Action |
|---------|--------|
| "Robot, hand me the roller" | Fetch roller from tool station |
| "More resin" | Bring resin container for additional application |
| "Check this area" | Move camera to inspect indicated region |
| "Pause" | Halt robot motion, maintain state |
| "Continue" | Resume from pause |
| "Emergency stop" | Immediate halt, disengage |

---

## Troubleshooting

### Scale Reading Unstable
- Check for vibration isolation
- Increase averaging window in config
- Verify scale is on flat surface

### Defect False Positives
- Adjust confidence threshold upward
- Check lighting consistency
- Retrain on site-specific samples

### Pour Overshooting Target
- Reduce pour angle
- Increase scale read frequency
- Start slowing earlier (adjust remaining_g thresholds)

### Human Not Detected
- Check camera coverage
- Verify pose model loaded
- Adjust human detection confidence

---

## Next Steps

After completing this demo:
1. Collect data for fine-tuning defect detection
2. Record successful sequences for demonstration
3. Analyze timing to optimize robot motion
4. Extend to multi-ply automated scheduling

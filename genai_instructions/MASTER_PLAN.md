# AURA Master Development Plan

## Overview

This document outlines the complete development plan for AURA, broken into manageable tasks for AI coding agents. Each task is designed to be:
- **Self-contained**: Can be completed independently
- **Testable**: Has clear validation criteria
- **Incremental**: Builds on previous work

## Task Assignment Strategy

Tasks are organized into **Sprints** with clear dependencies. Each sprint has multiple parallel tracks that can be worked on simultaneously.

---

## Sprint 0: Foundation Setup ✅ (Current)

### Task 0.1: Project Structure
**Agent**: Any  
**Status**: ✅ Complete  
**Deliverables**:
- [x] Folder structure created
- [x] Documentation framework
- [x] pyproject.toml configured

### Task 0.2: Dependency Installation
**Agent**: Any  
**Instructions**: See [agents/00_setup_agent.md](agents/00_setup_agent.md)  
**Validation**: `uv sync` completes without errors

---

## Sprint 1: Core Interfaces & State Management

### Task 1.1: Define Core Data Types
**Agent**: `01_core_types_agent`  
**Instructions**: [agents/01_core_types_agent.md](agents/01_core_types_agent.md)  
**Deliverables**:
- `src/aura/core/types.py` - All dataclasses (State, Object, Pose, etc.)
- `src/aura/core/enums.py` - Status enums, action types
- Unit tests in `tests/test_core/`

**Validation**:
```bash
uv run pytest tests/test_core/ -v
```

### Task 1.2: Base Monitor Interface
**Agent**: `02_monitor_interface_agent`  
**Instructions**: [agents/02_monitor_interface_agent.md](agents/02_monitor_interface_agent.md)  
**Dependencies**: Task 1.1  
**Deliverables**:
- `src/aura/monitors/base_monitor.py` - Abstract base class
- `src/aura/monitors/monitor_bus.py` - Event bus for monitor communication

### Task 1.3: Configuration System
**Agent**: `03_config_agent`  
**Instructions**: [agents/03_config_agent.md](agents/03_config_agent.md)  
**Deliverables**:
- `src/aura/utils/config.py` - Config loading with Pydantic
- `config/default.yaml` - Default configuration
- `config/game_demo.yaml` - Game demo config

---

## Sprint 2: Perception Pipeline

### Task 2.1: Perception Module (SAM3 Integration)
**Agent**: `perception_agent`  
**Instructions**: [agents/04_perception_agent.md](agents/04_perception_agent.md)  
**Dependencies**: Sprint 1  
**Deliverables**:
- `src/aura/monitors/perception_module.py`
- Object detection and tracking
- Integration with existing `testing.py`

**Validation**:
```bash
uv run python scripts/test_perception.py --input webcam --duration 10
```

### Task 2.2: Human Intention and Motion Predictor
**Agent**: `human_motion_agent`  
**Instructions**: [agents/05_human_motion_agent.md](agents/05_motion_agent.md)  
**Dependencies**: Sprint 1  
**Reference**: `/home/mani/Repos/hcdt/LLMcalls/run_phase_two.py`  
**Deliverables**:
- `src/aura/monitors/motion_predictor.py`
- Human Intention Hand/pose prediction from video frames

### Task 2.3: Sound Monitor (Gemini Live)
**Agent**: `sound_agent`  
**Instructions**: [agents/06_sound_agent.md](agents/06_sound_agent.md)  
**Dependencies**: Sprint 1  
**Reference**: `/home/mani/Repos/aura/gemini_live_test.py`  
**Deliverables**:
- `src/aura/monitors/sound_monitor.py`
- Real-time speech to intent

---

## Sprint 3: Decision Engine (Brain)

### Task 3.1: State Manager
**Agent**: `brain_agent`  
**Instructions**: [agents/07_brain_agent.md](agents/07_brain_agent.md)  
**Dependencies**: Sprint 1, Sprint 2  
**Deliverables**:
- `src/aura/brain/state_manager.py`
- Aggregates all monitor outputs
- Thread-safe state updates

### Task 3.2: SOP Parser
**Agent**: `sop_agent`  
**Instructions**: [agents/08_sop_agent.md](agents/08_sop_agent.md)  
**Deliverables**:
- `src/aura/brain/sop_parser.py`
- Task graph from JSON
- Progress tracking

### Task 3.3: Decision Engine (LLM Integration)
**Agent**: `decision_agent`  
**Instructions**: [agents/09_decision_agent.md](agents/09_decision_agent.md)  
**Dependencies**: Task 3.1, 3.2  
**Deliverables**:
- `src/aura/brain/decision_engine.py`
- Gemini-based reasoning
- Explainable outputs

### Task 3.4: Brain Orchestrator
**Agent**: `brain_agent`  
**Dependencies**: Task 3.1, 3.2, 3.3  
**Deliverables**:
- `src/aura/brain/brain.py`
- Main loop coordinating all components

---

## Sprint 4: Action & Communication

### Task 4.1: Affordance Monitor
**Agent**: `affordance_agent`  
**Instructions**: [agents/10_affordance_agent.md](agents/10_affordance_agent.md)  
**Deliverables**:
- `src/aura/monitors/affordance_monitor.py`
- Scene-based action possibilities

### Task 4.2: Action Executor
**Agent**: `action_agent`  
**Instructions**: [agents/11_action_agent.md](agents/11_action_agent.md)  
**Deliverables**:
- `src/aura/actions/executor.py`
- Abstract action interface
- Sim and real implementations

### Task 4.3: Communication Module
**Agent**: `comm_agent`  
**Instructions**: [agents/12_comm_agent.md](agents/12_comm_agent.md)  
**Deliverables**:
- `src/aura/actions/communication.py`
- Speech synthesis
- Visual feedback

---

## Sprint 5: Integration - Game Demo

### Task 5.1: Game Interface Adapter
**Agent**: `game_agent`  
**Instructions**: [agents/13_game_agent.md](agents/13_game_agent.md)  
**Reference**: `/home/mani/Repos/proactive_hcdt/tests/collaborative_game.py`  
**Deliverables**:
- `src/aura/interfaces/game_interface.py`
- Bridge between AURA and collaborative game
- File-based command protocol

### Task 5.2: Intent Monitor (Game-Specific)
**Agent**: `intent_agent`  
**Instructions**: [agents/14_intent_agent.md](agents/14_intent_agent.md)  
**Deliverables**:
- `src/aura/monitors/intent_monitor.py`
- Detect which object human is targeting

### Task 5.3: Full Game Demo
**Agent**: Integration Agent  
**Dependencies**: All Sprint 5 tasks  
**Deliverables**:
- `scripts/run_game_demo.py`
- End-to-end demo script

**Validation**:
```bash
# Terminal 1: Start game
cd /home/mani/Repos/proactive_hcdt
conda run -n ur5_python python tests/collaborative_game.py

# Terminal 2: Run AURA
cd /home/mani/Repos/aura
uv run python scripts/run_game_demo.py
```

---

## Sprint 6: Robot Integration

### Task 6.1: Robot Interface Abstraction
**Agent**: `robot_agent`  
**Instructions**: [agents/15_robot_agent.md](agents/15_robot_agent.md)  
**Reference**: `/home/mani/Repos/ur_ws/robot_interface.py`  
**Deliverables**:
- `src/aura/interfaces/robot_interface.py`
- Unified interface for sim/real

### Task 6.2: Isaac Sim Digital Twin
**Agent**: `isaac_agent`  
**Instructions**: [agents/16_isaac_agent.md](agents/16_isaac_agent.md)  
**Reference**: Isaac Sim UR10 Cortex extension  
**Deliverables**:
- `src/aura/visualization/digital_twin.py`
- Isaac Sim standalone script

### Task 6.3: Full Robot Demo
**Agent**: Integration Agent  
**Dependencies**: All Sprint 6 tasks  
**Deliverables**:
- `scripts/run_robot_demo.py`

---

## Sprint 7: Composite Layup Demo 🆕

### Task 7.1: Scale & Pot Life Monitors
**Agent**: `07_composite_layup_agent`  
**Instructions**: [agents/07_composite_layup_agent.md](agents/07_composite_layup_agent.md)  
**Dependencies**: Sprint 2 monitors  
**Deliverables**:
- `src/aura/monitors/scale_monitor.py` - Weight reading from scale
- `src/aura/monitors/pot_life_monitor.py` - Epoxy working time tracker

**Validation**:
```bash
uv run pytest tests/test_demos/test_composite_layup/test_scale_monitor.py -v
uv run pytest tests/test_demos/test_composite_layup/test_pot_life.py -v
```

### Task 7.2: Defect Detection
**Agent**: `07_composite_layup_agent`  
**Dependencies**: Perception monitor (Sprint 2)  
**Deliverables**:
- `src/aura/monitors/defect_detector.py` - SAM3-based defect detection
- Defect types: dry_spot, air_bubble, wrinkle, fiber_misalignment

**Validation**:
```bash
uv run pytest tests/test_demos/test_composite_layup/test_defect_detector.py -v
```

### Task 7.3: Controlled Pour & Handoff Actions
**Agent**: `07_composite_layup_agent`  
**Dependencies**: Robot interface (Sprint 6)  
**Deliverables**:
- `src/aura/actions/pour_action.py` - Weight-feedback controlled pouring
- `src/aura/actions/handoff_action.py` - Safe tool handoff to human

### Task 7.4: Demo Integration
**Agent**: `07_composite_layup_agent`  
**Dependencies**: Tasks 7.1-7.3  
**Configuration Files**:
- `config/composite_layup.yaml` - Demo-specific config
- `sops/composite_layup.json` - Task graph with 13 nodes

**Deliverables**:
- `src/aura/demos/composite_layup/main.py` - Main entry point
- End-to-end demo with UR5 + Robotiq 2F-85

**Run Demo**:
```bash
# Simulation mode
python -m aura.demos.composite_layup.main --config config/composite_layup.yaml --simulate

# Real robot
python -m aura.demos.composite_layup.main --config config/composite_layup.yaml
```

**Key Features**:
- Resin/hardener mixing at 100:30 ratio by weight
- 4-ply fiberglass layup with alternating 0°/90° orientation
- Real-time defect detection (dry spots, bubbles, wrinkles)
- Pot life tracking with 30/40 minute warnings
- Voice commands for tool requests
- Safe human-robot handoffs

---

## Validation Checklist

Each sprint must pass these validation gates:

### Unit Tests
```bash
uv run pytest tests/ -v --cov=src/aura
```

### Type Checking
```bash
uv run mypy src/aura/
```

### Linting
```bash
uv run ruff check src/aura/
```

### Integration Tests
```bash
uv run pytest tests/integration/ -v
```

---

## Agent Handoff Protocol

When an agent completes a task:

1. **Code Complete**: All files created/modified
2. **Tests Pass**: Validation commands succeed
3. **Documentation Updated**: Relevant docstrings and README updates
4. **Handoff Notes**: Create `genai_instructions/handoff/<task_id>.md` with:
   - What was implemented
   - Known limitations
   - Suggested next steps
   - Any blockers for dependent tasks

---

## Quick Reference: File Locations

| Component | Primary File | Config |
|-----------|-------------|--------|
| Brain | `src/aura/brain/brain.py` | `config/default.yaml` |
| Perception | `src/aura/monitors/perception_module.py` | - |
| Sound | `src/aura/monitors/sound_monitor.py` | - |
| Motion | `src/aura/monitors/motion_predictor.py` | - |
| Intent | `src/aura/monitors/intent_monitor.py` | - |
| Affordance | `src/aura/monitors/affordance_monitor.py` | - |
| Performance | `src/aura/monitors/performance_monitor.py` | - |
| Robot | `src/aura/interfaces/robot_interface.py` | - |
| Game | `src/aura/interfaces/game_interface.py` | - |
| State | `src/aura/core/types.py` | - |

---

## Environment Variables

Required for full functionality:

```bash
export GEMINI_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"  # Optional, for backup
```

---

## Development Tips

1. **Start with tests**: Write test cases before implementation
2. **Use type hints**: All functions should have type annotations
3. **Async where needed**: Monitors should be async-capable
4. **Logging**: Use `src/aura/utils/logging.py` for consistent logs
5. **Config over hardcode**: No hardcoded paths, use config system

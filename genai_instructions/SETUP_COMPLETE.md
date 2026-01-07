# AURA Project Setup Complete

**Date**: 2026-01-06  
**Status**: Foundation Complete ✅

## What Was Created

### 1. Project Structure
```
aura/
├── src/aura/           # Main package
│   ├── core/           # Types and enums (IMPLEMENTED)
│   ├── brain/          # Decision engine (TODO)
│   ├── monitors/       # Perception, sound, etc. (TODO)
│   ├── actions/        # Robot actions (TODO)
│   ├── interfaces/     # Game, robot interfaces (TODO)
│   ├── visualization/  # Digital twin (TODO)
│   └── utils/          # Config, logging (TODO)
├── config/             # YAML configs
├── sops/               # Task graphs (1 example)
├── tests/              # Pytest tests
├── scripts/            # Demo scripts (TODO)
└── genai_instructions/ # AI agent tasks
```

### 2. Core Types Implemented
- `Pose2D`, `Pose3D`, `BoundingBox`, `Trajectory`
- `TrackedObject`, `SceneGraph`
- `HumanPose`, `Intent`, `PredictedMotion`
- `Action`, `Affordance`
- `TaskNode`, `TaskGraph`
- `AuraState` - Central state container
- `MonitorOutput` types for all monitors

### 3. Configuration
- `config/default.yaml` - Default settings
- `sops/sorting_task.json` - Example SOP

### 4. Documentation
- [Documentation.md](Documentation.md) - Architecture overview
- [MASTER_PLAN.md](MASTER_PLAN.md) - Development roadmap
- [agents/](agents/) - 6 agent task files created
- [validation/test_checklists.md](validation/test_checklists.md)

### 5. Dependencies
All dependencies installed via `uv sync`:
- PyTorch, OpenCV, SAM3
- google-genai for Gemini
- pyaudio for sound
- pytest for testing

## How to Run Tests

```bash
cd /home/mani/Repos/aura
./run_tests.sh tests/test_core/ -v
```

Or with clean environment:
```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/ -v
```

## Next Steps for AI Agents

### Sprint 1 (Ready to Start)
1. **Task 1.2**: Implement `BaseMonitor` and `MonitorEventBus`
   - See: [agents/02_monitor_interface_agent.md](agents/02_monitor_interface_agent.md)

2. **Task 1.3**: Implement config loading with Pydantic
   - Create: `src/aura/utils/config.py`

### Sprint 2 (After Sprint 1)
1. **Task 2.1**: Perception Module
   - See: [agents/04_perception_agent.md](agents/04_perception_agent.md)
   - Reference: `/home/mani/Repos/aura/testing.py`

2. **Task 2.3**: Sound Monitor
   - See: [agents/06_sound_agent.md](agents/06_sound_agent.md)
   - Reference: `/home/mani/Repos/aura/gemini_live_test.py`

### Sprint 5 (Game Demo)
1. **Task 5.1**: Game Interface
   - See: [agents/13_game_agent.md](agents/13_game_agent.md)
   - Reference: `/home/mani/Repos/proactive_hcdt/tests/COLLABORATIVE_GAME_README.md`

## Key Files for Agents

| Purpose | File |
|---------|------|
| Core types | `src/aura/core/types.py` |
| Enums | `src/aura/core/enums.py` |
| Default config | `config/default.yaml` |
| Example SOP | `sops/sorting_task.json` |
| Test runner | `run_tests.sh` |

## Environment Variables Required

```bash
export GEMINI_API_KEY="your-api-key"
```

## Validation Complete

- ✅ Project structure created
- ✅ Core types implemented and tested
- ✅ Dependencies installed with uv
- ✅ 10/10 core type tests passing
- ✅ Agent task files created

## Notes

- ROS pytest plugins conflict with regular pytest. Use `./run_tests.sh` or set `PYTHONPATH=""` before running pytest.
- SAM3 is installed from `third_party/sam3` as editable.
- The collaborative game is in a separate repo at `/home/mani/Repos/proactive_hcdt/`.

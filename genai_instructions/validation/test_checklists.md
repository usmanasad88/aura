# Validation Test Checklists

This document provides comprehensive validation checklists for each AURA component.

---

## Sprint 0: Foundation

### Task 0.1: Project Structure
- [ ] All directories exist as per Documentation.md
- [ ] All `__init__.py` files present
- [ ] pyproject.toml has correct dependencies

### Task 0.2: Dependencies
```bash
cd /home/mani/Repos/aura
uv sync
```

**Expected**: No errors, `.venv` created

**Verification**:
```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "from google import genai; print('Gemini OK')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

---

## Sprint 1: Core Interfaces

### Task 1.1: Core Types
```bash
uv run pytest tests/test_core/test_types.py -v
```

**Expected**: All tests pass

**Manual checks**:
```python
from aura.core import AuraState, TrackedObject, Pose3D
state = AuraState()
obj = TrackedObject(id="1", name="cup", category="container")
pose = Pose3D.from_position(1, 2, 3)
print(f"State: {state}")
print(f"Object: {obj}")
print(f"Pose: {pose}")
```

### Task 1.2: Monitor Interface
```bash
uv run pytest tests/test_monitors/test_base_monitor.py -v
```

**Expected**: All async tests pass

**Manual checks**:
```python
import asyncio
from aura.monitors import MonitorEventBus
bus = MonitorEventBus()
print(f"Bus history: {bus.get_history()}")
```

### Task 1.3: Configuration
```bash
uv run python -c "from aura.utils.config import load_config; cfg = load_config(); print(cfg)"
```

**Expected**: Default config loaded without errors

---

## Sprint 2: Perception Pipeline

### Task 2.1: Perception Module
```bash
uv run python scripts/test_perception.py --input webcam --duration 5
```

**Expected**:
- [ ] SAM3 model loads (may take 20-30s first time)
- [ ] Webcam opens
- [ ] Objects detected and visualized
- [ ] FPS displayed (should be >5)

**Without display**:
```bash
uv run python scripts/test_perception.py --input /path/to/image.jpg
```

### Task 2.2: Motion Predictor
```bash
uv run pytest tests/test_monitors/test_motion.py -v
```

**Expected**: Predictions generated for sample data

### Task 2.3: Sound Monitor
```bash
# Requires microphone and GEMINI_API_KEY
uv run python scripts/test_sound_monitor.py --mode conversation
```

**Expected**:
- [ ] Microphone captures audio
- [ ] Gemini responds to speech
- [ ] Text displayed in terminal

---

## Sprint 3: Decision Engine

### Task 3.1: State Manager
```bash
uv run pytest tests/test_brain/test_state.py -v
```

### Task 3.2: SOP Parser
```bash
uv run python -c "
from aura.brain.sop_parser import load_sop
sop = load_sop('sops/sorting_task.json')
print(f'Task: {sop.name}')
print(f'Nodes: {len(sop.nodes)}')
"
```

### Task 3.3: Decision Engine
```bash
uv run pytest tests/test_brain/test_decision.py -v
```

### Task 3.4: Brain Orchestrator
```bash
uv run pytest tests/test_brain/test_brain.py -v
```

---

## Sprint 4: Actions

### Task 4.1: Affordance Monitor
```bash
uv run pytest tests/test_monitors/test_affordance.py -v
```

### Task 4.2: Action Executor
```bash
uv run pytest tests/test_actions/test_executor.py -v
```

### Task 4.3: Communication
```bash
uv run python scripts/test_communication.py
```

---

## Sprint 5: Game Integration

### Task 5.1: Game Interface
```bash
# Terminal 1: Start game
cd /home/mani/Repos/proactive_hcdt
conda run -n ur5_python python tests/collaborative_game.py

# Terminal 2: Test interface
cd /home/mani/Repos/aura
uv run python scripts/run_game_demo.py --mode monitor
```

**Expected**:
- [ ] Game state read correctly
- [ ] Objects parsed
- [ ] Human/AI positions shown

### Task 5.2: Full Game Demo
```bash
# Terminal 1: Game running
# Terminal 2: Agent
uv run python scripts/run_game_demo.py --mode agent
```

**Expected**:
- [ ] Agent waits for game
- [ ] Selects objects to push
- [ ] Avoids human's targets
- [ ] Displays reasoning in game UI
- [ ] Scores objects

---

## Sprint 6: Robot Integration

### Task 6.1: Robot Interface
```bash
# Simulation mode
uv run python -c "
from aura.interfaces.robot_interface import RobotInterface
robot = RobotInterface(sim_mode=True)
robot.move_to_pose(0.5, 0.2, 0.3)
robot.close_gripper()
"
```

### Task 6.2: Isaac Sim Digital Twin
```bash
# Requires Isaac Sim
cd /home/mani/isaac-sim-standalone-5.0.0-linux-x86_64
./python.sh /home/mani/Repos/aura/scripts/test_digital_twin.py
```

---

## Integration Tests

### Full Pipeline Test
```bash
uv run pytest tests/integration/test_full_pipeline.py -v
```

### End-to-End with Game
```bash
uv run pytest tests/integration/test_game_e2e.py -v
```

---

## Performance Benchmarks

### Perception FPS
```bash
uv run python scripts/benchmark_perception.py
```

**Target**: >10 FPS on GPU

### Decision Latency
```bash
uv run python scripts/benchmark_decision.py
```

**Target**: <500ms per decision

---

## Code Quality

### Type Checking
```bash
uv run mypy src/aura/ --ignore-missing-imports
```

**Expected**: No errors (or only warnings)

### Linting
```bash
uv run ruff check src/aura/
```

**Expected**: No errors

### Test Coverage
```bash
uv run pytest tests/ --cov=src/aura --cov-report=html
```

**Target**: >70% coverage

---

## Common Issues

### SAM3 CUDA Errors
If SAM3 fails with CUDA errors:
```bash
# Check CUDA version
nvidia-smi
# Reinstall PyTorch for correct CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Gemini API Errors
```bash
# Verify API key
echo $GEMINI_API_KEY
# Test connection
uv run python -c "
from google import genai
client = genai.Client(api_key='$GEMINI_API_KEY')
print('Connected')
"
```

### PyAudio Installation
```bash
# Ubuntu
sudo apt-get install portaudio19-dev
uv pip install pyaudio
```

### Game Not Detected
Ensure game is writing to `ai_runtime_status.json`:
```bash
ls -la /home/mani/Repos/proactive_hcdt/ai_runtime_status.json
cat /home/mani/Repos/proactive_hcdt/ai_runtime_status.json | head
```

# AURA - Proactive Robotic Assistance Framework

## Vision Statement

AURA (Autonomous Understanding for Robotic Assistance) is a modular, explainable framework for proactive robotic assistance. Rather than waiting for explicit commands, AURA enables robots to:
- Understand human intent through multi-modal perception
- Predict human actions and workspace changes
- Decide what to do based on affordances and task state
- Communicate or execute actions appropriately
- Monitor performance and adapt

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AURA BRAIN                                      │
│  (Central Decision Engine - LLM-based reasoning with structured state)      │
├─────────────────────────────────────────────────────────────────────────────┤
│                          INPUT MONITORS                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │   Intent    │ │   Motion    │ │ Perception  │ │        Sound            ││
│  │   Monitor   │ │  Predictor  │ │   Module    │ │        Monitor          ││
│  │             │ │             │ │  (SAM3/VLM) │ │    (Gemini Live)        ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Gesture Monitor (MediaPipe)                       ││
│  │                   (Safety Control & Gesture Recognition)                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                          CAPABILITY LAYER                                    │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │  Affordance Monitor │           │ Performance Monitor │                  │
│  │  (What can I do?)   │           │ (Is it working?)    │                  │
│  └─────────────────────┘           └─────────────────────┘                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                          OUTPUT LAYER                                        │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │   Action Executor   │           │  Communication      │                  │
│  │  (Robot Interface)  │           │  (Speech/Visual)    │                  │
│  └─────────────────────┘           └─────────────────────┘                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                          VISUALIZATION                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Isaac Sim Digital Twin / Game Visualization / RViz                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Brain (Central Decision Engine)
- **Location**: `src/aura/brain/`
- **Purpose**: Orchestrates all monitors, maintains state, makes decisions
- **Key Features**:
  - State machine for task progression
  - LLM-based reasoning with structured prompts
  - Explainable decision logging
  - SOP-guided behavior

### 2. Intent Monitor
- **Location**: `src/aura/monitors/intent_monitor.py`
- **Purpose**: Detect what the human is trying to do
- **Inputs**: Vision, pose, historical actions
- **Outputs**: Intent classification, confidence score

### 3. Motion Predictor
- **Location**: `src/aura/monitors/motion_predictor.py`
- **Purpose**: Predict future human/object positions
- **Based on**: Previous HCDT motion prediction work
- **Outputs**: Predicted trajectories, collision risks

### 4. Gesture Monitor (NEW)
- **Location**: `src/aura/monitors/gesture_monitor.py`
- **Purpose**: Real-time hand gesture recognition for safety and interaction
- **Technology**: MediaPipe Gesture Recognizer
- **Features**:
  - Multi-hand tracking (up to 2 hands)
  - Safety control (stop/resume gestures)
  - Intent mapping from gestures
  - Debouncing for stable recognition
- **Outputs**: Detected gestures, safety status, mapped intent
- **Documentation**: See [Gesture Monitor Guide](docs/gesture_monitor.md)

### 5. Perception Module
- **Location**: `src/aura/monitors/perception_module.py`
- **Purpose**: Object detection, segmentation, tracking
- **Components**:
  - SAM3 for object masks
  - Gemini for scene understanding
  - Object pose estimation

### 6. Sound Monitor
- **Location**: `src/aura/monitors/sound_monitor.py`
- **Purpose**: Speech recognition and intent extraction
- **Technology**: Gemini Live API
- **Features**: Real-time transcription, command extraction

### 7. Affordance Monitor
- **Location**: `src/aura/monitors/affordance_monitor.py`
- **Purpose**: Determine what actions are possible
- **Outputs**: List of available actions with parameters

### 8. Performance Monitor
- **Location**: `src/aura/monitors/performance_monitor.py`
- **Purpose**: Track execution success/failure
- **Features**: Timeout detection, error recovery

## State Management

The system maintains a comprehensive state object:

```python
@dataclass
class AuraState:
    # Scene understanding
    objects: List[TrackedObject]
    object_positions: Dict[str, Pose]
    
    # Human tracking
    human_pose: Optional[HumanPose]
    human_intent: Optional[Intent]
    predicted_trajectory: Optional[Trajectory]
    
    # Task state
    task_graph: TaskGraph
    current_task_node: str
    task_progress: float
    
    # Communication
    verbal_history: List[Utterance]
    pending_clarifications: List[str]
    
    # Robot capabilities
    available_actions: List[Action]
    current_action: Optional[Action]
    action_status: ActionStatus
    
    # Performance
    performance_metrics: PerformanceMetrics
    error_log: List[Error]
```

## Development Phases

### Phase 1: Foundation (Current)
- [x] Project structure setup
- [ ] Core interfaces defined
- [ ] Collaborative game integration
- [ ] Basic perception with SAM3

### Phase 2: Perception Pipeline
- [ ] Intent Monitor implementation
- [ ] Motion Predictor integration
- [ ] Sound Monitor with Gemini Live
- [ ] Object tracking pipeline

### Phase 3: Decision Engine
- [ ] Brain implementation
- [ ] SOP parsing and following
- [ ] Explainable decision logging
- [ ] Affordance reasoning

### Phase 4: Robot Integration
- [ ] Robot interface abstraction
- [ ] cuRobo motion planning
- [ ] Isaac Sim digital twin
- [ ] Real robot deployment

### Phase 5: Case Studies
- [ ] Collaborative sorting game
- [ ] Composite layup assistance
- [ ] Additional domains

## Directory Structure

```
aura/
├── src/aura/
│   ├── __init__.py
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── brain.py              # Main brain orchestrator
│   │   ├── state.py              # State definitions
│   │   ├── decision_engine.py    # LLM reasoning
│   │   └── sop_parser.py         # SOP/task graph handling
│   ├── monitors/
│   │   ├── __init__.py
│   │   ├── base_monitor.py       # Abstract monitor interface
│   │   ├── intent_monitor.py
│   │   ├── motion_predictor.py
│   │   ├── perception_module.py
│   │   ├── sound_monitor.py
│   │   ├── affordance_monitor.py
│   │   └── performance_monitor.py
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── base_action.py
│   │   ├── robot_actions.py
│   │   └── communication.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── visualizer.py
│   │   └── digital_twin.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── robot_interface.py
│   │   └── game_interface.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── config.py
├── config/
│   ├── default.yaml
│   ├── game_demo.yaml
│   └── robot_demo.yaml
├── sops/
│   ├── sorting_task.json
│   └── composite_layup.json
├── tests/
│   ├── test_monitors/
│   ├── test_brain/
│   └── integration/
├── scripts/
│   ├── run_game_demo.py
│   ├── run_perception_test.py
│   └── run_robot_demo.py
├── genai_instructions/
│   ├── Documentation.md           # This file
│   ├── MASTER_PLAN.md             # Overall development plan
│   ├── agents/                    # Instructions for AI agents
│   │   ├── 01_perception_agent.md
│   │   ├── 02_motion_agent.md
│   │   ├── 03_brain_agent.md
│   │   ├── 04_robot_agent.md
│   │   └── 05_game_agent.md
│   └── validation/
│       └── test_checklists.md
└── third_party/
    └── sam3/                      # SAM3 submodule
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Object Segmentation | SAM3 | Real-time object masks |
| Scene Understanding | Gemini 2.0 | VLM reasoning |
| Speech | Gemini Live API | Real-time audio |
| Motion Planning | cuRobo | Collision-free paths |
| Simulation | Isaac Sim 5.0 | Digital twin |
| Robot Control | ROS 2 Humble | Real robot interface |
| Visualization | PyGame / Isaac Sim | Demo and debug |

## Configuration

AURA uses YAML configuration files:

```yaml
# config/default.yaml
brain:
  llm_model: "gemini-2.0-flash"
  decision_interval_ms: 500
  max_thinking_time_ms: 2000

perception:
  sam3_enabled: true
  gemini_enabled: true
  update_rate_hz: 10

sound:
  enabled: true
  gemini_live_model: "gemini-2.5-flash-native-audio-preview"

motion:
  prediction_horizon_sec: 2.0
  update_rate_hz: 15

robot:
  type: "ur5"  # or "ur10", "sim"
  interface: "ros2"  # or "http", "sim"
```

## Getting Started

See [MASTER_PLAN.md](MASTER_PLAN.md) for the development roadmap and agent task assignments.

## Demo Applications

### Composite Layup Assistance Demo

A UR5 robot with Robotiq 2F-85 gripper assists a human operator with fiberglass hand layup on a mold.

**Files**:
- Configuration: `config/composite_layup.yaml`
- Task Graph: `sops/composite_layup.json`
- Demo Code: `src/aura/demos/composite_layup/`
- Agent Instructions: `genai_instructions/agents/07_composite_layup_agent.md`
- Implementation Guide: `genai_instructions/demos/composite_layup_demo.md`

**Key Features**:
- Resin/hardener measurement at 100:30 ratio using weigh scale
- Pot life tracking (45 min) with warning thresholds
- 4-ply fiberglass layup with quality inspection
- Defect detection (dry spots, bubbles, wrinkles)
- Voice-activated tool handoffs
- Safe human-robot collaboration

**Run Demo**:
```bash
# Simulation mode
python -m aura.demos.composite_layup.main --simulate

# Real robot
python -m aura.demos.composite_layup.main --config config/composite_layup.yaml
```

## References

- [proactive_hcdt](../../../proactive_hcdt/): Original collaborative game
- [hcdt](../../../hcdt/): Motion prediction research
- [ur_ws](../../../ur_ws/): Robot control workspace
- [Isaac Sim](../../../isaac-sim-standalone-5.0.0-linux-x86_64/): Simulation


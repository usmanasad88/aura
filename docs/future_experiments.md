# AURA Future Experiments & Extensions

This document captures experiment ideas that are not yet implemented but could strengthen the paper's evaluation section.

---

## Part B: Live Robot Experiments (Not Yet Implemented)

### Wizard-of-Oz Data Collection

Follow the `weigh_bottles` demo_data pattern to collect ground truth from expert operators:

1. **Recording script** (`scripts/record_woz_session.py`):
   - Launch video recording (webcam/GoPro)
   - Connect to robot API at `localhost:5050`
   - Present operator UI (Streamlit from `ur_ws`) showing camera feed
   - Log operator-triggered programs with timestamps → `program_events.json`
   - Record robot joint states (via `/api/status` polling) → `joint_states.json`
   - Save `metadata.json` with session info
   - Output: `demo_data/hand_layup_woz/session_YYYYMMDD_HHMMSS/`

2. **Ground truth conversion**:
   - Convert `program_events.json` → `ground_truth.json` format
   - Multiple operators capture inter-operator variance
   - These become additional benchmarks beyond the existing annotated video

3. **Live experiment runner** (`scripts/run_live_experiment.sh`):
   - Launches `ur_ws/launch_all.sh` (robot stack)
   - Launches `run_aura.py --live --robot-url http://localhost:5050 --webcam 0`
   - Records video + logs simultaneously
   - Requires human performing the layup task in real-time

### Cross-Task Generalization (Tier 5)

Run the best 2 models on `weigh_bottles` task using existing demo data:

```bash
# weigh_bottles already has Wizard-of-Oz ground truth
./scripts/run_experiments.sh --task weigh_bottles \
    --video demo_data/weigh_bottles/video.mp4 --tier 1 --reps 2
```

Would demonstrate task-agnostic capabilities of the framework.

---

## Additional Experiment Tiers

### Split Model Experiments (Tier 2)

Test cost-accuracy tradeoffs by using different models for intent vs decision:

| Intent Model | Decision Model | Hypothesis |
|-------------|---------------|------------|
| gemini-2.5-flash | gemini-2.5-pro | Cheap perception, expensive reasoning |
| gemini-2.5-pro | gemini-2.5-flash | Expensive perception, cheap reasoning |
| Local VLM (Qwen3.5-VL-4B) | gemini-2.5-pro | Zero-cost perception, cloud reasoning |

Already supported by `run_experiments.sh --tier 2` and `run_aura.py` flags:
`--intent-backend`, `--intent-model`, `--decision-backend`, `--decision-model`

### Frame Skip Ablation (Tier 3)

Test temporal resolution effect on prediction quality:

| Frame Skip | Effective Rate | Expected Effect |
|-----------|---------------|-----------------|
| 15 | 2 fps | More context, higher API cost |
| 30 | 1 fps | Baseline |
| 60 | 0.5 fps | Reduced cost, possible missed transitions |
| 90 | 0.33 fps | Minimal context, fastest |

Already supported by `run_experiments.sh --tier 3`.

### Ground Truth Ablation (Tier 4)

Compare system performance with vs without oracle robot status:
- `--use-ground-truth-robot-status`: Robot state (idle/busy, active program) is injected from task config
- Without: System must infer robot state from context alone

Already supported by `run_experiments.sh --tier 4`.

---

## Additional Monitor Integration

### Affordance Monitor

Create a rule-based (no VLM) affordance checker for hand_layup:
- Evaluates `preconditions` from `robot_skills.json` against current SSG state
- Returns list of currently available skills
- Could reduce false positive robot actions
- Implementation:
  - Create `src/aura/monitors/hand_layup_affordance_monitor.py` extending `AffordanceMonitor`
  - Add `"affordance"` to `active_monitors` in `task_profile.json`
  - Add `run_affordance_node` to workflow in `builder.py` (after `update_ssg_node`, before `decide_action_node`)

### Performance Monitor

VLM-based defect detection during layup:
- Detect wrinkles, air bubbles, fiber misalignment
- Adds another evaluation dimension (quality monitoring)
- Higher latency — would need careful integration to not slow the main loop

### Body Pose Monitor

3D human body pose tracking for safety:
- Detect proximity to robot workspace
- Trigger safety pauses before collisions
- Uses SAM-3D-Body via ZMQ (already implemented, needs hand_layup config)

---

## Local VLM Experiments

Test open-source models via SGLang for cost-free inference:

```bash
# Start SGLang server first
./scripts/start_sglang_server.sh --model Qwen/Qwen3.5-VL-4B-Instruct

# Run with local backend
./scripts/run_experiments.sh --tier 1 --reps 1 \
    # Modify MODELS array to use local models
```

Requires GPU with sufficient VRAM (4B model needs ~10GB).

---

## Evaluation Extensions

### Subjective A_mod Component

The paper defines A_mod as combining objective + subjective assessment.
Current implementation only uses objective (action type match).
Future work: add post-hoc human ratings via annotation UI.

### Confidence Calibration Analysis

Plot predicted confidence vs actual accuracy to assess calibration:
- Well-calibrated: 80% confidence → 80% correct
- Overconfident: High confidence but low accuracy
- Data already available in `response_parsed.json` → `prediction_confidence`

### Multi-Video Evaluation

Record additional hand layup sessions with different:
- Operators (skill levels)
- Task variations (different number of layers, different materials)
- Environmental conditions (lighting, camera angle)
- Error scenarios (operator makes mistakes the robot should detect)

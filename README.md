# AURA — Agentic Unified Robotic Assistance

Reference implementation of the AURA framework for proactive human–robot
collaboration described in *"Modular framework for responsive and explainable
robotic assistance with intention prediction using human-centric digital
twins"* (Asad et al., 2026).

AURA couples a VLM-driven Intent Monitor and a modular set of perception,
pose, gesture, sound, and affordance monitors with a hybrid behaviour-tree /
LLM decision engine. The runtime is a LangGraph state machine executing a
continuous *sense → decide → act* loop over a shared Semantic Scene Graph
(SSG), grounded by task-specific JSON config (DAG, state schema, skills,
initial scene).

The robot side of the paper (UR5 + Robotiq 2F-85, `.prog` DSL, cuRobo planner,
Quest/SpaceMouse teleop, Isaac Sim digital twin) lives in the companion
[ur_ws](../ur_ws) repository.

---

## 1. Install

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> aura
cd aura
uv sync                       # core dependencies
uv sync --extra pose-tracking # optional: 6-DoF pose pipeline
```

Create a `.env` in the repo root with your API keys:

```env
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...     # optional, for OpenAI backend
```

---

## 2. Demo data

The framework expects task videos and reference images under
[`demo_data/`](demo_data/). This directory is **not** tracked in git.

> **TODO:** Download `demo_data/` from Google Drive — `<link to be added>`
> and extract into the repo root so the layout matches:
>
> ```
> demo_data/
>   layup_demo/    layup_gesture_demo*.mp4, anchor_image_layup_stationary.png
>   tea/           tea_making*.mp4
>   sorting/       ...
>   weigh_bottles/ ...
> ```

Ground-truth intent annotations referenced by `--intent-source ground_truth`
live under `tasks/<task>/ground_truth/<video_stem>.intent_gt.json` and are
included with the source tree.

---

## 3. Running the framework — `scripts/run_aura.py`

The single entry point is [scripts/run_aura.py](scripts/run_aura.py). It
builds the task-specific LangGraph, initialises the SSG from
`tasks/<task>/config/initial_scene.json`, and runs the perception–decision
loop. A real-time web dashboard is served at <http://localhost:5555>.

### Web launcher (no CLI args)

```bash
uv run python scripts/run_aura.py --ui
```

Opens a browser launcher to pick task, video/webcam, models, monitors.

### CLI — typical configurations

```bash
# Offline replay of a demo video, dry-run (no robot), with dashboard
uv run python scripts/run_aura.py \
    --task hand_layup \
    --video demo_data/layup_demo/layup_gesture_demo_stationary_with_overlay.mp4 \
    --dry-run

# Live webcam, real robot (UR5 controlled via ur_ws REST API)
uv run python scripts/run_aura.py \
    --task hand_layup --webcam 0 --live \
    --robot-url http://localhost:5050

# Local VLM via SGLang (start server first: ./scripts/start_sglang_server.sh)
uv run python scripts/run_aura.py \
    --task hand_layup --video demo_data/layup_demo/layup_gesture_demo.mp4 \
    --llm-backend sglang --model Qwen/Qwen3.5-VL-4B-Instruct \
    --sglang-url http://localhost:8100/v1

# Voice control (Gemini Live audio in parallel with the visual loop)
uv run python scripts/run_aura.py \
    --task hand_layup --webcam 0 --live --audio \
    --audio-input-device USB --audio-output-device Analog
```

### Key flags

| Flag | Purpose |
|---|---|
| `--task <name>` | Selects `tasks/<name>/config/` (required for CLI mode). Available: `hand_layup`, `cuboid_manipulation`, `tea_making`, `sorting`, `weigh_bottles`. |
| `--video <path>` / `--webcam <idx>` / `--screen-capture` / `--gopro-stream` | Frame source. Exactly one. |
| `--dry-run` (default) / `--live` | Log vs. dispatch robot commands. `--live` requires a reachable `--robot-url`. |
| `--robot-url <url>` | Base URL of the ur_ws External Control API (default `http://localhost:5050`). |
| `--model <id>` | VLM used by both intent and decision (default `gemini-3.1-pro-preview`). Per-component overrides: `--intent-model`, `--decision-model`. |
| `--llm-backend {gemini,openai,sglang,vllm,ollama,local}` | Backend (default `gemini`). Per-component: `--intent-backend`, `--decision-backend`. |
| `--no-realtime` `--frame-skip N` `--max-cycles N` | Offline replay: process every Nth frame, cap LLM calls. Used by the experiment runner. |
| `--intent-source {llm,ground_truth}` | Source of intent predictions. `ground_truth` replays annotated keyframes from `tasks/<task>/ground_truth/`, isolating the decision engine. |
| `--enable-pose` / `--no-pose`, `--pose-endpoint` | Enable the SAM-3D-Body activity gate on the intent monitor (requires the pose server, below). |
| `--no-dashboard`, `--dashboard-port` | Disable / re-port the web dashboard. |

Run `uv run python scripts/run_aura.py --help` for the full list.

### Pose server (optional)

The Body Pose / Activity monitor delegates inference to an external
Fast-SAM-3D-Body ZMQ service. In a separate shell:

```bash
./scripts/run_aura_server.sh           # default tcp://localhost:5556
```

Then add `--enable-pose` to the `run_aura.py` invocation.

### Experiment runner

To reproduce the paper's evaluation matrix (A-Score, intent F1, ablations):

```bash
./scripts/run_experiments.sh                # all tiers
./scripts/run_experiments.sh --tier 1       # decision-engine isolation
./scripts/run_experiments.sh --eval-only    # re-score existing logs
```

Outputs land in `logs/experiments/<experiment_id>/rep_NNN/` with aggregate
tables in `results/`.

---

## 4. Repository layout

```
src/aura/        framework source (monitors, workflow, brain, interfaces)
scripts/         entry points: run_aura.py, run_experiments.sh, eval/, ...
tasks/<name>/    per-task config (dag.json, robot_skills.json, state_schema.json,
                 task_profile.json, initial_scene.json) + ground_truth/
config/          defaults (backend max tokens, perception thresholds)
demo_data/       (gitignored) videos and reference images — see §2
logs/            run logs, experiment results
third_party/     vendored SAM3
```

---

## 5. Citing

If you use this code or the released dataset, please cite the paper:

```bibtex
@article{Asad2026AURA,
  title  = {Modular framework for responsive and explainable robotic
            assistance with intention prediction using human-centric
            digital twins},
  author = {Asad, Usman and Khalid, Azfar and Lughmani, Waqas Akbar
            and Rasheed, Shummaila and Khan, Muhammad Mahabat},
  year   = {2026}
}
```

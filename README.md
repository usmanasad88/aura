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
[ur_ws](https://github.com/usmanasad88/ur5-robotiq-ros2-control) repository.

---

## 1. Install

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> aura
cd aura
uv sync                       # core dependencies
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

Fetch it from the public Google Drive
[folder](https://drive.google.com/drive/folders/1baUVBFkgLUW8HMS8z6C7hPzUAPxtRd_U)
with the bundled helper (uses `gdown`; the script installs it on the fly if
missing):

```bash
./scripts/download_demo_data.sh           # default location
./scripts/download_demo_data.sh --force   # re-download / overwrite
```

This will download the demo_data for the framework. Complete data will be provided upon request, and made available publicly after publication. Resulting layout:

```
demo_data/
  layup_demo/    layup_gesture_demo*.mp4, anchor_image_layup_stationary.png
  tea/           tea_making*.mp4
  sorting/       ...
  weigh_bottles/ ...
```

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

### CLI — hand layup demo

```bash
uv run python scripts/run_aura.py \
    --task hand_layup \
    --video demo_data/layup_demo/layup_gesture_demo_stationary_with_overlay.mp4 \
    --dry-run
```

For live operation against the real UR5, add `--live --robot-url
http://<ur_ws-host>:5050` (see the [ur_ws](https://github.com/usmanasad88/ur_ws)
repo for the External Control API).

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
| `--enable-pose` / `--no-pose`, `--pose-endpoint` | Enable the activity gate on the intent monitor (requires an external pose server, below). |
| `--no-dashboard`, `--dashboard-port` | Disable / re-port the web dashboard. |

Run `uv run python scripts/run_aura.py --help` for the full list.

### Pose server (optional)

The Body Pose / Activity monitor delegates inference to an external
Fast-SAM-3D-Body ZMQ service. [`scripts/run_aura_server.sh`](scripts/run_aura_server.sh)
is provided as **sample wiring** showing how to launch such a service from a
conda env — it is not standalone and requires a working Fast-SAM-3D-Body
install on the host. Once a pose server is reachable, add `--enable-pose` to
the `run_aura.py` invocation.

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

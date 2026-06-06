#!/usr/bin/env python3
"""AURA Unified Workflow — LangGraph + SSG runtime.

Config-driven entry point that builds a task-specific LangGraph,
initialises the SSG from ``initial_scene.json``, and runs the
continuous ``sense → decide → act`` loop.

The system runs continuously — gesture detection on every frame
sets a ``human_requesting_help`` flag that the decision engine uses
to decide when proactive robot intervention is needed.

Includes a real-time web dashboard (http://localhost:5555) that
displays frames, monitor outputs, and decision engine state.

Usage
-----
::

    # Launch the web UI (recommended — no CLI args needed)
    uv run python scripts/run_aura.py --ui

    # Video file, dry-run, with dashboard
    uv run python scripts/run_aura.py \\
        --task hand_layup \\
        --video demo_data/layup_demo/layup_gesture_demo.mp4 \\
        --dry-run

    # Without dashboard
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --dry-run --no-dashboard

    # Live robot
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --live \\
        --robot-url http://192.168.1.100:5050

    # Gemini Live API (low-latency streaming with persistent session)
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --dry-run \\
        --model gemini-3.1-flash-live-preview

    # Local VLM via SGLang (start server first: ./scripts/start_sglang_server.sh)
    uv run python scripts/run_aura.py \\
        --task hand_layup \\
        --video demo_data/layup_demo/layup_gesture_demo.mp4 \\
        --llm-backend sglang \\
        --model Qwen/Qwen3.5-VL-4B-Instruct \\
        --sglang-url http://localhost:8100/v1

    # With voice control (Gemini Live audio alongside visual workflow)
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --live \\
        --audio --audio-input-device USB --audio-output-device Analog
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.dashboard.server import DashboardServer

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Disable LangSmith tracing (force-set to avoid noisy SSL/422 warnings)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env from project root (API keys, etc.)
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Printing helpers ───────────────────────────────────────────────────────

def _print_intent(intent: dict, cycle: int) -> None:
    phase = intent.get("current_phase", "?")
    action = intent.get("current_action", "?")
    predicted = intent.get("predicted_next_action", "?")
    conf = intent.get("prediction_confidence", 0.0)
    completed = len(intent.get("steps_completed", []))
    gen_time = intent.get("generation_time_sec", 0.0)
    reasoning = intent.get("reasoning", "")

    state_vals = " | ".join(
        f"{k}: {v}"
        for k, v in intent.get("state", {}).items()
        if k not in {
            "current_phase", "current_action", "predicted_next_action",
            "prediction_confidence", "steps_completed", "steps_in_progress",
            "steps_pending", "reasoning", "human_state",
        }
    )

    print(
        f"\n{'─' * 60}\n"
        f"  Frame {intent.get('frame_num', '?')} | "
        f"{intent.get('timestamp', 0):.1f}s | cycle #{cycle} ({gen_time:.1f}s)\n"
        f"  Phase: {phase}  |  Action: {action}\n"
        f"  Completed: {completed} steps  |  State: {state_vals}\n"
        f"  Predicted next: {predicted} ({conf:.0%})\n"
        f"  Reasoning: {reasoning}\n"
        f"{'─' * 60}"
    )


def _print_actions(actions: list) -> None:
    for a in actions:
        status = "OK" if a.get("success") else "PEND"
        mode = f" [{a.get('mode', 'live')}]" if "mode" in a else ""
        print(
            f"  >> ROBOT [{status}]{mode} "
            f"{a.get('action_type')} {a.get('object_name')} "
            f"(trigger: {a.get('trigger_step')})"
        )


# ─── Audio monitor setup ───────────────────────────────────────────────────

async def _start_audio_monitor(
    config_dir: Path,
    robot_url: str,
    dry_run: bool,
    task_profile: dict,
    input_device: str | None,
    output_device: str | None,
    sample_rate: int,
    voice_name: str,
) -> tuple:
    """Initialise and start the SoundMonitor as a background async task.

    Returns ``(AudioWorkflowBridge, SoundMonitor)`` on success, or
    ``(None, None)`` if audio initialisation fails.
    """
    try:
        from aura.brain.skill_registry import SkillRegistry
        from aura.interfaces.audio_workflow_bridge import AudioWorkflowBridge
        from aura.interfaces.skill_action_bridge import SkillActionBridge
        from aura.monitors.sound_monitor import SoundMonitor
        from aura.workflow.nodes import set_audio_bridge, _get_ssg

        # Load skills from the task's robot_skills.json
        skills = SkillRegistry()
        skills_path = config_dir / "robot_skills.json"
        if skills_path.exists():
            skills.load_from_file(str(skills_path))
        else:
            logger.warning("No robot_skills.json found at %s", skills_path)

        # Create the bridge and skill action handler
        bridge = AudioWorkflowBridge()

        skill_bridge = SkillActionBridge(
            skills=skills,
            robot_url=robot_url,
            dry_run=dry_run,
            ssg=None,  # Will be set after first SSG init in the graph
            on_action=lambda entry: logger.info(
                "[VoiceAction] %s → %s", entry.function_name, "OK" if entry.success else "FAIL"
            ),
            on_ssg_update=bridge.on_ssg_update,
            on_context_message=bridge.on_context_message,
        )
        # Stash reference so we can print the action log on exit
        bridge._skill_bridge = skill_bridge  # type: ignore[attr-defined]

        # Build system instruction from task profile
        task_instruction = task_profile.get("system_instruction", "")
        system_instruction = (
            f"{task_instruction}\n\n"
            "You are a voice interface for a robot assistant. "
            "The human is working on a task and may ask you to execute "
            "robot skills, tell you about the scene, or ask about task "
            "progress. You can also relay context to the decision engine."
        )

        sound_config = skill_bridge.build_sound_config(
            system_instruction=system_instruction,
            voice_name=voice_name,
            enable_speech_output=True,
            input_device_name=input_device,
            output_device_name=output_device,
            input_sample_rate=sample_rate,
        )

        def on_response(text: str):
            bridge.on_robot_utterance(text)
            logger.info("[Gemini Voice]: %s", text[:100])

        monitor = SoundMonitor(
            config=sound_config,
            on_response=on_response,
            tool_handlers=skill_bridge.tool_handlers,
        )

        # Start listening (opens Gemini Live session + audio streams)
        await monitor.start_listening()

        # Register bridge with the workflow nodes module
        loop = asyncio.get_running_loop()
        bridge.set_sound_monitor(monitor, loop)
        set_audio_bridge(bridge)

        logger.info("Audio monitor started successfully")
        return bridge, monitor

    except Exception as e:
        logger.error("Failed to start audio monitor: %s", e)
        return None, None


# ─── Main workflow runner ───────────────────────────────────────────────────

async def run_workflow(
    task_name: str,
    video_path: str | None,
    webcam_device: int | str | None,
    robot_url: str,
    speed: float,
    model: str,
    dry_run: bool,
    screen_capture: bool = False,
    screen_monitor: int = 1,
    screen_region: list[int] | None = None,
    gopro_stream: bool = False,
    gopro_ip: str = "172.29.170.51",
    gopro_lens: str = "front",
    dashboard_port: int = 5555,
    no_dashboard: bool = False,
    offline_realtime: bool = True,
    frame_skip: int = 30,
    max_cycles: int | None = None,
    use_ground_truth_robot_status: bool = False,
    intent_source: str = "llm",
    intent_gt_path: str | None = None,
    llm_backend: str = "gemini",
    sglang_base_url: str = "http://localhost:8100/v1",
    intent_backend: str | None = None,
    intent_model: str | None = None,
    decision_backend: str | None = None,
    decision_model: str | None = None,
    intent_max_tokens: int | None = None,
    dashboard: "DashboardServer | None" = None,
    # Audio / voice control
    enable_audio: bool = False,
    audio_input_device: str | None = None,
    audio_output_device: str | None = None,
    audio_sample_rate: int = 16000,
    audio_voice: str = "Zephyr",
    # Monitor toggles — ``None`` means "use task_profile default".
    enable_gesture: bool | None = None,
    enable_perception: bool | None = None,
    enable_pose: bool | None = None,
    enable_activity: bool | None = None,
    intent_blocking: bool | None = None,
    pose_endpoint: str = "tcp://localhost:5556",
    intent_num_frames: int = 5,
    frame_buffer_size: int = 300,
    intent_include_previous_state: bool = True,
    intent_previous_state_source: str = "self",
) -> None:
    """Build and run the LangGraph workflow loop."""
    from aura.workflow.builder import build_task_graph
    from aura.utils.config import load_config

    config_dir = _project_root / "tasks" / task_name / "config"

    # Run-mode classification (mirrors workflow.nodes.resolve_source_mode):
    #   offline_eval     — video file, as-fast-as-possible (offline_realtime=False)
    #   offline_realtime — video file, wall-clock paced, fast/small-context intent
    #   live             — webcam/screen/gopro, continuous, responsive intent
    offline_eval = bool(video_path) and not offline_realtime

    # Resolve backend-specific defaults from config/default.yaml
    if intent_max_tokens is None:
        aura_cfg = load_config()
        effective_backend = intent_backend or llm_backend
        backend_defs = aura_cfg.backend_defaults.get(effective_backend, {})
        intent_max_tokens = backend_defs.get("intent_max_tokens", 4096)

    # ── Create unified per-run log directory ────────────────────────
    run_log_dir = _project_root / "logs" / (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_name}"
    )
    run_log_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "started_at": datetime.now().isoformat(),
        "task_name": task_name,
        "video_path": video_path,
        "webcam_device": webcam_device,
        "robot_url": robot_url,
        "speed": speed,
        "model": model,
        "dry_run": dry_run,
        "screen_capture": screen_capture,
        "screen_monitor": screen_monitor,
        "screen_region": screen_region,
        "gopro_stream": gopro_stream,
        "gopro_ip": gopro_ip,
        "gopro_lens": gopro_lens,
        "dashboard_port": dashboard_port,
        "no_dashboard": no_dashboard,
        "offline_realtime": offline_realtime,
        "frame_skip": frame_skip,
        "max_cycles": max_cycles,
        "use_ground_truth_robot_status": use_ground_truth_robot_status,
        "intent_source": intent_source,
        "intent_gt_path": intent_gt_path,
        "llm_backend": llm_backend,
        "sglang_base_url": sglang_base_url,
        "intent_backend": intent_backend,
        "intent_model": intent_model,
        "decision_backend": decision_backend,
        "decision_model": decision_model,
        "intent_max_tokens": intent_max_tokens,
        "enable_audio": enable_audio,
        "audio_input_device": audio_input_device,
        "audio_output_device": audio_output_device,
        "audio_sample_rate": audio_sample_rate,
        "audio_voice": audio_voice,
        "enable_gesture": enable_gesture,
        "enable_perception": enable_perception,
        "enable_pose": enable_pose,
        "enable_activity": enable_activity,
        "intent_blocking": intent_blocking,
        "pose_endpoint": pose_endpoint,
        "intent_num_frames": intent_num_frames,
        "frame_buffer_size": frame_buffer_size,
        "intent_include_previous_state": intent_include_previous_state,
        "intent_previous_state_source": intent_previous_state_source,
    }
    (run_log_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )
    logger.info("Run log directory: %s", run_log_dir)

    extra = {
        "run_log_dir": str(run_log_dir),
        "offline_realtime": offline_realtime,
        "frame_skip": frame_skip,
        "use_ground_truth_robot_status": use_ground_truth_robot_status,
        "intent_source": intent_source,
        "intent_gt_path": intent_gt_path,
        "llm_backend": llm_backend,
        "sglang_base_url": sglang_base_url,
        "screen_capture": screen_capture,
        "screen_monitor": screen_monitor,
        "screen_region": screen_region,
        "gopro_stream": gopro_stream,
        "gopro_ip": gopro_ip,
        "gopro_lens": gopro_lens,
        # Per-component overrides (fall back to shared llm_backend / model)
        "intent_backend": intent_backend or llm_backend,
        "intent_model": intent_model or model,
        "decision_backend": decision_backend or llm_backend,
        "decision_model": decision_model or model,
        "intent_max_tokens": intent_max_tokens,
        "pose_server_endpoint": pose_endpoint,
        "intent_num_frames": intent_num_frames,
        "frame_buffer_size": frame_buffer_size,
        "intent_include_previous_state": intent_include_previous_state,
        "intent_previous_state_source": intent_previous_state_source,
        # Default: block only in offline-eval (every cycle needs a fresh
        # prediction); offline-realtime and live dispatch in the background.
        # Override explicitly via --intent-blocking / --no-intent-blocking.
        "intent_blocking": (
            intent_blocking if intent_blocking is not None else offline_eval
        ),
    }
    if max_cycles is not None:
        extra["max_cycles"] = max_cycles

    # Surface audio flag so the dashboard header can show it.
    extra["enable_audio"] = bool(enable_audio)

    # Build active_monitors override from per-monitor toggles. Intent is
    # always on; gesture/perception follow the flags. Audio lives outside
    # the LangGraph and is toggled separately via ``enable_audio``.
    if (
        enable_gesture is not None
        or enable_perception is not None
        or enable_pose is not None
        or enable_activity is not None
    ):
        profile_path = config_dir / "task_profile.json"
        try:
            wf_profile = json.loads(profile_path.read_text()).get("workflow_config", {})
            profile_monitors = set(wf_profile.get("active_monitors", ["intent", "gesture"]))
        except Exception:
            profile_monitors = {"intent", "gesture"}
        monitors = {"intent"}
        want_gesture = enable_gesture if enable_gesture is not None else ("gesture" in profile_monitors)
        want_perception = enable_perception if enable_perception is not None else ("perception" in profile_monitors)
        want_pose = enable_pose if enable_pose is not None else ("pose" in profile_monitors)
        # Activity auto-enables when pose is on (it's the cheap classifier on
        # top of pose that gates the intent branch). Explicit flag overrides.
        if enable_activity is None:
            want_activity = want_pose or ("activity" in profile_monitors)
        else:
            want_activity = enable_activity
        if want_gesture:
            monitors.add("gesture")
        if want_perception:
            monitors.add("perception")
        if want_pose:
            monitors.add("pose")
        if want_activity:
            monitors.add("activity")
        extra["active_monitors_override"] = sorted(monitors)

    compiled_graph, initial_state = build_task_graph(
        config_dir=config_dir,
        dry_run=dry_run,
        video_path=video_path,
        webcam_device=webcam_device,
        robot_url=robot_url,
        speed=speed,
        model=model,
        extra_config=extra,
    )

    # ── Export LangGraph graph visualization ─────────────────────────
    graph_image_path = run_log_dir / f"graph_{task_name}.png"
    try:
        graph_image_path.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        graph_image_path.write_bytes(png_bytes)
        logger.info("Graph image written to %s", graph_image_path)
    except Exception as e:
        logger.warning("Failed to render graph image: %s", e)
        try:
            mermaid_path = graph_image_path.with_suffix(".mmd")
            mermaid_path.write_text(compiled_graph.get_graph().draw_mermaid())
            logger.info("Graph mermaid source written to %s", mermaid_path)
        except Exception as e2:
            logger.warning("Failed to render graph mermaid source: %s", e2)

    # ── Start dashboard ──────────────────────────────────────────────
    dash = dashboard  # Use existing dashboard if provided (launcher mode)
    owns_dashboard = False
    if dash is None and not no_dashboard:
        try:
            from aura.dashboard import DashboardServer
            dash = DashboardServer(port=dashboard_port)
            dash.start()
            owns_dashboard = True
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)
            dash = None

    if dash:
        dash.publish("init", {"config": initial_state.get("config", {})})

    # ── Start audio / voice monitor (background) ────────────────────
    audio_bridge = None
    sound_monitor = None
    if enable_audio:
        audio_bridge, sound_monitor = await _start_audio_monitor(
            config_dir=config_dir,
            robot_url=robot_url,
            dry_run=dry_run,
            task_profile=initial_state.get("task_profile", {}),
            input_device=audio_input_device,
            output_device=audio_output_device,
            sample_rate=audio_sample_rate,
            voice_name=audio_voice,
        )

    task_display = initial_state["config"].get("task_name", task_name)

    ib = extra["intent_backend"]
    im = extra["intent_model"]
    db = extra["decision_backend"]
    dm = extra["decision_model"]

    print("\n" + "=" * 60)
    print(f"  AURA Workflow [{task_display}]")
    print(f"  Mode: {'dry-run' if dry_run else 'LIVE'}  |  Continuous")
    if intent_source == "ground_truth":
        gt_tag = intent_gt_path if intent_gt_path else "auto-resolved from video"
        print(f"  Intent   : GROUND TRUTH  |  {gt_tag}")
    else:
        print(f"  Intent   : {ib}  |  {im}")
    print(f"  Decision : {db}  |  {dm}")
    if ib != "gemini" or db != "gemini":
        print(f"  SGLang   : {sglang_base_url}")
    if video_path:
        print(f"  Video: {video_path}  |  Speed: {speed}x")
    elif screen_capture:
        print(f"  Screen Capture: monitor {screen_monitor}")
    elif gopro_stream:
        print(f"  GoPro Stream: {gopro_ip}")
    elif webcam_device is not None:
        print(f"  Webcam: {webcam_device}")
    active_mon_summary = extra.get("active_monitors_override")
    if active_mon_summary:
        print(f"  Monitors: {', '.join(active_mon_summary)}")
    if enable_audio:
        print(f"  Audio: {'ACTIVE' if sound_monitor else 'FAILED'}")
    if dash and owns_dashboard:
        print(f"  Dashboard: http://localhost:{dashboard_port}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # LangGraph counts every node execution as a "superstep" toward the
    # recursion limit, and one sense→decide→act loop iteration spends several
    # supersteps (capture → sensing branches → update_ssg → routing →
    # decide/execute → check_complete). The real stop condition is
    # ``max_cycles`` (enforced in ``check_complete_node``), so the recursion
    # limit must comfortably exceed ``max_cycles × supersteps-per-cycle`` or
    # the stream aborts mid-run with a "recursion limit reached" error.
    #
    # Offline-eval runs terminate on video EOF well before the loop count
    # matters, so they keep the original fixed limit (behaviour unchanged).
    # Continuous runs (live, and offline-realtime which can play a long video
    # at 1x) only stop on ``max_cycles`` (or Ctrl+C / dashboard stop), so the
    # limit is scaled to make that reachable — at ~12 supersteps/cycle this
    # keeps the loop alive long enough for a slow VLM call to return and intent
    # to keep re-dispatching.
    if not offline_eval:
        effective_max_cycles = int(initial_state["config"].get("max_cycles", 1500))
        recursion_limit = effective_max_cycles * 12 + 200
    else:
        recursion_limit = 6000

    thread_config = {
        "configurable": {
            "thread_id": f"aura_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        },
        "recursion_limit": recursion_limit,
    }

    # ── Run the LangGraph fast perception loop ───────────────────────
    # Intent prediction is a real graph node now (``run_intent_node``)
    # but it dispatches the slow VLM call to a worker thread and returns
    # immediately in realtime mode — so the parallel sensing branches
    # (pose / activity / intent, perception, gesture) all converge at
    # ``update_ssg`` without stalling on the LLM. In offline / eval mode
    # the intent node blocks (``intent_blocking=True``) so each cycle has
    # a fresh prediction.
    last_cycle = 0
    try:
         async for event in compiled_graph.astream(initial_state, thread_config, stream_mode="updates"):
            for node_name, node_state in event.items():
                print(f"[LangGraph] Node executed: {node_name}")
                # LangGraph may emit events where the node returned ``None``
                # or a non-dict payload; treat them as empty updates.
                if not isinstance(node_state, dict):
                    node_state = {}

                # Annotated perception frame: feed it to the dashboard frame
                # slot and pop it so the numpy image never hits the SSE JSON.
                # Perception runs after capture_frame in the cycle, so this
                # overrides the raw frame with the detection overlay.
                perception_vis = node_state.pop("perception_vis", None)
                if dash is not None and perception_vis is not None:
                    dash.set_frame(perception_vis)

                # ── Publish to dashboard ─────────────────────────────
                if dash:
                    dash.publish(node_name, node_state)
                    # Push latest frame after capture
                    if node_name == "capture_frame":
                        buf = node_state.get("frames_buffer") or []
                        if buf:
                            dash.set_frame(buf[-1])

                # Intent results land in the slot during run_intent and
                # get applied to the SSG inside update_ssg_node — that's
                # where the freshest serialised dict shows up.
                if node_name == "update_ssg":
                    intent_res = node_state.get("intent_result")
                    if intent_res:
                        cycle = node_state.get("cycle_count", last_cycle)
                        _print_intent(intent_res, cycle)

                # After execute, print actions
                if node_name == "execute_action":
                    history = node_state.get("decision_history") or []
                    if history:
                        _print_actions(history)

                # After check_complete, update cycle counter
                if node_name == "check_complete":
                    last_cycle = node_state.get("cycle_count", last_cycle)

                # Propagate completion
                if node_state.get("is_complete"):
                    break
                if node_state.get("error"):
                    logger.error("Workflow error: %s", node_state["error"])
                    break

            if dash and hasattr(dash, "stop_requested") and dash.stop_requested:
                logger.info("Workflow stop requested from dashboard.")
                dash._state["is_complete"] = True
                dash._state["error"] = "Stopped by user"
                dash.publish("system", {})
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Stop audio monitor
        if sound_monitor is not None:
            try:
                await sound_monitor.stop_listening()
            except Exception as e:
                logger.debug("Audio monitor cleanup: %s", e)
        if dash and owns_dashboard:
            dash.stop()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  AURA Workflow — Summary")
    print(f"  Cycles: {last_cycle}")
    if audio_bridge:
        from aura.interfaces.skill_action_bridge import SkillActionBridge
        # Print voice action log if any actions were dispatched
        for entry in (audio_bridge._skill_bridge.get_action_log()
                      if hasattr(audio_bridge, '_skill_bridge') else []):
            s = "OK" if entry["success"] else "FAIL"
            print(f"  Voice [{s}] {entry['function']}({entry['args']})")
    print("=" * 60)


# ─── UI Launcher ──────────────────────────────────────────────────────────

def run_launcher(dashboard_port: int = 5555) -> None:
    """Start the web-based launcher UI.

    Opens a browser-accessible configuration page at
    http://localhost:<port> where the user can select task, source,
    model, etc. and launch the workflow with a single click.
    """
    from aura.dashboard.server import DashboardServer

    server = DashboardServer(
        port=dashboard_port,
        launcher_mode=True,
        project_root=_project_root,
    )

    def launch_callback(config: dict) -> None:
        """Map launcher UI config to run_workflow kwargs."""
        source = config.get("source_type", "video")

        video_path = None
        webcam_device = None
        screen_capture = False
        gopro_stream = False

        if source == "video":
            rel = config.get("video_path", "")
            if rel:
                video_path = str(_project_root / rel)
        elif source == "webcam":
            webcam_device = config.get("webcam_device", 0)
        elif source == "screen":
            screen_capture = True
        elif source == "gopro":
            gopro_stream = True

        asyncio.run(
            run_workflow(
                task_name=config.get("task", "hand_layup"),
                video_path=video_path,
                webcam_device=webcam_device,
                screen_capture=screen_capture,
                screen_monitor=config.get("screen_monitor", 1),
                screen_region=config.get("screen_region"),
                gopro_stream=gopro_stream,
                gopro_ip=config.get("gopro_ip", "172.29.170.51"),
                gopro_lens=config.get("gopro_lens", "front"),
                robot_url=config.get("robot_url", "http://localhost:5050"),
                speed=config.get("speed", 1.0),
                model=config.get("model", "gemini-3.1-pro-preview"),
                dry_run=config.get("dry_run", True),
                no_dashboard=True,  # Don't create a second dashboard
                # Launcher UI sends "realtime" (paced playback for video);
                # accept the new key too. Only affects video-file sources.
                offline_realtime=config.get(
                    "offline_realtime", config.get("realtime", True)
                ),
                frame_skip=config.get("frame_skip", 30),
                max_cycles=config.get("max_cycles"),
                use_ground_truth_robot_status=config.get(
                    "use_ground_truth_robot_status", False
                ),
                intent_source=config.get("intent_source", "llm"),
                intent_gt_path=config.get("intent_gt_path"),
                llm_backend=config.get("llm_backend", "gemini"),
                sglang_base_url=config.get(
                    "sglang_url", "http://localhost:8100/v1"
                ),
                intent_backend=config.get("intent_backend"),
                intent_model=config.get("intent_model"),
                decision_backend=config.get("decision_backend"),
                decision_model=config.get("decision_model"),
                intent_max_tokens=config.get("intent_max_tokens"),
                dashboard=server,  # Reuse the launcher's server
                enable_audio=config.get("enable_audio", False),
                audio_input_device=config.get("audio_input_device"),
                audio_output_device=config.get("audio_output_device"),
                audio_sample_rate=config.get("audio_sample_rate", 16000),
                audio_voice=config.get("audio_voice", "Zephyr"),
                enable_gesture=config.get("enable_gesture"),
                enable_perception=config.get("enable_perception"),
                enable_pose=config.get("enable_pose"),
                enable_activity=config.get("enable_activity"),
                intent_blocking=config.get("intent_blocking"),
                pose_endpoint=config.get("pose_endpoint", "tcp://localhost:5556"),
                intent_num_frames=config.get("intent_num_frames", 5),
                frame_buffer_size=config.get("frame_buffer_size", 300),
                intent_include_previous_state=config.get(
                    "intent_include_previous_state", True
                ),
                intent_previous_state_source=config.get(
                    "intent_previous_state_source", "self"
                ),
            )
        )

    server.set_launch_callback(launch_callback)
    server.start()

    print("\n" + "=" * 60)
    print(f"  AURA Launcher")
    print(f"  Open http://localhost:{dashboard_port} to configure and launch")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.stop()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AURA Unified Workflow — LangGraph + SSG runtime",
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch the web-based configuration UI (no other args needed)",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task name (directory under tasks/)",
    )
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument(
        "--webcam", default=None, nargs="?", const=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--screen-capture", action="store_true", default=False,
        help="Use live screen capture as frame source (requires mss package)",
    )
    parser.add_argument(
        "--screen-monitor", type=int, default=1,
        help="Monitor index for screen capture (0=all, 1=primary, 2=secondary, default: 1)",
    )
    parser.add_argument(
        "--screen-region", type=int, nargs=4, default=None,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Sub-region of screen to capture: LEFT TOP WIDTH HEIGHT (pixels)",
    )
    parser.add_argument(
        "--gopro-stream", action="store_true", default=False,
        help="Use GoPro realtime UDP video stream (~30 fps, low latency)",
    )
    parser.add_argument(
        "--gopro-ip", default="172.29.170.51", metavar="IP",
        help="GoPro camera IP address (default: 172.29.170.51)",
    )
    parser.add_argument(
        "--gopro-lens", choices=["front", "back", "both"], default="front",
        help="Which GoPro lens to use: front (ultrawide), back, or both (raw dual-fisheye). Default: front",
    )
    parser.add_argument(
        "--robot-url", default="http://localhost:5050",
        help="Robot HTTP API base URL",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Log robot actions without executing (default)",
    )
    parser.add_argument(
        "--live", dest="dry_run", action="store_false",
        help="Execute robot actions for real",
    )
    parser.add_argument(
        "--offline-eval", "--no-realtime", dest="offline_eval", action="store_true",
        help="Offline-eval mode: process a video file as fast as possible "
             "(no wall-clock pacing), with blocking intent and full context. "
             "Without this, a video file plays at wall-clock pace "
             "(offline-realtime: non-blocking intent, reduced context for "
             "faster inference). Ignored for live sources (webcam/screen/gopro), "
             "which are always continuous. (--no-realtime is a deprecated alias.)",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=30,
        help="In offline-eval mode, yield every N-th frame (default: 30)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help="Stop after N workflow cycles (limits LLM calls)",
    )
    parser.add_argument(
        "--use-ground-truth-robot-status", action="store_true",
        help="Populate robot_state from tasks/<task>/config/ground_truth.json",
    )
    parser.add_argument(
        "--intent-source", default="llm", choices=["llm", "ground_truth"],
        help="Source of intent predictions: 'llm' runs the VLM (default), "
             "'ground_truth' serves pre-annotated keyframes from the GT file "
             "(see scripts/annotate_ground_truth.py).",
    )
    parser.add_argument(
        "--intent-gt-path", default=None,
        help="Explicit path to the intent GT JSON. Defaults to "
             "tasks/<task>/ground_truth/<video_stem>.intent_gt.json when omitted.",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the web dashboard UI",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=5555,
        help="Dashboard server port (default: 5555)",
    )
    _backend_choices = ["gemini", "openai", "sglang", "vllm", "ollama", "local"]
    parser.add_argument(
        "--llm-backend", default="gemini", choices=_backend_choices,
        help="Default LLM backend for both monitors (default: gemini)",
    )
    parser.add_argument(
        "--sglang-url", default="http://localhost:8100/v1",
        help="SGLang / OpenAI-compatible server base URL (default: http://localhost:8100/v1)",
    )
    parser.add_argument(
        "--intent-backend", default=None, choices=_backend_choices,
        help="LLM backend for intent monitor only (overrides --llm-backend)",
    )
    parser.add_argument(
        "--intent-model", default=None,
        help="Model for intent monitor only (overrides --model)",
    )
    parser.add_argument(
        "--decision-backend", default=None, choices=_backend_choices,
        help="LLM backend for decision engine only (overrides --llm-backend)",
    )
    parser.add_argument(
        "--decision-model", default=None,
        help="Model for decision engine only (overrides --model)",
    )
    parser.add_argument(
        "--intent-max-tokens", type=int, default=None,
        help="Max completion tokens for intent monitor (default from config/default.yaml per backend)",
    )
    # Audio / voice control
    parser.add_argument(
        "--audio", action="store_true", default=False,
        help="Enable voice control via Gemini Live (microphone + speaker)",
    )
    parser.add_argument(
        "--audio-input-device", default=None,
        help="Substring of input audio device name (e.g. 'C310', 'USB')",
    )
    parser.add_argument(
        "--audio-output-device", default=None,
        help="Substring of output audio device name (e.g. 'Line Out', 'Analog')",
    )
    parser.add_argument(
        "--audio-rate", type=int, default=16000,
        help="Audio input sample rate (default: 16000)",
    )
    parser.add_argument(
        "--audio-voice", default="Zephyr",
        help="Gemini voice name for spoken responses (default: Zephyr)",
    )
    # Body-pose monitor / intent gating
    parser.add_argument(
        "--enable-pose", dest="enable_pose", action="store_true", default=None,
        help="Enable SAM-3D-Body pose monitor (gates the background intent runner)",
    )
    parser.add_argument(
        "--no-pose", dest="enable_pose", action="store_false",
        help="Disable pose monitor even if the task profile enables it",
    )
    parser.add_argument(
        "--pose-endpoint", default="tcp://localhost:5556",
        help="ZMQ endpoint for the SAM-3D-Body server (default: tcp://localhost:5556)",
    )
    parser.add_argument(
        "--intent-num-frames", type=int, default=5,
        help="Number of frames per intent prompt (default: 5)",
    )
    parser.add_argument(
        "--frame-buffer-size", type=int, default=300,
        help="Max frames kept in the rolling capture buffer (default: 300)",
    )
    parser.add_argument(
        "--enable-activity", dest="enable_activity", action="store_true", default=None,
        help="Enable the activity-detection node (gates the intent branch)",
    )
    parser.add_argument(
        "--no-activity", dest="enable_activity", action="store_false",
        help="Disable activity gate even if pose is on (intent fires unconditionally)",
    )
    parser.add_argument(
        "--intent-blocking", dest="intent_blocking", action="store_true", default=None,
        help="Force run_intent_node to block until the VLM call finishes "
             "(use for offline / eval; defaults to True in offline-eval mode).",
    )
    parser.add_argument(
        "--no-intent-blocking", dest="intent_blocking", action="store_false",
        help="Force non-blocking intent dispatch even in offline mode "
             "(threadpool fire-and-forget; results land on later cycles).",
    )
    parser.add_argument(
        "--no-intent-previous-state", dest="intent_include_previous_state",
        action="store_false", default=True,
        help="Drop the previous-cycle state section from the intent prompt "
             "(turns RCWPS into a stateless rolling-window prompt).",
    )
    parser.add_argument(
        "--intent-previous-state-source", choices=["self", "ground_truth"],
        default="self",
        help="Where the previous-cycle state in the intent prompt comes from: "
             "'self' (default, uses the monitor's last LLM output) or "
             "'ground_truth' (reads GT annotations at the previous predict's "
             "frame; requires a GT file — see --intent-gt-path).",
    )
    args = parser.parse_args()

    # ── UI launcher mode ────────────────────────────────────────────
    if args.ui:
        run_launcher(dashboard_port=args.dashboard_port)
        return

    # ── CLI mode (requires --task) ──────────────────────────────────
    if not args.task:
        parser.error("--task is required (or use --ui for the web launcher)")

    webcam_dev: int | str | None = None
    if args.webcam is not None:
        try:
            webcam_dev = int(args.webcam)
        except ValueError:
            webcam_dev = args.webcam

    asyncio.run(
        run_workflow(
            task_name=args.task,
            video_path=args.video,
            webcam_device=webcam_dev,
            screen_capture=args.screen_capture,
            screen_monitor=args.screen_monitor,
            screen_region=args.screen_region,
            gopro_stream=args.gopro_stream,
            gopro_ip=args.gopro_ip,
            gopro_lens=args.gopro_lens,
            robot_url=args.robot_url,
            speed=args.speed,
            model=args.model,
            dry_run=args.dry_run,
            dashboard_port=args.dashboard_port,
            no_dashboard=args.no_dashboard,
            offline_realtime=not args.offline_eval,
            frame_skip=args.frame_skip,
            max_cycles=args.max_cycles,
            use_ground_truth_robot_status=args.use_ground_truth_robot_status,
            intent_source=args.intent_source,
            intent_gt_path=args.intent_gt_path,
            llm_backend=args.llm_backend,
            sglang_base_url=args.sglang_url,
            intent_backend=args.intent_backend,
            intent_model=args.intent_model,
            decision_backend=args.decision_backend,
            decision_model=args.decision_model,
            intent_max_tokens=args.intent_max_tokens,
            enable_audio=args.audio,
            audio_input_device=args.audio_input_device,
            audio_output_device=args.audio_output_device,
            audio_sample_rate=args.audio_rate,
            audio_voice=args.audio_voice,
            enable_pose=args.enable_pose,
            enable_activity=args.enable_activity,
            intent_blocking=args.intent_blocking,
            pose_endpoint=args.pose_endpoint,
            intent_num_frames=args.intent_num_frames,
            frame_buffer_size=args.frame_buffer_size,
            intent_include_previous_state=args.intent_include_previous_state,
            intent_previous_state_source=args.intent_previous_state_source,
        )
    )


if __name__ == "__main__":
    main()

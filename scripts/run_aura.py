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
"""

from __future__ import annotations

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


# ─── Main workflow runner ───────────────────────────────────────────────────

async def run_workflow(
    task_name: str,
    video_path: str | None,
    webcam_device: int | str | None,
    robot_url: str,
    speed: float,
    model: str,
    dry_run: bool,
    gopro_stream: bool = False,
    gopro_ip: str = "172.29.170.51",
    gopro_lens: str = "front",
    dashboard_port: int = 5555,
    no_dashboard: bool = False,
    realtime: bool = True,
    frame_skip: int = 30,
    max_cycles: int | None = None,
    use_ground_truth_robot_status: bool = False,
    llm_backend: str = "gemini",
    sglang_base_url: str = "http://localhost:8100/v1",
    intent_backend: str | None = None,
    intent_model: str | None = None,
    decision_backend: str | None = None,
    decision_model: str | None = None,
) -> None:
    """Build and run the LangGraph workflow loop."""
    from aura.workflow.builder import build_task_graph

    config_dir = _project_root / "tasks" / task_name / "config"

    extra = {
        "realtime": realtime,
        "frame_skip": frame_skip,
        "use_ground_truth_robot_status": use_ground_truth_robot_status,
        "llm_backend": llm_backend,
        "sglang_base_url": sglang_base_url,
        "gopro_stream": gopro_stream,
        "gopro_ip": gopro_ip,
        "gopro_lens": gopro_lens,
        # Per-component overrides (fall back to shared llm_backend / model)
        "intent_backend": intent_backend or llm_backend,
        "intent_model": intent_model or model,
        "decision_backend": decision_backend or llm_backend,
        "decision_model": decision_model or model,
    }
    if max_cycles is not None:
        extra["max_cycles"] = max_cycles

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

    # ── Start dashboard ──────────────────────────────────────────────
    dash = None
    if not no_dashboard:
        try:
            from aura.dashboard import DashboardServer
            dash = DashboardServer(port=dashboard_port)
            dash.start()
            # Publish initial config
            dash.publish("init", {"config": initial_state.get("config", {})})
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)
            dash = None

    task_display = initial_state["config"].get("task_name", task_name)

    ib = extra["intent_backend"]
    im = extra["intent_model"]
    db = extra["decision_backend"]
    dm = extra["decision_model"]

    print("\n" + "=" * 60)
    print(f"  AURA Workflow [{task_display}]")
    print(f"  Mode: {'dry-run' if dry_run else 'LIVE'}  |  Continuous")
    print(f"  Intent   : {ib}  |  {im}")
    print(f"  Decision : {db}  |  {dm}")
    if ib != "gemini" or db != "gemini":
        print(f"  SGLang   : {sglang_base_url}")
    if video_path:
        print(f"  Video: {video_path}  |  Speed: {speed}x")
    elif gopro_stream:
        print(f"  GoPro Stream: {gopro_ip}")
    elif webcam_device is not None:
        print(f"  Webcam: {webcam_device}")
    if dash:
        print(f"  Dashboard: http://localhost:{dashboard_port}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    thread_config = {
        "configurable": {
            "thread_id": f"aura_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        },
        "recursion_limit": 2000,
    }

    last_cycle = 0
    try:
        async for event in compiled_graph.astream(initial_state, thread_config):
            for node_name, node_state in event.items():
                # ── Publish to dashboard ─────────────────────────────
                if dash:
                    dash.publish(node_name, node_state)
                    # Push latest frame after capture
                    if node_name == "capture_frame":
                        buf = node_state.get("frames_buffer") or []
                        if buf:
                            dash.set_frame(buf[-1])

                # After intent node, print result
                if node_name == "run_intent" and node_state.get("intent_result"):
                    cycle = node_state.get("cycle_count", last_cycle)
                    _print_intent(node_state["intent_result"], cycle)
                    last_cycle = cycle

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
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if dash:
            dash.stop()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  AURA Workflow — Summary")
    print(f"  Cycles: {last_cycle}")
    print("=" * 60)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AURA Unified Workflow — LangGraph + SSG runtime",
    )
    parser.add_argument(
        "--task", required=True,
        help="Task name (directory under tasks/)",
    )
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument(
        "--webcam", default=None, nargs="?", const=0,
        help="Webcam device index (default: 0)",
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
        "--no-realtime", action="store_true",
        help="Process video as fast as possible (no wall-clock pacing)",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=30,
        help="In non-realtime mode, yield every N-th frame (default: 30)",
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
    args = parser.parse_args()

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
            gopro_stream=args.gopro_stream,
            gopro_ip=args.gopro_ip,
            gopro_lens=args.gopro_lens,
            robot_url=args.robot_url,
            speed=args.speed,
            model=args.model,
            dry_run=args.dry_run,
            dashboard_port=args.dashboard_port,
            no_dashboard=args.no_dashboard,
            realtime=not args.no_realtime,
            frame_skip=args.frame_skip,
            max_cycles=args.max_cycles,
            use_ground_truth_robot_status=args.use_ground_truth_robot_status,
            llm_backend=args.llm_backend,
            sglang_base_url=args.sglang_url,
            intent_backend=args.intent_backend,
            intent_model=args.intent_model,
            decision_backend=args.decision_backend,
            decision_model=args.decision_model,
        )
    )


if __name__ == "__main__":
    main()

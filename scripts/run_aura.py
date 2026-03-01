#!/usr/bin/env python3
"""AURA Unified Workflow — LangGraph + SSG runtime.

Config-driven entry point that builds a task-specific LangGraph,
initialises the SSG from ``initial_scene.json``, and runs the
continuous ``sense → decide → act`` loop.

The system runs continuously — gesture detection on every frame
sets a ``human_requesting_help`` flag that the decision engine uses
to decide when proactive robot intervention is needed.

This replaces ``run_aura_assistant.py`` while maintaining CLI
compatibility.  The old flat-state pipeline is preserved as the
rule-engine inside ``decide_action_node``.

Usage
-----
::

    # Video file, dry-run
    uv run python scripts/run_aura.py \\
        --task hand_layup \\
        --video demo_data/layup_demo/layup_gesture_demo.mp4 \\
        --dry-run

    # Webcam, dry-run
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --dry-run

    # Live robot
    uv run python scripts/run_aura.py \\
        --task hand_layup --webcam 0 --live \\
        --robot-url http://192.168.1.100:5050
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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
) -> None:
    """Build and run the LangGraph workflow loop."""
    from aura.workflow.builder import build_task_graph

    config_dir = _project_root / "tasks" / task_name / "config"

    compiled_graph, initial_state = build_task_graph(
        config_dir=config_dir,
        dry_run=dry_run,
        video_path=video_path,
        webcam_device=webcam_device,
        robot_url=robot_url,
        speed=speed,
        model=model,
    )

    task_display = initial_state["config"].get("task_name", task_name)

    print("\n" + "=" * 60)
    print(f"  AURA Workflow [{task_display}]")
    print(f"  Mode: {'dry-run' if dry_run else 'LIVE'}  |  Continuous")
    if video_path:
        print(f"  Video: {video_path}  |  Speed: {speed}x")
    elif webcam_device is not None:
        print(f"  Webcam: {webcam_device}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    thread_config = {
        "configurable": {
            "thread_id": f"aura_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
    }

    last_cycle = 0
    try:
        async for event in compiled_graph.astream(initial_state, thread_config):
            for node_name, node_state in event.items():
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
        "--robot-url", default="http://localhost:5050",
        help="Robot HTTP API base URL",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Log robot actions without executing (default)",
    )
    parser.add_argument(
        "--live", dest="dry_run", action="store_false",
        help="Execute robot actions for real",
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
            robot_url=args.robot_url,
            speed=args.speed,
            model=args.model,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()

"""LangGraph state and graph definition for Hand Layup task.

This module defines the state schema and workflow graph for processing
the fiberglass hand layup task through the AURA framework monitors.

The graph implements the following workflow:
1. Capture frames from video
2. Run monitors in parallel (perception, intent, performance, affordance)
3. Update the Semantic Scene Graph (SSG)
4. Decide on robot action via the brain/decision engine
5. Execute or wait
6. Check for task completion

Usage:
    from tasks.hand_layup.workflow import (
        create_hand_layup_graph,
        run_hand_layup_workflow,
        HandLayupState,
    )
    
    results = await run_hand_layup_workflow(
        video_path="demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4",
        config_path="tasks/hand_layup/config/hand_layup.yaml"
    )
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

logger = logging.getLogger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class SSGState(TypedDict, total=False):
    """Semantic Scene Graph state."""
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]
    task_state: Dict[str, Any]
    last_updated: str


class MonitorOutputs(TypedDict, total=False):
    """Outputs from all monitors."""
    perception: Dict[str, Any]
    intent: Dict[str, Any]
    performance: Dict[str, Any]
    affordance: Dict[str, Any]


class HandLayupState(TypedDict, total=False):
    """Complete state for the hand layup LangGraph workflow.
    
    This state is passed between all nodes in the graph and represents
    the complete context for decision making.
    """
    # --- Video/Frame State ---
    video_path: str
    current_frame_num: int
    current_timestamp_sec: float
    frames_buffer: List[str]  # base64 encoded frames
    fps: float
    total_frames: int
    
    # --- Monitor Outputs ---
    monitor_outputs: MonitorOutputs
    
    # --- Semantic Scene Graph ---
    ssg: SSGState
    
    # --- Task-Specific State ---
    current_phase: str
    current_action: str
    robot_state: str
    gripper_state: str
    human_state: str
    
    # Object locations
    cup_location: str
    resin_bottle_location: str
    hardener_bottle_location: str
    brush_small_location: str
    brush_medium_location: str
    roller_location: str
    weigh_scale_location: str
    
    # Process tracking
    resin_added: bool
    hardener_added: bool
    mixture_weighed: bool
    mixture_mixed: bool
    mixture_shelf_life_start: Optional[float]
    layers_placed: int
    layers_resined: int
    consolidated: bool
    human_wearing_gloves: bool
    safety_alerts: List[str]
    
    # --- Decision State ---
    available_skills: List[str]
    current_skill: Optional[str]
    completed_skills: List[str]
    robot_instructions: List[Dict[str, Any]]
    next_action: Optional[str]
    action_confidence: float
    decision_reasoning: str
    should_abort: bool
    
    # --- Ground Truth ---
    ground_truth_events: List[Dict[str, Any]]
    current_gt_event: Optional[Dict[str, Any]]
    
    # --- History ---
    decision_history: Annotated[List[Dict[str, Any]], operator.add]
    frame_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # --- Workflow Control ---
    workflow_step: str
    error: Optional[str]
    is_complete: bool
    
    # --- Configuration ---
    config: Dict[str, Any]


def create_initial_state(
    video_path: str,
    config: Dict[str, Any] = None,
    ground_truth: List[Dict[str, Any]] = None,
) -> HandLayupState:
    """Create initial state for the workflow."""
    return HandLayupState(
        # Video
        video_path=video_path,
        current_frame_num=0,
        current_timestamp_sec=0.0,
        frames_buffer=[],
        fps=30.0,
        total_frames=0,
        
        # Monitor outputs
        monitor_outputs={
            "perception": {},
            "intent": {},
            "performance": {},
            "affordance": {},
        },
        
        # SSG
        ssg={
            "nodes": {},
            "edges": [],
            "task_state": {},
            "last_updated": datetime.now().isoformat(),
        },
        
        # Task state
        current_phase="initialization",
        current_action="idle",
        robot_state="at_home",
        gripper_state="empty",
        human_state="idle",
        
        # Object locations (all start on workplace)
        cup_location="workplace",
        resin_bottle_location="workplace",
        hardener_bottle_location="workplace",
        brush_small_location="workplace",
        brush_medium_location="workplace",
        roller_location="workplace",
        weigh_scale_location="workplace",
        
        # Process
        resin_added=False,
        hardener_added=False,
        mixture_weighed=False,
        mixture_mixed=False,
        mixture_shelf_life_start=None,
        layers_placed=0,
        layers_resined=0,
        consolidated=False,
        human_wearing_gloves=False,
        safety_alerts=[],
        
        # Decision
        available_skills=[],
        current_skill=None,
        completed_skills=[],
        robot_instructions=[],
        next_action=None,
        action_confidence=0.0,
        decision_reasoning="",
        should_abort=False,
        
        # Ground truth
        ground_truth_events=ground_truth or [],
        current_gt_event=None,
        
        # History
        decision_history=[],
        frame_results=[],
        
        # Control
        workflow_step="initialized",
        error=None,
        is_complete=False,
        
        # Config
        config=config or {},
    )


# =============================================================================
# Graph Definition
# =============================================================================

def create_hand_layup_graph() -> Optional["StateGraph"]:
    """Create the LangGraph workflow for hand layup task.
    
    Returns:
        Compiled StateGraph or None if LangGraph not available
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not installed. Install with: pip install langgraph")
        return None
    
    from .nodes import (
        capture_frames_node,
        run_perception_node,
        run_intent_node,
        run_affordance_node,
        run_performance_node,
        update_ssg_node,
        decide_action_node,
        execute_action_node,
        check_complete_node,
    )
    
    workflow = StateGraph(HandLayupState)
    
    workflow.add_node("capture_frames", capture_frames_node)
    workflow.add_node("run_perception", run_perception_node)
    workflow.add_node("run_intent", run_intent_node)
    workflow.add_node("run_affordance", run_affordance_node)
    workflow.add_node("run_performance", run_performance_node)
    workflow.add_node("update_ssg", update_ssg_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("check_complete", check_complete_node)
    
    workflow.set_entry_point("capture_frames")
    
    workflow.add_edge("capture_frames", "run_perception")
    workflow.add_edge("run_perception", "run_intent")
    workflow.add_edge("run_intent", "run_affordance")
    workflow.add_edge("run_affordance", "run_performance")
    workflow.add_edge("run_performance", "update_ssg")
    workflow.add_edge("update_ssg", "decide_action")
    
    def should_execute(state: HandLayupState) -> str:
        if state.get("should_abort"):
            return "check_complete"
        if state.get("next_action") and state.get("action_confidence", 0) > 0.5:
            return "execute_action"
        return "check_complete"
    
    workflow.add_conditional_edges(
        "decide_action",
        should_execute,
        {
            "execute_action": "execute_action",
            "check_complete": "check_complete",
        }
    )
    
    workflow.add_edge("execute_action", "check_complete")
    
    def is_complete(state: HandLayupState) -> str:
        if state.get("is_complete") or state.get("error"):
            return END
        return "capture_frames"
    
    workflow.add_conditional_edges(
        "check_complete",
        is_complete,
        {
            END: END,
            "capture_frames": "capture_frames",
        }
    )
    
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Created hand_layup LangGraph workflow")
    return compiled


# =============================================================================
# Workflow Runner
# =============================================================================

async def run_hand_layup_workflow(
    video_path: str,
    config_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    frame_skip: int = 30,
    max_frames: Optional[int] = None,
    headless: bool = True,
) -> Dict[str, Any]:
    """Run the complete hand layup workflow on a video.
    
    Args:
        video_path: Path to video file
        config_path: Path to hand_layup.yaml config
        ground_truth_path: Path to ground_truth.json
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process
        headless: Don't show visualization
        
    Returns:
        Dictionary with results including decision history and metrics
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not installed. Install with: pip install langgraph")
    
    config = {}
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    
    ground_truth = []
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            gt_data = json.load(f)
            ground_truth = gt_data.get("events", [])
    
    config["frame_skip"] = frame_skip
    config["max_frames"] = max_frames
    config["headless"] = headless
    
    graph = create_hand_layup_graph()
    if graph is None:
        raise RuntimeError("Failed to create workflow graph")
    
    initial_state = create_initial_state(
        video_path=video_path,
        config=config,
        ground_truth=ground_truth,
    )
    
    thread_config = {"configurable": {"thread_id": f"hand_layup_{datetime.now().isoformat()}"}}
    
    logger.info(f"Starting hand_layup workflow on: {video_path}")
    
    final_state = None
    async for state in graph.astream(initial_state, thread_config):
        for node_name, node_state in state.items():
            logger.debug(f"Completed node: {node_name}")
            final_state = node_state
    
    results = {
        "video_path": video_path,
        "total_frames_processed": final_state.get("current_frame_num", 0) if final_state else 0,
        "decision_history": final_state.get("decision_history", []) if final_state else [],
        "frame_results": final_state.get("frame_results", []) if final_state else [],
        "robot_instructions": final_state.get("robot_instructions", []) if final_state else [],
        "final_task_state": {
            "layers_placed": final_state.get("layers_placed", 0) if final_state else 0,
            "layers_resined": final_state.get("layers_resined", 0) if final_state else 0,
            "consolidated": final_state.get("consolidated", False) if final_state else False,
            "mixture_mixed": final_state.get("mixture_mixed", False) if final_state else False,
        },
        "completed_skills": final_state.get("completed_skills", []) if final_state else [],
        "is_complete": final_state.get("is_complete", False) if final_state else False,
        "error": final_state.get("error") if final_state else None,
    }
    
    logger.info(f"Workflow complete. Processed {results['total_frames_processed']} frames.")
    
    return results

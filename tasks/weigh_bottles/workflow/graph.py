"""LangGraph state and graph definition for Weigh Bottles task.

This module defines the state schema and workflow graph for processing
the bottle weighing task through the AURA framework monitors.

The graph implements the following workflow:
1. Capture frames from video
2. Run monitors in parallel (perception, intent, performance, affordance)
3. Update the Semantic Scene Graph (SSG)
4. Decide on robot action via the brain/decision engine
5. Execute or wait
6. Check for task completion

Usage:
    from tasks.weigh_bottles.workflow import (
        create_weigh_bottles_graph,
        run_weigh_bottles_workflow,
        WeighBottlesState,
    )
    
    # Process a video
    results = await run_weigh_bottles_workflow(
        video_path="demo_data/weigh_bottles/video.mp4",
        config_path="tasks/weigh_bottles/config/weigh_bottles.yaml"
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
    nodes: Dict[str, Dict[str, Any]]  # node_id -> node data
    edges: List[Dict[str, Any]]  # list of edge data
    task_state: Dict[str, Any]  # task-specific state variables
    last_updated: str  # ISO timestamp


class MonitorOutputs(TypedDict, total=False):
    """Outputs from all monitors."""
    perception: Dict[str, Any]
    intent: Dict[str, Any]
    performance: Dict[str, Any]
    affordance: Dict[str, Any]


class WeighBottlesState(TypedDict, total=False):
    """Complete state for the weigh bottles LangGraph workflow.
    
    This state is passed between all nodes in the graph and represents
    the complete context for decision making.
    """
    # --- Video/Frame State ---
    video_path: str
    gripper_video_path: Optional[str]
    current_frame_num: int
    current_timestamp_sec: float
    frames_buffer: List[str]  # base64 encoded frames
    fps: float
    total_frames: int
    
    # --- Monitor Outputs ---
    monitor_outputs: MonitorOutputs
    
    # --- Semantic Scene Graph ---
    ssg: SSGState
    
    # --- Task-Specific State (from weigh_bottles_state.json) ---
    current_phase: str  # initialization, hardener_delivery, etc.
    current_action: str  # idle, pick_hardener, etc.
    robot_state: str  # at_home, moving_to_storage, etc.
    gripper_state: str  # empty, holding_hardener, holding_resin
    hardener_location: str  # storage_table, robot_gripper, human_hands
    resin_location: str
    hardener_weighed: bool
    resin_weighed: bool
    hardener_returned: bool
    resin_returned: bool
    human_state: str  # waiting, receiving_bottle, etc.
    
    # --- Decision State ---
    available_programs: List[str]
    current_program: Optional[str]
    completed_programs: List[str]
    next_action: Optional[str]
    action_confidence: float
    decision_reasoning: str
    should_abort: bool
    
    # --- Ground Truth (for evaluation) ---
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
) -> WeighBottlesState:
    """Create initial state for the workflow.
    
    Args:
        video_path: Path to the video file
        config: Task configuration dictionary
        ground_truth: List of ground truth events
        
    Returns:
        Initialized WeighBottlesState
    """
    return WeighBottlesState(
        # Video
        video_path=video_path,
        gripper_video_path=None,
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
        
        # Task state (defaults from weigh_bottles_state.json)
        current_phase="initialization",
        current_action="idle",
        robot_state="at_home",
        gripper_state="empty",
        hardener_location="storage_table",
        resin_location="storage_table",
        hardener_weighed=False,
        resin_weighed=False,
        hardener_returned=False,
        resin_returned=False,
        human_state="waiting",
        
        # Decision
        available_programs=[],
        current_program=None,
        completed_programs=[],
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

def create_weigh_bottles_graph() -> Optional["StateGraph"]:
    """Create the LangGraph workflow for weigh bottles task.
    
    Returns:
        Compiled StateGraph or None if LangGraph not available
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not installed. Install with: pip install langgraph")
        return None
    
    # Import nodes
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
    
    # Create graph
    workflow = StateGraph(WeighBottlesState)
    
    # Add nodes
    workflow.add_node("capture_frames", capture_frames_node)
    workflow.add_node("run_perception", run_perception_node)
    workflow.add_node("run_intent", run_intent_node)
    workflow.add_node("run_affordance", run_affordance_node)
    workflow.add_node("run_performance", run_performance_node)
    workflow.add_node("update_ssg", update_ssg_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("check_complete", check_complete_node)
    
    # Define edges
    # Start -> capture frames
    workflow.set_entry_point("capture_frames")
    
    # After capture, run monitors (could be parallel in enhanced version)
    workflow.add_edge("capture_frames", "run_perception")
    workflow.add_edge("run_perception", "run_intent")
    workflow.add_edge("run_intent", "run_affordance")
    workflow.add_edge("run_affordance", "run_performance")
    
    # After monitors, update SSG
    workflow.add_edge("run_performance", "update_ssg")
    
    # After SSG update, decide action
    workflow.add_edge("update_ssg", "decide_action")
    
    # Conditional: execute or skip
    def should_execute(state: WeighBottlesState) -> str:
        """Determine if we should execute an action."""
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
    
    # After execute, check complete
    workflow.add_edge("execute_action", "check_complete")
    
    # Conditional: continue or end
    def is_complete(state: WeighBottlesState) -> str:
        """Check if task is complete or should continue."""
        if state.get("is_complete") or state.get("error"):
            return END
        return "capture_frames"  # Continue processing
    
    workflow.add_conditional_edges(
        "check_complete",
        is_complete,
        {
            END: END,
            "capture_frames": "capture_frames",
        }
    )
    
    # Compile with memory checkpointer
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Created weigh_bottles LangGraph workflow")
    return compiled


# =============================================================================
# Workflow Runner
# =============================================================================

async def run_weigh_bottles_workflow(
    video_path: str,
    config_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    frame_skip: int = 30,
    max_frames: Optional[int] = None,
    headless: bool = True,
) -> Dict[str, Any]:
    """Run the complete weigh bottles workflow on a video.
    
    Args:
        video_path: Path to video file
        config_path: Path to weigh_bottles.yaml config
        ground_truth_path: Path to ground_truth.json
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process (None = all)
        headless: Don't show visualization
        
    Returns:
        Dictionary with results including decision history and metrics
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not installed. Install with: pip install langgraph")
    
    # Load config
    config = {}
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    
    # Load ground truth
    ground_truth = []
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            gt_data = json.load(f)
            ground_truth = gt_data.get("events", [])
    
    # Add workflow parameters to config
    config["frame_skip"] = frame_skip
    config["max_frames"] = max_frames
    config["headless"] = headless
    
    # Create graph
    graph = create_weigh_bottles_graph()
    if graph is None:
        raise RuntimeError("Failed to create workflow graph")
    
    # Create initial state
    initial_state = create_initial_state(
        video_path=video_path,
        config=config,
        ground_truth=ground_truth,
    )
    
    # Run workflow with thread_id for checkpointing
    thread_config = {"configurable": {"thread_id": f"weigh_bottles_{datetime.now().isoformat()}"}}
    
    logger.info(f"Starting weigh_bottles workflow on: {video_path}")
    
    # Run the graph
    final_state = None
    async for state in graph.astream(initial_state, thread_config):
        # State is returned as dict with node name -> state
        for node_name, node_state in state.items():
            logger.debug(f"Completed node: {node_name}")
            final_state = node_state
    
    # Extract results
    results = {
        "video_path": video_path,
        "total_frames_processed": final_state.get("current_frame_num", 0) if final_state else 0,
        "decision_history": final_state.get("decision_history", []) if final_state else [],
        "frame_results": final_state.get("frame_results", []) if final_state else [],
        "final_task_state": {
            "hardener_weighed": final_state.get("hardener_weighed", False) if final_state else False,
            "resin_weighed": final_state.get("resin_weighed", False) if final_state else False,
            "hardener_returned": final_state.get("hardener_returned", False) if final_state else False,
            "resin_returned": final_state.get("resin_returned", False) if final_state else False,
        },
        "completed_programs": final_state.get("completed_programs", []) if final_state else [],
        "is_complete": final_state.get("is_complete", False) if final_state else False,
        "error": final_state.get("error") if final_state else None,
    }
    
    logger.info(f"Workflow complete. Processed {results['total_frames_processed']} frames.")
    
    return results

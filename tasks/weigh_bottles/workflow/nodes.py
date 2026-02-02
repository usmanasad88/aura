"""LangGraph node functions for Weigh Bottles workflow.

Each node function receives the current state and returns an updated state.
Nodes are designed to be composable and handle errors gracefully.
"""

import asyncio
import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image

from .graph import WeighBottlesState

logger = logging.getLogger(__name__)


# =============================================================================
# Frame Capture Node
# =============================================================================

def capture_frames_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Capture next frame(s) from video.
    
    Reads the next frame from the video file and updates the frame buffer.
    """
    video_path = state.get("video_path")
    if not video_path or not Path(video_path).exists():
        return {
            "error": f"Video not found: {video_path}",
            "workflow_step": "capture_failed",
        }
    
    config = state.get("config", {})
    frame_skip = config.get("frame_skip", 30)
    max_frames = config.get("max_frames")
    
    current_frame = state.get("current_frame_num", 0)
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties on first call
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate next frame number
        next_frame = current_frame + frame_skip if current_frame > 0 else 0
        
        # Check if we've reached the end
        if next_frame >= total_frames:
            cap.release()
            return {
                "is_complete": True,
                "workflow_step": "video_complete",
            }
        
        # Check max frames limit
        if max_frames and next_frame // frame_skip >= max_frames:
            cap.release()
            return {
                "is_complete": True,
                "workflow_step": "max_frames_reached",
            }
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return {
                "is_complete": True,
                "workflow_step": "read_failed",
            }
        
        # Convert frame to base64
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if needed
        max_dim = 640
        if max(pil_image.size) > max_dim:
            ratio = max_dim / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        frame_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Update frame buffer (keep last N frames for temporal context)
        frames_buffer = state.get("frames_buffer", [])[-4:]  # Keep last 4
        frames_buffer.append(frame_b64)
        
        # Calculate timestamp
        timestamp = next_frame / fps
        
        # Get current ground truth event
        current_gt = _get_current_gt_event(
            state.get("ground_truth_events", []),
            timestamp
        )
        
        cap.release()
        
        return {
            "current_frame_num": next_frame,
            "current_timestamp_sec": timestamp,
            "frames_buffer": frames_buffer,
            "fps": fps,
            "total_frames": total_frames,
            "current_gt_event": current_gt,
            "workflow_step": "frames_captured",
        }
        
    except Exception as e:
        logger.error(f"Frame capture error: {e}")
        return {
            "error": str(e),
            "workflow_step": "capture_failed",
        }


def _get_current_gt_event(events: List[Dict], timestamp: float) -> Optional[Dict]:
    """Get ground truth event at timestamp."""
    current = None
    for event in events:
        if event.get("timestamp", 0) <= timestamp:
            current = event
        else:
            break
    return current


# =============================================================================
# Perception Node
# =============================================================================

def run_perception_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Run perception monitor on current frame.
    
    Uses SAM3 for object detection and segmentation.
    """
    frames_buffer = state.get("frames_buffer", [])
    if not frames_buffer:
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "perception": {"error": "No frames available"},
            },
            "workflow_step": "perception_skipped",
        }
    
    try:
        # Import perception module
        from aura.monitors.perception_module import PerceptionModule
        from aura.utils.config import PerceptionConfig
        from aura.core import MonitorType
        
        # Get latest frame
        frame_b64 = frames_buffer[-1]
        frame_bytes = base64.b64decode(frame_b64)
        pil_image = Image.open(BytesIO(frame_bytes))
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Objects of interest for bottle weighing
        objects_of_interest = [
            "bottle", "resin bottle", "hardener bottle",
            "scale", "person", "hand", "robot", "gripper"
        ]
        
        # Run perception (sync wrapper for async)
        config = PerceptionConfig(
            monitor_type=MonitorType.PERCEPTION,
            enabled=True,
            use_sam3=False,  # Start with Gemini-only for speed
            use_gemini_detection=True,
            default_prompts=objects_of_interest,
            confidence_threshold=0.3,
        )
        
        perception = PerceptionModule(config)
        
        # Run synchronously
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(perception.process_frame(frame_bgr))
        finally:
            loop.close()
        
        # Extract results
        detected_objects = []
        if output and hasattr(output, 'objects'):
            for obj in output.objects:
                detected_objects.append({
                    "id": getattr(obj, 'id', str(len(detected_objects))),
                    "name": getattr(obj, 'name', getattr(obj, 'label', 'unknown')),
                    "confidence": getattr(obj, 'confidence', 0.0),
                    "bbox": None,  # Add if available
                })
        
        perception_output = {
            "detected_objects": detected_objects,
            "timestamp": state.get("current_timestamp_sec", 0),
            "frame_num": state.get("current_frame_num", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "perception": perception_output,
            },
            "workflow_step": "perception_complete",
        }
        
    except Exception as e:
        logger.error(f"Perception error: {e}")
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "perception": {"error": str(e)},
            },
            "workflow_step": "perception_failed",
        }


# =============================================================================
# Intent Node
# =============================================================================

def run_intent_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Run intent monitor to predict current and next actions.
    
    Uses Gemini with task DAG for action recognition.
    """
    frames_buffer = state.get("frames_buffer", [])
    if not frames_buffer:
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "intent": {"error": "No frames available"},
            },
            "workflow_step": "intent_skipped",
        }
    
    try:
        # Import intent monitor
        from aura.monitors.intent_monitor import IntentMonitor
        from aura.utils.config import IntentMonitorConfig
        
        # Load task DAG
        task_dir = Path(__file__).parent.parent
        dag_path = task_dir / "config" / "weigh_bottles_dag.json"
        state_path = task_dir / "config" / "weigh_bottles_state.json"
        
        dag_nodes = []
        if dag_path.exists():
            with open(dag_path) as f:
                dag_data = json.load(f)
                dag_nodes = list(dag_data.get("nodes", {}).keys())
        
        # Create system prompt for bottle weighing
        system_prompt = """You are analyzing video frames from a robot-assisted bottle weighing task.
The UR5 robot helps a human weigh hardener and resin bottles.

Task sequence:
1. Robot picks hardener bottle, delivers to human
2. Human weighs hardener
3. Robot picks resin bottle, delivers to human
4. Human weighs resin
5. Robot returns hardener to storage
6. Robot returns resin to storage

Analyze the frames to determine the current action and predict the next action."""

        config = IntentMonitorConfig(
            enabled=True,
            fps=2.0,
            capture_duration=2.0,
            prediction_interval=2.0,
            dag_file=str(dag_path) if dag_path.exists() else None,
            state_file=str(state_path) if state_path.exists() else None,
            system_prompt=system_prompt,
            task_name="Bottle Weighing",
        )
        
        monitor = IntentMonitor(config)
        
        # Decode frames for intent analysis
        frames = []
        for frame_b64 in frames_buffer[-3:]:  # Use last 3 frames
            frame_bytes = base64.b64decode(frame_b64)
            pil_image = Image.open(BytesIO(frame_bytes))
            frames.append(np.array(pil_image))
        
        # Run prediction
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                monitor.predict_from_frames(frames)
            )
        finally:
            loop.close()
        
        intent_output = {
            "current_action": getattr(result, 'current_action', 'unknown') if result else 'unknown',
            "current_confidence": getattr(result, 'current_action_confidence', 0.0) if result else 0.0,
            "predicted_next": getattr(result, 'predicted_next_action', 'unknown') if result else 'unknown',
            "predicted_confidence": getattr(result, 'predicted_next_confidence', 0.0) if result else 0.0,
            "reasoning": getattr(result, 'reasoning', '') if result else '',
            "task_state": getattr(result, 'task_state', {}) if result else {},
            "timestamp": state.get("current_timestamp_sec", 0),
        }
        
        # Update task state variables from intent
        task_updates = {}
        if result and hasattr(result, 'task_state') and result.task_state:
            ts = result.task_state
            if 'current_action' in ts:
                task_updates['current_action'] = ts['current_action']
            if 'robot_state' in ts:
                task_updates['robot_state'] = ts['robot_state']
            if 'gripper_state' in ts:
                task_updates['gripper_state'] = ts['gripper_state']
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "intent": intent_output,
            },
            **task_updates,
            "workflow_step": "intent_complete",
        }
        
    except Exception as e:
        logger.error(f"Intent error: {e}")
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "intent": {"error": str(e)},
            },
            "workflow_step": "intent_failed",
        }


# =============================================================================
# Affordance Node
# =============================================================================

def run_affordance_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Run affordance monitor to determine available robot programs.
    
    Checks prerequisites and returns available actions.
    """
    try:
        # Import task-specific affordance monitor
        import sys
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir.parent.parent))
        
        from tasks.weigh_bottles.monitors.affordance_monitor import (
            WeighBottlesAffordanceMonitor,
        )
        
        monitor = WeighBottlesAffordanceMonitor()
        
        # Update completed programs from state
        for prog_id in state.get("completed_programs", []):
            monitor.mark_program_complete(prog_id)
        
        # Get available programs
        available = []
        for prog_id, prog in monitor.programs.items():
            if monitor.is_program_available(prog_id):
                available.append(prog_id)
        
        affordance_output = {
            "available_programs": available,
            "current_program": state.get("current_program"),
            "completed_programs": state.get("completed_programs", []),
            "all_programs": list(monitor.programs.keys()),
            "timestamp": state.get("current_timestamp_sec", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "affordance": affordance_output,
            },
            "available_programs": available,
            "workflow_step": "affordance_complete",
        }
        
    except Exception as e:
        logger.error(f"Affordance error: {e}")
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "affordance": {"error": str(e)},
            },
            "workflow_step": "affordance_failed",
        }


# =============================================================================
# Performance Node
# =============================================================================

def run_performance_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Run performance monitor to check for failures.
    
    Analyzes frames for task execution issues.
    """
    current_program = state.get("current_program")
    
    # Skip if no program is running
    if not current_program:
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "performance": {
                    "status": "OK",
                    "message": "No program running",
                    "timestamp": state.get("current_timestamp_sec", 0),
                },
            },
            "workflow_step": "performance_complete",
        }
    
    try:
        # Import performance monitor
        import sys
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir.parent.parent))
        
        from tasks.weigh_bottles.monitors.performance_monitor import (
            WeighBottlesPerformanceMonitor,
            PerformanceStatus,
        )
        
        frames_buffer = state.get("frames_buffer", [])
        if not frames_buffer:
            return {
                "monitor_outputs": {
                    **state.get("monitor_outputs", {}),
                    "performance": {"error": "No frames available"},
                },
                "workflow_step": "performance_skipped",
            }
        
        # Decode frames
        frames = []
        for frame_b64 in frames_buffer[-3:]:
            frame_bytes = base64.b64decode(frame_b64)
            pil_image = Image.open(BytesIO(frame_bytes))
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
        
        # Create monitor and run check
        monitor = WeighBottlesPerformanceMonitor()
        monitor.set_current_instruction(current_program)
        
        for frame in frames:
            monitor.add_frame(frame)
        
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(monitor.check_performance())
        finally:
            loop.close()
        
        should_abort = False
        if result and hasattr(result, 'should_abort'):
            should_abort = result.should_abort
        
        performance_output = {
            "status": result.status.name if result else "UNKNOWN",
            "failure_type": result.failure_type.name if result else "NONE",
            "confidence": result.confidence if result else 0.0,
            "reasoning": result.reasoning if result else "",
            "timestamp": state.get("current_timestamp_sec", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "performance": performance_output,
            },
            "should_abort": should_abort,
            "workflow_step": "performance_complete",
        }
        
    except Exception as e:
        logger.error(f"Performance error: {e}")
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "performance": {"error": str(e)},
            },
            "workflow_step": "performance_failed",
        }


# =============================================================================
# SSG Update Node
# =============================================================================

def update_ssg_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Update Semantic Scene Graph from monitor outputs.
    
    Integrates all monitor outputs into the SSG.
    """
    try:
        from aura.core.scene_graph import (
            SemanticSceneGraph, 
            ObjectNode, AgentNode, RegionNode,
            SSGEdge, SpatialRelation, SemanticRelation,
            NodeType, EdgeType,
        )
        from aura.core.scene_graph.nodes import ObjectState, AgentState
        
        # Create or restore SSG
        ssg = SemanticSceneGraph(name="weigh_bottles")
        
        # Restore from state if exists
        ssg_state = state.get("ssg", {})
        
        # Update from perception
        perception = state.get("monitor_outputs", {}).get("perception", {})
        detected_objects = perception.get("detected_objects", [])
        
        for obj_data in detected_objects:
            obj_id = obj_data.get("id", f"obj_{len(ssg.nodes)}")
            obj_name = obj_data.get("name", "unknown")
            
            # Create or update node
            existing = ssg.get_node(obj_id)
            if existing:
                existing.confidence = obj_data.get("confidence", 0.0)
                existing.last_updated = datetime.now()
            else:
                node = ObjectNode(
                    id=obj_id,
                    name=obj_name,
                    node_type=NodeType.OBJECT,
                    confidence=obj_data.get("confidence", 0.0),
                    state=ObjectState.AVAILABLE,
                )
                ssg.add_node(node)
        
        # Update from intent
        intent = state.get("monitor_outputs", {}).get("intent", {})
        current_action = intent.get("current_action", "unknown")
        
        # Update human agent
        human = ssg.get_node("human")
        if not human:
            human = AgentNode(
                id="human",
                name="Human Operator",
                node_type=NodeType.AGENT,
                agent_type="human",
            )
            ssg.add_node(human)
        
        # Update robot agent
        robot = ssg.get_node("robot")
        if not robot:
            robot = AgentNode(
                id="robot",
                name="UR5 Robot",
                node_type=NodeType.AGENT,
                agent_type="robot",
                capabilities=[
                    "pick_hardener_bottle.prog",
                    "pick_resin_bottle.prog",
                    "return_hardener_bottle.prog",
                    "return_resin_bottle.prog",
                ],
            )
            ssg.add_node(robot)
        
        # Add regions
        for region_id, region_name in [
            ("storage_table", "Storage Table"),
            ("weighing_station", "Weighing Station"),
            ("work_area", "Work Area"),
        ]:
            if not ssg.has_node(region_id):
                region = RegionNode(
                    id=region_id,
                    name=region_name,
                    node_type=NodeType.REGION,
                )
                ssg.add_node(region)
        
        # Update task state in SSG
        task_state = {
            "current_action": current_action,
            "current_phase": state.get("current_phase", "initialization"),
            "hardener_location": state.get("hardener_location", "storage_table"),
            "resin_location": state.get("resin_location", "storage_table"),
            "hardener_weighed": state.get("hardener_weighed", False),
            "resin_weighed": state.get("resin_weighed", False),
            "gripper_state": state.get("gripper_state", "empty"),
        }
        
        for key, value in task_state.items():
            ssg.set_task_state(key, value)
        
        # Create edges based on current state
        hardener_loc = state.get("hardener_location", "storage_table")
        resin_loc = state.get("resin_location", "storage_table")
        
        # Object location edges
        if ssg.has_node("hardener_bottle"):
            ssg.set_location("hardener_bottle", hardener_loc, SpatialRelation.AT)
        if ssg.has_node("resin_bottle"):
            ssg.set_location("resin_bottle", resin_loc, SpatialRelation.AT)
        
        # Serialize SSG state
        new_ssg_state = {
            "nodes": {n.id: n.to_dict() for n in ssg.nodes.values()},
            "edges": [e.to_dict() for e in ssg.edges],
            "task_state": ssg.task_state,
            "last_updated": datetime.now().isoformat(),
        }
        
        return {
            "ssg": new_ssg_state,
            "workflow_step": "ssg_updated",
        }
        
    except Exception as e:
        logger.error(f"SSG update error: {e}")
        return {
            "workflow_step": "ssg_update_failed",
            "error": str(e),
        }


# =============================================================================
# Decision Node
# =============================================================================

def decide_action_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Decide what action the robot should take.
    
    Uses the brain/decision engine with SSG context.
    """
    try:
        # Get context from state
        available_programs = state.get("available_programs", [])
        current_program = state.get("current_program")
        intent = state.get("monitor_outputs", {}).get("intent", {})
        performance = state.get("monitor_outputs", {}).get("performance", {})
        
        # Check for abort condition
        if performance.get("status") == "CRITICAL":
            return {
                "should_abort": True,
                "decision_reasoning": f"Critical failure detected: {performance.get('reasoning')}",
                "workflow_step": "decision_abort",
            }
        
        # Ground truth for timing reference
        gt_event = state.get("current_gt_event", {})
        gt_program = gt_event.get("robot_program") if gt_event else None
        
        # Simple rule-based decision for now
        # In production, this would use the full DecisionEngine with Gemini
        next_action = None
        confidence = 0.0
        reasoning = ""
        
        # If no program running and programs available, pick first available
        if not current_program and available_programs:
            # Prioritize based on task order
            program_order = [
                "pick_hardener_bottle.prog",
                "pick_resin_bottle.prog",
                "return_hardener_bottle.prog",
                "return_resin_bottle.prog",
            ]
            
            for prog in program_order:
                if prog in available_programs:
                    next_action = prog
                    confidence = 0.8
                    reasoning = f"Program {prog} is available and next in sequence"
                    break
        
        # If ground truth says we should be running something
        if gt_program and gt_program in available_programs:
            next_action = gt_program
            confidence = 0.95
            reasoning = f"Ground truth indicates {gt_program} should execute at this timestamp"
        
        # Record decision
        decision_record = {
            "timestamp": state.get("current_timestamp_sec", 0),
            "frame_num": state.get("current_frame_num", 0),
            "action": next_action,
            "confidence": confidence,
            "reasoning": reasoning,
            "gt_action": gt_event.get("action") if gt_event else None,
            "gt_program": gt_program,
            "available_programs": available_programs,
        }
        
        return {
            "next_action": next_action,
            "action_confidence": confidence,
            "decision_reasoning": reasoning,
            "decision_history": [decision_record],
            "workflow_step": "decision_complete",
        }
        
    except Exception as e:
        logger.error(f"Decision error: {e}")
        return {
            "next_action": None,
            "decision_reasoning": f"Error: {e}",
            "workflow_step": "decision_failed",
        }


# =============================================================================
# Execute Action Node
# =============================================================================

def execute_action_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Execute the decided action.
    
    In real deployment, this would send commands to the robot.
    For testing, it simulates execution and updates state.
    """
    next_action = state.get("next_action")
    
    if not next_action:
        return {
            "workflow_step": "execute_skipped",
        }
    
    try:
        # Update current program
        current_program = next_action
        
        # Simulate execution - update task state based on program
        updates = {}
        completed_programs = list(state.get("completed_programs", []))
        
        # For simulation, we assume instant completion
        # In real deployment, this would monitor actual robot execution
        if next_action == "pick_hardener_bottle.prog":
            updates["hardener_location"] = "robot_gripper"
            updates["gripper_state"] = "holding_hardener"
            updates["current_phase"] = "hardener_delivery"
            completed_programs.append(next_action)
            
        elif next_action == "pick_resin_bottle.prog":
            updates["resin_location"] = "robot_gripper"
            updates["gripper_state"] = "holding_resin"
            updates["current_phase"] = "resin_delivery"
            completed_programs.append(next_action)
            
        elif next_action == "return_hardener_bottle.prog":
            updates["hardener_location"] = "storage_table"
            updates["gripper_state"] = "empty"
            updates["hardener_returned"] = True
            updates["current_phase"] = "hardener_return"
            completed_programs.append(next_action)
            
        elif next_action == "return_resin_bottle.prog":
            updates["resin_location"] = "storage_table"
            updates["gripper_state"] = "empty"
            updates["resin_returned"] = True
            updates["current_phase"] = "resin_return"
            completed_programs.append(next_action)
        
        logger.info(f"Executed action: {next_action}")
        
        return {
            "current_program": None,  # Program completed
            "completed_programs": completed_programs,
            **updates,
            "workflow_step": "execute_complete",
        }
        
    except Exception as e:
        logger.error(f"Execute error: {e}")
        return {
            "error": str(e),
            "workflow_step": "execute_failed",
        }


# =============================================================================
# Check Complete Node
# =============================================================================

def check_complete_node(state: WeighBottlesState) -> Dict[str, Any]:
    """Check if the task is complete.
    
    Evaluates task state to determine if workflow should continue.
    """
    # Check for errors
    if state.get("error"):
        return {
            "is_complete": True,
            "workflow_step": "error_exit",
        }
    
    # Check if video processing is complete
    if state.get("workflow_step") in ["video_complete", "max_frames_reached"]:
        return {
            "is_complete": True,
            "workflow_step": "video_done",
        }
    
    # Check if all task objectives are met
    hardener_returned = state.get("hardener_returned", False)
    resin_returned = state.get("resin_returned", False)
    
    if hardener_returned and resin_returned:
        logger.info("Task complete: All bottles returned")
        return {
            "is_complete": True,
            "workflow_step": "task_complete",
        }
    
    # Record frame result for analysis
    frame_result = {
        "frame_num": state.get("current_frame_num", 0),
        "timestamp": state.get("current_timestamp_sec", 0),
        "current_action": state.get("current_action", "unknown"),
        "decision": state.get("next_action"),
        "gt_event": state.get("current_gt_event", {}).get("action"),
        "completed_programs": list(state.get("completed_programs", [])),
    }
    
    return {
        "is_complete": False,
        "frame_results": [frame_result],
        "workflow_step": "continue",
    }

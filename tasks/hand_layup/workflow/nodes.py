"""LangGraph node functions for Hand Layup workflow.

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

from .graph import HandLayupState

logger = logging.getLogger(__name__)


# =============================================================================
# Frame Capture Node
# =============================================================================

def capture_frames_node(state: HandLayupState) -> Dict[str, Any]:
    """Capture next frame(s) from video."""
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        next_frame = current_frame + frame_skip if current_frame > 0 else 0
        
        if next_frame >= total_frames:
            cap.release()
            return {
                "is_complete": True,
                "workflow_step": "video_complete",
            }
        
        if max_frames and next_frame // frame_skip >= max_frames:
            cap.release()
            return {
                "is_complete": True,
                "workflow_step": "max_frames_reached",
            }
        
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
        
        max_dim = 640
        if max(pil_image.size) > max_dim:
            ratio = max_dim / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        frame_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        frames_buffer = state.get("frames_buffer", [])[-4:]
        frames_buffer.append(frame_b64)
        
        timestamp = next_frame / fps
        
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

def run_perception_node(state: HandLayupState) -> Dict[str, Any]:
    """Run perception monitor on current frame.
    
    Detects objects of interest for the hand layup task.
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
        from aura.monitors.perception_module import PerceptionModule
        from aura.utils.config import PerceptionConfig
        from aura.core import MonitorType
        
        frame_b64 = frames_buffer[-1]
        frame_bytes = base64.b64decode(frame_b64)
        pil_image = Image.open(BytesIO(frame_bytes))
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        objects_of_interest = [
            "fiberglass sheet", "metal mold", "resin bottle", "hardener bottle",
            "cup", "weigh scale", "small brush", "medium brush", "roller",
            "person", "hand", "gloves",
        ]
        
        config = PerceptionConfig(
            monitor_type=MonitorType.PERCEPTION,
            enabled=True,
            use_sam3=False,
            use_gemini_detection=True,
            default_prompts=objects_of_interest,
            confidence_threshold=0.3,
        )
        
        perception = PerceptionModule(config)
        
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(perception.process_frame(frame_bgr))
        finally:
            loop.close()
        
        detected_objects = []
        gloves_detected = False
        if output and hasattr(output, 'objects'):
            for obj in output.objects:
                name = getattr(obj, 'name', getattr(obj, 'label', 'unknown'))
                detected_objects.append({
                    "id": getattr(obj, 'id', str(len(detected_objects))),
                    "name": name,
                    "confidence": getattr(obj, 'confidence', 0.0),
                })
                if "glove" in name.lower():
                    gloves_detected = True
        
        perception_output = {
            "detected_objects": detected_objects,
            "gloves_detected": gloves_detected,
            "timestamp": state.get("current_timestamp_sec", 0),
            "frame_num": state.get("current_frame_num", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "perception": perception_output,
            },
            "human_wearing_gloves": gloves_detected,
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
# Intent Node  (RCWPS – Rolling Context Window with Previous State)
# =============================================================================

# Module-level singleton so that rolling state persists across graph iterations
_rcwps_intent_monitor = None


def _get_rcwps_monitor(model: str = "gemini-3-pro-preview") -> "HandLayupIntentMonitor":
    """Return (or create) the module-level RCWPS intent monitor."""
    global _rcwps_intent_monitor
    if _rcwps_intent_monitor is None:
        from tasks.hand_layup.monitors.intent_monitor import HandLayupIntentMonitor
        _rcwps_intent_monitor = HandLayupIntentMonitor(
            model=model,
            max_frames=5,
            max_image_dimension=640,
            temperature=0.3,
            enable_logging=True,
        )
    return _rcwps_intent_monitor


def run_intent_node(state: HandLayupState) -> Dict[str, Any]:
    """Run RCWPS intent monitor to predict current/next actions and step tracking.

    Uses the Rolling Context Window with Previous State approach:
    each call sends the DAG, state schema, previous predicted state, and a
    window of recent frames to Gemini.  The model returns an updated state
    dict with steps_completed / in_progress / pending lists.
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
        config = state.get("config", {})
        model = config.get("brain", {}).get("model", "gemini-2.5-flash")
        monitor = _get_rcwps_monitor(model)

        # Decode base64 frames → numpy BGR (last 5)
        frames: List[np.ndarray] = []
        for frame_b64 in frames_buffer[-5:]:
            frame_bytes = base64.b64decode(frame_b64)
            pil_image = Image.open(BytesIO(frame_bytes))
            frames.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

        timestamp = state.get("current_timestamp_sec", 0.0)
        frame_num = state.get("current_frame_num", 0)

        result = monitor.predict(
            frames=frames,
            timestamp=timestamp,
            frame_num=frame_num,
        )

        intent_output = {
            "current_action": result.current_action,
            "current_confidence": result.prediction_confidence,
            "predicted_next": result.predicted_next_action,
            "predicted_confidence": result.prediction_confidence,
            "reasoning": result.reasoning,
            "steps_completed": result.steps_completed,
            "steps_in_progress": result.steps_in_progress,
            "steps_pending": result.steps_pending,
            "current_phase": result.current_phase,
            "layers_placed": result.layers_placed,
            "layers_resined": result.layers_resined,
            "mixture_mixed": result.mixture_mixed,
            "consolidated": result.consolidated,
            "human_state": result.human_state,
            "generation_time_sec": result.generation_time_sec,
            "timestamp": timestamp,
        }

        task_updates = {
            "current_action": result.current_action,
            "human_state": result.human_state,
            "current_phase": result.current_phase,
            "layers_placed": result.layers_placed,
            "layers_resined": result.layers_resined,
            "mixture_mixed": result.mixture_mixed,
            "consolidated": result.consolidated,
            "human_wearing_gloves": result.human_wearing_gloves,
        }

        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "intent": intent_output,
            },
            **task_updates,
            "workflow_step": "intent_complete",
        }

    except Exception as e:
        logger.error(f"Intent RCWPS error: {e}")
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

def run_affordance_node(state: HandLayupState) -> Dict[str, Any]:
    """Run affordance monitor to determine available robot skills."""
    try:
        import sys
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir.parent.parent))
        
        from tasks.hand_layup.monitors.affordance_monitor import (
            HandLayupAffordanceMonitor,
        )
        
        monitor = HandLayupAffordanceMonitor()
        
        # Sync object locations from state
        location_keys = {
            "weigh_scale": "weigh_scale_location",
            "resin_bottle": "resin_bottle_location",
            "hardener_bottle": "hardener_bottle_location",
            "cup": "cup_location",
            "brush_small": "brush_small_location",
            "brush_medium": "brush_medium_location",
            "roller": "roller_location",
        }
        
        for obj_id, state_key in location_keys.items():
            loc = state.get(state_key, "workplace")
            monitor.update_object_location(obj_id, loc)
        
        # Sync completed task nodes
        gt_event = state.get("current_gt_event")
        if gt_event:
            action = gt_event.get("action", "")
            monitor.mark_task_node_complete(action)
        
        # Get available skills
        available_skills = monitor.get_available_skills()
        housekeeping = monitor.get_housekeeping_suggestions()
        
        affordance_output = {
            "available_skills": available_skills,
            "housekeeping_suggestions": housekeeping,
            "object_locations": dict(monitor.object_locations),
            "timestamp": state.get("current_timestamp_sec", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "affordance": affordance_output,
            },
            "available_skills": [s["skill_id"] for s in available_skills],
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

def run_performance_node(state: HandLayupState) -> Dict[str, Any]:
    """Run performance monitor to check for safety and quality issues."""
    try:
        import sys
        task_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(task_dir.parent.parent))
        
        # Check safety: gloves
        safety_alerts = list(state.get("safety_alerts", []))
        current_phase = state.get("current_phase", "initialization")
        wearing_gloves = state.get("human_wearing_gloves", False)
        
        resin_phases = [
            "resin_preparation", "mixing",
            "layer_1_placement", "layer_1_resin",
            "layer_2_placement", "layer_2_resin",
            "layer_3_placement", "layer_3_resin",
            "layer_4_placement", "layer_4_resin",
            "consolidation",
        ]
        
        if current_phase in resin_phases and not wearing_gloves:
            alert = "WARNING: Gloves not detected during resin handling phase"
            if alert not in safety_alerts:
                safety_alerts.append(alert)
                logger.warning(alert)
        
        # Check shelf life
        shelf_life_start = state.get("mixture_shelf_life_start")
        if shelf_life_start is not None:
            elapsed = state.get("current_timestamp_sec", 0) - shelf_life_start
            if elapsed > 1800:
                alert = f"CRITICAL: Mixture shelf life exceeded ({elapsed/60:.0f} min)"
                if alert not in safety_alerts:
                    safety_alerts.append(alert)
            elif elapsed > 1200:
                alert = f"WARNING: Mixture shelf life at {elapsed/60:.0f} min (max 30 min)"
                if alert not in safety_alerts:
                    safety_alerts.append(alert)
        
        performance_output = {
            "status": "WARNING" if safety_alerts else "OK",
            "safety_alerts": safety_alerts,
            "human_wearing_gloves": wearing_gloves,
            "shelf_life_elapsed_sec": (
                state.get("current_timestamp_sec", 0) - shelf_life_start
                if shelf_life_start is not None else None
            ),
            "timestamp": state.get("current_timestamp_sec", 0),
        }
        
        return {
            "monitor_outputs": {
                **state.get("monitor_outputs", {}),
                "performance": performance_output,
            },
            "safety_alerts": safety_alerts,
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

def update_ssg_node(state: HandLayupState) -> Dict[str, Any]:
    """Update Semantic Scene Graph from monitor outputs."""
    try:
        from aura.core.scene_graph import (
            SemanticSceneGraph,
            ObjectNode, AgentNode, RegionNode,
            SSGEdge, SpatialRelation, SemanticRelation,
            NodeType, EdgeType,
        )
        from aura.core.scene_graph.nodes import ObjectState, AgentState
        
        ssg = SemanticSceneGraph(name="hand_layup")
        
        # Update from perception
        perception = state.get("monitor_outputs", {}).get("perception", {})
        detected_objects = perception.get("detected_objects", [])
        
        for obj_data in detected_objects:
            obj_id = obj_data.get("id", f"obj_{len(ssg.nodes)}")
            obj_name = obj_data.get("name", "unknown")
            
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
        
        # Add agents
        human = AgentNode(
            id="human",
            name="Human Operator",
            node_type=NodeType.AGENT,
            agent_type="human",
        )
        ssg.add_node(human)
        
        robot = AgentNode(
            id="robot",
            name="UR5 Robot",
            node_type=NodeType.AGENT,
            agent_type="robot",
            capabilities=[
                "move_to_workplace",
                "move_to_storage",
            ],
        )
        ssg.add_node(robot)
        
        # Add regions
        for region_id, region_name in [
            ("workplace", "Workplace Table"),
            ("storage", "Storage Table"),
            ("mold_surface", "Mold Surface"),
        ]:
            if not ssg.has_node(region_id):
                region = RegionNode(
                    id=region_id,
                    name=region_name,
                    node_type=NodeType.REGION,
                )
                ssg.add_node(region)
        
        # Update task state in SSG
        intent = state.get("monitor_outputs", {}).get("intent", {})
        task_state = {
            "current_action": intent.get("current_action", state.get("current_action", "idle")),
            "current_phase": state.get("current_phase", "initialization"),
            "layers_placed": state.get("layers_placed", 0),
            "layers_resined": state.get("layers_resined", 0),
            "consolidated": state.get("consolidated", False),
            "mixture_mixed": state.get("mixture_mixed", False),
            "human_wearing_gloves": state.get("human_wearing_gloves", False),
        }
        
        for key, value in task_state.items():
            ssg.set_task_state(key, value)
        
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

def decide_action_node(state: HandLayupState) -> Dict[str, Any]:
    """Decide what action the robot should take.
    
    For the hand layup task, the robot's main decisions are:
    1. Proactively deliver objects that will be needed soon
    2. Housekeep objects that are no longer needed
    3. Alert on safety issues (missing gloves, shelf life)
    """
    try:
        affordance = state.get("monitor_outputs", {}).get("affordance", {})
        performance = state.get("monitor_outputs", {}).get("performance", {})
        intent = state.get("monitor_outputs", {}).get("intent", {})
        
        # Check for critical safety issues
        safety_alerts = state.get("safety_alerts", [])
        if any("CRITICAL" in a for a in safety_alerts):
            return {
                "should_abort": True,
                "decision_reasoning": f"Critical safety alert: {safety_alerts}",
                "workflow_step": "decision_abort",
            }
        
        # Get ground truth for offline evaluation
        gt_event = state.get("current_gt_event", {})
        gt_robot_action = gt_event.get("robot_action") if gt_event else None
        
        next_action = None
        confidence = 0.0
        reasoning = ""
        
        # Check housekeeping suggestions from affordance monitor
        housekeeping = affordance.get("housekeeping_suggestions", [])
        if housekeeping and not state.get("current_skill"):
            suggestion = housekeeping[0]
            next_action = suggestion["skill_id"]
            confidence = 0.7
            reasoning = (
                f"Housekeeping: {suggestion['reason']}. "
                f"Moving {suggestion['object']} from workplace to storage."
            )
        
        # If ground truth says robot should act, use that for evaluation
        if gt_robot_action:
            confidence = 0.9
            reasoning = f"Ground truth robot action: {gt_robot_action}"
            # Parse GT action into skill_id
            if "move_to_storage" in gt_robot_action:
                # Extract object name from e.g. "move_to_storage(weigh_scale)"
                obj_part = gt_robot_action.split("(")[-1].rstrip(")")
                objects = [o.strip() for o in obj_part.split(",")]
                if objects:
                    next_action = f"move_to_storage__{objects[0]}"
        
        # Build robot instruction
        robot_instruction = None
        if next_action or gt_robot_action:
            robot_instruction = {
                "timestamp": state.get("current_timestamp_sec", 0),
                "action": next_action or gt_robot_action,
                "confidence": confidence,
                "reasoning": reasoning,
                "gt_action": gt_robot_action,
                "safety_alerts": safety_alerts,
                "executed": False,  # Not executed in dummy video
            }
        
        decision_record = {
            "timestamp": state.get("current_timestamp_sec", 0),
            "frame_num": state.get("current_frame_num", 0),
            "action": next_action,
            "confidence": confidence,
            "reasoning": reasoning,
            "gt_action": gt_event.get("action") if gt_event else None,
            "gt_robot_action": gt_robot_action,
            "safety_alerts": safety_alerts,
        }
        
        updates = {
            "next_action": next_action,
            "action_confidence": confidence,
            "decision_reasoning": reasoning,
            "decision_history": [decision_record],
            "workflow_step": "decision_complete",
        }
        
        if robot_instruction:
            updates["robot_instructions"] = state.get("robot_instructions", []) + [robot_instruction]
        
        return updates
        
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

def execute_action_node(state: HandLayupState) -> Dict[str, Any]:
    """Execute the decided action.
    
    In the dummy video, robot is not active. We simulate the action
    and record what the robot WOULD do.
    """
    next_action = state.get("next_action")
    
    if not next_action:
        return {"workflow_step": "execute_skipped"}
    
    try:
        updates = {}
        completed_skills = list(state.get("completed_skills", []))
        
        # Parse the action: move_to_storage__object or move_to_workplace__object
        parts = next_action.split("__")
        if len(parts) == 2:
            direction, obj = parts
            
            # Map object to state key
            location_map = {
                "weigh_scale": "weigh_scale_location",
                "resin_bottle": "resin_bottle_location",
                "hardener_bottle": "hardener_bottle_location",
                "cup": "cup_location",
                "brush_small": "brush_small_location",
                "brush_medium": "brush_medium_location",
                "roller": "roller_location",
            }
            
            state_key = location_map.get(obj)
            if state_key:
                if direction == "move_to_storage":
                    updates[state_key] = "storage"
                    logger.info(f"[SIMULATED] Moved {obj} to storage")
                elif direction == "move_to_workplace":
                    updates[state_key] = "workplace"
                    logger.info(f"[SIMULATED] Moved {obj} to workplace")
        
        completed_skills.append(next_action)
        
        return {
            "current_skill": None,
            "completed_skills": completed_skills,
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

def check_complete_node(state: HandLayupState) -> Dict[str, Any]:
    """Check if the task is complete."""
    if state.get("error"):
        return {
            "is_complete": True,
            "workflow_step": "error_exit",
        }
    
    if state.get("workflow_step") in ["video_complete", "max_frames_reached"]:
        return {
            "is_complete": True,
            "workflow_step": "video_done",
        }
    
    # Check ground truth for task completion
    gt_event = state.get("current_gt_event")
    if gt_event and gt_event.get("action") == "task_complete":
        return {
            "is_complete": True,
            "workflow_step": "task_complete",
        }
    
    # Update task state from ground truth for tracking
    task_updates = {}
    if gt_event:
        action = gt_event.get("action", "")
        # Map GT action to phase and layer counts
        phase_map = {
            "place_cup_on_scale": ("resin_preparation", 0, 0),
            "add_resin": ("resin_preparation", 0, 0),
            "add_hardener": ("resin_preparation", 0, 0),
            "weigh_mixture": ("resin_preparation", 0, 0),
            "mix_resin_hardener": ("mixing", 0, 0),
            "place_layer_1": ("layer_1_placement", 1, 0),
            "apply_resin_layer_1": ("layer_1_resin", 1, 1),
            "place_layer_2": ("layer_2_placement", 2, 1),
            "apply_resin_layer_2": ("layer_2_resin", 2, 2),
            "place_layer_3": ("layer_3_placement", 3, 2),
            "apply_resin_layer_3": ("layer_3_resin", 3, 3),
            "place_layer_4": ("layer_4_placement", 4, 3),
            "apply_resin_layer_4": ("layer_4_resin", 4, 4),
            "consolidate_with_roller": ("consolidation", 4, 4),
            "cleanup": ("cleanup", 4, 4),
        }
        
        if action in phase_map:
            phase, layers_p, layers_r = phase_map[action]
            task_updates["current_phase"] = phase
            task_updates["layers_placed"] = layers_p
            task_updates["layers_resined"] = layers_r
            task_updates["current_action"] = action
            
            if action == "mix_resin_hardener":
                task_updates["mixture_mixed"] = True
                if state.get("mixture_shelf_life_start") is None:
                    task_updates["mixture_shelf_life_start"] = state.get("current_timestamp_sec", 0)
            if action == "add_resin":
                task_updates["resin_added"] = True
            if action == "add_hardener":
                task_updates["hardener_added"] = True
            if action == "weigh_mixture":
                task_updates["mixture_weighed"] = True
            if action == "consolidate_with_roller":
                task_updates["consolidated"] = True
    
    frame_result = {
        "frame_num": state.get("current_frame_num", 0),
        "timestamp": state.get("current_timestamp_sec", 0),
        "current_action": state.get("current_action", "unknown"),
        "current_phase": state.get("current_phase", "initialization"),
        "layers_placed": state.get("layers_placed", 0),
        "layers_resined": state.get("layers_resined", 0),
        "decision": state.get("next_action"),
        "gt_event": gt_event.get("action") if gt_event else None,
        "safety_alerts": state.get("safety_alerts", []),
    }
    
    return {
        "is_complete": False,
        "frame_results": [frame_result],
        **task_updates,
        "workflow_step": "continue",
    }

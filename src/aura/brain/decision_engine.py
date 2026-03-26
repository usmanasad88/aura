"""Decision Engine - The Brain of AURA.

The Decision Engine is responsible for:
1. Receiving monitor outputs and updating the SSG
2. Using LLM reasoning to decide on proactive actions
3. Generating explainable decisions
4. Predicting when robot skills should be executed

The engine uses a configurable Gemini model for reasoning.
"""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from aura.core.scene_graph import (
    SemanticSceneGraph, GraphReasoner,
    SSGNode, ObjectNode, AgentNode, RegionNode,
    SSGEdge, SpatialRelation, SemanticRelation,
    NodeType, EdgeType
)
from aura.core.scene_graph.nodes import ObjectState, AgentState, Affordance
from aura.core import (
    MonitorOutput, PerceptionOutput, IntentOutput,
    MotionOutput, SoundOutput, TrackedObject
)
from .skill_registry import SkillRegistry, RobotSkill
from .explainer import DecisionExplainer, DecisionRecord


logger = logging.getLogger(__name__)


# ─── Prompt / Response Logger ────────────────────────────────────────────────

class DecisionPromptLogger:
    """Logs every LLM prompt/response exchange to disk.

    Mirrors the ``PromptLogger`` used by ``AURAIntentMonitor`` so that
    decision engine calls can be reviewed and debugged in the same way.

    Directory layout per call::

        <session_dir>/
          call_0001/
            prompt.txt
            response.txt
            response_parsed.json
            ssg_snapshot.json
            meta.json
          call_0002/
            ...
    """

    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.session_dir: Optional[Path] = None
            return

        base = Path(log_dir) if log_dir else Path("logs/decision_engine")
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = base / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.call_counter = 0
        logger.info("Decision prompt logger session: %s", self.session_dir)

    def log_call(
        self,
        *,
        prompt_text: str,
        response_text: str,
        parsed_response: Optional[Dict[str, Any]],
        model: str,
        generation_time_sec: float,
        timestamp_sec: float,
        frame_num: int = 0,
        decision: Optional[str] = None,
        ssg_snapshot: Optional[Dict[str, Any]] = None,
        available_actions: Optional[List[Dict]] = None,
        proactive_opportunities: Optional[List[Dict]] = None,
    ) -> None:
        """Persist one LLM call to the session directory."""
        if not self.enabled or self.session_dir is None:
            return

        self.call_counter += 1
        call_dir = self.session_dir / f"call_{self.call_counter:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

        if parsed_response is not None:
            with open(call_dir / "response_parsed.json", "w") as f:
                json.dump(parsed_response, f, indent=2, default=str)

        if ssg_snapshot is not None:
            with open(call_dir / "ssg_snapshot.json", "w") as f:
                json.dump(ssg_snapshot, f, indent=2, default=str)

        meta: Dict[str, Any] = {
            "call_number": self.call_counter,
            "model": model,
            "generation_time_sec": round(generation_time_sec, 3),
            "timestamp_sec": round(timestamp_sec, 3),
            "frame_num": frame_num,
            "decision": decision,
            "response_length_chars": len(response_text),
            "num_available_actions": len(available_actions) if available_actions else 0,
            "num_proactive_opportunities": len(proactive_opportunities) if proactive_opportunities else 0,
            "logged_at": datetime.now().isoformat(),
        }
        with open(call_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def get_session_dir(self) -> Optional[Path]:
        return self.session_dir


# Lazy import for backwards compatibility
_gemini_client = None

def _get_gemini_client(api_key: str = None):
    """Get or create Gemini client (legacy helper, prefer create_llm_client)."""
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                _gemini_client = genai.Client(
                    http_options={"api_version": "v1beta"},
                    api_key=key,
                )
            else:
                logger.warning("GEMINI_API_KEY not set")
        except ImportError:
            logger.warning("google-genai not installed")
    return _gemini_client


@dataclass
class DecisionEngineConfig:
    """Configuration for the Decision Engine."""
    gemini_model: str = "gemini-2.5-pro-preview-06-05"  # Default model
    enable_llm_reasoning: bool = True
    max_reasoning_time_sec: float = 10.0
    decision_interval_sec: float = 1.0
    enable_explainability: bool = True
    proactive_threshold: float = 0.7  # Min confidence for proactive actions
    timing_prediction_enabled: bool = True

    # LLM backend: "gemini", "openai", "sglang", "vllm", "ollama"
    llm_backend: str = "gemini"
    sglang_base_url: str = "http://localhost:8100/v1"

    # Logging
    enable_logging: bool = True
    log_dir: Optional[str] = None

    # Paths
    sop_path: Optional[str] = None
    skills_path: Optional[str] = None
    initial_scene_path: Optional[str] = None


@dataclass
class ActionPrediction:
    """A predicted action with timing."""
    action_id: str
    target_id: Optional[str]
    predicted_time_sec: float  # Time from start of task
    confidence: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """Central decision-making component of AURA.
    
    Maintains the Semantic Scene Graph, processes monitor outputs,
    and uses LLM reasoning to decide on proactive robot actions.
    """
    
    def __init__(self, config: DecisionEngineConfig = None):
        """Initialize the Decision Engine.

        Args:
            config: Engine configuration
        """
        self.config = config or DecisionEngineConfig()

        # Core components
        self.graph = SemanticSceneGraph(name="aura_ssg")
        self.reasoner = GraphReasoner(self.graph)
        self.explainer = DecisionExplainer(self.graph)
        self.skills = SkillRegistry()
        self.prompt_logger = DecisionPromptLogger(
            log_dir=self.config.log_dir,
            enabled=self.config.enable_logging,
        )

        # Load additional skills if path provided
        if self.config.skills_path and Path(self.config.skills_path).exists():
            self.skills.load_from_file(self.config.skills_path)

        # State tracking
        self.is_running = False
        self.task_start_time: Optional[datetime] = None
        self.current_video_time_sec: float = 0.0
        self.pending_actions: List[ActionPrediction] = []
        self.executed_actions: List[Dict[str, Any]] = []

        # Ground truth for evaluation
        self.ground_truth: List[Dict[str, Any]] = []

        # LLM client (unified abstraction)
        self._llm_client = None

        logger.info(
            "DecisionEngine initialized — model: %s, backend: %s",
            self.config.gemini_model, self.config.llm_backend,
        )

    @property
    def llm_client(self):
        """Get LLM client (lazy initialization via unified abstraction)."""
        if self._llm_client is None:
            try:
                from aura.utils.llm_client import create_llm_client
                self._llm_client = create_llm_client(
                    self.config.llm_backend,
                    model=self.config.gemini_model,
                    base_url=self.config.sglang_base_url,
                )
            except Exception as e:
                logger.warning("Failed to create LLM client (%s): %s", self.config.llm_backend, e)
        return self._llm_client
    
    # =========================================================================
    # Scene Graph Updates
    # =========================================================================
    
    def update_from_perception(self, output: PerceptionOutput) -> None:
        """Update SSG from perception monitor output."""
        if not output or not output.is_valid:
            return
        
        for obj in output.objects:
            # Convert TrackedObject to ObjectNode
            existing = self.graph.get_node(obj.id)
            
            if existing and isinstance(existing, ObjectNode):
                # Update existing node
                existing.bbox = (obj.bbox.x_min, obj.bbox.y_min, 
                                obj.bbox.x_max, obj.bbox.y_max) if obj.bbox else None
                existing.confidence = obj.confidence
                existing.last_updated = datetime.now()
                if obj.pose:
                    existing.position = (obj.pose.x, obj.pose.y, obj.pose.z)
            else:
                # Create new node
                node = ObjectNode(
                    id=obj.id,
                    name=obj.name,
                    node_type=NodeType.OBJECT,
                    category=obj.category,
                    bbox=(obj.bbox.x_min, obj.bbox.y_min,
                          obj.bbox.x_max, obj.bbox.y_max) if obj.bbox else None,
                    confidence=obj.confidence,
                    position=(obj.pose.x, obj.pose.y, obj.pose.z) if obj.pose else None,
                    state=ObjectState.AVAILABLE,
                )
                self.graph.add_node(node)
    
    def update_from_intent(self, output: IntentOutput) -> None:
        """Update SSG from intent monitor output."""
        if not output or not output.is_valid or not output.intent:
            return
        
        intent = output.intent
        
        # Find or create human agent
        human = None
        for agent in self.graph.get_agents():
            if agent.agent_type == "human":
                human = agent
                break
        
        if not human:
            human = AgentNode(
                id="human",
                name="Human Operator",
                node_type=NodeType.AGENT,
                agent_type="human",
            )
            self.graph.add_node(human)
        
        # Update human state based on intent
        intent_to_state = {
            "IDLE": AgentState.IDLE,
            "REACHING": AgentState.REACHING,
            "GRASPING": AgentState.GRASPING,
            "MOVING": AgentState.MOVING,
            "PLACING": AgentState.PLACING,
            "SPEAKING": AgentState.SPEAKING,
        }
        
        intent_name = intent.type.name if hasattr(intent.type, 'name') else str(intent.type)
        human.state = intent_to_state.get(intent_name, AgentState.BUSY)
        human.last_updated = datetime.now()
        
        # Update target edge if applicable
        if intent.target_object:
            self.graph.set_agent_target(
                "human", 
                intent.target_object,
                confidence=intent.confidence,
                reasoning=intent.reasoning
            )
    
    def update_from_motion(self, output: MotionOutput) -> None:
        """Update SSG from motion predictor output."""
        if not output or not output.is_valid:
            return
        
        # Update predicted actions on objects based on motion predictions
        for pred in output.predictions:
            entity = self.graph.get_node(pred.entity_id)
            if entity:
                # Add predicted motion info to node metadata
                entity.metadata["predicted_trajectory"] = {
                    "confidence": pred.confidence,
                    "horizon_sec": pred.prediction_horizon_sec,
                }
    
    def update_from_sound(self, output: SoundOutput) -> None:
        """Update SSG from sound monitor output.

        Stores each utterance in ``task_state["recent_utterances"]`` so
        the LLM prompt can reference them for context-aware decisions.
        """
        if not output or not output.is_valid:
            return

        recent: List[Dict[str, Any]] = list(
            self.graph.task_state.get("recent_utterances", [])
        )
        for utterance in output.utterances:
            recent.append({
                "text": utterance.text,
                "timestamp": getattr(utterance, "timestamp", None),
            })

        # Keep only the last 10 utterances
        self.graph.set_task_state("recent_utterances", recent[-10:])
    
    def process_monitor_outputs(self, outputs: Dict[str, MonitorOutput]) -> None:
        """Process outputs from all monitors."""
        if "perception" in outputs:
            self.update_from_perception(outputs["perception"])
        if "intent" in outputs:
            self.update_from_intent(outputs["intent"])
        if "motion" in outputs:
            self.update_from_motion(outputs["motion"])
        if "sound" in outputs:
            self.update_from_sound(outputs["sound"])
        
        self.graph.take_snapshot()
    
    # =========================================================================
    # Decision Making
    # =========================================================================
    
    async def decide_action(self, current_time_sec: float = None) -> Optional[ActionPrediction]:
        """Decide what action the robot should take (if any).
        
        Args:
            current_time_sec: Current time in the task
        
        Returns:
            ActionPrediction if robot should act, None otherwise
        """
        if current_time_sec is not None:
            self.current_video_time_sec = current_time_sec
        
        # Get available actions from reasoner
        robot = self.graph.get_node("robot")
        if not robot:
            # Create robot agent if not exists
            robot = AgentNode(
                id="robot",
                name="Robot Assistant",
                node_type=NodeType.AGENT,
                agent_type="robot",
                capabilities=self.skills.list_skill_ids(),
            )
            self.graph.add_node(robot)
        
        available_actions = self.reasoner.get_available_actions("robot")
        proactive_opportunities = self.reasoner.get_proactive_opportunities("robot")
        
        # If LLM reasoning enabled, use it to select action
        if self.config.enable_llm_reasoning and self.llm_client:
            return await self._llm_decide_action(
                available_actions, 
                proactive_opportunities,
                current_time_sec
            )
        else:
            # Rule-based fallback
            return self._rule_based_decide(
                available_actions,
                proactive_opportunities,
                current_time_sec
            )
    
    def _format_recent_decisions(self, n: int = 5) -> str:
        """Format the last *n* decisions for inclusion in the LLM prompt."""
        recent = self.explainer.decision_history[-n:]
        if not recent:
            return "No actions taken yet."
        lines: List[str] = []
        for d in recent:
            if d.decision_type == "action":
                lines.append(
                    f"- Executed: {d.action_id} on {d.target or 'N/A'} "
                    f"(confidence {d.confidence:.1f}) — {d.reasoning[:100]}"
                )
            else:
                lines.append(f"- Waited — {d.reasoning[:100]}")
        return "\n".join(lines)

    async def _llm_decide_action(self, available_actions: List[Dict],
                                  opportunities: List[Dict],
                                  current_time_sec: float) -> Optional[ActionPrediction]:
        """Use LLM to decide on action."""
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to rules")
            return self._rule_based_decide(available_actions, opportunities, current_time_sec)

        # Build prompt
        scene_state = self.graph.get_state_summary_for_llm()
        skills_desc = self.skills.get_skills_for_llm()

        prompt = f"""You are a proactive robot assistant helping a human with a task.
Your goal is to anticipate what the human needs and provide timely assistance.

{scene_state}

{skills_desc}

## Available Actions Now
{json.dumps(available_actions[:5], indent=2) if available_actions else "No immediately available actions."}

## Proactive Opportunities
{json.dumps(opportunities[:3], indent=2) if opportunities else "No proactive opportunities identified."}

## Current Time
Task time: {current_time_sec:.1f} seconds

## Recent Robot Actions
{self._format_recent_decisions()}

## Your Task
Decide whether the robot should:
1. Execute an action now
2. Wait for a better moment
3. Ask the human a question

If you decide to act, respond with JSON:
{{
    "decision": "act",
    "action_id": "<skill_id>",
    "target_id": "<target_object_id or null>",
    "parameters": {{}},
    "confidence": 0.0-1.0,
    "reasoning": "Explain why this action, why now"
}}

If waiting is better:
{{
    "decision": "wait",
    "reasoning": "Explain what we're waiting for"
}}

Respond with ONLY the JSON object, no other text."""

        try:
            t0 = time.monotonic()
            response_text = await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm_client.generate,
                    prompt,
                    temperature=0.3,
                    json_mode=True,
                ),
                timeout=self.config.max_reasoning_time_sec,
            )
            generation_time = time.monotonic() - t0

            result = json.loads(response_text)

            # ── Log the call ────────────────────────────────────
            self.prompt_logger.log_call(
                prompt_text=prompt,
                response_text=response_text,
                parsed_response=result,
                model=self.config.gemini_model,
                generation_time_sec=generation_time,
                timestamp_sec=current_time_sec,
                decision=result.get("decision"),
                ssg_snapshot=self.graph.to_dict(),
                available_actions=available_actions,
                proactive_opportunities=opportunities,
            )

            if result.get("decision") == "act":
                prediction = ActionPrediction(
                    action_id=result["action_id"],
                    target_id=result.get("target_id"),
                    predicted_time_sec=current_time_sec,
                    confidence=result.get("confidence", 0.5),
                    reasoning=result.get("reasoning", ""),
                    parameters=result.get("parameters", {}),
                )

                # Record decision
                if self.config.enable_explainability:
                    self.explainer.record_decision(DecisionRecord(
                        timestamp=datetime.now(),
                        decision_type="action",
                        action_id=prediction.action_id,
                        target=prediction.target_id,
                        parameters=prediction.parameters,
                        reasoning=prediction.reasoning,
                        confidence=prediction.confidence,
                    ))

                return prediction
            else:
                # Record wait decision
                if self.config.enable_explainability:
                    self.explainer.record_decision(DecisionRecord(
                        timestamp=datetime.now(),
                        decision_type="wait",
                        reasoning=result.get("reasoning", "Waiting"),
                        confidence=1.0,
                    ))
                return None

        except asyncio.TimeoutError:
            logger.warning("LLM reasoning timed out")
            return self._rule_based_decide(available_actions, opportunities, current_time_sec)
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return self._rule_based_decide(available_actions, opportunities, current_time_sec)
    
    def _rule_based_decide(self, available_actions: List[Dict],
                           opportunities: List[Dict],
                           current_time_sec: float) -> Optional[ActionPrediction]:
        """Simple rule-based decision making as fallback."""
        # Check proactive opportunities first
        for opp in opportunities:
            if opp.get("priority", 0) >= self.config.proactive_threshold:
                return ActionPrediction(
                    action_id=opp["action_id"],
                    target_id=opp.get("target"),
                    predicted_time_sec=current_time_sec,
                    confidence=opp.get("priority", 0.5),
                    reasoning=opp.get("reasoning", "Rule-based decision"),
                )
        
        # Check available actions
        for action in available_actions:
            if action.get("feasibility", 0) >= self.config.proactive_threshold:
                return ActionPrediction(
                    action_id=action["action_id"],
                    target_id=action.get("target_object"),
                    predicted_time_sec=current_time_sec,
                    confidence=action.get("feasibility", 0.5),
                    reasoning=action.get("reasoning", "High feasibility action"),
                )
        
        return None
    
    # =========================================================================
    # Task and SOP Management
    # =========================================================================
    
    def load_task(self, dag_path: str, state_path: str = None,
                  skills_path: str = None, initial_scene_path: str = None) -> None:
        """Load a task definition.
        
        Args:
            dag_path: Path to task DAG JSON
            state_path: Path to state schema JSON
            skills_path: Path to robot skills JSON
            initial_scene_path: Path to initial scene graph JSON
        """
        # Load DAG
        with open(dag_path, 'r') as f:
            dag = json.load(f)
        
        # Store DAG in graph metadata
        self.graph.metadata = {"dag": dag}
        dag_name = dag.get("name", "unknown") if isinstance(dag, dict) else f"{len(dag)} steps"
        logger.info(f"Loaded task DAG: {dag_name}")
        
        # Load state schema
        if state_path and Path(state_path).exists():
            with open(state_path, 'r') as f:
                state_schema = json.load(f)
            self.graph.initialize_task_state(state_schema)
            logger.info(f"Initialized {len(self.graph.task_state)} state variables")
        
        # Load additional skills
        if skills_path and Path(skills_path).exists():
            self.skills.load_from_file(skills_path)
        
        # Load initial scene
        if initial_scene_path and Path(initial_scene_path).exists():
            with open(initial_scene_path, 'r') as f:
                scene_data = json.load(f)
            self._initialize_scene(scene_data)
    
    def _initialize_scene(self, scene_data: Dict[str, Any]) -> None:
        """Initialize scene graph from scene definition."""
        # Add regions
        for region_data in scene_data.get("regions", []):
            region = RegionNode.from_dict({
                "id": region_data["id"],
                "name": region_data["name"],
                "node_type": "REGION",
                **region_data
            })
            self.graph.add_node(region)
        
        # Add objects
        for obj_data in scene_data.get("objects", []):
            obj = ObjectNode.from_dict({
                "id": obj_data["id"],
                "name": obj_data["name"],
                "node_type": "OBJECT",
                **obj_data
            })
            
            # Add affordances
            for aff_data in obj_data.get("affordances", []):
                obj.add_affordance(Affordance(**aff_data))
            
            self.graph.add_node(obj)
            
            # Set initial location
            if "initial_location" in obj_data:
                self.graph.set_location(obj.id, obj_data["initial_location"])
        
        # Add agents
        for agent_data in scene_data.get("agents", []):
            agent = AgentNode.from_dict({
                "id": agent_data["id"],
                "name": agent_data["name"],
                "node_type": "AGENT",
                **agent_data
            })
            
            # Set robot capabilities from skills registry
            if agent.agent_type == "robot":
                agent.capabilities = self.skills.list_skill_ids()
            
            self.graph.add_node(agent)
        
        logger.info(f"Initialized scene with {self.graph.node_count} nodes")
    
    def load_ground_truth(self, ground_truth_path: str) -> None:
        """Load ground truth timing data for evaluation.
        
        Ground truth format:
        [
            {"time_sec": 10.5, "action_id": "retrieve_object", "target": "obj_a"},
            {"time_sec": 25.0, "action_id": "ask_preference", "target": null},
            ...
        ]
        """
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        logger.info(f"Loaded {len(self.ground_truth)} ground truth events")
    
    def evaluate_predictions(self, tolerance_sec: float = 2.0) -> Dict[str, Any]:
        """Evaluate predictions against ground truth.
        
        Returns evaluation metrics.
        """
        if not self.ground_truth:
            return {"error": "No ground truth loaded"}
        
        decisions = self.explainer.decision_history
        action_decisions = [d for d in decisions if d.decision_type == "action"]
        
        results = {
            "total_predictions": len(action_decisions),
            "total_ground_truth": len(self.ground_truth),
            "matches": [],
            "missed": [],
            "false_positives": [],
            "timing_errors": [],
        }
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred in action_decisions:
            pred_time = self.current_video_time_sec  # Approximate
            
            best_match = None
            best_error = float('inf')
            
            for i, gt in enumerate(self.ground_truth):
                if i in matched_gt:
                    continue
                if gt["action_id"] != pred.action_id:
                    continue
                
                error = abs(gt["time_sec"] - pred_time)
                if error < best_error and error <= tolerance_sec * 2:
                    best_error = error
                    best_match = (i, gt)
            
            if best_match:
                matched_gt.add(best_match[0])
                within_tolerance = best_error <= tolerance_sec
                results["matches"].append({
                    "predicted": pred.action_id,
                    "ground_truth": best_match[1],
                    "error_sec": best_error,
                    "correct": within_tolerance,
                })
                results["timing_errors"].append(best_error)
            else:
                results["false_positives"].append({
                    "action_id": pred.action_id,
                    "time": pred_time,
                })
        
        # Find missed ground truth events
        for i, gt in enumerate(self.ground_truth):
            if i not in matched_gt:
                results["missed"].append(gt)
        
        # Calculate metrics
        if results["matches"]:
            results["accuracy"] = sum(1 for m in results["matches"] if m["correct"]) / len(results["matches"])
            results["avg_timing_error"] = sum(results["timing_errors"]) / len(results["timing_errors"])
        else:
            results["accuracy"] = 0.0
            results["avg_timing_error"] = float('inf')
        
        results["precision"] = len(results["matches"]) / max(1, len(action_decisions))
        results["recall"] = len(results["matches"]) / max(1, len(self.ground_truth))
        
        return results
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start_task(self) -> None:
        """Start task execution."""
        self.is_running = True
        self.task_start_time = datetime.now()
        self.current_video_time_sec = 0.0
        logger.info("Task started")
    
    def stop_task(self) -> Dict[str, Any]:
        """Stop task and return summary."""
        self.is_running = False
        
        summary = {
            "duration_sec": (datetime.now() - self.task_start_time).total_seconds() if self.task_start_time else 0,
            "decisions_made": len(self.explainer.decision_history),
            "actions_executed": len(self.executed_actions),
        }
        
        if self.ground_truth:
            summary["evaluation"] = self.evaluate_predictions()
        
        logger.info(f"Task stopped: {summary}")
        return summary
    
    def get_state_summary(self) -> str:
        """Get current state summary."""
        return self.graph.get_state_summary_for_llm()
    
    def get_decision_report(self) -> str:
        """Get decision history report."""
        return self.explainer.generate_decision_report()

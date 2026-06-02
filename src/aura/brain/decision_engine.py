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
from aura.core.scene_graph.nodes import AgentState
from aura.core.types import Affordance
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

        # When log_dir is provided, use it directly as the session dir
        # (the caller owns the unique per-run layout). Otherwise fall back
        # to the legacy ``logs/decision_engine/session_<timestamp>/`` scheme
        # so standalone usage (tests, ad-hoc scripts) still works.
        if log_dir:
            self.session_dir = Path(log_dir)
        else:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.session_dir = Path("logs/decision_engine") / session_name
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
        images: Optional[List[Any]] = None,
    ) -> None:
        """Persist one LLM call to the session directory."""
        if not self.enabled or self.session_dir is None:
            return

        self.call_counter += 1
        call_dir = self.session_dir / f"call_{self.call_counter:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        (call_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

        if images:
            try:
                from PIL import Image
                for i, img in enumerate(images):
                    if isinstance(img, Image.Image):
                        # Save in RGB mode to ensure JPG compatibility
                        img_path = call_dir / f"image_{i}.jpg"
                        if img.mode in ("RGBA", "P"):
                            img.convert("RGB").save(img_path)
                        else:
                            img.save(img_path)
            except Exception as e:
                logger.warning(f"Failed to save images to prompt log: {e}")

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
    # "llm"     — every cycle routed to the LLM (legacy behaviour)
    # "bt"      — BT only, no LLM; ambiguity escalates to the human
    # "hybrid"  — BT with LLM fallback on ambiguity (default)
    decision_mode: str = "hybrid"
    # When True (hybrid mode), defer to the LLM on idle ticks where no BT
    # branch fired, instead of defaulting to wait. Easy global toggle.
    llm_fallback_on_idle: bool = False

    # LLM backend: "gemini", "openai", "sglang", "vllm", "ollama"
    llm_backend: str = "gemini"
    sglang_base_url: str = "http://localhost:8100/v1"
    max_completion_tokens: int = 1024

    # Logging
    enable_logging: bool = True
    log_dir: Optional[str] = None

    # Task-specific system instruction (from task_profile.json)
    task_system_instruction: str = ""

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
    
    def __init__(self, config_dir: str, config: DecisionEngineConfig = None):
        """Initialize the Decision Engine.

        Args:
            config_dir: Path to task config directory containing dag.json, etc.
            config: Engine configuration
        """
        self.config = config or DecisionEngineConfig()
        self.config_dir = Path(config_dir)
        dag_path = self.config_dir / "dag.json"

        # Core components
        self.graph = SemanticSceneGraph(name="aura_ssg")
        self.skills = SkillRegistry()
        self.reasoner = GraphReasoner(self.graph, skills=self.skills)
        self.explainer = DecisionExplainer(self.graph)
        self.task_graph_string = dag_path.read_text(encoding="utf-8") if dag_path.exists() else "{}"

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

        # BT policy — built lazily in load_task() once configs are loaded.
        self._bt_policy = None
        self._bt_ctx = None

        # Last BT tick introspection (surfaced on the dashboard).
        self._last_bt_reasoning: str = ""
        self._last_bt_llm_invoked: bool = False
        self._last_bt_branch: str = ""

        logger.info(
            "DecisionEngine initialized — model: %s, backend: %s, mode: %s",
            self.config.gemini_model, self.config.llm_backend, self.config.decision_mode,
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
    
    async def decide_action(self, current_time_sec: float = None, current_frame: Optional[Any] = None) -> Optional[ActionPrediction]:
        """Decide what action the robot should take (if any).

        Delegates to the compiled BT policy (built by ``load_task``).
        The BT decides — deterministic branches fire first (safety,
        reactive, scheduled deliveries), and only escalate to the LLM
        fallback on ambiguity. See ``aura.brain.bt_policy`` for the
        full topology.
        """
        if current_time_sec is not None:
            self.current_video_time_sec = current_time_sec

        # Ensure robot agent exists in the SSG (legacy invariant).
        robot = self.graph.get_node("robot")
        if not robot:
            robot = AgentNode(
                id="robot",
                name="Robot Assistant",
                node_type=NodeType.AGENT,
                agent_type="robot",
                capabilities=self.skills.list_skill_ids(),
            )
            self.graph.add_node(robot)

        # If BT not built (no task loaded), fall back to legacy LLM path.
        if self._bt_policy is None:
            if self.config.enable_llm_reasoning and self.llm_client:
                return await self._llm_decide_action([], current_time_sec or 0.0, current_frame)
            return self._rule_based_decide([], current_time_sec or 0.0)

        # Gather tick inputs from SSG / task_state.
        task_state = dict(self.graph.task_state)
        intent = {
            "current_phase": task_state.get("current_phase", ""),
            "current_action": task_state.get("current_action", ""),
            "predicted_next_action": task_state.get("predicted_next_action", ""),
            "prediction_confidence": task_state.get("prediction_confidence", 0.0),
        }
        steps_completed = list(task_state.get("steps_completed") or [])
        # Intent monitor also writes steps_completed onto the SSG via
        # update_from_intent_result; but the BT also accepts it from the
        # graph snapshot directly when present.
        if not steps_completed:
            steps_completed = list(getattr(self.graph, "_steps_completed", []) or [])
        human_help = bool(task_state.get("human_requesting_help", False))

        # Run the async LLM-fallback through the BT's sync bridge.
        self._bt_pending_current_time = current_time_sec or 0.0

        t0 = time.monotonic()
        prediction, reasoning, llm_invoked = self._bt_policy.tick(
            current_time_sec=current_time_sec or 0.0,
            intent=intent,
            task_state=task_state,
            steps_completed=steps_completed,
            human_requesting_help=human_help,
        )
        generation_time = time.monotonic() - t0

        # Log the decision to disk only if it was resolved deterministically by BT
        # (If LLM was invoked, the prompt and text were already logged by _llm_decide_action)
        if not llm_invoked:
            self.prompt_logger.log_call(
                prompt_text=json.dumps({
                    "intent": intent,
                    "task_state": task_state,
                    "steps_completed": steps_completed,
                    "human_requesting_help": human_help
                }, indent=2, default=str),
                response_text=reasoning,
                parsed_response={
                    "decision": prediction.action_id if prediction else "wait",
                    "target": prediction.target_id if prediction else None,
                    "parameters": prediction.parameters if prediction else {},
                    "confidence": prediction.confidence if prediction else 1.0,
                    "reasoning": reasoning
                },
                model="BehaviorTree",
                generation_time_sec=generation_time,
                timestamp_sec=current_time_sec or 0.0,
                decision=prediction.action_id if prediction else "wait",
                ssg_snapshot=self.graph.to_dict(),
            )

        # Cache the trail so the dashboard can display BT state.
        self._last_bt_reasoning = reasoning
        self._last_bt_llm_invoked = bool(llm_invoked)
        # Branch = prefix of the first trail entry (e.g. "safety", "delivery",
        # "timer", "reactive", "llm_fallback", "escalate_human", "wait").
        first = reasoning.split(" | ", 1)[0] if reasoning else ""
        self._last_bt_branch = first.split(":", 1)[0] if first else ""

        # Record the decision with the BT reasoning trail.
        if self.config.enable_explainability:
            if prediction is not None:
                self.explainer.record_decision(DecisionRecord(
                    timestamp=datetime.now(),
                    decision_type="action" if prediction.action_id not in (
                        "ask_question", "speak", "abort"
                    ) else prediction.action_id,
                    action_id=prediction.action_id,
                    target=prediction.target_id,
                    parameters=prediction.parameters,
                    reasoning=f"{reasoning} — {prediction.reasoning}",
                    confidence=prediction.confidence,
                ))
            else:
                self.explainer.record_decision(DecisionRecord(
                    timestamp=datetime.now(),
                    decision_type="wait",
                    reasoning=reasoning,
                    confidence=1.0,
                ))

        if llm_invoked:
            logger.debug("BT tick invoked LLM fallback: %s", reasoning)
        else:
            logger.debug("BT tick resolved deterministically: %s", reasoning)

        return prediction

    # -------------------------------------------------------------------
    # BT construction & LLM-fallback adapter
    # -------------------------------------------------------------------

    def _build_bt_policy(self) -> None:
        """Compile the BT from the loaded task configs.

        Must be called after ``load_task`` so ``skills``, ``graph`` and
        ``self._task_profile`` are populated.
        """
        from aura.brain.bt_policy import BTContext, BTPolicy

        dag: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(self.task_graph_string)
            if isinstance(parsed, list):
                dag = [s for s in parsed if isinstance(s, dict)]
            elif isinstance(parsed, dict):
                dag = [s for s in parsed.get("steps", []) if isinstance(s, dict)]
        except Exception:
            dag = []

        ctx = BTContext(
            ssg=self.graph,
            skills=self.skills,
            task_profile=getattr(self, "_task_profile", {}) or {},
            dag=dag,
            explainer=self.explainer,
            decision_mode=self.config.decision_mode,
            proactive_threshold=self.config.proactive_threshold,
            defer_to_llm_when_idle=self.config.llm_fallback_on_idle,
            llm_fallback=self._bt_llm_fallback_hook,
        )
        self._bt_ctx = ctx
        self._bt_policy = BTPolicy(ctx, decision_mode=self.config.decision_mode)
        logger.info(
            "BT policy compiled (mode=%s, %d safety rules, %d timers, %d scheduled skills)",
            self.config.decision_mode,
            len(ctx.task_profile.get("safety_rules", []) or []),
            len(ctx.task_profile.get("timers", []) or []),
            sum(
                1 for s in self.skills.list_skills()
                if s.trigger_steps or s.trigger_after_steps
            ),
        )

    async def _bt_llm_fallback_hook(self, reason: str, ctx) -> Optional[ActionPrediction]:
        """Async hook invoked by the BT's LLM fallback leaf.

        Reuses the legacy LLM prompt verbatim; ``reason`` is prepended
        to the reasoning trail so the dashboard/A-Score pipeline can
        see why the LLM was called.
        """
        if not (self.config.enable_llm_reasoning and self.llm_client):
            return None

        available = self.reasoner.get_available_actions("robot")
        prediction = await self._llm_decide_action(
            available, ctx.current_time_sec, getattr(ctx, "current_frame", None)
        )
        if prediction is not None:
            prediction.reasoning = f"[llm_fallback:{reason}] {prediction.reasoning}"
        return prediction
    
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
                                  current_time_sec: float,
                                  current_frame: Optional[Any] = None) -> Optional[ActionPrediction]:
        """Use LLM to decide on action."""
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to rules")
            return self._rule_based_decide(available_actions, current_time_sec)

        # Build prompt
        scene_state = self.graph.get_state_summary_for_llm()
        skills_desc = self.skills.get_skills_for_llm()

        task_instruction = self.config.task_system_instruction
        
        INCLUDE_RECENT_ACTIONS = False
        recent_actions_str = f"\n## Recent Robot Actions\n{self._format_recent_decisions()}\n" if INCLUDE_RECENT_ACTIONS else ""

        prompt = f"""You are a proactive robot assistant helping a human with a task.
Your goal is to anticipate what the human needs and provide timely assistance.

## Task Instructions

{task_instruction if task_instruction else "No specific task instructions."}
## Task Graph Definition
```json
{self.task_graph_string}
```
{scene_state}

{skills_desc}

## Current Time
Task time: {current_time_sec:.1f} seconds
{recent_actions_str}
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

# Removed from Prompt:
# ## Available Actions Now
# {json.dumps(available_actions[:5], indent=2) if available_actions else "No immediately available actions."}

        generate_kwargs = {
            "prompt": prompt,
            "temperature": 0.3,
            "json_mode": True,
            "max_tokens": self.config.max_completion_tokens,
        }

        # Check if we should pass the captured frame and/or the anchor image
        task_profile = getattr(self, "_task_profile", {}) or {}
        images_to_pass = []
        if task_profile.get("workflow_config", {}).get("pass_captured_frame_to_vlm", True) and current_frame is not None:
            # We assume current_frame is an np.ndarray (OpenCV format) or PIL Image.
            # Convert to PIL Image for the llm_client
            try:
                import numpy as np
                from PIL import Image
                if isinstance(current_frame, np.ndarray):
                    # usually BGR from cv2, convert to RGB
                    import cv2
                    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    images_to_pass.append(pil_img)
                elif isinstance(current_frame, Image.Image):
                    images_to_pass.append(current_frame)
            except Exception as e:
                logger.warning(f"Engine: Failed to parse current_frame for LLM: {e}")

        # Check if anchor image should be passed
        anchor_cfg = task_profile.get("anchor_image", {})
        if anchor_cfg.get("enabled", False) and anchor_cfg.get("path"):
            try:
                from PIL import Image
                import os
                # Path relative to workspace or absolute
                anchor_path = anchor_cfg["path"]
                if os.path.exists(anchor_path):
                    anchor_img = Image.open(anchor_path).convert("RGB")
                    images_to_pass.append(anchor_img)
                else:
                    logger.warning(f"Engine: Anchor image {anchor_path} not found.")
            except Exception as e:
                logger.warning(f"Engine: Failed to load anchor image for LLM: {e}")

        if images_to_pass:
            generate_kwargs["images"] = images_to_pass

        try:
            t0 = time.monotonic()
            response_text = await asyncio.wait_for(
                asyncio.to_thread(self.llm_client.generate, **generate_kwargs),
                timeout=self.config.max_reasoning_time_sec,
            )
            generation_time = time.monotonic() - t0

            result = json.loads(response_text)
            if isinstance(result, list):
                result = result[0] if result else {}

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
                images=images_to_pass,
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
            return self._rule_based_decide(available_actions, current_time_sec)
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return self._rule_based_decide(available_actions, current_time_sec)

    def _rule_based_decide(self, available_actions: List[Dict],
                           current_time_sec: float) -> Optional[ActionPrediction]:
        """Simple rule-based decision making as fallback."""
        # Fire the first skill whose preconditions clear the threshold.
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

        # Load task profile (safety_rules, timers) so the BT can compile.
        profile_path = self.config_dir / "task_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    self._task_profile = json.load(f)
            except Exception as exc:
                logger.warning("Failed to parse task_profile.json: %s", exc)
                self._task_profile = {}
        else:
            self._task_profile = {}

        # Sync decision_mode from task_profile if the engine config
        # still has the default. Workflow-level plumbing may have
        # already set it — but this allows a task to override.
        wf = (self._task_profile.get("workflow_config") or {}) if isinstance(
            self._task_profile, dict
        ) else {}
        tp_mode = wf.get("decision_mode")
        if tp_mode in ("llm", "bt", "hybrid"):
            self.config.decision_mode = tp_mode

        # Compile the BT now that skills, SSG and task_profile are loaded.
        try:
            self._build_bt_policy()
        except Exception as exc:
            logger.exception("Failed to compile BT policy: %s", exc)
            self._bt_policy = None
    
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

"""Decision Explainer for generating human-readable explanations.

Provides utilities to explain why the Brain made specific decisions
by citing evidence from the Semantic Scene Graph.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from aura.core.scene_graph import (
    SemanticSceneGraph, SSGNode, SSGEdge,
    ObjectNode, AgentNode, RegionNode
)


logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Record of a decision made by the Brain.
    
    Attributes:
        timestamp: When the decision was made
        decision_type: Type of decision (action, wait, ask, etc.)
        action_id: Skill/action ID if applicable
        target: Target object/agent ID
        parameters: Action parameters
        reasoning: LLM reasoning text
        evidence: List of SSG edges supporting the decision
        confidence: Confidence in the decision (0-1)
        ground_truth_timing: Actual timing from Wizard-of-Oz data (if available)
        was_correct: Whether decision matched ground truth
    """
    timestamp: datetime
    decision_type: str
    action_id: Optional[str] = None
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)  # Edge descriptions
    confidence: float = 0.0
    ground_truth_timing: Optional[float] = None
    was_correct: Optional[bool] = None
    latency_error_sec: Optional[float] = None


class DecisionExplainer:
    """Generates explanations for Brain decisions.
    
    Uses the SSG state to construct human-readable explanations
    that cite specific graph edges as evidence.
    """
    
    def __init__(self, graph: SemanticSceneGraph):
        self.graph = graph
        self.decision_history: List[DecisionRecord] = []
    
    def record_decision(self, record: DecisionRecord) -> None:
        """Record a decision for history tracking."""
        self.decision_history.append(record)
        logger.debug(f"Recorded decision: {record.decision_type} -> {record.action_id}")
    
    def explain_action(self, action_id: str, target_id: str = None,
                       reasoning: str = "", additional_context: Dict = None) -> str:
        """Generate explanation for an action decision.
        
        Args:
            action_id: The skill/action being executed
            target_id: Target object ID
            reasoning: LLM reasoning (if available)
            additional_context: Extra context to include
        
        Returns:
            Human-readable explanation string
        """
        lines = []
        lines.append(f"## Decision: Execute `{action_id}`\n")
        
        # Target information
        if target_id:
            target = self.graph.get_node(target_id)
            if target:
                lines.append(f"**Target:** {target.name}")
                
                # Location
                location = self.graph.get_location(target_id)
                if location:
                    region = self.graph.get_node(location)
                    region_name = region.name if region else location
                    lines.append(f"**Location:** {region_name}")
                
                # State
                if isinstance(target, ObjectNode):
                    state = target.state.name if hasattr(target.state, 'name') else str(target.state)
                    lines.append(f"**State:** {state}")
        
        # Evidence from graph
        lines.append("\n### Evidence from Scene Graph\n")
        evidence = self._collect_evidence(target_id)
        for edge_str in evidence:
            lines.append(f"- {edge_str}")
        
        # Task state context
        task_state = self.graph.task_state
        if task_state:
            lines.append("\n### Relevant Task State\n")
            for key, value in task_state.items():
                lines.append(f"- `{key}`: {value}")
        
        # LLM reasoning
        if reasoning:
            lines.append(f"\n### LLM Reasoning\n{reasoning}")
        
        # Additional context
        if additional_context:
            lines.append("\n### Additional Context\n")
            for key, value in additional_context.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def _collect_evidence(self, target_id: str = None, max_edges: int = 10) -> List[str]:
        """Collect relevant edge descriptions as evidence."""
        evidence = []
        
        if target_id:
            # Edges involving target
            for edge in self.graph.get_incoming_edges(target_id):
                evidence.append(edge.to_explanation_string())
            for edge in self.graph.get_outgoing_edges(target_id):
                evidence.append(edge.to_explanation_string())
        
        # Agent targeting edges
        for agent in self.graph.get_agents():
            if agent.attention_target:
                edges = self.graph.get_edges(
                    source_id=agent.id,
                    relation="targets"
                )
                for edge in edges:
                    evidence.append(edge.to_explanation_string())
        
        return evidence[:max_edges]
    
    def explain_wait(self, reason: str, waiting_for: str = None) -> str:
        """Explain why the system decided to wait."""
        lines = ["## Decision: Wait\n"]
        lines.append(f"**Reason:** {reason}")
        
        if waiting_for:
            lines.append(f"**Waiting for:** {waiting_for}")
        
        # Current state
        lines.append("\n### Current State\n")
        for agent in self.graph.get_agents():
            state = agent.state.name if hasattr(agent.state, 'name') else str(agent.state)
            lines.append(f"- {agent.name}: {state}")
        
        return "\n".join(lines)
    
    def explain_proactive_opportunity(self, opportunity: Dict[str, Any]) -> str:
        """Explain a proactive assistance opportunity."""
        lines = []
        lines.append(f"## Proactive Opportunity: {opportunity.get('type', 'unknown')}\n")
        lines.append(f"**Suggested Action:** `{opportunity.get('action_id', 'N/A')}`")
        lines.append(f"**Target:** {opportunity.get('target', 'N/A')}")
        lines.append(f"**Priority:** {opportunity.get('priority', 0):.2f}")
        lines.append(f"\n**Reasoning:** {opportunity.get('reasoning', 'N/A')}")
        
        return "\n".join(lines)
    
    def generate_decision_report(self, start_time: datetime = None,
                                  end_time: datetime = None) -> str:
        """Generate a summary report of decisions in a time window."""
        decisions = self.decision_history
        
        if start_time:
            decisions = [d for d in decisions if d.timestamp >= start_time]
        if end_time:
            decisions = [d for d in decisions if d.timestamp <= end_time]
        
        if not decisions:
            return "No decisions recorded in the specified time window."
        
        lines = ["# Decision Report\n"]
        lines.append(f"**Total Decisions:** {len(decisions)}")
        
        # Count by type
        type_counts = {}
        for d in decisions:
            type_counts[d.decision_type] = type_counts.get(d.decision_type, 0) + 1
        
        lines.append("\n## Decision Types\n")
        for dtype, count in sorted(type_counts.items()):
            lines.append(f"- {dtype}: {count}")
        
        # Accuracy if ground truth available
        with_gt = [d for d in decisions if d.was_correct is not None]
        if with_gt:
            correct = sum(1 for d in with_gt if d.was_correct)
            accuracy = correct / len(with_gt) * 100
            lines.append(f"\n## Accuracy (vs Ground Truth)\n")
            lines.append(f"- Correct: {correct}/{len(with_gt)} ({accuracy:.1f}%)")
            
            # Timing errors
            timing_errors = [d.latency_error_sec for d in with_gt 
                           if d.latency_error_sec is not None]
            if timing_errors:
                avg_error = sum(abs(e) for e in timing_errors) / len(timing_errors)
                lines.append(f"- Average Timing Error: {avg_error:.2f}s")
        
        # Recent decisions
        lines.append("\n## Recent Decisions\n")
        for d in decisions[-10:]:
            timestamp = d.timestamp.strftime("%H:%M:%S")
            lines.append(f"- [{timestamp}] {d.decision_type}: {d.action_id or 'N/A'} "
                        f"(conf: {d.confidence:.2f})")
        
        return "\n".join(lines)
    
    def compare_to_ground_truth(self, predicted_time: float, 
                                 ground_truth_time: float,
                                 action_id: str,
                                 tolerance_sec: float = 2.0) -> Dict[str, Any]:
        """Compare a prediction to ground truth.
        
        Returns:
            Dict with comparison metrics
        """
        error = predicted_time - ground_truth_time
        is_correct = abs(error) <= tolerance_sec
        
        return {
            "predicted_time": predicted_time,
            "ground_truth_time": ground_truth_time,
            "error_sec": error,
            "abs_error_sec": abs(error),
            "is_within_tolerance": is_correct,
            "tolerance_sec": tolerance_sec,
            "action_id": action_id,
            "assessment": "CORRECT" if is_correct else ("EARLY" if error < 0 else "LATE"),
        }

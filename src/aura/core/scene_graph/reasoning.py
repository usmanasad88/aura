"""Graph reasoning utilities for Semantic Scene Graph.

Provides query builders and reasoning helpers for extracting
insights from the scene graph.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

from .nodes import SSGNode, ObjectNode, AgentNode, RegionNode, NodeType
from .edges import SSGEdge, EdgeType, SpatialRelation, SemanticRelation


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a graph query."""
    nodes: List[SSGNode] = field(default_factory=list)
    edges: List[SSGEdge] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class GraphQuery:
    """Builder for graph queries.
    
    Example usage:
        query = GraphQuery(graph)
        result = (query
            .find_objects()
            .with_state("AVAILABLE")
            .at_region("working_area")
            .with_affordance("pickable")
            .execute())
    """
    
    def __init__(self, graph: "SemanticSceneGraph"):
        self.graph = graph
        self._node_filters: List[Callable[[SSGNode], bool]] = []
        self._edge_filters: List[Callable[[SSGEdge], bool]] = []
        self._node_type: Optional[NodeType] = None
        self._limit: Optional[int] = None
    
    def find_objects(self) -> "GraphQuery":
        """Filter to object nodes."""
        self._node_type = NodeType.OBJECT
        return self
    
    def find_agents(self) -> "GraphQuery":
        """Filter to agent nodes."""
        self._node_type = NodeType.AGENT
        return self
    
    def find_regions(self) -> "GraphQuery":
        """Filter to region nodes."""
        self._node_type = NodeType.REGION
        return self
    
    def with_state(self, state: str) -> "GraphQuery":
        """Filter nodes by state attribute."""
        def filter_fn(node: SSGNode) -> bool:
            node_state = getattr(node, 'state', None)
            if node_state:
                state_name = node_state.name if hasattr(node_state, 'name') else str(node_state)
                return state_name == state
            return node.attributes.get('state') == state
        self._node_filters.append(filter_fn)
        return self
    
    def with_attribute(self, key: str, value: Any) -> "GraphQuery":
        """Filter nodes by attribute value."""
        def filter_fn(node: SSGNode) -> bool:
            return node.attributes.get(key) == value
        self._node_filters.append(filter_fn)
        return self
    
    def with_affordance(self, action_id: str) -> "GraphQuery":
        """Filter nodes that have a specific affordance."""
        def filter_fn(node: SSGNode) -> bool:
            return node.has_affordance(action_id)
        self._node_filters.append(filter_fn)
        return self
    
    def at_region(self, region_id: str) -> "GraphQuery":
        """Filter objects at a specific region."""
        def filter_fn(node: SSGNode) -> bool:
            location = self.graph.get_location(node.id)
            return location == region_id
        self._node_filters.append(filter_fn)
        return self
    
    def with_category(self, category: str) -> "GraphQuery":
        """Filter objects by category."""
        def filter_fn(node: SSGNode) -> bool:
            return isinstance(node, ObjectNode) and node.category == category
        self._node_filters.append(filter_fn)
        return self
    
    def agent_type(self, agent_type: str) -> "GraphQuery":
        """Filter agents by type (human/robot)."""
        def filter_fn(node: SSGNode) -> bool:
            return isinstance(node, AgentNode) and node.agent_type == agent_type
        self._node_filters.append(filter_fn)
        return self
    
    def can_perform(self, action_id: str) -> "GraphQuery":
        """Filter agents that can perform a specific action."""
        def filter_fn(node: SSGNode) -> bool:
            return isinstance(node, AgentNode) and node.can_perform(action_id)
        self._node_filters.append(filter_fn)
        return self
    
    def where(self, predicate: Callable[[SSGNode], bool]) -> "GraphQuery":
        """Add custom filter predicate."""
        self._node_filters.append(predicate)
        return self
    
    def limit(self, n: int) -> "GraphQuery":
        """Limit number of results."""
        self._limit = n
        return self
    
    def execute(self) -> QueryResult:
        """Execute the query and return results."""
        # Get initial node set
        if self._node_type:
            nodes = self.graph.get_nodes_by_type(self._node_type)
        else:
            nodes = list(self.graph.nodes.values())
        
        # Apply filters
        for filter_fn in self._node_filters:
            nodes = [n for n in nodes if filter_fn(n)]
        
        # Apply limit
        if self._limit:
            nodes = nodes[:self._limit]
        
        return QueryResult(nodes=nodes)


def _eval_ssg_precondition(
    key: str, expected: Any, graph: "SemanticSceneGraph"
) -> Tuple[bool, Any]:
    """Evaluate one precondition clause against the SSG.

    Keys are ``"<node_id>.<attr>"`` or a bare ``"<task_state_key>"``.
    ``.location`` is resolved via ``graph.get_location``; any other
    attr falls back to ``task_state["<node>_<attr>"]`` then
    ``task_state["<key>"]``. Returns ``(matches, actual_value)``.
    """
    if "." in key:
        node_id, attr = key.split(".", 1)
    else:
        node_id, attr = key, ""

    actual: Any = None
    if attr == "location":
        try:
            actual = graph.get_location(node_id)
        except Exception:
            actual = None
        if actual is None:
            actual = graph.task_state.get(
                f"{node_id}_location", graph.task_state.get(key)
            )
    else:
        actual = graph.task_state.get(key)
        if actual is None and attr:
            actual = graph.task_state.get(f"{node_id}_{attr}")
        if actual is None and not attr:
            actual = graph.task_state.get(node_id)

    return actual == expected, actual


class GraphReasoner:
    """Reasoning utilities for the scene graph.

    Provides methods to extract actionable insights from the graph
    state, supporting the decision engine. When constructed with a
    ``SkillRegistry``, ``get_available_actions`` consults skills and
    their preconditions from the task config rather than affordances
    on object nodes.
    """

    def __init__(
        self,
        graph: "SemanticSceneGraph",
        skills: Optional[Any] = None,
    ):
        self.graph = graph
        self.skills = skills

    def get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all actions available to an agent given current state.

        Walks the ``SkillRegistry``, filters by the agent's
        ``capabilities``, and checks each skill's ``preconditions``
        against the live SSG / ``task_state``. Returns list of dicts:
            - action_id: skill id
            - action_name: skill human-readable name
            - target_object: best-effort id parsed from the first
              dotted ``effects`` key (may be ``None``)
            - feasibility: 1.0 if preconditions hold, else 0.0
            - reasoning: human-readable summary of the check
            - effects: raw effects dict from the skill
        """
        agent = self.graph.get_node(agent_id)
        if not isinstance(agent, AgentNode):
            return []

        if self.skills is None:
            return []

        capabilities = set(getattr(agent, "capabilities", []) or [])
        available: List[Dict[str, Any]] = []

        for skill in self.skills.list_skills():
            if capabilities and skill.id not in capabilities:
                continue

            preconditions_met, why = self._check_preconditions(
                skill.preconditions or {}
            )
            if not preconditions_met:
                continue

            target_id: Optional[str] = None
            for effect_key in (skill.effects or {}).keys():
                if "." in effect_key:
                    target_id = effect_key.split(".", 1)[0]
                    break

            target_name: Optional[str] = None
            if target_id is not None:
                target_node = self.graph.get_node(target_id)
                if target_node is not None:
                    target_name = target_node.name

            available.append({
                "action_id": skill.id,
                "action_name": skill.name,
                "target_object": target_id,
                "target_name": target_name,
                "feasibility": 1.0,
                "reasoning": why or "preconditions met",
                "effects": dict(skill.effects or {}),
            })

        return sorted(available, key=lambda x: x["feasibility"], reverse=True)

    def _check_preconditions(
        self, preconditions: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Evaluate skill preconditions against the SSG.

        Returns ``(all_match, summary)``.
        """
        if not preconditions:
            return True, "no preconditions"

        matched: List[str] = []
        for key, expected in preconditions.items():
            ok, actual = _eval_ssg_precondition(key, expected, self.graph)
            if not ok:
                return False, f"{key}={actual!r}≠{expected!r}"
            matched.append(f"{key}={actual!r}")
        return True, "; ".join(matched)
    
    def get_blocking_objects(self, target_id: str) -> List[str]:
        """Get all objects blocking access to target."""
        blockers = []
        for edge in self.graph.get_edges(target_id=target_id, 
                                         relation=SemanticRelation.BLOCKS.value):
            blockers.append(edge.source_id)
        return blockers
    
    def find_path_to_goal(self, current_state: Dict[str, Any], 
                          goal_state: Dict[str, Any]) -> List[str]:
        """Find sequence of actions to reach goal state.
        
        Simple BFS planner. Returns list of action_ids.
        """
        # This is a simplified planner - in practice would use more
        # sophisticated planning algorithms
        missing = []
        for key, expected in goal_state.items():
            current = current_state.get(key, self.graph.get_task_state(key))
            if current != expected:
                missing.append((key, expected))
        
        # For now, just return what's missing (not a real plan)
        return [f"achieve_{k}={v}" for k, v in missing]
    
    def explain_decision(self, action: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> str:
        """Generate explanation for why an action was selected.
        
        Args:
            action: The selected action dict
            context: Additional context for explanation
        
        Returns:
            Human-readable explanation citing graph edges
        """
        lines = [f"Decision: Execute '{action.get('action_name', action['action_id'])}'"]
        
        target = action.get("target_object")
        if target:
            obj = self.graph.get_node(target)
            if obj:
                lines.append(f"\nTarget: {obj.name}")
                location = self.graph.get_location(target)
                if location:
                    lines.append(f"  Location: {location}")
                    # Cite the edge
                    edges = self.graph.get_edges(source_id=target, target_id=location)
                    if edges:
                        lines.append(f"  Evidence: {edges[0].to_explanation_string()}")
        
        lines.append(f"\nReasoning: {action.get('reasoning', 'No specific reasoning')}")
        
        # Add relevant edges
        if target:
            lines.append("\nRelevant relationships:")
            for edge in self.graph.get_incoming_edges(target)[:5]:
                lines.append(f"  - {edge.to_explanation_string()}")
        
        # Add effects
        effects = action.get("effects", {})
        if effects:
            lines.append("\nExpected effects:")
            for key, value in effects.items():
                lines.append(f"  - {key} → {value}")
        
        return "\n".join(lines)

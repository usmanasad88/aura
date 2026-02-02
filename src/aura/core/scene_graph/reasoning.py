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


class GraphReasoner:
    """Reasoning utilities for the scene graph.
    
    Provides methods to extract actionable insights from the graph
    state, supporting the decision engine.
    """
    
    def __init__(self, graph: "SemanticSceneGraph"):
        self.graph = graph
    
    def get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all actions available to an agent given current state.
        
        Returns list of dicts with:
            - action_id: Action identifier
            - target_object: Object to act on (if applicable)
            - feasibility: How feasible the action is (0-1)
            - reasoning: Why this action is available
        """
        agent = self.graph.get_node(agent_id)
        if not isinstance(agent, AgentNode):
            return []
        
        available = []
        
        # Check each capability against objects with matching affordances
        for capability in agent.capabilities:
            for obj in self.graph.get_objects():
                aff = obj.get_affordance(capability)
                if not aff:
                    continue
                
                # Check preconditions
                preconditions_met, reasoning = self._check_preconditions(
                    aff.preconditions, obj
                )
                
                if preconditions_met:
                    # Check if object is blocked
                    is_blocked, blocker = self.graph.is_blocked(obj.id)
                    if is_blocked:
                        feasibility = 0.2
                        reasoning += f" (blocked by {blocker})"
                    else:
                        feasibility = aff.feasibility
                    
                    available.append({
                        "action_id": capability,
                        "action_name": aff.name,
                        "target_object": obj.id,
                        "target_name": obj.name,
                        "feasibility": feasibility,
                        "reasoning": reasoning,
                        "effects": aff.effects,
                    })
        
        return sorted(available, key=lambda x: x["feasibility"], reverse=True)
    
    def _check_preconditions(self, preconditions: Dict[str, Any], 
                             obj: ObjectNode) -> Tuple[bool, str]:
        """Check if preconditions are met for an action.
        
        Returns (are_met, reasoning).
        """
        if not preconditions:
            return True, "No preconditions required"
        
        reasons = []
        for key, expected in preconditions.items():
            # Check object attributes
            if key in obj.attributes:
                actual = obj.attributes[key]
                if actual != expected:
                    return False, f"Precondition failed: {key}={actual}, expected {expected}"
                reasons.append(f"{key}={actual}")
            
            # Check object state
            elif key == "state":
                actual = obj.state.name if hasattr(obj.state, 'name') else str(obj.state)
                if actual != expected:
                    return False, f"Object state is {actual}, expected {expected}"
                reasons.append(f"state={actual}")
            
            # Check task state
            elif key.startswith("task."):
                task_key = key[5:]
                actual = self.graph.get_task_state(task_key)
                if actual != expected:
                    return False, f"Task state {task_key}={actual}, expected {expected}"
                reasons.append(f"{task_key}={actual}")
            
            # Check location
            elif key == "at_region":
                location = self.graph.get_location(obj.id)
                if location != expected:
                    return False, f"Object at {location}, expected at {expected}"
                reasons.append(f"at {location}")
        
        return True, "Preconditions met: " + ", ".join(reasons)
    
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
    
    def get_proactive_opportunities(self, robot_id: str = "robot") -> List[Dict[str, Any]]:
        """Identify opportunities for proactive robot assistance.
        
        Looks for:
        - Objects human is targeting that robot can prepare
        - Missing prerequisites for predicted human actions
        - Objects that should be retrieved/returned
        
        Returns list of opportunity dicts.
        """
        opportunities = []
        robot = self.graph.get_node(robot_id)
        if not isinstance(robot, AgentNode):
            return opportunities
        
        human = None
        for agent in self.graph.get_agents():
            if agent.agent_type == "human":
                human = agent
                break
        
        if not human:
            return opportunities
        
        # Check what human is targeting
        human_target = self.graph.get_agent_target(human.id)
        if human_target:
            target_obj = self.graph.get_node(human_target)
            if target_obj:
                # Can robot prepare this object?
                for aff in target_obj.affordances:
                    if robot.can_perform(aff.action_id):
                        opportunities.append({
                            "type": "prepare_for_human",
                            "action_id": aff.action_id,
                            "target": human_target,
                            "reasoning": f"Human targeting {target_obj.name}, robot can {aff.name}",
                            "priority": 0.8,
                        })
        
        # Check predicted actions for missing prerequisites
        for obj in self.graph.get_objects():
            for pred in obj.predicted_actions:
                if pred.agent_id == human.id:
                    # Human predicted to use this object
                    location = self.graph.get_location(obj.id)
                    if location and location != "working_area":
                        # Object not in working area
                        if robot.can_perform("retrieve_object"):
                            opportunities.append({
                                "type": "retrieve_needed_object",
                                "action_id": "retrieve_object",
                                "target": obj.id,
                                "reasoning": f"{obj.name} predicted to be needed, currently at {location}",
                                "priority": 0.7,
                            })
        
        return sorted(opportunities, key=lambda x: x["priority"], reverse=True)

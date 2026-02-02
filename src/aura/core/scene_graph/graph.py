"""Semantic Scene Graph main class.

The SemanticSceneGraph is the central "Shared Truth" that all monitors
read from and write to. It maintains nodes (objects, agents, regions)
and edges (spatial and semantic relationships).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set, Tuple, Iterator
from pathlib import Path

from .nodes import (
    NodeType, SSGNode, ObjectNode, AgentNode, RegionNode,
    ObjectState, AgentState, Affordance, PredictedAction
)
from .edges import (
    EdgeType, SpatialRelation, SemanticRelation, SSGEdge
)


logger = logging.getLogger(__name__)


@dataclass
class GraphSnapshot:
    """A snapshot of the graph state at a point in time."""
    timestamp: datetime
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]
    task_state: Dict[str, Any]


class SemanticSceneGraph:
    """Dynamic Spatio-Temporal Scene Graph.
    
    The central data structure for AURA that represents:
    - All entities (objects, agents, regions) as nodes
    - All relationships (spatial, semantic) as edges
    - State variables for task tracking
    
    This graph is updated by monitors and queried by the decision engine.
    All decisions can be explained by citing specific edges.
    """
    
    def __init__(self, name: str = "default"):
        """Initialize empty scene graph.
        
        Args:
            name: Name identifier for this graph
        """
        self.name = name
        self._nodes: Dict[str, SSGNode] = {}
        self._edges: List[SSGEdge] = []
        self._task_state: Dict[str, Any] = {}
        self._history: List[GraphSnapshot] = []
        self._max_history = 100
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def add_node(self, node: SSGNode) -> None:
        """Add or update a node in the graph."""
        self._nodes[node.id] = node
        self.last_updated = datetime.now()
        logger.debug(f"Added/updated node: {node.id} ({node.node_type.name})")
    
    def get_node(self, node_id: str) -> Optional[SSGNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node and all its edges."""
        if node_id not in self._nodes:
            return False
        
        del self._nodes[node_id]
        # Remove all edges involving this node
        self._edges = [e for e in self._edges 
                       if e.source_id != node_id and e.target_id != node_id]
        self.last_updated = datetime.now()
        logger.debug(f"Removed node: {node_id}")
        return True
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[SSGNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]
    
    def get_objects(self) -> List[ObjectNode]:
        """Get all object nodes."""
        return [n for n in self._nodes.values() if isinstance(n, ObjectNode)]
    
    def get_agents(self) -> List[AgentNode]:
        """Get all agent nodes."""
        return [n for n in self._nodes.values() if isinstance(n, AgentNode)]
    
    def get_regions(self) -> List[RegionNode]:
        """Get all region nodes."""
        return [n for n in self._nodes.values() if isinstance(n, RegionNode)]
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes
    
    @property
    def nodes(self) -> Dict[str, SSGNode]:
        """Get all nodes."""
        return self._nodes
    
    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    def add_edge(self, edge: SSGEdge) -> None:
        """Add or update an edge in the graph.
        
        If an edge with the same source, target, and relation exists,
        it will be updated.
        """
        # Remove existing edge with same signature
        self._edges = [e for e in self._edges if e.edge_id != edge.edge_id]
        self._edges.append(edge)
        self.last_updated = datetime.now()
        logger.debug(f"Added/updated edge: {edge.edge_id}")
    
    def remove_edge(self, source_id: str, target_id: str, relation: str = None) -> int:
        """Remove edges matching criteria. Returns count of removed edges."""
        original_count = len(self._edges)
        if relation:
            self._edges = [e for e in self._edges 
                          if not (e.source_id == source_id and 
                                  e.target_id == target_id and
                                  e.relation == relation)]
        else:
            self._edges = [e for e in self._edges 
                          if not (e.source_id == source_id and 
                                  e.target_id == target_id)]
        
        removed = original_count - len(self._edges)
        if removed > 0:
            self.last_updated = datetime.now()
            logger.debug(f"Removed {removed} edge(s)")
        return removed
    
    def get_edges(self, source_id: str = None, target_id: str = None,
                  relation: str = None, edge_type: EdgeType = None) -> List[SSGEdge]:
        """Query edges by criteria."""
        return [e for e in self._edges 
                if e.matches(source_id, target_id, relation, edge_type)]
    
    def get_outgoing_edges(self, node_id: str) -> List[SSGEdge]:
        """Get all edges originating from a node."""
        return [e for e in self._edges if e.source_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[SSGEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self._edges if e.target_id == node_id]
    
    def get_neighbors(self, node_id: str, relation: str = None) -> List[str]:
        """Get IDs of nodes connected to given node."""
        neighbors = set()
        for edge in self._edges:
            if relation and edge.relation != relation:
                continue
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)
        return list(neighbors)
    
    def has_edge(self, source_id: str, target_id: str, relation: str = None) -> bool:
        """Check if edge exists."""
        return len(self.get_edges(source_id, target_id, relation)) > 0
    
    @property
    def edges(self) -> List[SSGEdge]:
        """Get all edges."""
        return self._edges
    
    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self._edges)
    
    # =========================================================================
    # Spatial Convenience Methods
    # =========================================================================
    
    def set_location(self, object_id: str, region_id: str, 
                     relation: SpatialRelation = SpatialRelation.AT) -> None:
        """Set object location to a region."""
        # Remove old location edges
        for edge in self.get_edges(source_id=object_id, edge_type=EdgeType.SPATIAL):
            if edge.relation in [SpatialRelation.AT.value, SpatialRelation.ON.value,
                                 SpatialRelation.INSIDE.value]:
                self.remove_edge(edge.source_id, edge.target_id, edge.relation)
        
        # Add new location edge
        self.add_edge(SSGEdge.spatial(object_id, region_id, relation))
        
        # Update region's contained objects
        region = self.get_node(region_id)
        if isinstance(region, RegionNode):
            region.add_object(object_id)
    
    def get_location(self, object_id: str) -> Optional[str]:
        """Get the region ID where an object is located."""
        for edge in self.get_edges(source_id=object_id, edge_type=EdgeType.SPATIAL):
            if edge.relation in [SpatialRelation.AT.value, SpatialRelation.ON.value,
                                 SpatialRelation.INSIDE.value]:
                return edge.target_id
        return None
    
    def get_objects_in_region(self, region_id: str) -> List[str]:
        """Get all object IDs in a region."""
        region = self.get_node(region_id)
        if isinstance(region, RegionNode):
            return region.contained_objects.copy()
        return []
    
    # =========================================================================
    # Semantic Convenience Methods
    # =========================================================================
    
    def set_agent_target(self, agent_id: str, target_id: str, 
                         confidence: float = 1.0, reasoning: str = "") -> None:
        """Set what an agent is targeting/intending to interact with."""
        # Remove old target edges
        for edge in self.get_edges(source_id=agent_id):
            if edge.relation == SemanticRelation.TARGETS.value:
                self.remove_edge(edge.source_id, edge.target_id, edge.relation)
        
        # Add new target edge
        self.add_edge(SSGEdge.semantic(
            agent_id, target_id, SemanticRelation.TARGETS,
            confidence=confidence, 
            is_predicted=True,
            reasoning=reasoning
        ))
        
        # Update agent's attention target
        agent = self.get_node(agent_id)
        if isinstance(agent, AgentNode):
            agent.attention_target = target_id
    
    def get_agent_target(self, agent_id: str) -> Optional[str]:
        """Get what an agent is currently targeting."""
        edges = self.get_edges(source_id=agent_id, 
                               relation=SemanticRelation.TARGETS.value)
        if edges:
            return edges[0].target_id
        return None
    
    def set_blocks(self, blocker_id: str, blocked_id: str, 
                   reasoning: str = "") -> None:
        """Mark that one object blocks access to another."""
        self.add_edge(SSGEdge.semantic(
            blocker_id, blocked_id, SemanticRelation.BLOCKS,
            reasoning=reasoning
        ))
    
    def is_blocked(self, object_id: str) -> Tuple[bool, Optional[str]]:
        """Check if object is blocked. Returns (is_blocked, blocker_id)."""
        edges = self.get_edges(target_id=object_id, 
                               relation=SemanticRelation.BLOCKS.value)
        if edges:
            return True, edges[0].source_id
        return False, None
    
    # =========================================================================
    # Task State Management
    # =========================================================================
    
    def set_task_state(self, key: str, value: Any) -> None:
        """Set a task state variable."""
        self._task_state[key] = value
        self.last_updated = datetime.now()
    
    def get_task_state(self, key: str, default: Any = None) -> Any:
        """Get a task state variable."""
        return self._task_state.get(key, default)
    
    def update_task_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple task state variables."""
        self._task_state.update(updates)
        self.last_updated = datetime.now()
    
    @property
    def task_state(self) -> Dict[str, Any]:
        """Get full task state."""
        return self._task_state.copy()
    
    def initialize_task_state(self, schema: Dict[str, Any]) -> None:
        """Initialize task state from a schema definition.
        
        Args:
            schema: Dict with 'state_variables' containing variable definitions
        """
        state_vars = schema.get("state_variables", schema)
        for var_name, var_def in state_vars.items():
            if isinstance(var_def, dict):
                default = var_def.get("default")
            else:
                default = var_def
            self._task_state[var_name] = default
        self.last_updated = datetime.now()
        logger.info(f"Initialized task state with {len(self._task_state)} variables")
    
    # =========================================================================
    # Snapshot and History
    # =========================================================================
    
    def take_snapshot(self) -> GraphSnapshot:
        """Create a snapshot of current graph state."""
        snapshot = GraphSnapshot(
            timestamp=datetime.now(),
            nodes={nid: node.to_dict() for nid, node in self._nodes.items()},
            edges=[e.to_dict() for e in self._edges],
            task_state=self._task_state.copy()
        )
        
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return snapshot
    
    def get_history(self, count: int = None) -> List[GraphSnapshot]:
        """Get recent snapshots."""
        if count:
            return self._history[-count:]
        return self._history.copy()
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "task_state": self._task_state,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved graph to {path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticSceneGraph":
        """Deserialize graph from dictionary."""
        graph = cls(name=data.get("name", "loaded"))
        graph.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        graph.last_updated = datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now()
        
        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = SSGNode.from_dict(node_data)
            graph._nodes[node_id] = node
        
        # Load edges
        for edge_data in data.get("edges", []):
            edge = SSGEdge.from_dict(edge_data)
            graph._edges.append(edge)
        
        # Load task state
        graph._task_state = data.get("task_state", {})
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> "SemanticSceneGraph":
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: str) -> "SemanticSceneGraph":
        """Load graph from JSON file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    # =========================================================================
    # Query and Summary
    # =========================================================================
    
    def summary(self) -> str:
        """Generate a human-readable summary of the graph."""
        lines = [
            f"SemanticSceneGraph: {self.name}",
            f"  Nodes: {self.node_count}",
            f"    Objects: {len(self.get_objects())}",
            f"    Agents: {len(self.get_agents())}",
            f"    Regions: {len(self.get_regions())}",
            f"  Edges: {self.edge_count}",
            f"  Task State Variables: {len(self._task_state)}",
        ]
        return "\n".join(lines)
    
    def get_state_summary_for_llm(self) -> str:
        """Generate a summary suitable for LLM context.
        
        Returns a structured text representation of the graph state
        that can be included in LLM prompts for decision making.
        """
        lines = ["## Current Scene State\n"]
        
        # Agents
        agents = self.get_agents()
        if agents:
            lines.append("### Agents")
            for agent in agents:
                state = agent.state.name if hasattr(agent.state, 'name') else agent.state
                lines.append(f"- **{agent.name}** ({agent.agent_type}): {state}")
                if agent.attention_target:
                    lines.append(f"  - Attending to: {agent.attention_target}")
                if agent.capabilities:
                    lines.append(f"  - Capabilities: {', '.join(agent.capabilities)}")
        
        # Regions
        regions = self.get_regions()
        if regions:
            lines.append("\n### Regions")
            for region in regions:
                lines.append(f"- **{region.name}** ({region.region_type})")
                if region.contained_objects:
                    lines.append(f"  - Contains: {', '.join(region.contained_objects)}")
        
        # Objects
        objects = self.get_objects()
        if objects:
            lines.append("\n### Objects")
            for obj in objects:
                state = obj.state.name if hasattr(obj.state, 'name') else obj.state
                loc = self.get_location(obj.id) or "unknown"
                lines.append(f"- **{obj.name}**: {state}, at {loc}")
                if obj.affordances:
                    aff_names = [a.name for a in obj.affordances]
                    lines.append(f"  - Affordances: {', '.join(aff_names)}")
        
        # Key Relationships
        lines.append("\n### Key Relationships")
        for edge in self._edges[:20]:  # Limit for context length
            lines.append(f"- {edge.to_explanation_string()}")
        
        # Task State
        if self._task_state:
            lines.append("\n### Task State")
            for key, value in self._task_state.items():
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"SemanticSceneGraph(name={self.name}, nodes={self.node_count}, edges={self.edge_count})"

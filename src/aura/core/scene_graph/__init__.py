"""Semantic Scene Graph (SSG) module for AURA framework.

Provides a dynamic spatio-temporal scene graph that serves as the 
"Shared Truth" for all monitors and the decision engine.

Nodes: Objects, Agents (Human, Robot), Regions (Table, Bin, etc.)
Edges: Spatial relations (On, Near, Inside) and Semantic relations (OwnedBy, TargetedBy)
Attributes: State, Affordances, PredictedAction
"""

from .nodes import (
    NodeType,
    SSGNode,
    ObjectNode,
    AgentNode,
    RegionNode,
)

from .edges import (
    EdgeType,
    SpatialRelation,
    SemanticRelation,
    SSGEdge,
)

from .graph import SemanticSceneGraph

from .reasoning import (
    GraphQuery,
    GraphReasoner,
)

__all__ = [
    # Node types
    "NodeType",
    "SSGNode",
    "ObjectNode",
    "AgentNode",
    "RegionNode",
    # Edge types
    "EdgeType",
    "SpatialRelation",
    "SemanticRelation",
    "SSGEdge",
    # Graph
    "SemanticSceneGraph",
    # Reasoning
    "GraphQuery",
    "GraphReasoner",
]

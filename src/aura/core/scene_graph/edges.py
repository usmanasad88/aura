"""Edge types for Semantic Scene Graph.

Edges represent relationships between nodes:
- Spatial relations: On, Near, Inside, Above, Below, etc.
- Semantic relations: OwnedBy, TargetedBy, Blocks, Requires, etc.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple


class EdgeType(Enum):
    """High-level edge types."""
    SPATIAL = auto()     # Physical/spatial relationship
    SEMANTIC = auto()    # Semantic/functional relationship


class SpatialRelation(Enum):
    """Spatial relationships between nodes."""
    ON = "on"                    # Object is on top of another
    INSIDE = "inside"            # Object is inside another (container)
    NEAR = "near"                # Objects are close to each other
    ABOVE = "above"              # Object is above another (not touching)
    BELOW = "below"              # Object is below another
    LEFT_OF = "left_of"          # Object is to the left
    RIGHT_OF = "right_of"        # Object is to the right
    IN_FRONT_OF = "in_front_of"  # Object is in front
    BEHIND = "behind"            # Object is behind
    AT = "at"                    # Object is at a region/location
    ADJACENT = "adjacent"        # Objects are adjacent
    CONTAINS = "contains"        # Region contains object (inverse of AT/INSIDE)


class SemanticRelation(Enum):
    """Semantic/functional relationships between nodes."""
    # Agent-Object relations
    TARGETS = "targets"          # Agent intends to interact with object
    HOLDS = "holds"              # Agent is holding object
    USES = "uses"                # Agent is using object
    REACHES_FOR = "reaches_for"  # Agent is reaching for object
    
    # Object-Object relations
    BLOCKS = "blocks"            # Object blocks access to another
    REQUIRES = "requires"        # Object/action requires another
    PART_OF = "part_of"          # Object is part of another
    RELATED_TO = "related_to"    # Generic semantic relation
    
    # Task/State relations
    NEEDED_FOR = "needed_for"    # Object needed for task/action
    PRODUCED_BY = "produced_by"  # Object produced by action
    MODIFIED_BY = "modified_by"  # Object modified by action
    
    # Agent relations
    OWNED_BY = "owned_by"        # Object owned/controlled by agent
    ASSIGNED_TO = "assigned_to"  # Task assigned to agent
    ATTENDS_TO = "attends_to"    # Agent attending to object


@dataclass
class SSGEdge:
    """Edge connecting two nodes in the scene graph.
    
    Attributes:
        source_id: ID of source node
        target_id: ID of target node
        edge_type: SPATIAL or SEMANTIC
        relation: Specific relation (SpatialRelation or SemanticRelation)
        confidence: Confidence in this relationship (0-1)
        attributes: Additional edge attributes
        last_updated: When this edge was last updated
        is_predicted: Whether this is a predicted (not observed) relation
        reasoning: Why this edge was created/inferred
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    relation: str  # SpatialRelation or SemanticRelation value
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    is_predicted: bool = False
    reasoning: str = ""
    
    @property
    def edge_id(self) -> str:
        """Unique identifier for this edge."""
        return f"{self.source_id}--{self.relation}-->{self.target_id}"
    
    def matches(self, source: str = None, target: str = None, 
                relation: str = None, edge_type: EdgeType = None) -> bool:
        """Check if edge matches given criteria."""
        if source and self.source_id != source:
            return False
        if target and self.target_id != target:
            return False
        if relation and self.relation != relation:
            return False
        if edge_type and self.edge_type != edge_type:
            return False
        return True
    
    def to_explanation_string(self) -> str:
        """Generate human-readable explanation of this edge."""
        pred_marker = " (predicted)" if self.is_predicted else ""
        return f"[{self.source_id}] --{self.relation}--> [{self.target_id}]{pred_marker}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.name,
            "relation": self.relation,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "last_updated": self.last_updated.isoformat(),
            "is_predicted": self.is_predicted,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSGEdge":
        """Deserialize edge from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType[data["edge_type"]],
            relation=data["relation"],
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            is_predicted=data.get("is_predicted", False),
            reasoning=data.get("reasoning", ""),
        )
    
    @classmethod
    def spatial(cls, source_id: str, target_id: str, 
                relation: SpatialRelation, confidence: float = 1.0,
                **kwargs) -> "SSGEdge":
        """Create a spatial edge."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.SPATIAL,
            relation=relation.value,
            confidence=confidence,
            **kwargs
        )
    
    @classmethod
    def semantic(cls, source_id: str, target_id: str,
                 relation: SemanticRelation, confidence: float = 1.0,
                 **kwargs) -> "SSGEdge":
        """Create a semantic edge."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.SEMANTIC,
            relation=relation.value,
            confidence=confidence,
            **kwargs
        )

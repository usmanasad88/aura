"""Node types for Semantic Scene Graph.

Nodes represent entities in the scene: Objects, Agents, and Regions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np


class NodeType(Enum):
    """Types of nodes in the scene graph."""
    OBJECT = auto()      # Physical objects (cup, pan, sugar)
    AGENT = auto()       # Agents (human, robot)
    REGION = auto()      # Spatial regions (table, storage, stove)


class AgentState(Enum):
    """Agent states."""
    IDLE = auto()
    BUSY = auto()
    REACHING = auto()
    GRASPING = auto()
    MOVING = auto()
    PLACING = auto()
    SPEAKING = auto()
    WAITING = auto()


@dataclass
class SSGNode:
    """Base class for all scene graph nodes.
    
    Attributes:
        id: Unique node identifier
        name: Human-readable name
        node_type: Type of node (OBJECT, AGENT, REGION)
        last_updated: Timestamp of last update
    """
    id: str
    name: str
    node_type: NodeType = NodeType.OBJECT  # Default, overridden in subclasses
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.name,
            "last_updated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSGNode":
        """Deserialize node from dictionary."""
        node_type = NodeType[data["node_type"]]
        
        # Create appropriate subclass based on type
        if node_type == NodeType.OBJECT:
            return ObjectNode.from_dict(data)
        elif node_type == NodeType.AGENT:
            return AgentNode.from_dict(data)
        elif node_type == NodeType.REGION:
            return RegionNode.from_dict(data)
        
        # Fallback to base class
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=node_type,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
        )


@dataclass
class ObjectNode(SSGNode):
    """Node representing a physical object in the scene."""
    
    def __post_init__(self):
        self.node_type = NodeType.OBJECT
    
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectNode":
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.OBJECT,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
        )


@dataclass
class AgentNode(SSGNode):
    """Node representing an agent (human or robot).
    
    Additional attributes:
        agent_type: "human" or "robot"
        state: Current agent state
        current_action: Action currently being performed
    """
    agent_type: str = "unknown"
    state: AgentState = AgentState.IDLE
    current_action: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.AGENT
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "agent_type": self.agent_type,
            "state": self.state.name if isinstance(self.state, AgentState) else self.state,
            "current_action": self.current_action,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentNode":
        state = data.get("state", "IDLE")
        if isinstance(state, str):
            try:
                state = AgentState[state]
            except KeyError:
                state = AgentState.IDLE
        
        return cls(
            id=data["id"],
            name=data["name"],
            agent_type=data["agent_type"],
            node_type=NodeType.AGENT,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            state=state,
            current_action=data.get("current_action"),
        )


@dataclass
class RegionNode(SSGNode):
    """Node representing a spatial region (table, storage, stove).
    
    Additional attributes:
        region_type: Type of region (workspace, storage, appliance)
        contained_objects: List of object IDs in this region
    """
    region_type: str = "unknown"
    contained_objects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.REGION
    
    def contains(self, object_id: str) -> bool:
        """Check if region contains an object."""
        return object_id in self.contained_objects
    
    def add_object(self, object_id: str) -> None:
        """Add object to region."""
        if object_id not in self.contained_objects:
            self.contained_objects.append(object_id)
            self.last_updated = datetime.now()
    
    def remove_object(self, object_id: str) -> None:
        """Remove object from region."""
        if object_id in self.contained_objects:
            self.contained_objects.remove(object_id)
            self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "region_type": self.region_type,
            "contained_objects": self.contained_objects,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionNode":
        return cls(
            id=data["id"],
            name=data["name"],
            region_type=data["region_type"],
            node_type=NodeType.REGION,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            contained_objects=data.get("contained_objects", []),
        )

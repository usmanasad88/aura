"""Node types for Semantic Scene Graph.

Nodes represent entities in the scene: Objects, Agents, and Regions.
Each node has attributes for state, affordances, and predicted actions.
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


class ObjectState(Enum):
    """Common object states."""
    UNKNOWN = auto()
    EMPTY = auto()
    FULL = auto()
    PARTIAL = auto()
    DIRTY = auto()
    CLEAN = auto()
    HOT = auto()
    COLD = auto()
    BOILING = auto()
    READY = auto()
    IN_USE = auto()
    AVAILABLE = auto()


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
class Affordance:
    """An action that can be performed on/with this node.
    
    Attributes:
        action_id: Unique identifier for the action/skill
        name: Human-readable name
        preconditions: Dict of required state/conditions
        effects: Dict of state changes when action completes
        feasibility: 0-1 score of how feasible this action is now
        parameters: Additional action parameters
    """
    action_id: str
    name: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    feasibility: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedAction:
    """A predicted future action involving this node.
    
    Attributes:
        action_name: Name of the predicted action
        agent_id: Who will perform the action
        confidence: Prediction confidence (0-1)
        estimated_time_sec: When the action is expected
        reasoning: Why this action is predicted
    """
    action_name: str
    agent_id: str
    confidence: float = 0.0
    estimated_time_sec: float = 0.0
    reasoning: str = ""


@dataclass
class SSGNode:
    """Base class for all scene graph nodes.
    
    Attributes:
        id: Unique node identifier
        name: Human-readable name
        node_type: Type of node (OBJECT, AGENT, REGION)
        position: (x, y, z) coordinates in world frame
        bbox: Bounding box in image [x_min, y_min, x_max, y_max]
        confidence: Detection confidence (0-1)
        attributes: Custom key-value attributes
        affordances: Available actions for this node
        predicted_actions: Predicted future actions
        last_updated: Timestamp of last update
        metadata: Additional metadata
    """
    id: str
    name: str
    node_type: NodeType = NodeType.OBJECT  # Default, overridden in subclasses
    position: Optional[Tuple[float, float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    affordances: List[Affordance] = field(default_factory=list)
    predicted_actions: List[PredictedAction] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set an attribute value and update timestamp."""
        self.attributes[key] = value
        self.last_updated = datetime.now()
    
    def add_affordance(self, affordance: Affordance) -> None:
        """Add an available affordance."""
        # Remove existing with same ID
        self.affordances = [a for a in self.affordances if a.action_id != affordance.action_id]
        self.affordances.append(affordance)
    
    def get_affordance(self, action_id: str) -> Optional[Affordance]:
        """Get affordance by action ID."""
        for aff in self.affordances:
            if aff.action_id == action_id:
                return aff
        return None
    
    def has_affordance(self, action_id: str) -> bool:
        """Check if node has a specific affordance."""
        return self.get_affordance(action_id) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.name,
            "position": self.position,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "attributes": self.attributes,
            "affordances": [
                {
                    "action_id": a.action_id,
                    "name": a.name,
                    "preconditions": a.preconditions,
                    "effects": a.effects,
                    "feasibility": a.feasibility,
                    "parameters": a.parameters,
                }
                for a in self.affordances
            ],
            "predicted_actions": [
                {
                    "action_name": p.action_name,
                    "agent_id": p.agent_id,
                    "confidence": p.confidence,
                    "estimated_time_sec": p.estimated_time_sec,
                    "reasoning": p.reasoning,
                }
                for p in self.predicted_actions
            ],
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
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
            position=tuple(data["position"]) if data.get("position") else None,
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            affordances=[
                Affordance(**a) for a in data.get("affordances", [])
            ],
            predicted_actions=[
                PredictedAction(**p) for p in data.get("predicted_actions", [])
            ],
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ObjectNode(SSGNode):
    """Node representing a physical object in the scene.
    
    Additional attributes:
        category: Object category (e.g., "container", "tool", "ingredient")
        state: Current object state
        is_movable: Whether the object can be moved
        is_graspable: Whether the object can be grasped
        contents: What the object contains (if applicable)
    """
    category: str = "object"
    state: ObjectState = ObjectState.UNKNOWN
    is_movable: bool = True
    is_graspable: bool = True
    contents: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.OBJECT
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "category": self.category,
            "state": self.state.name if isinstance(self.state, ObjectState) else self.state,
            "is_movable": self.is_movable,
            "is_graspable": self.is_graspable,
            "contents": self.contents,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectNode":
        state = data.get("state", "UNKNOWN")
        if isinstance(state, str):
            try:
                state = ObjectState[state]
            except KeyError:
                state = ObjectState.UNKNOWN
        
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.OBJECT,
            position=tuple(data["position"]) if data.get("position") else None,
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            affordances=[Affordance(**a) for a in data.get("affordances", [])],
            predicted_actions=[PredictedAction(**p) for p in data.get("predicted_actions", [])],
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            metadata=data.get("metadata", {}),
            category=data.get("category", "object"),
            state=state,
            is_movable=data.get("is_movable", True),
            is_graspable=data.get("is_graspable", True),
            contents=data.get("contents"),
        )


@dataclass
class AgentNode(SSGNode):
    """Node representing an agent (human or robot).
    
    Additional attributes:
        agent_type: "human" or "robot"
        state: Current agent state
        capabilities: List of action IDs the agent can perform
        current_action: Action currently being performed
        attention_target: Node ID the agent is focused on
    """
    agent_type: str = "human"
    state: AgentState = AgentState.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_action: Optional[str] = None
    attention_target: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.AGENT
    
    def can_perform(self, action_id: str) -> bool:
        """Check if agent can perform a specific action."""
        return action_id in self.capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "agent_type": self.agent_type,
            "state": self.state.name if isinstance(self.state, AgentState) else self.state,
            "capabilities": self.capabilities,
            "current_action": self.current_action,
            "attention_target": self.attention_target,
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
            node_type=NodeType.AGENT,
            position=tuple(data["position"]) if data.get("position") else None,
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            affordances=[Affordance(**a) for a in data.get("affordances", [])],
            predicted_actions=[PredictedAction(**p) for p in data.get("predicted_actions", [])],
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            metadata=data.get("metadata", {}),
            agent_type=data.get("agent_type", "human"),
            state=state,
            capabilities=data.get("capabilities", []),
            current_action=data.get("current_action"),
            attention_target=data.get("attention_target"),
        )


@dataclass
class RegionNode(SSGNode):
    """Node representing a spatial region (table, storage, stove).
    
    Additional attributes:
        region_type: Type of region (workspace, storage, appliance)
        bounds: 3D bounding box [(x_min, y_min, z_min), (x_max, y_max, z_max)]
        contained_objects: List of object IDs in this region
        is_accessible: Whether the region is currently accessible
    """
    region_type: str = "workspace"
    bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    contained_objects: List[str] = field(default_factory=list)
    is_accessible: bool = True
    
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
            "bounds": self.bounds,
            "contained_objects": self.contained_objects,
            "is_accessible": self.is_accessible,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionNode":
        bounds = data.get("bounds")
        if bounds:
            bounds = (tuple(bounds[0]), tuple(bounds[1]))
        
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.REGION,
            position=tuple(data["position"]) if data.get("position") else None,
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            affordances=[Affordance(**a) for a in data.get("affordances", [])],
            predicted_actions=[PredictedAction(**p) for p in data.get("predicted_actions", [])],
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
            metadata=data.get("metadata", {}),
            region_type=data.get("region_type", "workspace"),
            bounds=bounds,
            contained_objects=data.get("contained_objects", []),
            is_accessible=data.get("is_accessible", True),
        )

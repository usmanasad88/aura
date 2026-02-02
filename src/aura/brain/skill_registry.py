"""Skill Registry for robot capabilities.

Defines and manages the robot skills that can be executed.
Skills are loaded from configuration files and matched against
scene graph affordances for action selection.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class SkillParameter:
    """Parameter for a robot skill."""
    name: str
    type: str  # "string", "number", "boolean", "object_id", "region_id"
    description: str = ""
    required: bool = True
    default: Any = None
    valid_values: List[Any] = field(default_factory=list)


@dataclass
class RobotSkill:
    """Definition of a robot skill.
    
    Attributes:
        id: Unique skill identifier
        name: Human-readable name
        description: What the skill does
        category: Skill category (manipulation, navigation, communication)
        parameters: Required/optional parameters
        preconditions: Conditions that must be true to execute
        effects: State changes when skill completes
        estimated_duration_sec: Expected execution time
        can_interrupt: Whether skill can be safely interrupted
    """
    id: str
    name: str
    description: str
    category: str = "manipulation"
    parameters: List[SkillParameter] = field(default_factory=list)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_sec: float = 5.0
    can_interrupt: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate that provided parameters are valid.
        
        Returns (is_valid, error_message).
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
            
            if param.name in params:
                value = params[param.name]
                if param.valid_values and value not in param.valid_values:
                    return False, f"Invalid value for {param.name}: {value}"
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "valid_values": p.valid_values,
                }
                for p in self.parameters
            ],
            "preconditions": self.preconditions,
            "effects": self.effects,
            "estimated_duration_sec": self.estimated_duration_sec,
            "can_interrupt": self.can_interrupt,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotSkill":
        """Deserialize from dictionary."""
        params = [
            SkillParameter(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
                valid_values=p.get("valid_values", []),
            )
            for p in data.get("parameters", [])
        ]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "manipulation"),
            parameters=params,
            preconditions=data.get("preconditions", {}),
            effects=data.get("effects", {}),
            estimated_duration_sec=data.get("estimated_duration_sec", 5.0),
            can_interrupt=data.get("can_interrupt", True),
            metadata=data.get("metadata", {}),
        )


class SkillRegistry:
    """Registry of available robot skills.
    
    Manages skill definitions and provides lookup functionality
    for the decision engine.
    """
    
    def __init__(self):
        self._skills: Dict[str, RobotSkill] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, skill: RobotSkill) -> None:
        """Register a skill."""
        self._skills[skill.id] = skill
        
        # Track by category
        if skill.category not in self._categories:
            self._categories[skill.category] = []
        if skill.id not in self._categories[skill.category]:
            self._categories[skill.category].append(skill.id)
        
        logger.debug(f"Registered skill: {skill.id}")
    
    def get(self, skill_id: str) -> Optional[RobotSkill]:
        """Get skill by ID."""
        return self._skills.get(skill_id)
    
    def has(self, skill_id: str) -> bool:
        """Check if skill is registered."""
        return skill_id in self._skills
    
    def get_by_category(self, category: str) -> List[RobotSkill]:
        """Get all skills in a category."""
        skill_ids = self._categories.get(category, [])
        return [self._skills[sid] for sid in skill_ids]
    
    def list_skills(self) -> List[RobotSkill]:
        """List all registered skills."""
        return list(self._skills.values())
    
    def list_skill_ids(self) -> List[str]:
        """List all skill IDs."""
        return list(self._skills.keys())
    
    def get_skills_for_llm(self) -> str:
        """Generate skill descriptions for LLM context."""
        lines = ["## Available Robot Skills\n"]
        
        for category in sorted(self._categories.keys()):
            lines.append(f"### {category.title()}\n")
            for skill_id in self._categories[category]:
                skill = self._skills[skill_id]
                lines.append(f"**{skill.name}** (`{skill.id}`)")
                lines.append(f"  {skill.description}")
                if skill.parameters:
                    params = ", ".join([
                        f"{p.name}: {p.type}" + ("*" if p.required else "")
                        for p in skill.parameters
                    ])
                    lines.append(f"  Parameters: {params}")
                lines.append("")
        
        return "\n".join(lines)
    
    def load_from_file(self, path: str) -> int:
        """Load skills from JSON file. Returns number loaded."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        skills = data if isinstance(data, list) else data.get("skills", [])
        count = 0
        for skill_data in skills:
            skill = RobotSkill.from_dict(skill_data)
            self.register(skill)
            count += 1
        
        logger.info(f"Loaded {count} skills from {path}")
        return count
    
    def save_to_file(self, path: str) -> None:
        """Save skills to JSON file."""
        data = {
            "skills": [s.to_dict() for s in self._skills.values()]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self._skills)} skills to {path}")
    
    @classmethod
    def create_default(cls) -> "SkillRegistry":
        """Create registry with default manipulation skills."""
        registry = cls()
        
        # Retrieve object from storage
        registry.register(RobotSkill(
            id="retrieve_object",
            name="Retrieve Object",
            description="Pick up an object from storage and place it in the working area",
            category="manipulation",
            parameters=[
                SkillParameter("object_id", "object_id", "Object to retrieve", True),
                SkillParameter("from_region", "region_id", "Region to pick from", False, "storage_area"),
                SkillParameter("to_region", "region_id", "Region to place in", False, "working_area"),
            ],
            preconditions={"object.is_graspable": True},
            effects={"object.location": "working_area"},
            estimated_duration_sec=8.0,
        ))
        
        # Return object to storage
        registry.register(RobotSkill(
            id="return_object",
            name="Return Object",
            description="Return an object from working area to storage",
            category="manipulation",
            parameters=[
                SkillParameter("object_id", "object_id", "Object to return", True),
                SkillParameter("to_region", "region_id", "Region to return to", False, "storage_area"),
            ],
            preconditions={"object.at_region": "working_area"},
            effects={"object.location": "storage_area"},
            estimated_duration_sec=6.0,
        ))
        
        # Add ingredient
        registry.register(RobotSkill(
            id="add_ingredient",
            name="Add Ingredient",
            description="Add an ingredient to a container",
            category="manipulation",
            parameters=[
                SkillParameter("ingredient_id", "object_id", "Ingredient to add", True),
                SkillParameter("target_id", "object_id", "Container to add to", True),
                SkillParameter("amount", "string", "Amount to add", False, "standard"),
            ],
            preconditions={
                "ingredient.state": "AVAILABLE",
                "target.at_region": "working_area",
            },
            effects={"task.ingredient_added": True},
            estimated_duration_sec=5.0,
        ))
        
        # Stir
        registry.register(RobotSkill(
            id="stir",
            name="Stir Contents",
            description="Stir the contents of a container",
            category="manipulation",
            parameters=[
                SkillParameter("target_id", "object_id", "Container to stir", True),
                SkillParameter("duration_sec", "number", "How long to stir", False, 5.0),
            ],
            preconditions={"target.at_region": "working_area"},
            effects={"target.contents_mixed": True},
            estimated_duration_sec=5.0,
        ))
        
        # Ask preference
        registry.register(RobotSkill(
            id="ask_preference",
            name="Ask Preference",
            description="Ask the human about their preference",
            category="communication",
            parameters=[
                SkillParameter("topic", "string", "What to ask about", True),
                SkillParameter("options", "string", "Available options", False),
            ],
            preconditions={},
            effects={"preference_known": True},
            estimated_duration_sec=10.0,
        ))
        
        # Wait
        registry.register(RobotSkill(
            id="wait",
            name="Wait",
            description="Wait for a condition or duration",
            category="utility",
            parameters=[
                SkillParameter("duration_sec", "number", "How long to wait", False, 5.0),
                SkillParameter("condition", "string", "Condition to wait for", False),
            ],
            preconditions={},
            effects={},
            estimated_duration_sec=5.0,
            can_interrupt=True,
        ))
        
        return registry

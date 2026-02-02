#!/usr/bin/env python3
"""Unit tests for Brain/Decision Engine components."""

import sys
import pytest
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aura.brain import SkillRegistry, RobotSkill, DecisionExplainer
from aura.brain.skill_registry import SkillParameter
from aura.brain.decision_engine import DecisionEngineConfig, DecisionEngine
from aura.core.scene_graph import SemanticSceneGraph, ObjectNode, AgentNode, RegionNode
from aura.core.scene_graph.nodes import ObjectState, AgentState, Affordance


class TestSkillRegistry:
    """Tests for robot skill registry."""
    
    def test_create_skill(self):
        """Test creating a robot skill."""
        skill = RobotSkill(
            id="pick_object",
            name="Pick Object",
            description="Pick up an object",
            category="manipulation",
            parameters=[
                SkillParameter("object_id", "object_id", "Target object", True),
            ],
            estimated_duration_sec=5.0,
        )
        
        assert skill.id == "pick_object"
        assert skill.category == "manipulation"
        assert len(skill.parameters) == 1
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        skill = RobotSkill(
            id="test",
            name="Test",
            description="Test skill",
            parameters=[
                SkillParameter("required_param", "string", "Required", True),
                SkillParameter("optional_param", "string", "Optional", False),
            ],
        )
        
        # Missing required parameter
        is_valid, error = skill.validate_parameters({})
        assert is_valid is False
        assert "required_param" in error
        
        # Valid
        is_valid, error = skill.validate_parameters({"required_param": "value"})
        assert is_valid is True
    
    def test_registry_operations(self):
        """Test registry add/get/list operations."""
        registry = SkillRegistry()
        
        skill = RobotSkill(
            id="test_skill",
            name="Test Skill",
            description="A test skill",
            category="test",
        )
        registry.register(skill)
        
        assert registry.has("test_skill") is True
        assert registry.has("nonexistent") is False
        
        retrieved = registry.get("test_skill")
        assert retrieved is not None
        assert retrieved.name == "Test Skill"
        
        assert len(registry.list_skills()) == 1
    
    def test_default_registry(self):
        """Test creating default registry with standard skills."""
        registry = SkillRegistry.create_default()
        
        assert registry.has("retrieve_object") is True
        assert registry.has("return_object") is True
        assert registry.has("add_ingredient") is True
        assert registry.has("stir") is True
        assert registry.has("ask_preference") is True
    
    def test_get_skills_for_llm(self):
        """Test generating skill descriptions for LLM."""
        registry = SkillRegistry.create_default()
        
        description = registry.get_skills_for_llm()
        
        assert "## Available Robot Skills" in description
        assert "retrieve_object" in description.lower() or "Retrieve" in description


class TestDecisionExplainer:
    """Tests for decision explainer."""
    
    @pytest.fixture
    def explainer_setup(self):
        """Create graph and explainer for testing."""
        graph = SemanticSceneGraph(name="explainer_test")
        
        graph.add_node(RegionNode(
            id="working_area", name="Working Area", region_type="workspace"
        ))
        graph.add_node(ObjectNode(
            id="cup", name="Cup", category="container",
            state=ObjectState.AVAILABLE
        ))
        graph.add_node(AgentNode(
            id="human", name="Human", agent_type="human"
        ))
        
        graph.set_location("cup", "working_area")
        graph.set_agent_target("human", "cup")
        
        explainer = DecisionExplainer(graph)
        return graph, explainer
    
    def test_explain_action(self, explainer_setup):
        """Test generating action explanation."""
        graph, explainer = explainer_setup
        
        explanation = explainer.explain_action(
            action_id="add_sugar",
            target_id="cup",
            reasoning="Cup is ready for sugar"
        )
        
        assert "add_sugar" in explanation
        assert "Cup" in explanation
        assert "Evidence" in explanation or "Reasoning" in explanation
    
    def test_explain_wait(self, explainer_setup):
        """Test generating wait explanation."""
        graph, explainer = explainer_setup
        
        explanation = explainer.explain_wait(
            reason="Waiting for water to boil",
            waiting_for="water_boiling"
        )
        
        assert "Wait" in explanation
        assert "boil" in explanation.lower()
    
    def test_compare_to_ground_truth(self, explainer_setup):
        """Test ground truth comparison."""
        graph, explainer = explainer_setup
        
        result = explainer.compare_to_ground_truth(
            predicted_time=10.5,
            ground_truth_time=11.0,
            action_id="add_sugar",
            tolerance_sec=2.0
        )
        
        assert result["error_sec"] == -0.5
        assert result["is_within_tolerance"] is True
        assert result["assessment"] == "CORRECT"
        
        # Test late prediction
        result = explainer.compare_to_ground_truth(
            predicted_time=15.0,
            ground_truth_time=10.0,
            action_id="add_sugar",
            tolerance_sec=2.0
        )
        
        assert result["is_within_tolerance"] is False
        assert result["assessment"] == "LATE"


class TestDecisionEngineConfig:
    """Tests for decision engine configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DecisionEngineConfig()
        
        assert config.gemini_model == "gemini-2.5-pro-preview-06-05"
        assert config.enable_llm_reasoning is True
        assert config.proactive_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DecisionEngineConfig(
            gemini_model="gemini-2.5-flash",
            enable_llm_reasoning=False,
            proactive_threshold=0.5,
        )
        
        assert config.gemini_model == "gemini-2.5-flash"
        assert config.enable_llm_reasoning is False
        assert config.proactive_threshold == 0.5


class TestDecisionEngine:
    """Tests for decision engine core functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create decision engine for testing."""
        config = DecisionEngineConfig(
            enable_llm_reasoning=False,  # Use rule-based for tests
        )
        return DecisionEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.graph is not None
        assert engine.skills is not None
        assert engine.reasoner is not None
        assert engine.explainer is not None
    
    def test_graph_state_management(self, engine):
        """Test graph state updates through engine."""
        engine.graph.set_task_state("test_var", True)
        
        assert engine.graph.get_task_state("test_var") is True
    
    def test_initialize_scene(self, engine):
        """Test scene initialization from data."""
        scene_data = {
            "regions": [
                {"id": "table", "name": "Table", "region_type": "workspace"}
            ],
            "objects": [
                {
                    "id": "cup", 
                    "name": "Cup", 
                    "category": "container",
                    "initial_location": "table",
                    "affordances": [
                        {"action_id": "fill", "name": "Fill Cup"}
                    ]
                }
            ],
            "agents": [
                {"id": "robot", "name": "Robot", "agent_type": "robot"}
            ]
        }
        
        engine._initialize_scene(scene_data)
        
        assert engine.graph.has_node("table")
        assert engine.graph.has_node("cup")
        assert engine.graph.has_node("robot")
        
        cup = engine.graph.get_node("cup")
        assert cup.has_affordance("fill")
        
        location = engine.graph.get_location("cup")
        assert location == "table"
    
    def test_task_lifecycle(self, engine):
        """Test task start/stop."""
        engine.start_task()
        assert engine.is_running is True
        assert engine.task_start_time is not None
        
        summary = engine.stop_task()
        assert engine.is_running is False
        assert "duration_sec" in summary


class TestDecisionEngineIntegration:
    """Integration tests for decision engine."""
    
    @pytest.fixture
    def loaded_engine(self):
        """Create engine loaded with tea-making task."""
        config = DecisionEngineConfig(
            enable_llm_reasoning=False,
        )
        engine = DecisionEngine(config)
        
        # Manually set up a simple scene for testing
        engine.graph.add_node(RegionNode(
            id="working_area", name="Working Area", region_type="workspace"
        ))
        engine.graph.add_node(RegionNode(
            id="storage_area", name="Storage", region_type="storage"
        ))
        
        cup = ObjectNode(
            id="cup", name="Cup", category="container",
            state=ObjectState.AVAILABLE
        )
        cup.add_affordance(Affordance(
            action_id="add_sugar",
            name="Add Sugar",
            preconditions={},
            effects={"task.sugar_added": True},
            feasibility=0.9
        ))
        engine.graph.add_node(cup)
        engine.graph.set_location("cup", "working_area")
        
        robot = AgentNode(
            id="robot", name="Robot", agent_type="robot",
            capabilities=["add_sugar", "stir", "retrieve_object"]
        )
        engine.graph.add_node(robot)
        
        human = AgentNode(
            id="human", name="Human", agent_type="human"
        )
        engine.graph.add_node(human)
        
        # Initialize task state
        engine.graph.set_task_state("cup_has_tea", True)
        engine.graph.set_task_state("sugar_preference_known", True)
        engine.graph.set_task_state("sugar_added", False)
        
        return engine
    
    def test_rule_based_decision(self, loaded_engine):
        """Test rule-based decision making."""
        import asyncio
        
        loaded_engine.start_task()
        
        # Run decision
        prediction = asyncio.run(loaded_engine.decide_action(current_time_sec=10.0))
        
        # Should get a prediction since cup is available and robot can add_sugar
        # The prediction comes from affordances, not the skill registry
        if prediction:
            # The action_id should be from an affordance on the cup
            cup = loaded_engine.graph.get_node("cup")
            affordance_ids = [a.action_id for a in cup.affordances]
            assert prediction.action_id in affordance_ids or prediction.action_id in loaded_engine.skills.list_skill_ids()
    
    def test_state_summary(self, loaded_engine):
        """Test state summary generation."""
        summary = loaded_engine.get_state_summary()
        
        assert "Cup" in summary or "cup" in summary.lower()
        assert "Robot" in summary or "robot" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

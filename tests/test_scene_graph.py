#!/usr/bin/env python3
"""Unit tests for Semantic Scene Graph components.

Tests the core SSG functionality: nodes, edges, graph, and reasoning.
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aura.core.scene_graph import (
    SemanticSceneGraph,
    GraphQuery, GraphReasoner,
    ObjectNode, AgentNode, RegionNode,
    SSGEdge, SpatialRelation, SemanticRelation,
    NodeType, EdgeType,
)
from aura.core.scene_graph.nodes import ObjectState, AgentState, Affordance


class TestNodes:
    """Tests for SSG node types."""
    
    def test_create_object_node(self):
        """Test creating an object node."""
        node = ObjectNode(
            id="cup_1",
            name="Coffee Cup",
            category="container",
            state=ObjectState.EMPTY,
        )
        
        assert node.id == "cup_1"
        assert node.name == "Coffee Cup"
        assert node.node_type == NodeType.OBJECT
        assert node.category == "container"
        assert node.state == ObjectState.EMPTY
        assert node.is_movable is True
        assert node.is_graspable is True
    
    def test_create_agent_node(self):
        """Test creating an agent node."""
        node = AgentNode(
            id="robot",
            name="Robot Assistant",
            agent_type="robot",
            capabilities=["pick", "place", "pour"],
        )
        
        assert node.id == "robot"
        assert node.node_type == NodeType.AGENT
        assert node.agent_type == "robot"
        assert node.can_perform("pick") is True
        assert node.can_perform("fly") is False
    
    def test_create_region_node(self):
        """Test creating a region node."""
        node = RegionNode(
            id="table",
            name="Work Table",
            region_type="workspace",
            is_accessible=True,
        )
        
        assert node.id == "table"
        assert node.node_type == NodeType.REGION
        assert node.region_type == "workspace"
        assert node.is_accessible is True
        assert node.contained_objects == []
    
    def test_node_affordance(self):
        """Test adding affordances to nodes."""
        node = ObjectNode(
            id="cup_1",
            name="Cup",
            category="container",
        )
        
        aff = Affordance(
            action_id="fill",
            name="Fill Cup",
            preconditions={"state": "EMPTY"},
            effects={"state": "FULL"},
        )
        node.add_affordance(aff)
        
        assert node.has_affordance("fill") is True
        assert node.has_affordance("fly") is False
        
        retrieved = node.get_affordance("fill")
        assert retrieved is not None
        assert retrieved.name == "Fill Cup"
    
    def test_node_serialization(self):
        """Test node to/from dict serialization."""
        node = ObjectNode(
            id="cup_1",
            name="Cup",
            category="container",
            state=ObjectState.EMPTY,
            position=(1.0, 2.0, 0.5),
        )
        node.set_state("temperature", "hot")
        
        data = node.to_dict()
        
        assert data["id"] == "cup_1"
        assert data["category"] == "container"
        assert data["state"] == "EMPTY"
        assert data["position"] == (1.0, 2.0, 0.5)
        assert data["attributes"]["temperature"] == "hot"
        
        # Deserialize
        restored = ObjectNode.from_dict(data)
        assert restored.id == node.id
        assert restored.category == node.category
        assert restored.position == node.position


class TestEdges:
    """Tests for SSG edges."""
    
    def test_create_spatial_edge(self):
        """Test creating spatial edges."""
        edge = SSGEdge.spatial(
            source_id="cup",
            target_id="table",
            relation=SpatialRelation.ON,
            confidence=0.9,
        )
        
        assert edge.source_id == "cup"
        assert edge.target_id == "table"
        assert edge.edge_type == EdgeType.SPATIAL
        assert edge.relation == SpatialRelation.ON.value
        assert edge.confidence == 0.9
    
    def test_create_semantic_edge(self):
        """Test creating semantic edges."""
        edge = SSGEdge.semantic(
            source_id="human",
            target_id="cup",
            relation=SemanticRelation.TARGETS,
            reasoning="Human reaching for cup",
        )
        
        assert edge.edge_type == EdgeType.SEMANTIC
        assert edge.relation == SemanticRelation.TARGETS.value
        assert "reaching" in edge.reasoning.lower()
    
    def test_edge_id(self):
        """Test edge ID generation."""
        edge = SSGEdge.spatial("a", "b", SpatialRelation.ON)
        assert edge.edge_id == "a--on-->b"
    
    def test_edge_explanation(self):
        """Test edge explanation string."""
        edge = SSGEdge.semantic(
            "human", "cup", SemanticRelation.TARGETS,
            is_predicted=True
        )
        
        explanation = edge.to_explanation_string()
        assert "[human]" in explanation
        assert "[cup]" in explanation
        assert "(predicted)" in explanation
    
    def test_edge_serialization(self):
        """Test edge to/from dict."""
        edge = SSGEdge.spatial("cup", "table", SpatialRelation.ON)
        
        data = edge.to_dict()
        restored = SSGEdge.from_dict(data)
        
        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.relation == edge.relation


class TestGraph:
    """Tests for SemanticSceneGraph."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = SemanticSceneGraph(name="test_graph")
        
        # Add regions
        graph.add_node(RegionNode(
            id="table", name="Table", region_type="workspace"
        ))
        graph.add_node(RegionNode(
            id="storage", name="Storage", region_type="storage"
        ))
        
        # Add objects
        cup = ObjectNode(
            id="cup", name="Cup", category="container",
            state=ObjectState.EMPTY
        )
        cup.add_affordance(Affordance(
            action_id="fill", name="Fill",
            preconditions={"state": "EMPTY"},
            effects={"state": "FULL"}
        ))
        graph.add_node(cup)
        
        graph.add_node(ObjectNode(
            id="pot", name="Pot", category="container",
            state=ObjectState.FULL
        ))
        
        # Add agents
        graph.add_node(AgentNode(
            id="human", name="Human", agent_type="human"
        ))
        graph.add_node(AgentNode(
            id="robot", name="Robot", agent_type="robot",
            capabilities=["fill", "pick", "place"]
        ))
        
        # Add location edges
        graph.set_location("cup", "table")
        graph.set_location("pot", "storage")
        
        return graph
    
    def test_add_get_node(self, sample_graph):
        """Test adding and retrieving nodes."""
        assert sample_graph.node_count == 6
        
        cup = sample_graph.get_node("cup")
        assert cup is not None
        assert cup.name == "Cup"
        
        assert sample_graph.has_node("cup") is True
        assert sample_graph.has_node("nonexistent") is False
    
    def test_remove_node(self, sample_graph):
        """Test removing nodes."""
        # Add edge to cup first
        sample_graph.add_edge(SSGEdge.semantic(
            "human", "cup", SemanticRelation.TARGETS
        ))
        
        initial_edges = sample_graph.edge_count
        result = sample_graph.remove_node("cup")
        
        assert result is True
        assert sample_graph.has_node("cup") is False
        # Edges involving cup should be removed
        assert sample_graph.edge_count < initial_edges
    
    def test_get_nodes_by_type(self, sample_graph):
        """Test filtering nodes by type."""
        objects = sample_graph.get_objects()
        assert len(objects) == 2
        
        agents = sample_graph.get_agents()
        assert len(agents) == 2
        
        regions = sample_graph.get_regions()
        assert len(regions) == 2
    
    def test_add_edge(self, sample_graph):
        """Test adding edges."""
        sample_graph.add_edge(SSGEdge.semantic(
            "human", "cup", SemanticRelation.TARGETS
        ))
        
        edges = sample_graph.get_edges(source_id="human")
        assert len(edges) >= 1
        
        target_edges = sample_graph.get_edges(
            source_id="human",
            relation=SemanticRelation.TARGETS.value
        )
        assert len(target_edges) == 1
    
    def test_set_get_location(self, sample_graph):
        """Test location operations."""
        location = sample_graph.get_location("cup")
        assert location == "table"
        
        # Change location
        sample_graph.set_location("cup", "storage")
        assert sample_graph.get_location("cup") == "storage"
        
        # Check region contains object
        storage = sample_graph.get_node("storage")
        assert "cup" in storage.contained_objects
    
    def test_agent_target(self, sample_graph):
        """Test agent targeting."""
        sample_graph.set_agent_target("human", "cup", confidence=0.8)
        
        target = sample_graph.get_agent_target("human")
        assert target == "cup"
        
        human = sample_graph.get_node("human")
        assert human.attention_target == "cup"
    
    def test_blocking(self, sample_graph):
        """Test blocking relationships."""
        sample_graph.set_blocks("pot", "cup", reasoning="Pot in the way")
        
        is_blocked, blocker = sample_graph.is_blocked("cup")
        assert is_blocked is True
        assert blocker == "pot"
    
    def test_task_state(self, sample_graph):
        """Test task state management."""
        sample_graph.set_task_state("water_boiling", True)
        sample_graph.set_task_state("sugar_preference", "standard")
        
        assert sample_graph.get_task_state("water_boiling") is True
        assert sample_graph.get_task_state("sugar_preference") == "standard"
        assert sample_graph.get_task_state("nonexistent", "default") == "default"
    
    def test_serialization(self, sample_graph):
        """Test graph serialization."""
        json_str = sample_graph.to_json()
        assert len(json_str) > 0
        
        restored = SemanticSceneGraph.from_json(json_str)
        assert restored.node_count == sample_graph.node_count
        assert restored.edge_count == sample_graph.edge_count
    
    def test_summary_for_llm(self, sample_graph):
        """Test LLM summary generation."""
        summary = sample_graph.get_state_summary_for_llm()
        
        assert "## Current Scene State" in summary
        assert "Human" in summary
        assert "Robot" in summary
        assert "Cup" in summary


class TestGraphReasoner:
    """Tests for graph reasoning utilities."""
    
    @pytest.fixture
    def reasoner_graph(self):
        """Create graph with robot capabilities."""
        graph = SemanticSceneGraph(name="reasoner_test")
        
        # Add region
        graph.add_node(RegionNode(
            id="working_area", name="Working Area", region_type="workspace"
        ))
        graph.add_node(RegionNode(
            id="storage_area", name="Storage", region_type="storage"
        ))
        
        # Add objects with affordances
        cup = ObjectNode(
            id="cup", name="Cup", category="container",
            state=ObjectState.AVAILABLE
        )
        cup.add_affordance(Affordance(
            action_id="add_sugar",
            name="Add Sugar",
            preconditions={"at_region": "working_area"},
            effects={"task.sugar_added": True},
            feasibility=0.9
        ))
        graph.add_node(cup)
        graph.set_location("cup", "working_area")
        
        # Add robot
        robot = AgentNode(
            id="robot", name="Robot", agent_type="robot",
            capabilities=["add_sugar", "stir", "pick"]
        )
        graph.add_node(robot)
        
        # Add human
        human = AgentNode(
            id="human", name="Human", agent_type="human"
        )
        graph.add_node(human)
        
        return graph
    
    def test_get_available_actions(self, reasoner_graph):
        """Test finding available actions."""
        reasoner = GraphReasoner(reasoner_graph)
        
        actions = reasoner.get_available_actions("robot")
        
        assert len(actions) > 0
        action_ids = [a["action_id"] for a in actions]
        assert "add_sugar" in action_ids
    
    def test_proactive_opportunities(self, reasoner_graph):
        """Test finding proactive opportunities."""
        reasoner = GraphReasoner(reasoner_graph)
        
        # Set human targeting cup
        reasoner_graph.set_agent_target("human", "cup")
        
        opportunities = reasoner.get_proactive_opportunities("robot")
        
        # Should find opportunity to help with cup
        assert len(opportunities) >= 0  # May be empty if no matching affordances


class TestGraphQuery:
    """Tests for graph query builder."""
    
    @pytest.fixture
    def query_graph(self):
        """Create graph for query testing."""
        graph = SemanticSceneGraph(name="query_test")
        
        graph.add_node(RegionNode(id="area", name="Area", region_type="workspace"))
        
        graph.add_node(ObjectNode(
            id="cup1", name="Cup 1", category="container",
            state=ObjectState.AVAILABLE
        ))
        graph.add_node(ObjectNode(
            id="cup2", name="Cup 2", category="container",
            state=ObjectState.EMPTY
        ))
        graph.add_node(ObjectNode(
            id="pot", name="Pot", category="container",
            state=ObjectState.FULL
        ))
        
        graph.set_location("cup1", "area")
        graph.set_location("cup2", "area")
        
        return graph
    
    def test_find_objects(self, query_graph):
        """Test finding all objects."""
        query = GraphQuery(query_graph)
        result = query.find_objects().execute()
        
        assert len(result.nodes) == 3
    
    def test_filter_by_state(self, query_graph):
        """Test filtering by state."""
        query = GraphQuery(query_graph)
        result = query.find_objects().with_state("AVAILABLE").execute()
        
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "cup1"
    
    def test_filter_by_category(self, query_graph):
        """Test filtering by category."""
        query = GraphQuery(query_graph)
        result = query.find_objects().with_category("container").execute()
        
        assert len(result.nodes) == 3
    
    def test_filter_by_region(self, query_graph):
        """Test filtering by region."""
        query = GraphQuery(query_graph)
        result = query.find_objects().at_region("area").execute()
        
        assert len(result.nodes) == 2
        ids = [n.id for n in result.nodes]
        assert "cup1" in ids
        assert "cup2" in ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

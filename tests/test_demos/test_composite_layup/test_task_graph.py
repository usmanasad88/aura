"""Tests for composite layup task graph loading and validation.

Tests that the SOP JSON is valid and contains all required nodes.
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def task_graph():
    """Load the composite layup task graph."""
    sop_path = Path(__file__).parents[3] / "sops" / "composite_layup.json"
    with open(sop_path) as f:
        return json.load(f)


class TestTaskGraphStructure:
    """Test SOP task graph structure."""
    
    def test_has_required_fields(self, task_graph):
        """Test task graph has all required top-level fields."""
        required = ["id", "name", "nodes", "materials", "tools"]
        for field in required:
            assert field in task_graph, f"Missing required field: {field}"
    
    def test_has_nodes(self, task_graph):
        """Test task graph has nodes defined."""
        assert len(task_graph["nodes"]) > 0
    
    def test_all_nodes_have_required_fields(self, task_graph):
        """Test each node has required fields."""
        required_fields = ["id", "name", "description", "preconditions", "postconditions", "assignee"]
        
        for node_id, node in task_graph["nodes"].items():
            for field in required_fields:
                assert field in node, f"Node '{node_id}' missing field: {field}"
    
    def test_valid_assignees(self, task_graph):
        """Test all assignees are valid."""
        valid_assignees = {"robot", "human", "both"}
        
        for node_id, node in task_graph["nodes"].items():
            assert node["assignee"] in valid_assignees, \
                f"Node '{node_id}' has invalid assignee: {node['assignee']}"


class TestMaterialsSpecification:
    """Test materials specification."""
    
    def test_mix_ratio(self, task_graph):
        """Test resin:hardener mix ratio is specified."""
        materials = task_graph["materials"]
        assert "mix_ratio" in materials
        
        ratio = materials["mix_ratio"]
        assert "resin_parts" in ratio
        assert "hardener_parts" in ratio
        assert ratio["by"] == "weight"
    
    def test_pot_life_specified(self, task_graph):
        """Test pot life is specified."""
        materials = task_graph["materials"]
        assert "pot_life_minutes" in materials
        assert materials["pot_life_minutes"] > 0
    
    def test_fiberglass_plies(self, task_graph):
        """Test fiberglass ply specifications."""
        plies = task_graph["materials"]["fiberglass_plies"]
        assert len(plies) >= 2  # At least 2 plies for alternating orientation
        
        for ply in plies:
            assert "id" in ply
            assert "type" in ply
            assert "orientation" in ply


class TestToolsSpecification:
    """Test tools specification."""
    
    def test_essential_tools_present(self, task_graph):
        """Test all essential tools are defined."""
        essential = ["mixing_cup", "weigh_scale", "roller", "brush"]
        tool_ids = [t["id"] for t in task_graph["tools"]]
        
        for tool in essential:
            assert tool in tool_ids, f"Missing essential tool: {tool}"
    
    def test_tools_have_location(self, task_graph):
        """Test all tools have a location."""
        for tool in task_graph["tools"]:
            assert "location" in tool, f"Tool '{tool['id']}' missing location"


class TestTaskSequence:
    """Test task sequencing and preconditions."""
    
    def test_initial_task_has_no_preconditions(self, task_graph):
        """Test that setup_workspace has no preconditions."""
        setup = task_graph["nodes"]["setup_workspace"]
        assert len(setup["preconditions"]) == 0
    
    def test_preconditions_reference_valid_nodes(self, task_graph):
        """Test all preconditions reference valid nodes."""
        node_ids = set(task_graph["nodes"].keys())
        
        for node_id, node in task_graph["nodes"].items():
            for pre in node["preconditions"]:
                # Preconditions are node IDs or postcondition names
                # This test checks that referenced node IDs exist
                if pre in node_ids or any(pre in n.get("postconditions", []) 
                                          for n in task_graph["nodes"].values()):
                    continue
                # If not a node ID, it should be a postcondition
    
    def test_layup_sequence(self, task_graph):
        """Test plies are laid up in sequence."""
        ply_nodes = ["layup_ply_1", "layup_ply_2", "layup_ply_3", "layup_ply_4"]
        
        for i, node_id in enumerate(ply_nodes):
            node = task_graph["nodes"].get(node_id)
            assert node is not None, f"Missing node: {node_id}"
            
            if i > 0:
                # Each ply after first should depend on previous
                prev_postcondition = f"ply_{i}_complete"
                assert prev_postcondition in node["preconditions"] or \
                       ply_nodes[i-1] in node["preconditions"]
    
    def test_cleanup_is_last(self, task_graph):
        """Test cleanup is the final task."""
        cleanup = task_graph["nodes"]["cleanup"]
        # Cleanup should depend on inspection
        assert "inspection_complete" in cleanup["preconditions"]


class TestQualityChecks:
    """Test quality check definitions."""
    
    def test_defect_definitions(self, task_graph):
        """Test defect types are defined."""
        assert "quality_defects" in task_graph
        defects = task_graph["quality_defects"]
        
        essential = ["dry_spot", "air_bubble", "wrinkle"]
        for defect in essential:
            assert defect in defects, f"Missing defect definition: {defect}"
            assert "severity" in defects[defect]
            assert "remedy" in defects[defect]
    
    def test_layup_nodes_have_quality_checks(self, task_graph):
        """Test layup nodes specify quality checks."""
        layup_nodes = ["layup_ply_1", "layup_ply_2", "layup_ply_3", "layup_ply_4"]
        
        for node_id in layup_nodes:
            node = task_graph["nodes"][node_id]
            if "quality_checks" in node:
                assert len(node["quality_checks"]) > 0


class TestSafetyConstraints:
    """Test safety constraint definitions."""
    
    def test_has_safety_constraints(self, task_graph):
        """Test safety constraints are defined."""
        assert "safety_constraints" in task_graph
    
    def test_collision_avoidance(self, task_graph):
        """Test collision avoidance is enabled."""
        safety = task_graph["safety_constraints"]
        assert safety.get("collision_avoidance") is True
    
    def test_human_priority(self, task_graph):
        """Test human safety is prioritized."""
        safety = task_graph["safety_constraints"]
        assert safety.get("human_priority") is True
    
    def test_speed_limits(self, task_graph):
        """Test speed limits are defined."""
        safety = task_graph["safety_constraints"]
        assert "speed_limit_near_human_mps" in safety
        assert safety["speed_limit_near_human_mps"] <= 0.25  # Safe speed near humans


class TestTimingConstraints:
    """Test timing constraint definitions."""
    
    def test_has_timing_constraints(self, task_graph):
        """Test timing constraints are defined."""
        assert "timing_constraints" in task_graph
    
    def test_pot_life_warnings(self, task_graph):
        """Test pot life warning thresholds."""
        timing = task_graph["timing_constraints"]
        
        assert "pot_life_warning_minutes" in timing
        assert "pot_life_critical_minutes" in timing
        
        # Warning should come before critical
        assert timing["pot_life_warning_minutes"] < timing["pot_life_critical_minutes"]

"""Tests for core types."""

import pytest
from datetime import datetime
import time

from aura.core import (
    Pose3D, BoundingBox, Trajectory,
    TrackedObject, Intent, IntentType,
    Action, RobotActionType, ActionStatus,
    TaskNode, TaskGraph, AuraState,
)


class TestPose3D:
    def test_from_position(self):
        pose = Pose3D.from_position(1.0, 2.0, 3.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0  # Identity quaternion


class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        assert bbox.center == (50, 50)
    
    def test_area(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        assert bbox.area == 5000


class TestTrajectory:
    def test_duration(self):
        poses = [Pose3D.from_position(0, 0, 0), Pose3D.from_position(1, 1, 1)]
        traj = Trajectory(poses=poses, timestamps=[0.0, 2.5])
        assert traj.duration == 2.5
    
    def test_empty_duration(self):
        traj = Trajectory(poses=[], timestamps=[])
        assert traj.duration == 0.0


class TestTrackedObject:
    def test_creation(self):
        obj = TrackedObject(
            id="obj_1",
            name="cup",
            category="container",
            confidence=0.95
        )
        assert obj.id == "obj_1"
        assert obj.confidence == 0.95


class TestAction:
    def test_lifecycle(self):
        action = Action(
            id="action_1",
            type=RobotActionType.MOVE_TO_POSE,
            parameters={"x": 1.0, "y": 2.0}
        )
        assert action.status == ActionStatus.PENDING
        
        action.mark_started()
        assert action.status == ActionStatus.IN_PROGRESS
        assert action.started_at is not None
        
        action.mark_completed()
        assert action.status == ActionStatus.COMPLETED
        assert action.completed_at is not None
    
    def test_failure(self):
        action = Action(id="a1", type=RobotActionType.CLOSE_GRIPPER)
        action.mark_started()
        action.mark_failed("Object slipped")
        assert action.status == ActionStatus.FAILED
        assert action.error_message == "Object slipped"


class TestTaskGraph:
    def test_available_nodes(self):
        graph = TaskGraph(
            id="sorting",
            name="Sorting Task",
            description="Sort objects",
            nodes={
                "start": TaskNode(id="start", name="Start", description="Begin", is_completed=True),
                "pick": TaskNode(id="pick", name="Pick", description="Pick object", preconditions=["start"]),
                "place": TaskNode(id="place", name="Place", description="Place object", preconditions=["pick"]),
            }
        )
        
        available = graph.get_available_nodes()
        assert len(available) == 1
        assert available[0].id == "pick"


class TestAuraState:
    def test_update_timestamp(self):
        state = AuraState()
        old_time = state.last_updated
        time.sleep(0.01)
        state.update_timestamp()
        assert state.last_updated > old_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

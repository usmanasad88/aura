"""AURA Composite Layup Assistance Demo.

This demo provides proactive robotic assistance for fiberglass
hand layup on a mold. The UR5 robot with Robotiq 2F-85 gripper:
- Measures and mixes resin/hardener
- Hands tools to the operator
- Monitors layup quality for defects
- Tracks pot life and issues warnings
"""

from .main import main

__all__ = ["main"]

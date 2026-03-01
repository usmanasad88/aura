"""Unified LangGraph-based workflow orchestration for AURA.

This package provides a generic, config-driven LangGraph runtime that
unifies the assistant pipeline (RCWPS intent + DAG rules) with the
brain pipeline (SSG + GraphReasoner + SkillRegistry).

Usage:
    from aura.workflow import build_task_graph, AuraGraphState

    graph = build_task_graph("tasks/hand_layup/config")
    result = await graph.ainvoke(initial_state, config)
"""

from .state import AuraGraphState
from .builder import build_task_graph

__all__ = ["AuraGraphState", "build_task_graph"]

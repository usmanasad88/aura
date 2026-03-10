"""AURA real-time dashboard — Flask + SSE live monitoring UI."""

from .server import DashboardServer, get_dashboard

__all__ = ["DashboardServer", "get_dashboard"]

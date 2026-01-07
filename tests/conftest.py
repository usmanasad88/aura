"""Pytest configuration for AURA tests."""

import sys

# Remove ROS paths that may interfere with testing
ros_paths = [p for p in sys.path if 'ros' in p.lower()]
for p in ros_paths:
    if p in sys.path:
        sys.path.remove(p)

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']

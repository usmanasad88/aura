"""Pytest configuration for AURA tests."""

import sys

# Remove ROS paths that may interfere with testing
ros_paths = [p for p in sys.path if 'ros' in p.lower()]
for p in ros_paths:
    if p in sys.path:
        sys.path.remove(p)

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


def pytest_configure(config):
    """Block ROS pytest plugins that leak from system site-packages."""
    pm = config.pluginmanager
    for name in (
        "launch_testing_ros",
        "launch_testing",
        "ament_copyright",
        "ament_flake8",
        "ament_pep257",
        "ament_xmllint",
        "ament_lint",
    ):
        pm.set_blocked(name)

    # Register custom markers
    config.addinivalue_line("markers", "gpu: mark test as requiring CUDA GPU")

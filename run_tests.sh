#!/bin/bash
# Run AURA tests with clean environment (avoids ROS pytest plugin conflicts)

cd "$(dirname "$0")"

# Run pytest with clean PYTHONPATH to avoid ROS conflicts
PYTHONPATH="" .venv/bin/python -m pytest "$@"

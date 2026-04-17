#!/bin/bash
# Launch the SAM-3D-Body ZMQ server for aura's BodyPoseMonitor.
#
# Usage:
#   ./run_aura_server.sh              # defaults (port 5556, image-size 512)
#   ./run_aura_server.sh --port 5557  # custom port
#
# Prerequisites:
#   conda activate sam_3d_body   (or run this script which activates it)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env if not already active
if [[ "$CONDA_DEFAULT_ENV" != "fast_sam_3d_body" ]]; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate sam_3d_body
fi

exec python sam3d_body_server.py "$@"

#!/usr/bin/env bash
# Launch a vLLM server for Qwen3.5-0.8B in an isolated venv.
#
# Usage:
#   ./scripts/start_vllm_server.sh                    # defaults (Qwen3.5-0.8B)
#   ./scripts/start_vllm_server.sh --port 8100        # custom port
#   MODEL=Qwen/Qwen3-VL-2B-Instruct ./scripts/start_vllm_server.sh  # alt model
#
# The server exposes an OpenAI-compatible API at http://localhost:${PORT}/v1
# that the LocalVLMMonitor connects to.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_VENV="${PROJECT_DIR}/.venv-vllm"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
PORT="${PORT:-8100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ── Isolate from ROS / conda polluting PYTHONPATH ─────────────────────
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO 2>/dev/null || true
export PATH="${VLLM_VENV}/bin:/usr/local/bin:/usr/bin:/bin"

# Parse CLI args (override env vars)
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)   PORT="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --gpu-mem) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  vLLM Server for AURA LocalVLMMonitor"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  GPU % : ${GPU_MEMORY_UTILIZATION}"
echo "═══════════════════════════════════════════════════════════"

# ── Create isolated venv if needed ────────────────────────────────────
if [ ! -d "${VLLM_VENV}" ]; then
    echo "Creating isolated vLLM venv at ${VLLM_VENV} …"
    uv venv "${VLLM_VENV}" --python 3.12
    echo "Installing vLLM …"
    uv pip install --python "${VLLM_VENV}/bin/python" vllm
    echo "vLLM venv ready."
fi

# ── Launch the server ─────────────────────────────────────────────────
echo ""
echo "Starting vLLM server …"
echo "  API endpoint: http://localhost:${PORT}/v1"
echo "  Stop with Ctrl+C"
echo ""

exec "${VLLM_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}'

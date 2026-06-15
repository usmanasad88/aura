#!/usr/bin/env bash
# Launch an SGLang server for local VLM inference in an isolated venv.
#
# Usage:
#   ./scripts/start_sglang_server.sh                          # defaults
#   ./scripts/start_sglang_server.sh --model Qwen/Qwen3-VL-4B-Instruct
#   PORT=8200 ./scripts/start_sglang_server.sh
#
# The server exposes an OpenAI-compatible API at http://localhost:${PORT}/v1
# that the LocalVLMMonitor connects to (same as vLLM — drop-in replacement).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SGLANG_VENV="${PROJECT_DIR}/.venv-sglang"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8100}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ── Isolate from ROS / conda polluting PYTHONPATH ─────────────────────
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO 2>/dev/null || true
export PATH="${SGLANG_VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"

# Parse CLI args (override env vars)
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)      PORT="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --mem-frac)  MEM_FRACTION="$2"; shift 2 ;;
        --max-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  SGLang Server for AURA LocalVLMMonitor"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  Mem % : ${MEM_FRACTION}"
echo "═══════════════════════════════════════════════════════════"

# ── Create isolated venv if needed ────────────────────────────────────
if [ ! -d "${SGLANG_VENV}" ]; then
    echo "Creating isolated SGLang venv at ${SGLANG_VENV} …"
    uv venv "${SGLANG_VENV}" --python 3.12
    echo "Installing SGLang with all extras …"
    uv pip install --python "${SGLANG_VENV}/bin/python" "sglang[all]>=0.4" qwen-vl-utils
    # Upgrade cuDNN to 9.16+ (torch bundles 9.10 which has Conv3d bug with PyTorch 2.9.1)
    uv pip install --python "${SGLANG_VENV}/bin/python" --no-deps nvidia-cudnn-cu12==9.16.0.29
    echo "SGLang venv ready."
fi

# ── Launch the server ─────────────────────────────────────────────────
echo ""
echo "Starting SGLang server …"
echo "  API endpoint: http://localhost:${PORT}/v1"
echo "  Stop with Ctrl+C"
echo ""

# Use the upgraded cuDNN from the venv (9.16+) and skip the strict check
export LD_LIBRARY_PATH="${SGLANG_VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_DISABLE_CUDNN_CHECK=1

exec "${SGLANG_VENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${MAX_MODEL_LEN}" \
    --dtype auto \
    --trust-remote-code \
    --enable-multimodal \
    --keep-mm-feature-on-device

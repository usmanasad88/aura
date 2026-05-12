#!/bin/bash
# ============================================================
# Download demo_data/ for the AURA framework.
#
# Fetches the demo videos + reference images from the public
# Google Drive folder and places them under demo_data/ in the
# repo root, producing:
#
#   demo_data/
#     layup_demo/    layup_gesture_demo*.mp4, anchor_image_layup_stationary.png
#     tea/           tea_making*.mp4
#     sorting/       ...
#     weigh_bottles/ ...
#
# Drive folders cannot be enumerated by wget, so this script
# uses gdown (pip install gdown) for the folder traversal.
#
# Usage:
#   ./scripts/download_demo_data.sh           # default location
#   ./scripts/download_demo_data.sh --force   # re-download / overwrite
# ============================================================

set -euo pipefail

DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1baUVBFkgLUW8HMS8z6C7hPzUAPxtRd_U"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/demo_data"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=true ;;
        --help|-h)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ -d "$DEST" ]] && [[ "$FORCE" != "true" ]]; then
    echo "$DEST already exists. Re-run with --force to overwrite."
    exit 0
fi

# Ensure gdown is available (preferred) or fall back to pip-installing it.
if ! command -v gdown &>/dev/null; then
    echo "gdown not found. Installing via pip..."
    if command -v uv &>/dev/null; then
        uv pip install --quiet gdown
    else
        python3 -m pip install --quiet --user gdown
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

if [[ "$FORCE" == "true" ]] && [[ -d "$DEST" ]]; then
    echo "Removing existing $DEST ..."
    rm -rf "$DEST"
fi

mkdir -p "$DEST"

echo "Downloading demo_data from $DRIVE_FOLDER_URL ..."
echo "Destination: $DEST"

# gdown --folder pulls every file in the public Drive folder, preserving the
# folder's internal directory structure. -O sets the output directory.
gdown --folder "$DRIVE_FOLDER_URL" -O "$DEST" --remaining-ok

# Some Drive folder dumps land their contents inside an extra wrapping
# directory (named after the folder on Drive). Flatten that case so the
# final layout is demo_data/layup_demo/, demo_data/tea/, etc.
shopt -s nullglob
entries=("$DEST"/*)
if [[ ${#entries[@]} -eq 1 ]] && [[ -d "${entries[0]}" ]]; then
    inner="${entries[0]}"
    echo "Flattening single wrapper directory: $(basename "$inner")"
    mv "$inner"/* "$DEST"/ 2>/dev/null || true
    mv "$inner"/.[!.]* "$DEST"/ 2>/dev/null || true
    rmdir "$inner" 2>/dev/null || true
fi

echo "Done. demo_data/ is ready at $DEST"
ls -1 "$DEST"

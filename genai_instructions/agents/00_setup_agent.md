# Agent 00: Setup Agent

## Task: Initialize Development Environment

### Objective
Set up the complete development environment for the AURA project, including:
- Python virtual environment with uv
- All required dependencies
- Directory structure
- Validation tests

### Prerequisites
- Python 3.12+ installed
- `uv` package manager installed (`pip install uv`)
- GEMINI_API_KEY environment variable set

### Steps

#### Step 1: Verify uv Installation
```bash
uv --version
# Should show uv 0.x.x or higher
```

If not installed:
```bash
pip install uv
```

#### Step 2: Navigate to Project
```bash
cd /home/mani/Repos/aura
```

#### Step 3: Sync Dependencies
```bash
uv sync
```

This will:
- Create/update `.venv/` virtual environment
- Install all dependencies from `pyproject.toml`
- Install `sam3` from `third_party/sam3` as editable

#### Step 4: Verify Installation
```bash
# Activate environment
source .venv/bin/activate

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from google import genai; print('Gemini SDK OK')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

#### Step 5: Create Missing Directories
Run the setup script (or execute these commands):

```bash
cd /home/mani/Repos/aura

# Create source directories
mkdir -p src/aura/brain
mkdir -p src/aura/monitors
mkdir -p src/aura/actions
mkdir -p src/aura/interfaces
mkdir -p src/aura/visualization
mkdir -p src/aura/utils
mkdir -p src/aura/core

# Create config and data directories
mkdir -p config
mkdir -p sops
mkdir -p tests/test_core
mkdir -p tests/test_monitors
mkdir -p tests/test_brain
mkdir -p tests/integration
mkdir -p scripts

# Create handoff directory for agent notes
mkdir -p genai_instructions/handoff
mkdir -p genai_instructions/validation
```

#### Step 6: Create __init__.py Files
Ensure all packages have `__init__.py`:

```bash
touch src/aura/brain/__init__.py
touch src/aura/monitors/__init__.py
touch src/aura/actions/__init__.py
touch src/aura/interfaces/__init__.py
touch src/aura/visualization/__init__.py
touch src/aura/utils/__init__.py
touch src/aura/core/__init__.py
touch tests/__init__.py
touch tests/test_core/__init__.py
touch tests/test_monitors/__init__.py
touch tests/test_brain/__init__.py
touch tests/integration/__init__.py
```

### Validation Checklist

- [ ] `uv sync` completes without errors
- [ ] All imports in Step 4 succeed
- [ ] Directory structure matches Documentation.md
- [ ] SAM3 model can be loaded (may take time on first run)

### Expected Output

After completion, `tree src/aura/` should show:
```
src/aura/
├── __init__.py
├── brain/
│   └── __init__.py
├── core/
│   └── __init__.py
├── monitors/
│   └── __init__.py
├── actions/
│   └── __init__.py
├── interfaces/
│   └── __init__.py
├── visualization/
│   └── __init__.py
└── utils/
    └── __init__.py
```

### Troubleshooting

#### CUDA Issues
If PyTorch CUDA is not detected:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### SAM3 Import Errors
If SAM3 fails to import, ensure it's properly linked:
```bash
uv pip install -e third_party/sam3
```

#### Gemini API Key
Ensure the key is set:
```bash
export GEMINI_API_KEY="your-key-here"
echo $GEMINI_API_KEY  # Should print your key
```

### Next Steps
After completing this setup:
1. Run Task 1.1: Core Types Agent
2. The environment is now ready for development

### Handoff Notes
Create `genai_instructions/handoff/00_setup.md` with:
- Any issues encountered during setup
- Actual versions of key packages
- Any deviations from these instructions

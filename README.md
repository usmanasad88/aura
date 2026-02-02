# Aura

A framework for proactive robotic assistance.

## Overview

This repository contains the Aura project, which aims to create a generic theoretical and programmatic framework for proactive robotic assistance.
## Features

- **Multi-Modal Perception**: Vision (SAM3), sound (Gemini Live), and gesture recognition (MediaPipe)
- **Motion Prediction**: Hand tracking and trajectory prediction
- **Intent Recognition**: Detect human intent through gestures, motion, and context
- **Safety Control**: Real-time safety monitoring with gesture-based stop/resume
- **Modular Architecture**: Extensible monitor system with event bus
- **Robot Integration**: Support for ROS 2, Isaac Sim, and real robots

## Quick Start

### Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Or if you need to add packages
uv add <package-name>
```

### Test Gesture Recognition

```bash
# Run gesture recognition test
python scripts/test_gesture_monitor.py --show_viz
```

See [GESTURE_INTEGRATION.md](GESTURE_INTEGRATION.md) for detailed gesture recognition setup and usage.
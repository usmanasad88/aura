"""AURA - Autonomous Understanding for Robotic Assistance.

A modular, explainable framework for proactive robotic assistance.
"""

__version__ = "0.1.0"


def main() -> None:
    """Entry point for the aura command."""
    print(f"AURA v{__version__}")
    print("Proactive Robotic Assistance Framework")
    print()
    print("Available commands:")
    print("  uv run python scripts/run_game_demo.py    - Run game demo")
    print("  uv run python scripts/test_perception.py  - Test perception")
    print("  uv run python scripts/test_sound_monitor.py - Test sound")
    print()
    print("See genai_instructions/Documentation.md for more info.")

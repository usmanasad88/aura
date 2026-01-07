# Agent 13: Game Interface Agent

## Task: Create Interface for Collaborative Game Integration

### Objective
Create a bridge between AURA and the collaborative game from proactive_hcdt, enabling AURA to control the AI agent and observe game state.

### Prerequisites
- Sprint 1-3 complete (core types, monitors, brain basics)
- Collaborative game available at `/home/mani/Repos/proactive_hcdt/`

### Reference
- Game documentation: `/home/mani/Repos/proactive_hcdt/tests/COLLABORATIVE_GAME_README.md`
- Game implementation: `/home/mani/Repos/proactive_hcdt/tests/collaborative_game.py`

### Key Integration Points

The game uses file-based communication:
1. **Commands**: Write to `ai_commands.txt`
2. **Status**: Read from `ai_status.txt`
3. **Runtime State**: Read from `ai_runtime_status.json`
4. **Reasoning Display**: Write to `ai_reasoning.txt`

### Files to Create

#### 1. `src/aura/interfaces/game_interface.py`

```python
"""Interface for collaborative game integration."""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from enum import Enum

from aura.core import (
    TrackedObject, Pose2D, Action, RobotActionType, ActionStatus
)


logger = logging.getLogger(__name__)


class GameObjectType(Enum):
    """Types of objects in the game."""
    BOX = "box"
    T_SHAPE = "t_shape"
    L_SHAPE = "l_shape"


@dataclass
class GameObject:
    """An object in the game world."""
    id: str
    name: str
    object_type: GameObjectType
    position: Pose2D
    in_goal: Optional[str] = None  # "Blue Goal", "Green Goal", or None
    is_scored: bool = False
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameObject":
        """Create from game JSON."""
        return cls(
            id=data.get("id", data.get("name", "")),
            name=data.get("name", ""),
            object_type=GameObjectType.BOX,  # Simplified
            position=Pose2D(
                x=data.get("x", data.get("position", {}).get("x", 0)),
                y=data.get("y", data.get("position", {}).get("y", 0))
            ),
            in_goal=data.get("in_goal"),
            is_scored=data.get("in_goal") is not None
        )


@dataclass
class AgentState:
    """State of an agent (human or AI)."""
    name: str
    position: Pose2D
    velocity: Pose2D = field(default_factory=lambda: Pose2D(0, 0))
    score: int = 0
    closest_object: Optional[str] = None
    goal_distance: float = 0.0


@dataclass
class GameState:
    """Complete game state."""
    tick: int = 0
    human_agent: Optional[AgentState] = None
    ai_agent: Optional[AgentState] = None
    objects: List[GameObject] = field(default_factory=list)
    human_score: int = 0
    ai_score: int = 0
    is_complete: bool = False
    
    # AI action state
    current_action: Optional[str] = None
    action_status: str = "idle"
    ready_for_command: bool = True
    
    # Last update time
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        """Parse from ai_runtime_status.json."""
        state = cls()
        
        state.tick = data.get("tick", 0)
        state.human_score = data.get("human_score", 0)
        state.ai_score = data.get("ai_score", 0)
        state.ready_for_command = data.get("ready_for_command", True)
        state.action_status = data.get("status", "idle")
        state.is_complete = data.get("game_complete", False)
        
        # Parse current action
        if "current_action" in data:
            action_data = data["current_action"]
            if action_data:
                state.current_action = action_data.get("command", "")
        
        # Parse objects
        for obj_data in data.get("objects", []):
            state.objects.append(GameObject.from_dict(obj_data))
        
        # Parse agents
        if "human" in data:
            human = data["human"]
            state.human_agent = AgentState(
                name="human",
                position=Pose2D(
                    x=human.get("x", human.get("position", {}).get("x", 0)),
                    y=human.get("y", human.get("position", {}).get("y", 0))
                ),
                velocity=Pose2D(
                    x=human.get("vx", 0),
                    y=human.get("vy", 0)
                ),
                score=state.human_score,
                closest_object=human.get("closest_object"),
                goal_distance=human.get("goal_distance", 0)
            )
        
        if "ai" in data:
            ai = data["ai"]
            state.ai_agent = AgentState(
                name="ai",
                position=Pose2D(
                    x=ai.get("x", ai.get("position", {}).get("x", 0)),
                    y=ai.get("y", ai.get("position", {}).get("y", 0))
                ),
                score=state.ai_score
            )
        
        state.timestamp = datetime.now()
        return state
    
    def get_unscored_objects(self) -> List[GameObject]:
        """Get objects not yet in a goal."""
        return [obj for obj in self.objects if not obj.is_scored]
    
    def get_objects_for_ai(self) -> List[GameObject]:
        """Get objects the AI should target (not in Green Goal)."""
        return [obj for obj in self.objects 
                if obj.in_goal != "Green Goal"]


class GameInterface:
    """Interface for communicating with the collaborative game.
    
    Handles:
    - Reading game state from JSON
    - Sending commands via text file
    - Displaying reasoning in game UI
    - Monitoring action execution
    """
    
    def __init__(
        self, 
        game_dir: str = "/home/mani/Repos/proactive_hcdt",
        poll_interval: float = 0.2
    ):
        self.game_dir = Path(game_dir)
        self.poll_interval = poll_interval
        
        # File paths
        self.commands_file = self.game_dir / "ai_commands.txt"
        self.status_file = self.game_dir / "ai_status.txt"
        self.runtime_file = self.game_dir / "ai_runtime_status.json"
        self.reasoning_file = self.game_dir / "ai_reasoning.txt"
        
        self._last_state: Optional[GameState] = None
        self._is_monitoring = False
        self._state_callbacks: List[Callable[[GameState], None]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Verify game directory exists
        if not self.game_dir.exists():
            logger.warning(f"Game directory not found: {self.game_dir}")
    
    def add_state_callback(self, callback: Callable[[GameState], None]):
        """Add callback for state updates."""
        self._state_callbacks.append(callback)
    
    def read_state(self) -> Optional[GameState]:
        """Read current game state from JSON file."""
        try:
            if not self.runtime_file.exists():
                return None
            
            with open(self.runtime_file, 'r') as f:
                data = json.load(f)
            
            state = GameState.from_json(data)
            self._last_state = state
            return state
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse game state: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read game state: {e}")
            return None
    
    def send_command(self, command: str, append: bool = False):
        """Send command to the game.
        
        Args:
            command: Command string (e.g., "push Box-1 to Green Goal")
            append: If True, append to queue; if False, replace
        """
        mode = 'a' if append else 'w'
        with open(self.commands_file, mode) as f:
            f.write(command + '\n')
        
        logger.info(f"Sent command: {command}")
    
    def send_reasoning(self, text: str):
        """Display reasoning in the game UI.
        
        Args:
            text: Reasoning text to display
        """
        with open(self.reasoning_file, 'w') as f:
            f.write(text)
    
    def stop_action(self):
        """Send stop command to interrupt current action."""
        self.send_command("stop")
    
    def observe(self):
        """Request world observation."""
        self.send_command("observe")
    
    def push_object(self, object_name: str, goal: str = "Green Goal"):
        """Command to push an object to a goal.
        
        Args:
            object_name: Name of object (e.g., "Box-1", "T-1")
            goal: Target goal (default: "Green Goal" for AI)
        """
        self.send_command(f"push {object_name} to {goal}")
    
    def move_to(self, x: int, y: int):
        """Command to move to a position.
        
        Args:
            x, y: Target coordinates
        """
        self.send_command(f"move to {x} {y}")
    
    def say(self, message: str):
        """Display a speech bubble message.
        
        Args:
            message: Message to display
        """
        self.send_command(f"say {message}")
    
    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """Wait until the AI is ready for a new command.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if ready, False if timeout
        """
        start = time.time()
        
        while time.time() - start < timeout:
            state = self.read_state()
            if state and state.ready_for_command:
                return True
            await asyncio.sleep(self.poll_interval)
        
        logger.warning("Timeout waiting for command readiness")
        return False
    
    async def execute_and_wait(
        self, 
        command: str, 
        timeout: float = 30.0
    ) -> bool:
        """Execute command and wait for completion.
        
        Args:
            command: Command to execute
            timeout: Maximum wait time
            
        Returns:
            True if completed successfully
        """
        # Wait for ready first
        if not await self.wait_for_ready(timeout=5.0):
            logger.warning("Not ready to execute command")
            return False
        
        # Send command
        self.send_command(command)
        
        # Wait for completion
        start = time.time()
        initial_tick = self._last_state.tick if self._last_state else 0
        
        while time.time() - start < timeout:
            state = self.read_state()
            if not state:
                await asyncio.sleep(self.poll_interval)
                continue
            
            # Check if action completed (ready again after starting)
            if state.tick > initial_tick + 10 and state.ready_for_command:
                return True
            
            # Check for failure
            if state.action_status == "failed":
                logger.warning(f"Action failed: {command}")
                return False
            
            await asyncio.sleep(self.poll_interval)
        
        logger.warning(f"Timeout executing: {command}")
        return False
    
    async def start_monitoring(self):
        """Start background state monitoring."""
        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started game state monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped game state monitoring")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._is_monitoring:
            try:
                state = self.read_state()
                if state:
                    for callback in self._state_callbacks:
                        try:
                            callback(state)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1.0)
    
    @property
    def last_state(self) -> Optional[GameState]:
        """Get the most recently read game state."""
        return self._last_state
    
    def is_game_running(self) -> bool:
        """Check if game seems to be running."""
        if not self.runtime_file.exists():
            return False
        
        # Check if file was updated recently
        try:
            mtime = self.runtime_file.stat().st_mtime
            age = time.time() - mtime
            return age < 5.0  # Updated within last 5 seconds
        except:
            return False


# ============================================================================
# AURA Integration Helpers
# ============================================================================

def game_objects_to_tracked(game_objects: List[GameObject]) -> List[TrackedObject]:
    """Convert game objects to AURA TrackedObjects."""
    tracked = []
    for obj in game_objects:
        tracked.append(TrackedObject(
            id=obj.id,
            name=obj.name,
            category=obj.object_type.value,
            pose=None,  # 2D game, no 3D pose
            metadata={
                "x": obj.position.x,
                "y": obj.position.y,
                "in_goal": obj.in_goal,
                "is_scored": obj.is_scored
            }
        ))
    return tracked


def create_push_action(object_name: str, goal: str = "Green Goal") -> Action:
    """Create a push action for the game."""
    return Action(
        id=f"push_{object_name}_{int(time.time()*1000)}",
        type=RobotActionType.MOVE_TO_POSE,  # Closest match
        parameters={
            "command": f"push {object_name} to {goal}",
            "object": object_name,
            "goal": goal
        }
    )
```

#### 2. `scripts/run_game_demo.py`

```python
#!/usr/bin/env python
"""Demo script for AURA controlling the collaborative game."""

import asyncio
import argparse
import logging
import sys
from datetime import datetime

from aura.interfaces.game_interface import (
    GameInterface, GameState, GameObject
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleGameAgent:
    """Simple AI agent for the collaborative game.
    
    Strategy:
    1. Wait for human to start moving
    2. Pick objects not targeted by human
    3. Push to Green Goal
    4. Avoid collisions with human
    """
    
    def __init__(self, game: GameInterface):
        self.game = game
        self.completed_objects = set()
        self.current_target = None
    
    def select_target(self, state: GameState) -> Optional[GameObject]:
        """Select next object to push."""
        available = state.get_objects_for_ai()
        
        # Filter out already completed
        available = [o for o in available if o.name not in self.completed_objects]
        
        if not available:
            return None
        
        # Avoid object human is targeting
        human_target = None
        if state.human_agent and state.human_agent.closest_object:
            human_target = state.human_agent.closest_object
        
        # Prefer objects not targeted by human
        preferred = [o for o in available if o.name != human_target]
        if preferred:
            available = preferred
        
        # Sort by distance to AI (if we have AI position)
        if state.ai_agent:
            ai_x, ai_y = state.ai_agent.position.x, state.ai_agent.position.y
            available.sort(key=lambda o: (
                (o.position.x - ai_x)**2 + (o.position.y - ai_y)**2
            ))
        
        return available[0] if available else None
    
    async def think_and_act(self, state: GameState):
        """Make a decision and act."""
        # Check if current action completed
        if self.current_target:
            obj = next((o for o in state.objects if o.name == self.current_target), None)
            if obj and obj.in_goal == "Green Goal":
                logger.info(f"✓ Completed: {self.current_target}")
                self.completed_objects.add(self.current_target)
                self.current_target = None
        
        # If not ready, wait
        if not state.ready_for_command:
            return
        
        # Select new target
        target = self.select_target(state)
        if not target:
            self.game.send_reasoning("All objects scored or being handled by human!")
            return
        
        self.current_target = target.name
        
        # Explain reasoning
        reasoning = f"Targeting {target.name} at ({target.position.x:.0f}, {target.position.y:.0f})"
        if state.human_agent and state.human_agent.closest_object:
            reasoning += f". Human is near {state.human_agent.closest_object}."
        
        self.game.send_reasoning(reasoning)
        logger.info(f"Strategy: {reasoning}")
        
        # Execute action
        self.game.push_object(target.name, "Green Goal")
    
    async def run(self, max_duration: float = 300.0):
        """Run the agent."""
        logger.info("Starting Simple Game Agent")
        logger.info("Waiting for game to start...")
        
        # Wait for game
        while not self.game.is_game_running():
            await asyncio.sleep(1.0)
        
        logger.info("Game detected! Starting AI control.")
        self.game.send_reasoning("AI Agent starting up... analyzing scene.")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_duration:
                    logger.info("Max duration reached")
                    break
                
                state = self.game.read_state()
                if not state:
                    await asyncio.sleep(0.5)
                    continue
                
                if state.is_complete:
                    logger.info("🎉 Game complete!")
                    self.game.send_reasoning("Victory! All objects sorted.")
                    break
                
                await self.think_and_act(state)
                await asyncio.sleep(0.3)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        # Summary
        logger.info(f"Session complete. AI scored: {len(self.completed_objects)} objects")


async def monitor_only(game: GameInterface):
    """Just monitor and display game state."""
    logger.info("Monitoring game state (no actions)...")
    logger.info("Press Ctrl+C to stop.")
    
    last_tick = -1
    
    try:
        while True:
            state = game.read_state()
            if state and state.tick != last_tick:
                last_tick = state.tick
                
                print(f"\n--- Tick {state.tick} ---")
                print(f"Score: Human {state.human_score} | AI {state.ai_score}")
                print(f"Ready: {state.ready_for_command} | Status: {state.action_status}")
                
                if state.human_agent:
                    h = state.human_agent
                    print(f"Human: ({h.position.x:.0f}, {h.position.y:.0f}) "
                          f"targeting {h.closest_object}")
                
                print("Objects:")
                for obj in state.objects:
                    status = f"in {obj.in_goal}" if obj.in_goal else "unscored"
                    print(f"  - {obj.name}: ({obj.position.x:.0f}, {obj.position.y:.0f}) [{status}]")
            
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="AURA Game Demo")
    parser.add_argument("--mode", type=str, default="agent",
                       choices=["agent", "monitor"],
                       help="Run mode: agent (control AI) or monitor (observe only)")
    parser.add_argument("--game-dir", type=str, 
                       default="/home/mani/Repos/proactive_hcdt",
                       help="Path to game directory")
    parser.add_argument("--duration", type=float, default=300.0,
                       help="Maximum run duration (seconds)")
    
    args = parser.parse_args()
    
    game = GameInterface(game_dir=args.game_dir)
    
    print("=" * 50)
    print("AURA Game Demo")
    print("=" * 50)
    print(f"Game directory: {args.game_dir}")
    print(f"Mode: {args.mode}")
    print()
    print("Make sure the game is running:")
    print(f"  cd {args.game_dir}")
    print("  conda run -n ur5_python python tests/collaborative_game.py")
    print()
    
    if args.mode == "agent":
        agent = SimpleGameAgent(game)
        asyncio.run(agent.run(max_duration=args.duration))
    else:
        asyncio.run(monitor_only(game))


if __name__ == "__main__":
    main()
```

### Validation

```bash
# Terminal 1: Start the game
cd /home/mani/Repos/proactive_hcdt
conda run -n ur5_python python tests/collaborative_game.py

# Terminal 2: Run AURA agent
cd /home/mani/Repos/aura
uv run python scripts/run_game_demo.py --mode agent

# Or just monitor
uv run python scripts/run_game_demo.py --mode monitor
```

### Expected Behavior

1. Agent waits for game to start
2. Reads game state from JSON
3. Selects object not targeted by human
4. Sends push command
5. Displays reasoning in game UI
6. Continues until all objects scored

### Handoff Notes

Create `genai_instructions/handoff/13_game_interface.md`:

```markdown
# Game Interface Complete

## Files Created
- src/aura/interfaces/game_interface.py
- scripts/run_game_demo.py

## Integration Points
- ai_commands.txt: Write commands
- ai_runtime_status.json: Read state
- ai_reasoning.txt: Display reasoning

## Key Classes
- GameInterface: File-based communication
- GameState: Parsed game state
- SimpleGameAgent: Basic decision making

## Next Steps
- Integrate with full AURA Brain
- Add intent prediction for human
- Add LLM reasoning for decisions
```

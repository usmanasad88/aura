"""DAG-driven Decision Engine for the Hand Layup task.

Converts intent analysis results into robot commands and voice instructions
by reading the task DAG and tracking which steps have been completed.

The DAG nodes may contain:
- ``robot_return_to_storage``: objects the robot should return when this step completes
- ``objects_needed_on_workplace``: objects needed for the current step (used for proactive delivery)

Usage::

    from tasks.hand_layup.decision_engine import HandLayupDecisionEngine

    engine = HandLayupDecisionEngine(dag_path="tasks/hand_layup/config/hand_layup_dag.json")
    # ... in your loop:
    engine.update(intent_result)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default DAG path relative to this file
_TASK_DIR = Path(__file__).parent
_DEFAULT_DAG_PATH = _TASK_DIR / "config" / "hand_layup_dag.json"

# Default directory where .prog files live (fallback when API is unavailable)
_DEFAULT_PROGRAMS_DIR = Path.home() / "ur5-robotiq-ros2-control" / "src" / "ur5_curobo_control" / "programs"

# Phases where gloves are required
_RESIN_PHASES = {
    "resin_preparation", "mixing",
    "layer_1_placement", "layer_1_resin",
    "layer_2_placement", "layer_2_resin",
    "layer_3_placement", "layer_3_resin",
    "layer_4_placement", "layer_4_resin",
    "consolidation",
}

# Objects the robot can move (excludes mold and fiberglass_sheet)
_MOVABLE_OBJECTS = {
    "weigh_scale", "resin_bottle", "hardener_bottle",
    "cup", "brush_small", "brush_medium", "roller",
}

# Objects the robot delivers from storage at the start and returns when done
_ROBOT_MANAGED_OBJECTS = {"resin_bottle", "hardener_bottle", "roller"}

# Static mapping: (action_type, object_name) → .prog file name
# Used as fallback when the robot API is not available for discovery.
_PROGRAM_MAP: Dict[tuple, str] = {
    ("return_to_storage", "resin_bottle"): "move_resin_from_workplace_to_storage.prog",
    ("return_to_storage", "hardener_bottle"): "move_hardener_from_workplace_to_storage.prog",
    ("return_to_storage", "roller"): "move_roller_from_workplace_to_storage.prog",
    ("deliver_to_workplace", "resin_bottle"): "move_resin_from_storage_to_workplace.prog",
    ("deliver_to_workplace", "hardener_bottle"): "move_hardener_from_storage_to_workplace.prog",
    ("deliver_to_workplace", "roller"): "move_roller_from_storage_to_workplace.prog",
}


@dataclass
class RobotAction:
    """A single robot action to execute."""
    action_type: str          # "return_to_storage" | "deliver_to_workplace"
    object_name: str          # e.g. "resin_bottle"
    trigger_step: str         # DAG step that triggered this action
    reason: str
    timestamp: float = 0.0
    executed: bool = False
    success: bool = False
    api_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceMessage:
    """A voice message to speak."""
    text: str
    priority: str = "normal"  # "normal" | "warning" | "critical"
    timestamp: float = 0.0


class HandLayupDecisionEngine:
    """DAG-driven decision engine for the hand layup task.

    Reads ``robot_return_to_storage`` and ``objects_needed_on_workplace``
    from the DAG to decide when the robot should move objects and what
    voice announcements to issue.
    """

    def __init__(
        self,
        dag_path: Optional[str] = None,
        robot_client=None,
        on_voice: Optional[Callable[[str], None]] = None,
        dry_run: bool = True,
        programs_dir: Optional[str] = None,
    ):
        """
        Args:
            dag_path: Path to hand_layup_dag.json.
            robot_client: A ``RobotControlClient`` instance for dispatching
                commands.  ``None`` means actions are only logged.
            on_voice: Callback ``(text) -> None`` invoked for each voice
                announcement.  If None, messages are only logged.
            dry_run: When True, robot commands are logged but not dispatched.
            programs_dir: Path to the directory containing ``.prog`` files.
                Used as fallback when the robot API is not available.
        """
        self.robot_client = robot_client
        self.on_voice = on_voice
        self.dry_run = dry_run
        self.programs_dir = Path(programs_dir) if programs_dir else _DEFAULT_PROGRAMS_DIR

        # Load DAG
        dp = Path(dag_path) if dag_path else _DEFAULT_DAG_PATH
        with open(dp, "r") as f:
            self.dag: Dict[str, Any] = json.load(f)
        self.nodes: Dict[str, Dict[str, Any]] = self.dag.get("nodes", {})

        # Build ordered step list from the DAG traversal
        self.step_order = self._build_step_order()

        # Extract return-to-storage triggers:  step_name → list of objects
        self.return_triggers: Dict[str, List[str]] = {}
        for step_name, node in self.nodes.items():
            rts = node.get("robot_return_to_storage")
            if rts and rts.get("objects"):
                self.return_triggers[step_name] = rts["objects"]

        # Discover available programs (API first, then filesystem fallback)
        self._available_programs: Set[str] = set()
        self._discover_programs()

        # State tracking
        self.completed_steps: Set[str] = set()
        # Robot-managed objects start in storage; others assumed on workplace
        self.object_locations: Dict[str, str] = {}
        for obj in _MOVABLE_OBJECTS:
            self.object_locations[obj] = "storage" if obj in _ROBOT_MANAGED_OBJECTS else "workplace"

        self.executed_actions: List[RobotAction] = []
        self.voice_log: List[VoiceMessage] = []
        self._pending_actions: List[RobotAction] = []
        self._initial_delivery_queued: bool = False
        self._gloves_warned: bool = False
        self._shelf_life_warned: bool = False
        self._mixture_start_time: Optional[float] = None

        logger.info(
            "HandLayupDecisionEngine loaded DAG with %d nodes, "
            "%d return-to-storage triggers: %s",
            len(self.nodes),
            len(self.return_triggers),
            {k: v for k, v in self.return_triggers.items()},
        )
        logger.info(
            "Available programs (%d): %s",
            len(self._available_programs),
            sorted(self._available_programs),
        )

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    def _build_step_order(self) -> List[str]:
        """Walk the DAG from start_node following next_possible to get order."""
        order = []
        visited = set()
        current = self.dag.get("start_node", "idle")
        while current and current not in visited:
            visited.add(current)
            order.append(current)
            node = self.nodes.get(current, {})
            nexts = node.get("next_possible", [])
            current = nexts[0] if nexts else None
        return order

    def _next_step(self, step_name: str) -> Optional[str]:
        """Return the step immediately after *step_name* in the DAG order."""
        try:
            idx = self.step_order.index(step_name)
            if idx + 1 < len(self.step_order):
                return self.step_order[idx + 1]
        except ValueError:
            pass
        return None

    # ------------------------------------------------------------------
    # Program discovery & robot status
    # ------------------------------------------------------------------

    def _discover_programs(self) -> None:
        """Populate ``_available_programs`` from the robot API or filesystem.

        1. If a ``robot_client`` is set and reachable, query ``/api/commands``
           to get the list of ``.prog`` files the API knows about.
        2. Otherwise, scan ``self.programs_dir`` for ``.prog`` files.
        3. As a last resort, use the keys from ``_PROGRAM_MAP``.
        """
        # Try API first
        if self.robot_client is not None:
            try:
                cmds = self.robot_client.get_commands()
                programs = cmds.get("programs", [])
                if programs:
                    self._available_programs = {
                        p["name"] for p in programs if isinstance(p, dict) and "name" in p
                    }
                    logger.info(
                        "Discovered %d programs from robot API",
                        len(self._available_programs),
                    )
                    return
            except Exception as e:
                logger.warning("Could not query robot API for programs: %s", e)

        # Fallback: scan filesystem
        if self.programs_dir.is_dir():
            self._available_programs = {
                p.name for p in self.programs_dir.glob("*.prog")
            }
            logger.info(
                "Discovered %d programs from filesystem (%s)",
                len(self._available_programs),
                self.programs_dir,
            )
            return

        # Last resort: use static map values
        self._available_programs = set(_PROGRAM_MAP.values())
        logger.warning(
            "No API or programs directory available — using static program map (%d programs)",
            len(self._available_programs),
        )

    def _resolve_program(self, action: RobotAction) -> Optional[str]:
        """Resolve a ``RobotAction`` to a ``.prog`` file name.

        Checks the static ``_PROGRAM_MAP`` first, then verifies the program
        exists in the discovered set.  Returns ``None`` if no match.
        """
        key = (action.action_type, action.object_name)
        prog = _PROGRAM_MAP.get(key)
        if prog and prog in self._available_programs:
            return prog
        if prog:
            logger.warning(
                "Program %s (for %s) is in the static map but not available on the robot",
                prog, key,
            )
        else:
            logger.warning("No program mapping for action %s", key)
        return prog  # return it anyway; execution will report the error

    def _is_robot_busy(self) -> bool:
        """Query the robot API to check if the executor is currently running.

        Returns ``False`` (not busy) if the API is unreachable or in dry-run
        mode, so that queued actions are not blocked indefinitely.
        """
        if self.dry_run or self.robot_client is None:
            return False
        try:
            status = self.robot_client.get_status()
            busy = status.get("executor_running", False)
            if busy:
                logger.debug("Robot is busy (executor_running=True)")
            return busy
        except Exception as e:
            logger.warning("Could not query robot status: %s — assuming not busy", e)
            return False

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, intent_result) -> List[RobotAction]:
        """Process an IntentResult and execute any resulting robot actions.

        Args:
            intent_result: An ``IntentResult`` from ``HandLayupIntentMonitor``.

        Returns:
            List of ``RobotAction`` objects that were queued (and possibly
            executed) during this update.
        """
        actions: List[RobotAction] = []
        timestamp = intent_result.timestamp

        # --- 0. Initial delivery: bring objects from storage to workplace ---
        if not self._initial_delivery_queued:
            self._initial_delivery_queued = True
            actions.extend(self._queue_initial_delivery(timestamp))

        # --- 1. Try to execute pending actions from previous updates ---
        self._drain_pending_actions()

        # --- 2. Detect newly completed steps ---
        new_completions = set(intent_result.steps_completed) - self.completed_steps
        if new_completions:
            # Process in DAG order
            for step in self.step_order:
                if step in new_completions:
                    logger.info("Step completed: %s", step)
                    self.completed_steps.add(step)
                    actions.extend(self._check_return_to_storage(step, timestamp))

        # --- 3. Proactive delivery for upcoming steps ---
        actions.extend(self._check_proactive_delivery(intent_result, timestamp))

        # --- 4. Safety checks ---
        self._check_safety(intent_result, timestamp)

        # --- 5. Mixture shelf-life tracking ---
        if intent_result.mixture_mixed and self._mixture_start_time is None:
            self._mixture_start_time = time.time()

        if self._mixture_start_time is not None:
            elapsed_min = (time.time() - self._mixture_start_time) / 60.0
            if elapsed_min > 20 and not self._shelf_life_warned:
                self._shelf_life_warned = True
                self._say(
                    f"Warning: mixture has been mixed for {elapsed_min:.0f} minutes. "
                    "Shelf life limit is 30 minutes.",
                    priority="warning",
                    timestamp=timestamp,
                )

        # --- 6. Execute new actions (or queue them if robot is busy) ---
        for action in actions:
            self._execute_action(action)

        # --- 7. Status announcement on new step in-progress ---
        if intent_result.steps_in_progress:
            current = intent_result.steps_in_progress[0]
            predicted = intent_result.predicted_next_action
            if predicted and predicted != "unknown":
                logger.info(
                    "Status — current: %s | next predicted: %s (%.0f%%)",
                    current, predicted,
                    intent_result.prediction_confidence * 100,
                )

        return actions

    # ------------------------------------------------------------------
    # Initial delivery
    # ------------------------------------------------------------------

    def _queue_initial_delivery(self, timestamp: float) -> List[RobotAction]:
        """Queue delivery of robot-managed objects from storage to workplace."""
        actions = []
        for obj in ("resin_bottle", "hardener_bottle", "roller"):
            if self.object_locations.get(obj) == "storage":
                action = RobotAction(
                    action_type="deliver_to_workplace",
                    object_name=obj,
                    trigger_step="idle",
                    reason="Initial setup — delivering to workplace",
                    timestamp=timestamp,
                )
                actions.append(action)
                # Mark as workplace immediately so proactive delivery doesn't
                # duplicate these actions before _execute_action runs.
                self.object_locations[obj] = "workplace"
                self._say(
                    f"Delivering {obj.replace('_', ' ')} from storage to workplace",
                    timestamp=timestamp,
                )
        return actions

    # ------------------------------------------------------------------
    # Pending action queue
    # ------------------------------------------------------------------

    def _drain_pending_actions(self) -> None:
        """Try to execute pending actions that were deferred because robot was busy."""
        if not self._pending_actions:
            return
        if self._is_robot_busy():
            logger.debug(
                "Robot still busy — %d actions pending", len(self._pending_actions)
            )
            return

        # Execute the next pending action (one at a time to avoid collisions)
        action = self._pending_actions.pop(0)
        logger.info(
            "Executing deferred action: %s %s", action.action_type, action.object_name
        )
        self._execute_action(action, from_queue=True)

    # ------------------------------------------------------------------
    # Return to storage
    # ------------------------------------------------------------------

    def _check_return_to_storage(
        self, step_name: str, timestamp: float
    ) -> List[RobotAction]:
        """Check if completing *step_name* triggers a return-to-storage."""
        objects_to_return = self.return_triggers.get(step_name, [])
        actions = []
        for obj in objects_to_return:
            if self.object_locations.get(obj) != "workplace":
                logger.debug(
                    "Skipping return of %s — not on workplace (loc=%s)",
                    obj, self.object_locations.get(obj),
                )
                continue

            node = self.nodes.get(step_name, {})
            reason = node.get("robot_return_to_storage", {}).get(
                "reason", f"{obj} no longer needed"
            )

            action = RobotAction(
                action_type="return_to_storage",
                object_name=obj,
                trigger_step=step_name,
                reason=reason,
                timestamp=timestamp,
            )
            actions.append(action)
            self._say(
                f"Returning {obj.replace('_', ' ')} to storage — {reason}",
                timestamp=timestamp,
            )
        return actions

    # ------------------------------------------------------------------
    # Proactive delivery
    # ------------------------------------------------------------------

    def _check_proactive_delivery(
        self, intent_result, timestamp: float
    ) -> List[RobotAction]:
        """Deliver objects needed for the next step that are in storage."""
        actions = []

        # Determine the next upcoming step
        predicted = intent_result.predicted_next_action
        if not predicted or predicted == "unknown":
            return actions

        next_node = self.nodes.get(predicted, {})
        needed = next_node.get("objects_needed_on_workplace", [])

        for obj in needed:
            if obj not in _MOVABLE_OBJECTS:
                continue
            if self.object_locations.get(obj) == "storage":
                action = RobotAction(
                    action_type="deliver_to_workplace",
                    object_name=obj,
                    trigger_step=predicted,
                    reason=f"Needed for upcoming step: {predicted}",
                    timestamp=timestamp,
                )
                actions.append(action)
                self._say(
                    f"Delivering {obj.replace('_', ' ')} to workplace — "
                    f"needed for {predicted.replace('_', ' ')}",
                    timestamp=timestamp,
                )
        return actions

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _check_safety(self, intent_result, timestamp: float) -> None:
        """Issue voice warnings for safety issues."""
        phase = intent_result.current_phase
        wearing_gloves = intent_result.human_wearing_gloves

        if phase in _RESIN_PHASES and not wearing_gloves and not self._gloves_warned:
            self._gloves_warned = True
            self._say(
                "Warning: gloves not detected during resin handling. "
                "Please wear protective gloves.",
                priority="warning",
                timestamp=timestamp,
            )
        elif wearing_gloves:
            # Reset so we can warn again if they take them off
            self._gloves_warned = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: RobotAction, from_queue: bool = False) -> None:
        """Dispatch a robot action.

        If the robot is currently busy executing another program, the action
        is added to ``_pending_actions`` and will be retried on the next
        ``update()`` call.
        """
        # Resolve the .prog file
        program = self._resolve_program(action)
        if program is None:
            logger.error(
                "No program found for %s %s — skipping",
                action.action_type, action.object_name,
            )
            action.executed = True
            action.success = False
            self.executed_actions.append(action)
            return

        # Check if robot is busy (skip check if this came from the queue —
        # _drain_pending_actions already verified the robot is free)
        if not from_queue and self._is_robot_busy():
            logger.info(
                "Robot busy — deferring %s (%s)", program, action.object_name
            )
            self._pending_actions.append(action)
            return

        if self.dry_run or self.robot_client is None:
            logger.info("[DRY-RUN] Would execute: %s", program)
            action.executed = True
            action.success = True
        else:
            try:
                resp = self.robot_client.execute_program(program)
                action.api_response = resp
                action.executed = True
                action.success = resp.get("success", False)
                logger.info(
                    "[ROBOT] %s → success=%s msg=%s",
                    program, action.success, resp.get("message", ""),
                )
            except Exception as e:
                logger.error("[ROBOT] %s failed: %s", program, e)
                action.executed = True
                action.success = False
                action.api_response = {"error": str(e)}

        # Update object location tracking
        if action.success:
            if action.action_type == "return_to_storage":
                self.object_locations[action.object_name] = "storage"
            elif action.action_type == "deliver_to_workplace":
                self.object_locations[action.object_name] = "workplace"

        self.executed_actions.append(action)

    # ------------------------------------------------------------------
    # Voice
    # ------------------------------------------------------------------

    def _say(
        self, text: str, priority: str = "normal", timestamp: float = 0.0
    ) -> None:
        """Issue a voice announcement."""
        msg = VoiceMessage(text=text, priority=priority, timestamp=timestamp)
        self.voice_log.append(msg)

        prefix = ""
        if priority == "warning":
            prefix = "[WARNING] "
        elif priority == "critical":
            prefix = "[CRITICAL] "
        logger.info("VOICE %s%s", prefix, text)

        if self.on_voice is not None:
            try:
                self.on_voice(text)
            except Exception as e:
                logger.error("Voice callback failed: %s", e)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Return a human-readable summary of all decisions made."""
        lines = ["=" * 60, "  Hand Layup Decision Engine — Summary", "=" * 60]

        lines.append(f"\nCompleted steps ({len(self.completed_steps)}):")
        for step in self.step_order:
            marker = "x" if step in self.completed_steps else " "
            lines.append(f"  [{marker}] {step}")

        lines.append(f"\nRobot actions executed ({len(self.executed_actions)}):")
        for a in self.executed_actions:
            status = "OK" if a.success else "FAIL"
            prog = self._resolve_program(a) or "?"
            lines.append(
                f"  [{status}] {a.action_type} {a.object_name} → {prog} "
                f"(trigger: {a.trigger_step}, t={a.timestamp:.1f}s)"
            )

        if self._pending_actions:
            lines.append(f"\nPending actions ({len(self._pending_actions)}):")
            for a in self._pending_actions:
                prog = self._resolve_program(a) or "?"
                lines.append(f"  {a.action_type} {a.object_name} → {prog}")

        lines.append(f"\nObject locations:")
        for obj, loc in sorted(self.object_locations.items()):
            lines.append(f"  {obj}: {loc}")

        lines.append(f"\nProgram mapping ({len(self._available_programs)} available):")
        for key, prog in sorted(_PROGRAM_MAP.items()):
            avail = "Y" if prog in self._available_programs else "N"
            lines.append(f"  {key[0]:20s} {key[1]:20s} → {prog} [{avail}]")

        lines.append(f"\nVoice messages ({len(self.voice_log)}):")
        for msg in self.voice_log:
            lines.append(f"  [{msg.priority}] {msg.text}")

        lines.append("=" * 60)
        return "\n".join(lines)

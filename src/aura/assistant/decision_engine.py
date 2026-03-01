"""Generic DAG-driven Decision Engine for any AURA task.

Reads a task_profile, dag, and dynamically makes decisions.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

@dataclass
class RobotAction:
    action_type: str
    object_name: str
    trigger_step: str
    reason: str
    timestamp: float = 0.0
    executed: bool = False
    success: bool = False
    api_response: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VoiceMessage:
    text: str
    priority: str = "normal"
    timestamp: float = 0.0

class AURADecisionEngine:
    """DAG-driven decision engine for generic tasks."""

    def __init__(
        self,
        config_dir: str,
        robot_client=None,
        on_voice: Optional[Callable[[str], None]] = None,
        dry_run: bool = True,
        programs_dir: Optional[str] = None,
    ):
        self.robot_client = robot_client
        self.on_voice = on_voice
        self.dry_run = dry_run
        
        # Determine fallback dir
        if programs_dir:
            self.programs_dir = Path(programs_dir)
        else:
            self.programs_dir = Path.home() / "ur5-robotiq-ros2-control" / "src" / "ur5_curobo_control" / "programs"

        self.config_dir = Path(config_dir)
        
        dag_path = self.config_dir / "dag.json"
        profile_path = self.config_dir / "task_profile.json"
        
        with open(dag_path, "r") as f:
            self.dag: Dict[str, Any] = json.load(f)
        self.nodes: Dict[str, Dict[str, Any]] = self.dag.get("nodes", {})

        self.task_profile = {}
        if profile_path.exists():
            with open(profile_path, "r") as f:
                self.task_profile = json.load(f)

        # Extract semantics from profile
        env = self.task_profile.get("environment", {})
        self.movable_objects = set(env.get("movable_objects", []))
        self.initial_delivery_objects = set(env.get("initial_delivery_objects", []))
        
        # Map: "action_type_object_name" -> "prog_name" e.g. "deliver_to_workplace_roller"
        raw_program_map = self.task_profile.get("program_map", {})
        self.program_map: Dict[tuple, str] = {}
        for k, v in raw_program_map.items():
            parts = k.split("|") # Format expected: "deliver_to_workplace|roller"
            if len(parts) == 2:
                self.program_map[(parts[0], parts[1])] = v

        self.step_order = self._build_step_order()

        self.return_triggers: Dict[str, List[str]] = {}
        for step_name, node in self.nodes.items():
            rts = node.get("robot_return_to_storage")
            if rts and rts.get("objects"):
                self.return_triggers[step_name] = rts["objects"]

        self._available_programs: Set[str] = set()
        self._discover_programs()

        self.completed_steps: Set[str] = set()
        self.object_locations: Dict[str, str] = {}
        for obj in self.movable_objects:
            self.object_locations[obj] = "storage" if obj in self.initial_delivery_objects else "workplace"

        self.executed_actions: List[RobotAction] = []
        self.voice_log: List[VoiceMessage] = []
        self._pending_actions: List[RobotAction] = []
        self._initial_delivery_queued: bool = False
        
        # State trackers for rule checking
        self._warned_rules = set()
        self._state_timers = {}

        self._log_dir: Optional[Path] = None
        self._update_counter: int = 0

    def set_log_dir(self, log_dir: Path | str) -> None:
        self._log_dir = Path(log_dir)

    def _build_step_order(self) -> List[str]:
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

    def _discover_programs(self) -> None:
        if self.robot_client is not None:
            try:
                cmds = self.robot_client.get_commands()
                programs = cmds.get("programs", [])
                if programs:
                    self._available_programs = {
                        p["name"] for p in programs if isinstance(p, dict) and "name" in p
                    }
                    return
            except Exception as e:
                logger.warning("Could not query robot API for programs: %s", e)

        if self.programs_dir.is_dir():
            self._available_programs = {p.name for p in self.programs_dir.glob("*.prog")}
            return

        self._available_programs = set(self.program_map.values())

    def _resolve_program(self, action: RobotAction) -> Optional[str]:
        key = (action.action_type, action.object_name)
        prog = self.program_map.get(key)
        if prog and prog in self._available_programs:
            return prog
        if prog:
            logger.warning("Program %s (for %s) map exists but is not available", prog, key)
        return prog

    def _is_robot_busy(self) -> bool:
        if self.dry_run or self.robot_client is None:
            return False
        try:
            return self.robot_client.get_status().get("executor_running", False)
        except Exception:
            return False

    def update(self, intent_result) -> List[RobotAction]:
        actions: List[RobotAction] = []
        timestamp = intent_result.timestamp
        voice_start_idx = len(self.voice_log)
        self._update_counter += 1

        if not self._initial_delivery_queued:
            self._initial_delivery_queued = True
            actions.extend(self._queue_initial_delivery(timestamp))

        self._drain_pending_actions()

        new_completions = set(intent_result.steps_completed) - self.completed_steps
        if new_completions:
            for step in self.step_order:
                if step in new_completions:
                    self.completed_steps.add(step)
                    actions.extend(self._check_return_to_storage(step, timestamp))

        actions.extend(self._check_proactive_delivery(intent_result, timestamp))
        self._evaluate_safety_and_timers(intent_result, timestamp)

        for action in actions:
            self._execute_action(action)

        if self._log_dir:
            self._log_update(intent_result, actions, self.voice_log[voice_start_idx:])

        return actions

    def _queue_initial_delivery(self, timestamp: float) -> List[RobotAction]:
        actions = []
        for obj in self.initial_delivery_objects:
            if self.object_locations.get(obj) == "storage":
                action = RobotAction(
                    action_type="deliver_to_workplace",
                    object_name=obj,
                    trigger_step="idle",
                    reason="Initial setup \u2014 delivering to workplace",
                    timestamp=timestamp,
                )
                actions.append(action)
                self.object_locations[obj] = "workplace"
                self._say(f"Delivering {obj.replace('_', ' ')} from storage to workplace", timestamp=timestamp)
        return actions

    def _drain_pending_actions(self) -> None:
        if not self._pending_actions or self._is_robot_busy():
            return
        action = self._pending_actions.pop(0)
        self._execute_action(action, from_queue=True)

    def _check_return_to_storage(self, step_name: str, timestamp: float) -> List[RobotAction]:
        actions = []
        for obj in self.return_triggers.get(step_name, []):
            if self.object_locations.get(obj) != "workplace":
                continue
            reason = self.nodes.get(step_name, {}).get("robot_return_to_storage", {}).get("reason", f"{obj} no longer needed")
            actions.append(RobotAction("return_to_storage", obj, step_name, reason, timestamp))
            self._say(f"Returning {obj.replace('_', ' ')} to storage \u2014 {reason}", timestamp=timestamp)
        return actions

    def _check_proactive_delivery(self, intent_result, timestamp: float) -> List[RobotAction]:
        actions = []
        predicted = intent_result.predicted_next_action
        if not predicted or predicted == "unknown":
            return actions

        needed = self.nodes.get(predicted, {}).get("objects_needed_on_workplace", [])
        for obj in needed:
            if obj in self.movable_objects and self.object_locations.get(obj) == "storage":
                actions.append(RobotAction("deliver_to_workplace", obj, predicted, f"Needed for {predicted}", timestamp))
                self._say(f"Delivering {obj.replace('_', ' ')} to workplace \u2014 needed for {predicted.replace('_', ' ')}", timestamp=timestamp)
        return actions

    def _evaluate_safety_and_timers(self, result, timestamp: float) -> None:
        safety_rules = self.task_profile.get("safety_rules", [])
        for i, rule in enumerate(safety_rules):
            field_name = rule.get("trigger_field")
            cond = rule.get("trigger_condition")
            active_phases = rule.get("active_phases", [])
            msg = rule.get("warning_message", "Warning")
            
            rule_id = f"rule_{i}_{field_name}"
            
            if result.current_phase in active_phases:
                field_val = result.state.get(field_name)
                # handle "Unknown" or standard bool matches
                if field_val == cond or (str(field_val).lower() == str(cond).lower()):
                    if rule_id not in self._warned_rules:
                        self._warned_rules.add(rule_id)
                        self._say(msg, priority="warning", timestamp=timestamp)
                else:
                    if rule_id in self._warned_rules:
                        self._warned_rules.remove(rule_id)

        timers = self.task_profile.get("timers", [])
        for i, timer in enumerate(timers):
            field_name = timer.get("trigger_field")
            cond = timer.get("trigger_condition")
            warn_at = timer.get("warning_interval_minutes", 20)
            msg = timer.get("warning_message", "Warning time limit")
            
            timer_id = f"timer_{i}"
            field_val = result.state.get(field_name)
            match = field_val == cond or (str(field_val).lower() == str(cond).lower())
            
            if match and timer_id not in self._state_timers:
                self._state_timers[timer_id] = time.time()
            elif not match and timer_id in self._state_timers:
                del self._state_timers[timer_id]
                if f"warn_{timer_id}" in self._warned_rules:
                    self._warned_rules.remove(f"warn_{timer_id}")
            
            if timer_id in self._state_timers:
                elapsed = (time.time() - self._state_timers[timer_id]) / 60.0
                if elapsed > warn_at and f"warn_{timer_id}" not in self._warned_rules:
                    self._warned_rules.add(f"warn_{timer_id}")
                    formatted_msg = msg.replace("{elapsed}", f"{elapsed:.0f}")
                    self._say(formatted_msg, priority="warning", timestamp=timestamp)

    def _execute_action(self, action: RobotAction, from_queue: bool = False) -> None:
        prog = self._resolve_program(action)
        if not prog:
            action.executed = True
            action.success = False
            self.executed_actions.append(action)
            return

        if not from_queue and self._is_robot_busy():
            self._pending_actions.append(action)
            return

        if self.dry_run or self.robot_client is None:
            action.executed = True
            action.success = True
        else:
            try:
                resp = self.robot_client.execute_program(prog)
                action.api_response = resp
                action.executed = True
                action.success = resp.get("success", False)
            except Exception as e:
                action.executed = True
                action.success = False
                action.api_response = {"error": str(e)}

        if action.success:
            if action.action_type == "return_to_storage":
                self.object_locations[action.object_name] = "storage"
            elif action.action_type == "deliver_to_workplace":
                self.object_locations[action.object_name] = "workplace"
        self.executed_actions.append(action)

    def _say(self, text: str, priority: str = "normal", timestamp: float = 0.0) -> None:
        self.voice_log.append(VoiceMessage(text, priority, timestamp))
        if self.on_voice:
            try:
                self.on_voice(text)
            except Exception:
                pass

    def _log_update(self, intent_result, actions: List[RobotAction], voice_messages_this_update: List[VoiceMessage]) -> None:
        if not self._log_dir:
            return
        call_dir = self._log_dir / f"call_{self._update_counter:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        decision = {
            "update_number": self._update_counter,
            "timestamp_sec": round(intent_result.timestamp, 3),
            "frame_num": intent_result.frame_num,
            "completed_steps": sorted(self.completed_steps),
            "object_locations": dict(sorted(self.object_locations.items())),
            "robot_actions": [dict(action_type=a.action_type, object_name=a.object_name, program=self._resolve_program(a), success=a.success) for a in actions],
            "voice_messages": [{"text": m.text, "priority": m.priority} for m in voice_messages_this_update],
            "dry_run": self.dry_run,
        }
        try:
            with open(call_dir / "decision.json", "w") as f:
                json.dump(decision, f, indent=2, default=str)
        except Exception:
            pass

    def get_summary(self) -> str:
        lines = ["=" * 60, f"  AURA Decision Engine \u2014 Summary", "=" * 60]
        lines.append(f"\nCompleted steps ({len(self.completed_steps)}):")
        for step in self.step_order:
            marker = "x" if step in self.completed_steps else " "
            lines.append(f"  [{marker}] {step}")
        return "\n".join(lines)

    def save_summary(self) -> Optional[Path]:
        if not self._log_dir: return None
        summary_path = self._log_dir / "decision_summary.json"
        summary = {
            "total_updates": self._update_counter,
            "completed_steps": sorted(self.completed_steps),
            "total_robot_actions": len(self.executed_actions),
            "object_locations": self.object_locations,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary_path

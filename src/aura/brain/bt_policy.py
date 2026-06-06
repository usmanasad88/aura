"""Behaviour Tree policy for the AURA decision engine.

Design note
===========

The BT is **compiled from config** at decision-engine init — there is no
hand-written topology per task. All branches read standard fields from
``task_profile.json``, ``robot_skills.json``, ``dag.json`` and the live
``SemanticSceneGraph.task_state``; nothing in this module hard-codes any task.

Topology (top-down priority selector)
-------------------------------------

::

    root : Selector
    ├── 1. safety        : Selector over task_profile.safety_rules
    ├── 2. reactive      : Selector over timers / gesture flag / utterances
    ├── 3. scheduled     : Selector over robot_skills with trigger_steps /
    │                      trigger_after_steps (conflicts escalate to LLM)
    ├── 4. llm_fallback  : wraps the legacy LLM prompt (hybrid mode only)
    └── 5. idle_wait     : always SUCCESS → returns WAIT

Each leaf writes into a shared :class:`BTContext` (``result``,
``reasoning_trail``). When a leaf fires it returns ``SUCCESS`` and the
parent selector stops — which gives strict priority order (safety beats
reactive beats scheduled beats LLM beats wait).

Branch 3 — how deterministic delivery is derived from config
-----------------------------------------------------------

Each skill in ``robot_skills.json`` may carry two optional fields:

* ``trigger_steps`` — DAG-step IDs whose *imminent execution* should
  fire this skill (e.g. a delivery skill fires when its consuming step
  is the predicted next action).
* ``trigger_after_steps`` — DAG-step IDs whose *completion* should fire
  this skill (e.g. a return-to-storage skill fires after the last step
  that used the object).

A ``SkillDeliveryLeaf`` succeeds iff **all** of:

1. ``task_state.robot_engagement == "continue"`` (vision-informed
   go/no-go from the intent monitor).
2. ``task_state.robot_state == "idle"``.
3. All ``preconditions`` match ``ssg.task_state`` /
   ``ssg.get_location()``.
4. Either
   (a) ``intent.predicted_next_action`` ∈ ``trigger_steps``, OR
   (b) every ID in ``trigger_after_steps`` ∈ ``steps_completed``.

Skills with neither field are **never** fired deterministically — they
remain available to the LLM fallback (by design, utility/motion skills
like ``gripper_open`` or ``move_to_named_position`` should not trigger
themselves).

LLM fallback conditions
-----------------------

Branch 4 runs only when *branches 1–3 all failed* AND one of:

* **conflict** — two or more scheduled-delivery leaves were eligible on
  the same tick (``ctx.scheduled_conflict``);
* **low intent confidence** —
  ``intent.prediction_confidence < proactive_threshold``;
* **unhandled utterance** — ``task_state.recent_utterances`` contains a
  message newer than ``ctx.last_utterance_handled_ts``;
* **off-SOP** — ``intent.predicted_next_action`` is empty or does not
  appear in the DAG.

In ``decision_mode="bt"`` branch 4 is replaced with an
``EscalateToHumanLeaf`` that emits an ``ask_question`` action — no LLM
ever runs.

In ``decision_mode="llm"`` the root is replaced with a single LLM leaf,
preserving the legacy per-cycle behaviour.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import py_trees

from aura.core.scene_graph import SemanticSceneGraph

logger = logging.getLogger(__name__)


# ── Shared context ───────────────────────────────────────────────────────────


@dataclass
class BTContext:
    """Per-tick shared state passed to every leaf.

    Most fields are reset at the start of each tick by
    :meth:`BTPolicy.tick`; ``last_utterance_handled_ts`` and
    ``timer_start_times`` persist across ticks.
    """

    ssg: SemanticSceneGraph
    skills: Any  # SkillRegistry — typed loosely to avoid import cycle
    task_profile: Dict[str, Any]
    dag: List[Dict[str, Any]]
    explainer: Any  # DecisionExplainer
    decision_mode: str = "hybrid"
    proactive_threshold: float = 0.7
    # When True and no BT branch fires, defer the tick to the LLM fallback
    # instead of defaulting to a wait. No effect in pure "bt" mode (which
    # has no LLM leaf). See LLMFallbackLeaf.
    defer_to_llm_when_idle: bool = False

    # LLM fallback hook; returns an ActionPrediction or None.
    # Signature: async (reason: str, ctx: BTContext) -> ActionPrediction | None
    llm_fallback: Optional[Callable[[str, "BTContext"], Awaitable[Any]]] = None

    # Per-tick inputs (populated by DecisionEngine.tick)
    current_time_sec: float = 0.0
    intent: Dict[str, Any] = field(default_factory=dict)
    task_state: Dict[str, Any] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)
    human_requesting_help: bool = False
    # Most recent captured camera frame (np.ndarray or PIL Image), passed
    # through to the LLM fallback so the VLM can see the live scene when
    # ``workflow_config.pass_captured_frame_to_vlm`` is enabled.
    current_frame: Optional[Any] = None

    # Per-tick outputs
    result: Optional[Any] = None  # ActionPrediction | None
    reasoning_trail: List[str] = field(default_factory=list)
    branch_fired: str = "none"
    eligible_deliveries: List[str] = field(default_factory=list)
    scheduled_conflict: bool = False
    llm_invoked: bool = False

    # Persistent
    last_utterance_handled_ts: float = 0.0
    timer_start_times: Dict[str, float] = field(default_factory=dict)

    def reset_tick(self) -> None:
        self.result = None
        self.reasoning_trail = []
        self.branch_fired = "none"
        self.eligible_deliveries = []
        self.scheduled_conflict = False
        self.llm_invoked = False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_preconditions(
    preconditions: Dict[str, Any], ssg: SemanticSceneGraph
) -> tuple[bool, str]:
    """Return (all_match, first_failure_description).

    Each key has the form ``"<node_id>.<attr>"``. ``.location`` is
    resolved via ``ssg.get_location``; any other ``.attr`` is resolved
    via ``ssg.task_state["<node_id>_<attr>"]`` or
    ``ssg.task_state["<key>"]`` (trying both).
    """
    for key, expected in preconditions.items():
        if "." in key:
            node_id, attr = key.split(".", 1)
        else:
            node_id, attr = key, ""

        actual: Any = None
        if attr == "location":
            try:
                actual = ssg.get_location(node_id)
            except Exception:
                actual = None
            if actual is None:
                # Fall back to task_state key conventions
                actual = ssg.task_state.get(
                    f"{node_id}_location", ssg.task_state.get(key)
                )
        else:
            # Try "<node>.<attr>" literal, "<node>_<attr>", and node.<attr>
            actual = ssg.task_state.get(key)
            if actual is None and attr:
                actual = ssg.task_state.get(f"{node_id}_{attr}")
            if actual is None and not attr:
                actual = ssg.task_state.get(node_id)

        if actual != expected:
            return False, f"{key}={actual!r}≠{expected!r}"
    return True, ""


# ── Parametric skill helpers ──────────────────────────────────────────────────
#
# A parametric skill (e.g. ``pick_and_place_item``) reuses one robot program
# with different arguments. Its preconditions/effects carry ``{var}``
# placeholders (e.g. ``"{item}.location": "storage_area"``). Rather than
# declaring N near-identical skills, the BT expands the single skill over the
# ``valid_values`` of its selection parameter and fires one concrete instance
# per tick. Multiple eligible values are NOT a conflict — the robot runs one
# program at a time, so the rest are picked up on later ticks as it frees up.

_TEMPLATE_RE = re.compile(r"\{(\w+)\}")


def _skill_is_parametric(skill: Any) -> bool:
    """True if the skill's preconditions reference any ``{var}`` placeholder."""
    for key, val in skill.preconditions.items():
        if "{" in key:
            return True
        if isinstance(val, str) and "{" in val:
            return True
    return False


def _parametric_candidates(
    skill: Any, ssg: SemanticSceneGraph
) -> tuple[Optional[str], List[str]]:
    """Expand a single-variable parametric skill over its candidate values.

    Returns ``(var_name, eligible_values)`` where ``eligible_values`` are the
    values (in the parameter's declared order) for which the substituted
    preconditions hold. Returns ``(None, [])`` when the skill references zero
    or more than one distinct template variable (unsupported here).
    """
    vars_: List[str] = []
    for key, val in skill.preconditions.items():
        for m in _TEMPLATE_RE.findall(key):
            if m not in vars_:
                vars_.append(m)
        if isinstance(val, str):
            for m in _TEMPLATE_RE.findall(val):
                if m not in vars_:
                    vars_.append(m)
    if len(vars_) != 1:
        return None, []

    var = vars_[0]
    param = next((p for p in skill.parameters if p.name == var), None)
    candidates = list(getattr(param, "valid_values", []) or []) if param else []

    eligible: List[str] = []
    for cand in candidates:
        concrete = {
            key.format(**{var: cand}): (
                val.format(**{var: cand}) if isinstance(val, str) else val
            )
            for key, val in skill.preconditions.items()
        }
        ok, _why = _check_preconditions(concrete, ssg)
        if ok:
            eligible.append(cand)
    return var, eligible


def _resolve_delivery_parameters(
    skill: Any, var: str, value: str
) -> Dict[str, Any]:
    """Fill the skill's call parameters for a chosen selection *value*.

    The selection parameter is set to *value*. Every other parameter is
    resolved by, in priority order: a ``valid_values`` entry scoped to the
    selected value (prefix ``"{value}_"`` — e.g. ``cup`` →
    ``cup_workplace``), then the parameter's ``default``, then its first
    ``valid_values`` entry.
    """
    params: Dict[str, Any] = {var: value}
    for p in skill.parameters:
        if p.name == var:
            continue
        vals = list(getattr(p, "valid_values", []) or [])
        scoped = [v for v in vals if isinstance(v, str) and v.startswith(f"{value}_")]
        if scoped:
            params[p.name] = scoped[0]
        elif getattr(p, "default", None) is not None:
            params[p.name] = p.default
        elif vals:
            params[p.name] = vals[0]
    return params


# ── Leaf base ────────────────────────────────────────────────────────────────


class _BTLeaf(py_trees.behaviour.Behaviour):
    """Shared boilerplate: holds a reference to the ctx."""

    def __init__(self, name: str, ctx: BTContext):
        super().__init__(name=name)
        self.ctx = ctx


# ── Branch 1: safety ─────────────────────────────────────────────────────────


class SafetyRuleLeaf(_BTLeaf):
    """One leaf per ``task_profile.safety_rules`` entry.

    Fires ABORT when the configured trigger field matches the
    configured condition, and (optionally) the current phase is in the
    rule's ``active_phases`` list.
    """

    def __init__(self, ctx: BTContext, rule: Dict[str, Any]):
        rid = rule.get("trigger_field", "rule")
        super().__init__(name=f"safety:{rid}", ctx=ctx)
        self.rule = rule

    def update(self) -> py_trees.common.Status:
        field_name = self.rule.get("trigger_field")
        if not field_name:
            return py_trees.common.Status.FAILURE
        actual = self.ctx.task_state.get(field_name)
        trigger_value = self.rule.get("trigger_condition")
        if actual != trigger_value:
            return py_trees.common.Status.FAILURE

        active_phases = self.rule.get("active_phases") or []
        if active_phases:
            phase = self.ctx.task_state.get("current_phase") or self.ctx.intent.get(
                "current_phase"
            )
            if phase not in active_phases:
                return py_trees.common.Status.FAILURE

        msg = self.rule.get("warning_message", f"Safety rule {field_name} fired")
        from aura.brain.decision_engine import ActionPrediction

        self.ctx.result = ActionPrediction(
            action_id="abort",
            target_id=None,
            predicted_time_sec=self.ctx.current_time_sec,
            confidence=1.0,
            reasoning=msg,
        )
        self.ctx.reasoning_trail.append(f"safety:{field_name}")
        self.ctx.branch_fired = "safety"
        return py_trees.common.Status.SUCCESS


# ── Branch 2: reactive ───────────────────────────────────────────────────────


class TimerWarningLeaf(_BTLeaf):
    """Emits a SPEAK action when a configured timer expires.

    The timer starts when ``trigger_field`` first matches
    ``trigger_condition`` and fires once it exceeds
    ``time_limit_minutes``.
    """

    def __init__(self, ctx: BTContext, timer: Dict[str, Any]):
        tid = timer.get("trigger_field", "timer")
        super().__init__(name=f"timer:{tid}", ctx=ctx)
        self.timer = timer

    def update(self) -> py_trees.common.Status:
        field_name = self.timer.get("trigger_field")
        if not field_name:
            return py_trees.common.Status.FAILURE
        actual = self.ctx.task_state.get(field_name)
        if actual != self.timer.get("trigger_condition"):
            # Reset if condition no longer holds
            self.ctx.timer_start_times.pop(field_name, None)
            return py_trees.common.Status.FAILURE

        start = self.ctx.timer_start_times.get(field_name)
        if start is None:
            self.ctx.timer_start_times[field_name] = self.ctx.current_time_sec
            return py_trees.common.Status.FAILURE

        elapsed_min = (self.ctx.current_time_sec - start) / 60.0
        limit = float(self.timer.get("time_limit_minutes", 0.0) or 0.0)
        if limit <= 0 or elapsed_min < limit:
            return py_trees.common.Status.FAILURE

        template = self.timer.get("warning_message", "Timer expired")
        try:
            msg = template.format(elapsed=f"{elapsed_min:.1f}")
        except Exception:
            msg = template
        from aura.brain.decision_engine import ActionPrediction

        self.ctx.result = ActionPrediction(
            action_id="speak",
            target_id=None,
            predicted_time_sec=self.ctx.current_time_sec,
            confidence=1.0,
            reasoning=msg,
        )
        self.ctx.reasoning_trail.append(f"timer:{field_name}")
        self.ctx.branch_fired = "timer"
        return py_trees.common.Status.SUCCESS


class HumanHelpGestureLeaf(_BTLeaf):
    """Escalate to the LLM (or ask the human in pure-BT mode) when
    ``human_requesting_help`` is set by the gesture monitor."""

    def __init__(self, ctx: BTContext):
        super().__init__(name="reactive:help_gesture", ctx=ctx)

    def update(self) -> py_trees.common.Status:
        if not self.ctx.human_requesting_help:
            return py_trees.common.Status.FAILURE

        self.ctx.reasoning_trail.append("reactive:help_gesture")
        if self.ctx.decision_mode == "bt" or self.ctx.llm_fallback is None:
            from aura.brain.decision_engine import ActionPrediction

            self.ctx.result = ActionPrediction(
                action_id="ask_question",
                target_id=None,
                predicted_time_sec=self.ctx.current_time_sec,
                confidence=1.0,
                reasoning="Human gesture requesting help — how can I assist?",
            )
            self.ctx.branch_fired = "reactive_gesture_bt"
            return py_trees.common.Status.SUCCESS

        # hybrid/llm: let the LLM fallback decide what to do
        self.ctx.branch_fired = "reactive_gesture_llm"
        # We return FAILURE so the selector proceeds — but mark that the
        # LLM MUST run. The LLM fallback leaf picks this up via the flag.
        self.ctx._force_llm_reason = "human_help_gesture"  # type: ignore[attr-defined]
        return py_trees.common.Status.FAILURE


class UtteranceLeaf(_BTLeaf):
    """Mark unhandled utterances so the LLM fallback picks them up."""

    def __init__(self, ctx: BTContext):
        super().__init__(name="reactive:utterance", ctx=ctx)

    def update(self) -> py_trees.common.Status:
        utterances = self.ctx.task_state.get("recent_utterances") or []
        unhandled = [
            u
            for u in utterances
            if float(u.get("timestamp") or 0.0) > self.ctx.last_utterance_handled_ts
        ]
        if not unhandled:
            return py_trees.common.Status.FAILURE

        # Advance watermark unconditionally so we don't keep re-firing.
        self.ctx.last_utterance_handled_ts = max(
            float(u.get("timestamp") or 0.0) for u in unhandled
        )
        self.ctx.reasoning_trail.append(
            f"reactive:utterance({len(unhandled)})"
        )
        if self.ctx.decision_mode == "bt" or self.ctx.llm_fallback is None:
            # Pure BT can't interpret speech — best we can do is acknowledge.
            from aura.brain.decision_engine import ActionPrediction

            self.ctx.result = ActionPrediction(
                action_id="speak",
                target_id=None,
                predicted_time_sec=self.ctx.current_time_sec,
                confidence=0.5,
                reasoning="Acknowledged human utterance (BT mode cannot interpret).",
            )
            self.ctx.branch_fired = "reactive_utterance_bt"
            return py_trees.common.Status.SUCCESS

        self.ctx._force_llm_reason = "unhandled_utterance"  # type: ignore[attr-defined]
        self.ctx.branch_fired = "reactive_utterance_llm"
        return py_trees.common.Status.FAILURE


# ── Branch 3: scheduled delivery ─────────────────────────────────────────────


class SkillDeliveryLeaf(_BTLeaf):
    """Fires when a skill's trigger_steps / trigger_after_steps match.

    Task-agnostic: reads the two optional config fields on ``RobotSkill``
    and gates on preconditions + robot_engagement + robot_state.
    """

    def __init__(self, ctx: BTContext, skill: Any):
        super().__init__(name=f"deliver:{skill.id}", ctx=ctx)
        self.skill = skill

    def update(self) -> py_trees.common.Status:
        skill = self.skill
        has_triggers = bool(skill.trigger_steps or skill.trigger_after_steps)
        has_preconditions = bool(skill.preconditions)

        # Utility skills with no triggers AND no preconditions are never
        # scheduled here — they remain available only via the LLM.
        if not has_triggers and not has_preconditions:
            return py_trees.common.Status.FAILURE

        # Gate 1: vision-informed engagement judgement.
        engagement = self.ctx.task_state.get("robot_engagement", "continue")
        if engagement != "continue":
            return py_trees.common.Status.FAILURE

        # Gate 2: robot must be idle.
        robot_state = self.ctx.task_state.get("robot_state", "idle")
        if robot_state not in ("idle", "unknown"):
            return py_trees.common.Status.FAILURE

        # Gate 2b: eligibility window closed. Once every step in
        # ``trigger_until_steps`` is completed this skill stops firing —
        # this bounds an open-ended ``trigger_after_steps`` so a delivery
        # skill does not overlap a later phase (e.g. a fetch skill yielding
        # to the return-to-storage skill once cleanup begins).
        until = getattr(skill, "trigger_until_steps", None)
        if until and all(s in set(self.ctx.steps_completed) for s in until):
            return py_trees.common.Status.FAILURE

        # Gate 3: preconditions. Parametric skills (templated preconditions)
        # are expanded over their selection parameter; static skills are
        # checked directly.
        parametric_var: Optional[str] = None
        eligible_values: List[str] = []
        if _skill_is_parametric(skill):
            parametric_var, eligible_values = _parametric_candidates(
                skill, self.ctx.ssg
            )
            if parametric_var is None or not eligible_values:
                # No satisfiable concrete instance (or unsupported template).
                return py_trees.common.Status.FAILURE
        else:
            ok, _why = _check_preconditions(skill.preconditions, self.ctx.ssg)
            if not ok:
                return py_trees.common.Status.FAILURE

        # Gate 4: trigger match. Skills without any trigger field but with
        # satisfied preconditions are ambiguous — the BT cannot tell when
        # to fire them. Escalate to the LLM rather than auto-firing or
        # silently ignoring.
        if not has_triggers:
            self.ctx._force_llm_reason = (  # type: ignore[attr-defined]
                f"eligible_no_trigger:{skill.id}"
            )
            self.ctx.reasoning_trail.append(f"delivery:eligible_no_trigger:{skill.id}")
            return py_trees.common.Status.FAILURE

        trigger_reason = ""
        if skill.trigger_steps:
            predicted = self.ctx.intent.get("predicted_next_action") or ""
            if predicted in skill.trigger_steps:
                trigger_reason = f"next_action={predicted}"
        if not trigger_reason and skill.trigger_after_steps:
            completed = set(self.ctx.steps_completed)
            if all(s in completed for s in skill.trigger_after_steps):
                trigger_reason = (
                    f"after_completed={','.join(skill.trigger_after_steps)}"
                )
        if not trigger_reason:
            return py_trees.common.Status.FAILURE

        # This skill is eligible. Record it ONCE — a parametric skill with
        # several eligible values still counts as a single eligible delivery
        # (the robot runs one program at a time). Multiple movable items
        # during e.g. setup_workspace therefore do NOT register as a
        # conflict; the rest are fetched on later ticks as the robot frees up.
        self.ctx.eligible_deliveries.append(skill.id)

        # If another delivery already claimed the result, mark conflict
        # and abort this leaf (selector will stop once the first one wins).
        if self.ctx.result is not None:
            self.ctx.scheduled_conflict = True
            return py_trees.common.Status.FAILURE

        if parametric_var is not None:
            # Selection policy: fire the first eligible value in the skill's
            # declared ``valid_values`` order (which acts as the priority
            # list). Per-item skip is already enforced upstream — only values
            # whose substituted preconditions hold reach ``eligible_values``,
            # so items already placed (or otherwise gated) are excluded.
            selected = eligible_values[0]
            params = _resolve_delivery_parameters(skill, parametric_var, selected)
            target_id = selected
            reason = f"{trigger_reason}; {parametric_var}={selected}"
            if len(eligible_values) > 1:
                self.ctx.reasoning_trail.append(
                    "delivery:multi_eligible("
                    f"{','.join(eligible_values)})->{selected}"
                )
        else:
            # Static skill: derive a primary target from effects (best-effort).
            params = {}
            target_id = None
            for effect_key in skill.effects.keys():
                if "." in effect_key:
                    target_id = effect_key.split(".", 1)[0]
                    break
            reason = trigger_reason

        from aura.brain.decision_engine import ActionPrediction

        self.ctx.result = ActionPrediction(
            action_id=skill.id,
            target_id=target_id,
            predicted_time_sec=self.ctx.current_time_sec,
            confidence=0.9,
            reasoning=f"BT scheduled delivery ({reason}).",
            parameters=params,
        )
        self.ctx.reasoning_trail.append(f"delivery:{skill.id}")
        self.ctx.branch_fired = "scheduled"
        return py_trees.common.Status.SUCCESS


class ScheduledSelector(py_trees.composites.Selector):
    """Selector over all SkillDeliveryLeaf children.

    Wraps the standard py_trees Selector to detect conflicts: if more
    than one child reported itself eligible during the same tick, flag
    ``ctx.scheduled_conflict`` and — in hybrid mode — FAIL the branch so
    the LLM fallback takes over instead of firing a guessed-first skill.
    """

    def __init__(self, ctx: BTContext, children: List[_BTLeaf]):
        super().__init__(name="scheduled", memory=False, children=children)
        self.ctx = ctx

    def tick(self):  # type: ignore[override]
        # Reset per-tick eligibility accounting before ticking children.
        self.ctx.eligible_deliveries = []
        self.ctx.scheduled_conflict = False
        yield from super().tick()

        # Conflict check: multiple children flagged themselves eligible.
        if len(self.ctx.eligible_deliveries) > 1:
            self.ctx.scheduled_conflict = True
            if self.ctx.decision_mode == "hybrid" and self.ctx.llm_fallback is not None:
                # Drop the tentatively-selected result and defer to LLM.
                self.ctx.result = None
                self.ctx.reasoning_trail.append(
                    f"delivery:conflict({','.join(self.ctx.eligible_deliveries)})"
                )
                self.ctx._force_llm_reason = "delivery_conflict"  # type: ignore[attr-defined]
                self.status = py_trees.common.Status.FAILURE


# ── Branch 4: LLM fallback ───────────────────────────────────────────────────


class LLMFallbackLeaf(_BTLeaf):
    """Runs the legacy LLM prompt. Invoked only on ambiguity.

    Triggers when *any* of:

    * ``ctx._force_llm_reason`` was set by an earlier branch (gesture,
      utterance, delivery conflict);
    * intent confidence below ``proactive_threshold``;
    * predicted_next_action missing or not in DAG;
    * ``ctx.defer_to_llm_when_idle`` is set and no other branch fired —
      i.e. the tree would otherwise default to waiting. This lets the LLM
      take over idle ticks instead of silently waiting.
    """

    def __init__(self, ctx: BTContext):
        super().__init__(name="llm_fallback", ctx=ctx)

    def update(self) -> py_trees.common.Status:
        if self.ctx.llm_fallback is None:
            return py_trees.common.Status.FAILURE

        robot_state = self.ctx.task_state.get("robot_state", "idle")
        if robot_state not in ("idle", "unknown"):
            self.ctx.reasoning_trail.append(
                f"llm_fallback_skipped:robot_busy({robot_state})"
            )
            return py_trees.common.Status.FAILURE

        forced = getattr(self.ctx, "_force_llm_reason", None)
        reason = forced
        if reason is None:
            confidence = float(self.ctx.intent.get("prediction_confidence") or 0.0)
            predicted = self.ctx.intent.get("predicted_next_action") or ""
            dag_ids = {step.get("id") for step in self.ctx.dag if isinstance(step, dict)}
            if confidence < self.ctx.proactive_threshold:
                reason = f"low_confidence({confidence:.2f})"
            elif not predicted or predicted not in dag_ids:
                reason = "off_sop"

        # Opt-in: when nothing else fired, defer the idle tick to the LLM
        # instead of letting the idle leaf default to waiting.
        if reason is None and self.ctx.defer_to_llm_when_idle:
            reason = "idle_default"

        if reason is None:
            return py_trees.common.Status.FAILURE

        self.ctx.llm_invoked = True
        self.ctx.reasoning_trail.append(f"llm_fallback:{reason}")
        # Run the async LLM hook from this sync tick.
        prediction = _run_async(self.ctx.llm_fallback(reason, self.ctx))
        if prediction is None:
            # LLM chose to wait — let the idle leaf handle it.
            return py_trees.common.Status.FAILURE

        self.ctx.result = prediction
        self.ctx.branch_fired = "llm_fallback"
        return py_trees.common.Status.SUCCESS


class EscalateToHumanLeaf(_BTLeaf):
    """Pure-BT substitute for the LLM fallback: ask the human."""

    def __init__(self, ctx: BTContext):
        super().__init__(name="escalate_human", ctx=ctx)

    def update(self) -> py_trees.common.Status:
        from aura.brain.decision_engine import ActionPrediction

        self.ctx.result = ActionPrediction(
            action_id="ask_question",
            target_id=None,
            predicted_time_sec=self.ctx.current_time_sec,
            confidence=0.5,
            reasoning="Pure-BT mode: no deterministic branch fired; escalating to human.",
        )
        self.ctx.reasoning_trail.append("escalate_human")
        self.ctx.branch_fired = "escalate_human"
        return py_trees.common.Status.SUCCESS


# ── Branch 5: idle ───────────────────────────────────────────────────────────


class IdleWaitLeaf(_BTLeaf):
    """Always SUCCESS — records a wait."""

    def __init__(self, ctx: BTContext):
        super().__init__(name="idle_wait", ctx=ctx)

    def update(self) -> py_trees.common.Status:
        self.ctx.reasoning_trail.append("wait:idle")
        self.ctx.branch_fired = "idle"
        self.ctx.result = None
        return py_trees.common.Status.SUCCESS


# ── Async-from-sync bridge ───────────────────────────────────────────────────


def _run_async(coro) -> Any:
    """Run an awaitable from a sync BT tick context.

    The BT is ticked from within ``DecisionEngine.decide_action``, which
    is itself an async method — so there *is* a running event loop. We
    can't call ``run_until_complete`` in that case. Instead, we block on
    a temporary helper thread.
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running loop present — offload to a helper thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


# ── Policy wrapper ──────────────────────────────────────────────────────────


class BTPolicy:
    """Owns the compiled tree and the per-tick plumbing.

    :meth:`tick` returns an ``ActionPrediction`` (or ``None`` for wait)
    and also populates ``DecisionExplainer`` via ``record_decision``.
    """

    def __init__(self, ctx: BTContext, decision_mode: str = "hybrid"):
        self.ctx = ctx
        self.ctx.decision_mode = decision_mode
        self.decision_mode = decision_mode
        self.root = self._build(decision_mode)
        self.tree = py_trees.trees.BehaviourTree(root=self.root)

    # Compilation ---------------------------------------------------------
    def _build(self, mode: str) -> py_trees.behaviour.Behaviour:
        if mode == "llm":
            return LLMFallbackLeaf(self.ctx)

        # Branch 1: safety
        safety_children: List[_BTLeaf] = [
            SafetyRuleLeaf(self.ctx, rule)
            for rule in (self.ctx.task_profile.get("safety_rules") or [])
        ]
        safety = py_trees.composites.Selector(
            name="safety", memory=False, children=safety_children
        ) if safety_children else None

        # Branch 2: reactive
        reactive_children: List[_BTLeaf] = [
            TimerWarningLeaf(self.ctx, t)
            for t in (self.ctx.task_profile.get("timers") or [])
        ]
        reactive_children.append(HumanHelpGestureLeaf(self.ctx))
        reactive_children.append(UtteranceLeaf(self.ctx))
        reactive = py_trees.composites.Selector(
            name="reactive", memory=False, children=reactive_children
        )

        # Branch 3: scheduled delivery (one leaf per skill with triggers)
        delivery_children: List[_BTLeaf] = []
        for skill in self.ctx.skills.list_skills():
            if skill.trigger_steps or skill.trigger_after_steps:
                delivery_children.append(SkillDeliveryLeaf(self.ctx, skill))
        scheduled = (
            ScheduledSelector(self.ctx, delivery_children)
            if delivery_children
            else None
        )

        # Branch 4: LLM (or escalation) fallback
        fallback: _BTLeaf
        if mode == "bt":
            fallback = EscalateToHumanLeaf(self.ctx)
        else:
            fallback = LLMFallbackLeaf(self.ctx)

        # Branch 5: idle wait (guaranteed SUCCESS)
        idle = IdleWaitLeaf(self.ctx)

        children: List[py_trees.behaviour.Behaviour] = []
        if safety is not None:
            children.append(safety)
        children.append(reactive)
        if scheduled is not None:
            children.append(scheduled)
        children.append(fallback)
        children.append(idle)

        return py_trees.composites.Selector(
            name="aura_root", memory=False, children=children
        )

    # Execution -----------------------------------------------------------
    def tick(
        self,
        *,
        current_time_sec: float,
        intent: Dict[str, Any],
        task_state: Dict[str, Any],
        steps_completed: List[str],
        human_requesting_help: bool,
    ) -> tuple[Optional[Any], str, bool]:
        """Run one BT tick.

        Returns ``(prediction, reasoning_string, llm_invoked)``.
        ``prediction`` is the ``ActionPrediction`` produced by the
        fired leaf (or ``None`` for a wait decision).
        """
        ctx = self.ctx
        ctx.reset_tick()
        # Clear any forced-LLM flag from a previous tick
        if hasattr(ctx, "_force_llm_reason"):
            delattr(ctx, "_force_llm_reason")

        ctx.current_time_sec = current_time_sec
        ctx.intent = intent or {}
        ctx.task_state = task_state or {}
        ctx.steps_completed = list(steps_completed or [])
        ctx.human_requesting_help = bool(human_requesting_help)

        self.tree.tick()

        reasoning = " | ".join(ctx.reasoning_trail) or "no_branch"
        return ctx.result, reasoning, ctx.llm_invoked


__all__ = [
    "BTContext",
    "BTPolicy",
    "SafetyRuleLeaf",
    "TimerWarningLeaf",
    "HumanHelpGestureLeaf",
    "UtteranceLeaf",
    "SkillDeliveryLeaf",
    "ScheduledSelector",
    "LLMFallbackLeaf",
    "EscalateToHumanLeaf",
    "IdleWaitLeaf",
]

"""Disjunctive ground-truth data structure for proactive-intervention scoring.

Each intervention is allowed to declare *alternative* time-window
specifications. A single alternative ("option") is a *conjunction* of one
or more time intervals ("phases") — all phases must be satisfied to claim
that option. Different options form a disjunction — only one needs to hold.

This generalises the v1.0 robot_gt.json schema, where every intervention
carried a single ``t_start``/``t_end`` interval. v1.0 files load as the
degenerate case: one option containing one phase.

Schema v2.0 example (Task 1 from the DTN-style scenario)::

    {
      "id": "task_1",
      "skill": "...",
      "options": [
        {"phases": [{"t_start": 5,  "t_end": 10},
                    {"t_start": 25, "t_end": 30}]},
        {"phases": [{"t_start": 10, "t_end": 15},
                    {"t_start": 35, "t_end": 40}]}
      ]
    }

The structure mirrors the disjunctive temporal constraints in Osanlou et
al., 2022 (DTN/DTNU): each intervention's allowed schedule is
``\\bigvee_k \\bigwedge_j v \\in [x_{k,j}, y_{k,j}]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class Phase:
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(
                f"Phase has t_end < t_start: {self.t_start} > {self.t_end}")

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def contains(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end


@dataclass(frozen=True)
class Option:
    """A single disjunctive alternative — a conjunction of phases."""
    phases: tuple[Phase, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("Option must contain at least one phase")

    @property
    def span(self) -> tuple[float, float]:
        """[earliest phase start, latest phase end]."""
        return (min(p.t_start for p in self.phases),
                max(p.t_end for p in self.phases))


@dataclass
class Intervention:
    """An intervention that may be satisfied by any one of several options."""
    id: str
    skill: str
    args: dict
    options: tuple[Option, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(
                f"Intervention {self.id!r} must declare at least one option")

    @property
    def label(self) -> str:
        if self.args:
            return f"{self.skill}({', '.join(f'{k}={v}' for k, v in self.args.items())})"
        return self.skill

    @property
    def is_disjunctive(self) -> bool:
        return len(self.options) > 1

    @property
    def is_multi_phase(self) -> bool:
        return any(len(o.phases) > 1 for o in self.options)


@dataclass
class RobotGT:
    """Top-level disjunctive robot ground truth."""
    task_id: str
    duration_sec: float
    interventions: list[Intervention]
    schema_version: str = SCHEMA_VERSION
    video: str = ""
    operator_id: str = ""
    description: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ── Parsing ──────────────────────────────────────────────────────────────────


def _parse_phases(raw_phases: list[dict]) -> tuple[Phase, ...]:
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("Each option must have a non-empty 'phases' list")
    phases: list[Phase] = []
    for ph in raw_phases:
        phases.append(Phase(t_start=float(ph["t_start"]),
                            t_end=float(ph["t_end"])))
    return tuple(phases)


def _parse_intervention(raw: dict) -> Intervention:
    iv_id = str(raw.get("id", ""))
    skill = str(raw.get("skill", ""))
    args = dict(raw.get("args") or {})
    rationale = str(raw.get("rationale", ""))

    if "options" in raw:
        options_raw = raw["options"]
        if not isinstance(options_raw, list) or not options_raw:
            raise ValueError(
                f"Intervention {iv_id!r}: 'options' must be a non-empty list")
        options: list[Option] = []
        for opt in options_raw:
            options.append(Option(
                phases=_parse_phases(opt.get("phases", [])),
                rationale=str(opt.get("rationale", "") or rationale),
            ))
    elif "t_start" in raw and "t_end" in raw:
        # v1.0 fallback — single option, single phase.
        options = [Option(
            phases=(Phase(t_start=float(raw["t_start"]),
                          t_end=float(raw["t_end"])),),
            rationale=rationale,
        )]
    else:
        raise ValueError(
            f"Intervention {iv_id!r}: needs either 'options' (v2.0) "
            f"or 't_start'/'t_end' (v1.0)")

    return Intervention(id=iv_id, skill=skill, args=args,
                        options=tuple(options), rationale=rationale)


def load_disjunctive_robot_gt(gt_path: Path) -> RobotGT:
    """Load a robot GT file. Accepts schema v1.0 or v2.0."""
    data = json.loads(Path(gt_path).read_text())
    interventions = [_parse_intervention(iv)
                     for iv in data.get("interventions", [])]
    duration = float(data.get("duration_sec", 0.0) or 0.0)
    if duration <= 0 and interventions:
        duration = max(o.span[1]
                       for iv in interventions for o in iv.options)
    return RobotGT(
        task_id=str(data.get("task_id", "")),
        duration_sec=duration,
        interventions=interventions,
        schema_version=str(data.get("schema_version", "1.0")),
        video=str(data.get("video", "")),
        operator_id=str(data.get("operator_id", "")),
        description=str(data.get("description", "")),
        notes=str(data.get("notes", "")),
    )


# ── Serialisation ────────────────────────────────────────────────────────────


def to_dict(gt: RobotGT) -> dict:
    """Serialise to the v2.0 JSON shape (round-trippable)."""
    out: dict = {
        "task_id": gt.task_id,
        "schema_version": SCHEMA_VERSION,
        "duration_sec": gt.duration_sec,
    }
    if gt.video:
        out["video"] = gt.video
    if gt.operator_id:
        out["operator_id"] = gt.operator_id
    if gt.description:
        out["description"] = gt.description

    iv_list: list[dict] = []
    for iv in gt.interventions:
        iv_d: dict = {
            "id": iv.id,
            "skill": iv.skill,
            "args": dict(iv.args),
            "options": [
                {
                    "phases": [{"t_start": p.t_start, "t_end": p.t_end}
                               for p in opt.phases],
                    **({"rationale": opt.rationale} if opt.rationale else {}),
                }
                for opt in iv.options
            ],
        }
        if iv.rationale:
            iv_d["rationale"] = iv.rationale
        iv_list.append(iv_d)
    out["interventions"] = iv_list

    if gt.notes:
        out["notes"] = gt.notes
    return out


def dump_json(gt: RobotGT, path: Path, *, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(to_dict(gt), indent=indent) + "\n")

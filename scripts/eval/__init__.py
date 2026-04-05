"""AURA evaluation pipeline — shared utilities for experiment management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Evaluation output for a single experiment repetition."""

    run_id: str
    model: str
    task: str
    frame_skip: int
    ground_truth_robot: bool
    # A-Score
    a_score: dict[str, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Intent accuracy
    intent_accuracy: dict[str, float] = field(default_factory=dict)
    # Latency
    latency: dict[str, float] = field(default_factory=dict)
    # Counts
    n_cycles: int = 0
    n_act: int = 0
    n_wait: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunResult:
        data = json.loads(path.read_text())
        return cls(**data)


def experiment_id(
    task: str,
    model: str,
    frame_skip: int,
    ground_truth_robot: bool,
    *,
    extra: dict[str, str] | None = None,
) -> str:
    """Deterministic experiment ID from configuration parameters.

    Returns a human-readable slug like ``hand_layup__gemini-2.5-flash__fs30__gt``.
    """
    model_slug = model.replace("/", "-").replace(" ", "-")
    parts = [task, model_slug, f"fs{frame_skip}"]
    if ground_truth_robot:
        parts.append("gt")
    if extra:
        for k, v in sorted(extra.items()):
            parts.append(f"{k}-{v}")
    return "__".join(parts)


def find_latest_session(log_dir: Path, component: str) -> Path | None:
    """Find the most recent session directory for a component.

    Args:
        log_dir: Root logs directory (e.g. ``logs/``).
        component: ``"intent_monitor"`` or ``"decision_engine"``.
    """
    comp_dir = log_dir / component
    if not comp_dir.is_dir():
        return None
    sessions = sorted(comp_dir.glob("session_*"))
    return sessions[-1] if sessions else None


def pair_sessions(
    log_dir: Path,
) -> list[tuple[Path | None, Path | None]]:
    """Pair intent_monitor and decision_engine sessions by timestamp proximity.

    Returns list of (intent_session, decision_session) tuples.
    Sessions created within ~2 minutes of each other are paired.
    """
    intent_dir = log_dir / "intent_monitor"
    decision_dir = log_dir / "decision_engine"

    def _parse_ts(p: Path) -> str:
        return p.name.replace("session_", "")

    intent_sessions = sorted(intent_dir.glob("session_*")) if intent_dir.is_dir() else []
    decision_sessions = sorted(decision_dir.glob("session_*")) if decision_dir.is_dir() else []

    if not intent_sessions and not decision_sessions:
        return []

    pairs: list[tuple[Path | None, Path | None]] = []
    used_decisions: set[int] = set()

    for isess in intent_sessions:
        its = _parse_ts(isess)
        best_idx = -1
        best_diff = 999999
        for j, dsess in enumerate(decision_sessions):
            if j in used_decisions:
                continue
            dts = _parse_ts(dsess)
            diff = abs(int(its.replace("_", "")) - int(dts.replace("_", "")))
            if diff < best_diff and diff < 200:  # within ~2 minutes
                best_diff = diff
                best_idx = j
        if best_idx >= 0:
            pairs.append((isess, decision_sessions[best_idx]))
            used_decisions.add(best_idx)
        else:
            pairs.append((isess, None))

    for j, dsess in enumerate(decision_sessions):
        if j not in used_decisions:
            pairs.append((None, dsess))

    return pairs

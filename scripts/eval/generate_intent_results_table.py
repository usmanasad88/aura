#!/usr/bin/env python3
"""Aggregate AURA intent-monitor runs into a comparison results table.

Default behaviour (no arguments): walks ``logs/`` and scores every
``run_<timestamp>_<task>/`` folder whose ``settings.json`` + ``intent_monitor/``
are populated. For each run, reads task/video/model/etc. from
``settings.json``, locates the matching ground truth under
``tasks/<task>/ground_truth/<video_stem>.intent_gt.json``, computes the
temporal F1 scores, and writes the per-run outputs *inside that run folder*::

    logs/run_20260425_031254_tea_making/
        predictions.json
        intent_results.json

Additionally it builds cross-run tables under ``results/intent_evaluation/``::

    intent_results_table.csv
    intent_results_pivot.csv
    intent_results_summary.json
    intent_results_table.tex   (when --latex is passed)

A run is skipped if ``intent_results.json`` already exists in it, unless
``--force`` is passed.

Mirrors the role that ``generate_results_table.py`` played in the hcdt repo.

Usage
-----
::

    # Default: score every run_*/ under logs/ that isn't already scored
    python scripts/eval/generate_intent_results_table.py

    # Recompute all, including already-scored runs
    python scripts/eval/generate_intent_results_table.py --force

    # Legacy layout (logs/experiments/<exp>/rep_*/intent_monitor/session_*)
    python scripts/eval/generate_intent_results_table.py \\
        --experiments-dir logs/experiments

    # Ad-hoc: score an arbitrary intent_monitor directory
    python scripts/eval/generate_intent_results_table.py \\
        --sessions logs/run_20260425_031254_tea_making/intent_monitor \\
        --task tea_making \\
        --video demo_data/layup_demo/tea_making.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session_to_predictions import session_to_predictions, get_session_model
from temporal_f1_state_eval import combined_f1, DEFAULT_WEIGHTS


AURA_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = AURA_ROOT / "logs"
DEFAULT_CROSS_OUTPUT_DIR = AURA_ROOT / "results" / "intent_evaluation"
PER_RUN_RESULTS_FILENAME = "intent_results.json"
PER_RUN_PREDICTIONS_FILENAME = "predictions.json"


# ── Row model ───────────────────────────────────────────────────────────────

@dataclass
class Row:
    run_id: str
    rep: str
    task: str
    model: str
    backend: str
    intent_source: str
    frame_skip: int | None
    session: str
    session_dir: str
    run_dir: str
    n_predictions: int
    gt_path: str
    scores: dict[str, float] = field(default_factory=dict)


# ── Ground-truth resolution ─────────────────────────────────────────────────

def resolve_gt_path(task: str, video: str | None) -> Path | None:
    """Find the intent-GT JSON for a task, preferring the matching video stem.

    Falls back to any ``<anything>.intent_gt.json`` in the task's
    ``ground_truth/`` directory when the video stem is unknown or the
    per-video file is absent.
    """
    gt_dir = AURA_ROOT / "tasks" / task / "ground_truth"
    if not gt_dir.is_dir():
        return None

    if video:
        stem = Path(video).stem
        candidate = gt_dir / f"{stem}.intent_gt.json"
        if candidate.exists():
            return candidate

    matches = sorted(gt_dir.glob("*.intent_gt.json"))
    return matches[0] if matches else None


# ── Run folder discovery (new layout) ───────────────────────────────────────

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _find_intent_dir(run_dir: Path) -> Path | None:
    """Locate the intent_monitor folder containing call_*/ in a run dir.

    Handles both direct layout ``run_dir/intent_monitor/call_*`` and the
    older nested ``run_dir/intent_monitor/session_*/call_*`` layout.
    """
    direct = run_dir / "intent_monitor"
    if direct.is_dir():
        if any(direct.glob("call_*")):
            return direct
        nested = sorted(direct.glob("session_*"))
        for s in nested:
            if any(s.glob("call_*")):
                return s
    return None


def _iter_run_folders(logs_dir: Path):
    """Yield ``(run_dir, settings, intent_dir)`` for every scorable run."""
    for run_dir in sorted(logs_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        settings_path = run_dir / "settings.json"
        if not settings_path.exists():
            continue
        intent_dir = _find_intent_dir(run_dir)
        if intent_dir is None:
            continue
        settings = _read_json(settings_path)
        if not settings:
            continue
        yield run_dir, settings, intent_dir


# ── Manifest / experiment discovery (legacy layout) ─────────────────────────

def _iter_experiment_sessions(experiments_dir: Path):
    """Yield ``(manifest, rep_dir, session_dir)`` for the legacy layout."""
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        for rep_dir in sorted(exp_dir.glob("rep_*")):
            intent_root = rep_dir / "intent_monitor"
            if not intent_root.is_dir():
                continue
            for session_dir in sorted(intent_root.glob("session_*")):
                if session_dir.is_dir() and any(session_dir.glob("call_*")):
                    yield manifest, rep_dir, session_dir


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_session(
    session_dir: Path,
    gt_path: Path,
    grace_frames: int,
    predictions_path: Path,
) -> tuple[dict[str, float], int]:
    """Convert a session (or run) to predictions and run combined F1."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preds = session_to_predictions(session_dir)
    predictions_path.write_text(json.dumps(preds, indent=2))
    scores = combined_f1(gt_path, predictions_path, grace_frames=grace_frames)
    return scores, len(preds)


def _row_from_run(
    run_dir: Path,
    settings: dict[str, Any],
    intent_dir: Path,
    scores: dict[str, float],
    n_predictions: int,
    gt_path: Path,
) -> Row:
    model = (
        settings.get("intent_model")
        or settings.get("model")
        or get_session_model(intent_dir)
    )
    backend = settings.get("intent_backend") or settings.get("llm_backend") or "gemini"
    return Row(
        run_id=run_dir.name,
        rep="rep_001",
        task=settings.get("task_name") or settings.get("task") or "unknown",
        model=model,
        backend=backend,
        intent_source=settings.get("intent_source") or "llm",
        frame_skip=settings.get("frame_skip"),
        session=intent_dir.name if intent_dir.parent == run_dir else intent_dir.parent.name,
        session_dir=str(intent_dir),
        run_dir=str(run_dir),
        n_predictions=n_predictions,
        gt_path=str(gt_path),
        scores=scores,
    )


def _row_from_manifest(
    manifest: dict[str, Any],
    rep_dir: Path,
    session_dir: Path,
    scores: dict[str, float],
    n_predictions: int,
    gt_path: Path,
) -> Row:
    model = manifest.get("model") or get_session_model(session_dir)
    backend = manifest.get("intent_backend") or manifest.get("llm_backend") or "gemini"
    return Row(
        run_id=manifest.get("experiment_id") or rep_dir.parent.name,
        rep=rep_dir.name,
        task=manifest.get("task") or "unknown",
        model=model,
        backend=backend,
        intent_source=manifest.get("intent_source") or "llm",
        frame_skip=manifest.get("frame_skip"),
        session=session_dir.name,
        session_dir=str(session_dir),
        run_dir=str(rep_dir),
        n_predictions=n_predictions,
        gt_path=str(gt_path),
        scores=scores,
    )


# ── Output formatters ───────────────────────────────────────────────────────

def format_model(model: str) -> str:
    """Shorten verbose model identifiers for display."""
    m = model.lower()
    if "gemini" in m:
        if "3.1" in m and "pro" in m:
            return "Gemini 3.1 Pro"
        if "3.1" in m and "flash-lite" in m:
            return "Gemini 3.1 Flash Lite"
        if "3.1" in m and "flash" in m:
            return "Gemini 3.1 Flash"
        if "2.5" in m and "pro" in m:
            return "Gemini 2.5 Pro"
        if "2.5" in m and "flash-lite" in m:
            return "Gemini 2.5 Flash Lite"
        if "2.5" in m and "flash" in m:
            return "Gemini 2.5 Flash"
    if "qwen" in m:
        return model.split("/")[-1]
    return model


FLAT_FIELDS = [
    "run_id", "rep", "task", "model", "backend", "intent_source",
    "frame_skip", "session", "n_predictions",
    "f1_steps_completed", "f1_steps_in_progress",
    "f1_steps_pending", "f1_predicted_next_action",
    "combined_f1",
]


def write_flat_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        writer.writeheader()
        for r in rows:
            record = {
                "run_id": r.run_id,
                "rep": r.rep,
                "task": r.task,
                "model": format_model(r.model),
                "backend": r.backend,
                "intent_source": r.intent_source,
                "frame_skip": r.frame_skip if r.frame_skip is not None else "",
                "session": r.session,
                "n_predictions": r.n_predictions,
            }
            for k in ("f1_steps_completed", "f1_steps_in_progress",
                     "f1_steps_pending", "f1_predicted_next_action",
                     "combined_f1"):
                record[k] = r.scores.get(k, "")
            writer.writerow(record)


def write_pivot_csv(rows: list[Row], path: Path) -> None:
    """Pivot the combined F1 by (model, backend, intent_source) × task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r.task for r in rows})
    pivot: "OrderedDict[tuple[str, str, str], dict[str, list[float]]]" = OrderedDict()

    for r in rows:
        key = (format_model(r.model), r.backend, r.intent_source)
        bucket = pivot.setdefault(key, {t: [] for t in tasks})
        score = r.scores.get("combined_f1")
        if isinstance(score, (int, float)):
            bucket[r.task].append(float(score))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "backend", "intent_source", *tasks])
        for (model, backend, intent_src), buckets in pivot.items():
            row = [model, backend, intent_src]
            for t in tasks:
                scores = buckets.get(t, [])
                row.append(f"{sum(scores) / len(scores):.3f}" if scores else "")
            writer.writerow(row)


def write_summary_json(rows: list[Row], skipped: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": DEFAULT_WEIGHTS,
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "rows": [
            {
                "run_id": r.run_id,
                "rep": r.rep,
                "task": r.task,
                "model": r.model,
                "model_display": format_model(r.model),
                "backend": r.backend,
                "intent_source": r.intent_source,
                "frame_skip": r.frame_skip,
                "session": r.session,
                "session_dir": r.session_dir,
                "run_dir": r.run_dir,
                "n_predictions": r.n_predictions,
                "gt_path": r.gt_path,
                **r.scores,
            }
            for r in rows
        ],
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, indent=2))


def write_latex(rows: list[Row], path: Path) -> None:
    """Minimal LaTeX table of per-run combined F1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{AURA intent-monitor temporal F1 per run.\\label{tab:intent_f1}}",
        "\\begin{tabular}{lllll c}",
        "\\toprule",
        "Task & Model & Backend & Source & Run & Combined F1 \\\\",
        "\\midrule",
    ]
    for r in rows:
        model_disp = format_model(r.model).replace("_", "\\_")
        run_id = r.run_id.replace("_", "\\_")
        score = r.scores.get("combined_f1", 0.0)
        lines.append(
            f"{r.task} & {model_disp} & {r.backend} & {r.intent_source} & "
            f"\\texttt{{{run_id}}} & {score:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines))


# ── Row collection ──────────────────────────────────────────────────────────

def collect_run_rows(
    logs_dir: Path,
    grace_frames: int,
    force: bool,
) -> tuple[list[Row], list[dict[str, str]]]:
    """Score every run_*/ under logs_dir. Writes per-run outputs in-place."""
    rows: list[Row] = []
    skipped: list[dict[str, str]] = []

    for run_dir, settings, intent_dir in _iter_run_folders(logs_dir):
        task = settings.get("task_name") or settings.get("task") or ""
        video = settings.get("video_path") or settings.get("video")

        if not task:
            skipped.append({"run": str(run_dir), "reason": "No task_name in settings.json"})
            continue

        gt_path = resolve_gt_path(task, video)
        if gt_path is None:
            skipped.append({
                "run": str(run_dir),
                "reason": f"No intent_gt.json under tasks/{task}/ground_truth/",
            })
            continue

        per_run_result = run_dir / PER_RUN_RESULTS_FILENAME
        per_run_preds = run_dir / PER_RUN_PREDICTIONS_FILENAME

        if per_run_result.exists() and not force:
            cached = _read_json(per_run_result)
            if cached:
                rows.append(Row(
                    run_id=cached.get("run_id", run_dir.name),
                    rep=cached.get("rep", "rep_001"),
                    task=cached.get("task", task),
                    model=cached.get("model", settings.get("model", "unknown")),
                    backend=cached.get("backend", "gemini"),
                    intent_source=cached.get("intent_source", "llm"),
                    frame_skip=cached.get("frame_skip"),
                    session=cached.get("session", intent_dir.name),
                    session_dir=cached.get("session_dir", str(intent_dir)),
                    run_dir=str(run_dir),
                    n_predictions=cached.get("n_predictions", 0),
                    gt_path=cached.get("gt_path", str(gt_path)),
                    scores=cached.get("scores", {}),
                ))
                continue

        try:
            scores, n = score_session(intent_dir, gt_path, grace_frames, per_run_preds)
        except Exception as e:
            skipped.append({"run": str(run_dir), "reason": f"Scoring error: {e}"})
            continue

        row = _row_from_run(run_dir, settings, intent_dir, scores, n, gt_path)
        rows.append(row)

        per_run_payload = {
            "run_id": row.run_id,
            "rep": row.rep,
            "task": row.task,
            "model": row.model,
            "model_display": format_model(row.model),
            "backend": row.backend,
            "intent_source": row.intent_source,
            "frame_skip": row.frame_skip,
            "session": row.session,
            "session_dir": row.session_dir,
            "n_predictions": row.n_predictions,
            "gt_path": row.gt_path,
            "grace_frames": grace_frames,
            "weights": DEFAULT_WEIGHTS,
            "scores": row.scores,
        }
        per_run_result.write_text(json.dumps(per_run_payload, indent=2))

    return rows, skipped


def collect_experiment_rows(
    experiments_dir: Path,
    grace_frames: int,
    tmp_dir: Path,
) -> tuple[list[Row], list[dict[str, str]]]:
    rows: list[Row] = []
    skipped: list[dict[str, str]] = []
    if not experiments_dir.is_dir():
        return rows, skipped

    for manifest, rep_dir, session_dir in _iter_experiment_sessions(experiments_dir):
        task = manifest.get("task") or ""
        video = manifest.get("video")
        gt_path = resolve_gt_path(task, video)
        if gt_path is None:
            skipped.append({
                "run": str(session_dir),
                "reason": f"No intent_gt.json under tasks/{task}/ground_truth/",
            })
            continue
        pred_path = tmp_dir / f"{rep_dir.parent.name}__{rep_dir.name}__{session_dir.name}.pred.json"
        try:
            scores, n = score_session(session_dir, gt_path, grace_frames, pred_path)
        except Exception as e:
            skipped.append({"run": str(session_dir), "reason": f"Scoring error: {e}"})
            continue
        rows.append(_row_from_manifest(manifest, rep_dir, session_dir, scores, n, gt_path))
    return rows, skipped


def collect_ad_hoc_rows(
    sessions: list[Path],
    task: str | None,
    video: str | None,
    grace_frames: int,
    tmp_dir: Path,
) -> tuple[list[Row], list[dict[str, str]]]:
    rows: list[Row] = []
    skipped: list[dict[str, str]] = []

    for session_dir in sessions:
        if not session_dir.is_dir():
            skipped.append({"run": str(session_dir), "reason": "Not a directory"})
            continue

        # If the user pointed at a run_*/ folder, dig into it.
        effective_intent = session_dir
        settings = {}
        if (session_dir / "settings.json").exists():
            settings = _read_json(session_dir / "settings.json")
            intent_dir = _find_intent_dir(session_dir)
            if intent_dir is not None:
                effective_intent = intent_dir

        eff_task = task or settings.get("task_name") or settings.get("task")
        eff_video = video or settings.get("video_path") or settings.get("video")

        if eff_task is None:
            skipped.append({
                "run": str(session_dir),
                "reason": "--task required (and no settings.json inside)",
            })
            continue

        gt_path = resolve_gt_path(eff_task, eff_video)
        if gt_path is None:
            skipped.append({
                "run": str(session_dir),
                "reason": f"No intent_gt.json under tasks/{eff_task}/ground_truth/",
            })
            continue

        pred_path = tmp_dir / f"{session_dir.name}.pred.json"
        try:
            scores, n = score_session(effective_intent, gt_path, grace_frames, pred_path)
        except Exception as e:
            skipped.append({"run": str(session_dir), "reason": f"Scoring error: {e}"})
            continue

        rows.append(Row(
            run_id=session_dir.name,
            rep="rep_001",
            task=eff_task,
            model=settings.get("model") or get_session_model(effective_intent),
            backend=settings.get("llm_backend") or "unknown",
            intent_source=settings.get("intent_source") or "llm",
            frame_skip=settings.get("frame_skip"),
            session=effective_intent.name,
            session_dir=str(effective_intent),
            run_dir=str(session_dir),
            n_predictions=n,
            gt_path=str(gt_path),
            scores=scores,
        ))

    return rows, skipped


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate intent-monitor runs into a comparison table",
    )
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR,
                        help="Root logs dir to scan for run_*/ folders "
                             "(default: logs/)")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Also aggregate the legacy "
                             "logs/experiments/<exp>/rep_*/intent_monitor/session_* layout")
    parser.add_argument("--sessions", type=Path, nargs="*", default=[],
                        help="Ad-hoc run_*/ or intent_monitor dirs to include")
    parser.add_argument("--task", default=None,
                        help="Task name for --sessions when no settings.json is present")
    parser.add_argument("--video", default=None,
                        help="Video path (for GT stem lookup) when using --sessions")
    parser.add_argument("--grace-frames", type=int, default=30,
                        help="Temporal F1 look-ahead tolerance (default 30)")
    parser.add_argument("--cross-output-dir", type=Path, default=DEFAULT_CROSS_OUTPUT_DIR,
                        help="Where to write the cross-run CSV / JSON tables "
                             "(default: results/intent_evaluation/)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute runs even if intent_results.json already exists")
    parser.add_argument("--latex", action="store_true",
                        help="Also emit a LaTeX table")
    parser.add_argument("--no-cross-output", action="store_true",
                        help="Skip writing the cross-run aggregate tables")
    args = parser.parse_args()

    tmp_dir = args.cross_output_dir / "predictions"
    all_rows: list[Row] = []
    all_skipped: list[dict[str, str]] = []

    if args.logs_dir.is_dir():
        rows, skipped = collect_run_rows(args.logs_dir, args.grace_frames, args.force)
        all_rows.extend(rows)
        all_skipped.extend(skipped)

    if args.experiments_dir is not None:
        rows, skipped = collect_experiment_rows(args.experiments_dir, args.grace_frames, tmp_dir)
        all_rows.extend(rows)
        all_skipped.extend(skipped)

    if args.sessions:
        rows, skipped = collect_ad_hoc_rows(
            args.sessions, args.task, args.video, args.grace_frames, tmp_dir,
        )
        all_rows.extend(rows)
        all_skipped.extend(skipped)

    all_rows.sort(key=lambda r: (r.task, r.model, r.backend, r.run_id))

    if not all_rows:
        print("No scorable intent-monitor runs found.")
        if all_skipped:
            print(f"Skipped {len(all_skipped)}:")
            for s in all_skipped:
                print(f"  - {s['run']}: {s['reason']}")
        sys.exit(0 if not all_skipped else 1)

    if not args.no_cross_output:
        out = args.cross_output_dir
        flat = out / "intent_results_table.csv"
        pivot = out / "intent_results_pivot.csv"
        summary = out / "intent_results_summary.json"
        write_flat_csv(all_rows, flat)
        write_pivot_csv(all_rows, pivot)
        write_summary_json(all_rows, all_skipped, summary)
        if args.latex:
            write_latex(all_rows, out / "intent_results_table.tex")
        print(f"Scored {len(all_rows)} run(s); skipped {len(all_skipped)}.")
        print(f"  flat  -> {flat}")
        print(f"  pivot -> {pivot}")
        print(f"  json  -> {summary}")
        if args.latex:
            print(f"  latex -> {out / 'intent_results_table.tex'}")

    print("\nTop rows:")
    for r in all_rows[:15]:
        print(f"  [{r.task}] {format_model(r.model):22s} "
              f"{r.backend:8s} {r.intent_source:13s} "
              f"combined_f1={r.scores.get('combined_f1', 0):.3f} "
              f"(n={r.n_predictions}) {r.run_id}")

    if all_skipped:
        print(f"\nSkipped {len(all_skipped)}:")
        for s in all_skipped:
            print(f"  - {s['run']}: {s['reason']}")


if __name__ == "__main__":
    main()

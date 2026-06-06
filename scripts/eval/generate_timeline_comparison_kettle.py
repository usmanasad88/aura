#!/usr/bin/env python3
"""Parameter-aware ablation timeline comparison for the kettle_tea_making task.

Same figure + A-Score / TWSS tables as
:mod:`generate_timeline_comparison`, but **argument-aware**: kettle skills are
parametric (``pick_and_place_item(item=cup)``, ``return_item_to_storage(item=
milk_container)``, …), so two interventions that share a skill id but move
different objects must be treated as *different* actions. The base comparison
script collapses ``skill(args)`` to the bare skill via ``_strip_args`` — that
is correct for hand_layup (no meaningful args) but wrong here.

What this script changes versus the base one
--------------------------------------------
1. Predictions: the base ``load_predictions`` keeps only ``action_id`` and
   discards the decision's ``parameters``. Here we read ``parameters`` from
   ``response_parsed.json`` and fold the meaningful args into a canonical
   action label, so distinct-object picks neither merge into one span nor
   match each other.
2. GT: interventions are loaded with the same canonical label from their
   ``args`` (mirrors ``robot_skills.json`` arg keys).
3. Matching / scoring: ``_strip_args`` is overridden to the identity, so
   A_act, TWSS per-skill unions, and PSA all compare the full parameterized
   label. Start-time matching (``match_predictions``) is unchanged.

Noise args (default: ``safe``, the retreat pose) are ignored on both sides so
they cannot cause spurious mismatches; tune with ``--ignore-args``.

Usage::

    python scripts/eval/generate_timeline_comparison_kettle.py \
        --gt tasks/kettle_tea_making/ground_truth/kettle_tea_making_2_overlay.robot_gt.json \
        --run logs/run_A="LLM intent + perception" \
        --run logs/run_B="GT intent + perception" \
        --output figures/generated/fig_ablation_timeline_comparison_kettle.pdf

Full command used for the kettle ablation comparison (all current runs)::

    cd /home/mani/Repos/aura/scripts/eval
    R=/home/mani/Repos/aura/logs/Kettle_Tea_Making
    python generate_timeline_comparison_kettle.py \
        --gt /home/mani/Repos/aura/tasks/kettle_tea_making/ground_truth/kettle_tea_making_2_overlay.robot_gt.json \
        --run "$R/gemini_3.5_standard_kettle_tea_making=Gemini 3.5 (standard)" \
        --run "$R/gemini_3.1_pro_standard_kettle_tea_making=Gemini 3.1 Pro (standard)" \
        --run "$R/gemini_3.1_flash_lite_prev_gt_kettle_tea_making=Gemini 3.1 Flash-Lite (prev GT)" \
        --run "$R/gemini_3.1_flash_lite_standard_kettle_tea_making=Gemini 3.1 Flash-Lite (standard)" \
        --output /home/mani/Repos/aura/figures/generated/fig_ablation_timeline_comparison_kettle
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Base comparison module: we reuse its scoring + plotting wholesale and only
# swap in parameter-aware data loaders + an args-preserving action key.
import generate_timeline_comparison as gtc
from generate_timeline_comparison import (
    DEFAULT_TWSS_BLIP_TAX,
    DEFAULT_TWSS_WAIT_WEIGHT,
    DEFAULT_WEIGHTS,
    MATCH_TOLERANCE_SEC,
    TUNED_TWSS_BETA,
    TUNED_TWSS_GAMMA,
    TUNED_TWSS_WAIT_WEIGHT,
    DEFAULT_TWSS_GAMMA,
    DEFAULT_TWSS_ALPHA,
    DEFAULT_TWSS_BETA,
    TUNED_TWSS_BLIP_TAX,
    AScore,
    TWSS,
    _parse_run_arg,
    compute_a_score,
    compute_twss,
    format_score_table,
    format_twss_table,
)
from generate_timeline import (
    AURA_ROOT,
    Event,
    _find_call_dir,
    _read_json,
    _resolve_robot_gt,
)

DEFAULT_IGNORE_ARGS = ("safe",)

# Kettle-tuned TWSS configuration. Differs from the hand-layup tuned config in
# two task-motivated ways:
#   * end_weight = 0  — no early-stop penalty. On the live robot a triggered
#     skill runs to completion, so the predicted span's *end* (an artifact of
#     how long the LLM kept re-asserting the action) is not graded; only the
#     trigger *start* time matters.
#   * heavier pollution penalty — false positives during wait time are
#     weighted much more (high wait_weight, sharper wait decay β, larger
#     per-blip tax δ) so spurious interventions are strongly penalised.
KETTLE_TWSS_GAMMA = 0.5
KETTLE_TWSS_BETA = 0.6
KETTLE_TWSS_WAIT_WEIGHT = 4.0
KETTLE_TWSS_BLIP_TAX = 6.0
KETTLE_TWSS_END_WEIGHT = 0.0


def _parse_action(action: str) -> tuple[str, dict[str, str]]:
    """Split a canonical ``skill(k=v, k=v)`` label back into (skill, args)."""
    if action.endswith(")") and "(" in action:
        skill, inner = action[:-1].split("(", 1)
        args = {}
        for part in inner.split(", "):
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip()
        return skill, args
    return action, {}


# Skills whose moved object distinguishes the action (so item enters the key /
# colour). All other skills colour by bare skill id.
_PARAMETRIC_SKILLS = {"pick_and_place_item", "return_item_to_storage"}


def action_key(action: str) -> str:
    """Colour/identity key for an event: bare skill, plus the item for the two
    genuinely parametric skills (e.g. ``pick_and_place_item(cup)``)."""
    skill, args = _parse_action(action)
    item = args.get("item")
    if skill in _PARAMETRIC_SKILLS and item:
        return f"{skill}({item})"
    return skill


def legend_label(key: str) -> str:
    """Human-readable legend text for an ``action_key``."""
    if key.startswith("pick_and_place_item(") and key.endswith(")"):
        return f"Pick & place: {key[len('pick_and_place_item('):-1]}"
    if key.startswith("return_item_to_storage(") and key.endswith(")"):
        return f"Return to storage: {key[len('return_item_to_storage('):-1]}"
    return {
        "bring_water_bottle": "Bring water bottle",
        "return_water_bottle": "Return water bottle",
        "close_lid_and_turn_on": "Close lid & turn on",
        "pick_and_place_item": "Pick & place",
        "return_item_to_storage": "Return to storage",
    }.get(key, key.replace("_", " "))


def _build_action_colors(gt_events: list[Event],
                         model_preds: dict[str, list[Event]]) -> dict[str, tuple]:
    """Deterministic colour per ``action_key``. GT actions are ordered first
    (by first appearance), then any prediction-only variants, so the GT's set
    keeps stable colours regardless of which runs are plotted."""
    order: list[str] = []
    for e in sorted(gt_events, key=lambda e: e.start):
        k = action_key(e.action)
        if k not in order:
            order.append(k)
    extra: set[str] = set()
    for evs in model_preds.values():
        for e in evs:
            k = action_key(e.action)
            if k not in order:
                extra.add(k)
    order += sorted(extra)
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)
    return {k: palette[i % len(palette)] for i, k in enumerate(order)}


def plot_action_colored_timeline(gt_events: list[Event],
                                 model_preds: dict[str, list[Event]],
                                 total_duration: float,
                                 title: str):
    """One track per source (GT + each model). Bars are coloured by action
    (no text on bars); a legend maps colour -> action."""
    colors = _build_action_colors(gt_events, model_preds)
    robot_gt = [e for e in gt_events if e.agent == "robot"]
    tracks = ["GT Robot"] + list(model_preds.keys())
    n = len(tracks)

    fig, ax = plt.subplots(figsize=(13, 1.4 + 0.95 * n))

    def yof(track_idx: int) -> int:   # GT (idx 0) on top
        return n - 1 - track_idx

    for ti in range(n):
        if ti % 2 == 1:
            y = yof(ti)
            ax.axhspan(y - 0.5, y + 0.5, color=(0.95, 0.95, 0.95), zorder=0)

    for ti, track in enumerate(tracks):
        y = yof(ti)
        evs = robot_gt if track == "GT Robot" else model_preds[track]
        for e in evs:
            c = colors.get(action_key(e.action), (0.6, 0.6, 0.6))
            ax.add_patch(mpatches.Rectangle(
                (e.start, y - 0.38), max(e.end - e.start, 0.6), 0.76,
                facecolor=c, edgecolor="white", linewidth=0.5, zorder=3))

    ax.set_xlim(0, total_duration * 1.02)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([yof(t) for t in range(n)])
    ax.set_yticklabels(tracks)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=colors[k], edgecolor="white",
                              label=legend_label(k)) for k in colors]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, title="Robot action")
    fig.tight_layout()
    return fig


# ── Canonical, argument-aware action labels ───────────────────────────────────

def canonical_label(skill: str, args: dict | None, ignore: set[str]) -> str:
    """``skill`` plus its meaningful args as ``skill(k=v, k=v)`` (keys sorted).

    Args whose key is in ``ignore`` or whose value is empty are dropped, so
    the label is identical for the same semantic action regardless of source
    (GT vs prediction) or original dict ordering.
    """
    kept = {
        k: v for k, v in (args or {}).items()
        if k not in ignore and v not in (None, "")
    }
    if not kept:
        return skill
    inner = ", ".join(f"{k}={kept[k]}" for k in sorted(kept))
    return f"{skill}({inner})"


def load_robot_gt_param(gt_path: Path, ignore: set[str]) -> tuple[list[Event], float]:
    """Like ``generate_timeline.load_robot_gt`` but with canonical arg labels."""
    data = json.loads(gt_path.read_text())
    events: list[Event] = []
    for iv in data.get("interventions", []):
        events.append(Event(
            action=canonical_label(iv.get("skill", ""), iv.get("args"), ignore),
            start=float(iv["t_start"]),
            end=float(iv["t_end"]),
            agent="robot",
        ))
    return events, float(data.get("duration_sec", 0.0) or 0.0)


def load_predictions_param(session_dir: Path, ignore: set[str]) -> list[Event]:
    """Load 'act' decisions, folding decision ``parameters`` into the label.

    Mirrors ``generate_timeline.load_predictions`` (same +2.0s span padding),
    with two differences: spans merge on the *parameterized* label so two
    picks of different items stay distinct, and ``wait`` (and any other
    non-``act``) cycles are kept as span *breakpoints* rather than dropped —
    so the same action separated by an intervening wait yields two spans with
    a gap, not one fused span.
    """
    # Walk all calls in time order; ``None`` marks a non-act (wait) cycle.
    seq: list[tuple[float, str | None]] = []
    for call_dir in sorted(session_dir.glob("call_*")):
        meta_path = call_dir / "meta.json"
        parsed_path = call_dir / "response_parsed.json"
        if not meta_path.exists() or not parsed_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            parsed = json.loads(parsed_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if parsed is None:
            continue
        decision = parsed.get("decision") or meta.get("decision", "")
        t = meta.get("timestamp_sec", 0.0)
        if decision == "act":
            action_id = parsed.get("action_id", "")
            params = parsed.get("parameters") or {}
            label = canonical_label(action_id, params, ignore) if action_id else None
        elif not decision or decision == "wait":
            label = None
        else:  # a non-act, non-wait decision id (rare) — treat as its own action
            label = canonical_label(decision, {}, ignore)
        seq.append((t, label))

    if not seq:
        return []

    seq.sort(key=lambda x: x[0])
    events: list[Event] = []
    cur_action: str | None = None
    cur_start = 0.0
    prev_t = 0.0
    for t, label in seq:
        if label != cur_action:
            if cur_action is not None:  # close the span that just ended
                events.append(Event(action=cur_action, start=cur_start,
                                    end=prev_t + 2.0, agent="robot"))
            cur_action = label
            cur_start = t
        prev_t = t
    if cur_action is not None:
        events.append(Event(action=cur_action, start=cur_start,
                            end=prev_t + 2.0, agent="robot"))
    return events


def _collect_predictions_param(run_dir: Path, ignore: set[str]) -> list[Event]:
    dec_dir = _find_call_dir(run_dir, "decision_engine")
    if dec_dir is None:
        print(f"  [skip] no decision_engine session in {run_dir}", file=sys.stderr)
        return []
    return load_predictions_param(dec_dir, ignore)


# ── Per-decision CSV (decision + intent + perception) ─────────────────────────

_LOC_RE = re.compile(r"-?\s*(\w+)_location:\s*(\S+)")


def _read_perception_locations(call_dir: Path) -> dict[str, str]:
    """Parse ``<obj>_location: <region>`` lines from a decision prompt.

    These are the perception-derived object locations the decision model
    actually saw at that cycle (e.g. ``cup -> storage_area``).
    """
    p = call_dir / "prompt.txt"
    if not p.exists():
        return {}
    locs: dict[str, str] = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = _LOC_RE.match(line.strip())
        if m:
            locs[m.group(1)] = m.group(2)
    return locs


def _load_intent_calls(run_dir: Path) -> list[dict]:
    """Time-sorted intent-monitor outputs (current_action, human_state, reasoning)."""
    idir = run_dir / "intent_monitor"
    calls: list[dict] = []
    for cd in sorted(idir.glob("call_*")):
        try:
            meta = json.loads((cd / "meta.json").read_text())
            parsed = json.loads((cd / "response_parsed.json").read_text())
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            continue
        if not parsed:
            continue
        calls.append({
            "call": cd.name,
            "t": float(meta.get("timestamp_sec", 0.0)),
            "current_action": parsed.get("current_action", ""),
            "human_state": parsed.get("human_state", ""),
            "reasoning": parsed.get("reasoning", ""),
        })
    calls.sort(key=lambda c: c["t"])
    return calls


def _match_intent(intent_calls: list[dict], t: float) -> dict | None:
    """Intent state in effect at decision time ``t`` (most recent at/before t)."""
    if not intent_calls:
        return None
    prior = [c for c in intent_calls if c["t"] <= t + 1e-6]
    if prior:
        return prior[-1]
    return min(intent_calls, key=lambda c: abs(c["t"] - t))


def write_decisions_csv(runs: list[tuple[Path, str]], csv_path: Path,
                        ignore: set[str]) -> int:
    """Write one row per decision cycle, joining the decision's reasoning with
    the concurrent intent reasoning and the perception object-locations.

    Returns the number of rows written.
    """
    fields = [
        "run", "dec_call", "t_sec", "decision", "action_id", "parameters",
        "confidence", "decision_reasoning",
        "intent_call", "intent_t_sec", "intent_current_action",
        "intent_human_state", "intent_reasoning",
        "perception_locations",
    ]
    rows: list[dict] = []
    for run_dir, label in runs:
        intent_calls = _load_intent_calls(run_dir)
        dec_dir = run_dir / "decision_engine"
        for cd in sorted(dec_dir.glob("call_*")):
            try:
                meta = json.loads((cd / "meta.json").read_text())
                parsed = json.loads((cd / "response_parsed.json").read_text())
            except (json.JSONDecodeError, OSError, FileNotFoundError):
                continue
            if not parsed:
                continue
            t = float(meta.get("timestamp_sec", 0.0))
            decision = parsed.get("decision") or meta.get("decision", "")
            action_id = parsed.get("action_id", "") if decision == "act" else ""
            params = parsed.get("parameters") or {}
            ic = _match_intent(intent_calls, t)
            locs = _read_perception_locations(cd)
            rows.append({
                "run": label,
                "dec_call": cd.name,
                "t_sec": f"{t:.2f}",
                "decision": decision,
                "action_id": action_id,
                "parameters": json.dumps(params, separators=(",", ":")) if params else "",
                "confidence": parsed.get("confidence", ""),
                "decision_reasoning": parsed.get("reasoning", ""),
                "intent_call": ic["call"] if ic else "",
                "intent_t_sec": f"{ic['t']:.2f}" if ic else "",
                "intent_current_action": ic["current_action"] if ic else "",
                "intent_human_state": ic["human_state"] if ic else "",
                "intent_reasoning": ic["reasoning"] if ic else "",
                "perception_locations": "; ".join(
                    f"{k}={v}" for k, v in sorted(locs.items())),
            })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _resolve_gt_for_runs(runs, task_override, gt_override) -> Path:
    if gt_override:
        return gt_override
    for run_dir, _ in runs:
        settings = _read_json(run_dir / "settings.json")
        if not settings:
            continue
        task = task_override or settings.get("task_name") or settings.get("task")
        video = settings.get("video_path") or settings.get("video")
        if not task:
            continue
        gt = _resolve_robot_gt(task, video)
        if gt and gt.exists():
            return gt
    if task_override:
        gt = _resolve_robot_gt(task_override, None)
        if gt and gt.exists():
            return gt
    raise FileNotFoundError(
        "Could not locate a robot_gt.json for the supplied runs. Pass --gt "
        "explicitly. (Kettle GT must be annotated first with "
        "scripts/annotate_robot_gt.py.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter-aware ablation timeline comparison (kettle_tea_making).")
    parser.add_argument("--run", action="append", default=None, required=False,
                        help="Ablation run dir, optionally `path=label`. Repeatable.")
    parser.add_argument("--task", default="kettle_tea_making",
                        help="Task name for GT lookup (default: kettle_tea_making).")
    parser.add_argument("--gt", type=Path, default=None,
                        help="Explicit robot_gt.json path (skips auto-resolve).")
    parser.add_argument("--ignore-args", type=str, default=",".join(DEFAULT_IGNORE_ARGS),
                        help="Comma-separated arg keys to ignore when comparing "
                             f"actions (default: {','.join(DEFAULT_IGNORE_ARGS)}).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (PNG sibling also written). Default: "
                             "figures/generated/fig_ablation_timeline_comparison_kettle.pdf")
    parser.add_argument("--title", type=str,
                        default="Kettle Tea Making — Ablation Intervention Timeline",
                        help="Figure title.")
    parser.add_argument("--show-legend", action="store_true",
                        help="Include the TP/FP/FN legend.")
    parser.add_argument("--weights", type=str, default=None,
                        help="A-Score weights 'w_t,w_a,w_n' (sum to 1).")
    parser.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE_SEC,
                        help=f"Start-time match tolerance (default: {MATCH_TOLERANCE_SEC}).")
    parser.add_argument("--twss-alpha", type=float, default=DEFAULT_TWSS_ALPHA)
    parser.add_argument("--twss-beta", type=float, default=DEFAULT_TWSS_BETA)
    parser.add_argument("--twss-beta-tuned", type=float, default=KETTLE_TWSS_BETA)
    parser.add_argument("--twss-gamma", type=float, default=KETTLE_TWSS_GAMMA)
    parser.add_argument("--twss-wait-weight", type=float, default=KETTLE_TWSS_WAIT_WEIGHT)
    parser.add_argument("--twss-blip-tax", type=float, default=KETTLE_TWSS_BLIP_TAX)
    parser.add_argument("--twss-end-weight", type=float, default=KETTLE_TWSS_END_WEIGHT,
                        help="Weight on the end-time term of the per-event timing "
                             "decay. 0 = no early-stop penalty (default for kettle: "
                             f"{KETTLE_TWSS_END_WEIGHT}); 1 = symmetric (hand-layup).")
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--decisions-csv", type=Path, default=None,
                        help="CSV of per-decision rows (decision + intent + "
                             "perception). Default: <output_stem>.decisions.csv")
    args = parser.parse_args()

    # Args-aware action comparison: full canonical label is the key.
    gtc._strip_args = lambda action: action

    if args.weights:
        try:
            ws = tuple(float(x) for x in args.weights.split(","))
        except ValueError:
            print(f"Error: --weights must be three floats, got {args.weights!r}",
                  file=sys.stderr)
            sys.exit(1)
        if len(ws) != 3 or abs(sum(ws) - 1.0) > 1e-6:
            print(f"Error: --weights must have 3 values summing to 1, got {ws}",
                  file=sys.stderr)
            sys.exit(1)
        weights = ws
    else:
        weights = DEFAULT_WEIGHTS

    ignore = {a.strip() for a in args.ignore_args.split(",") if a.strip()}

    if not args.run:
        print("Error: no --run given. Kettle has no pre-named ablation dirs, so "
              "you must pass run directories explicitly, e.g.\n"
              "  --run logs/run_..._kettle_tea_making=\"LLM intent + perception\"",
              file=sys.stderr)
        sys.exit(1)
    runs = [_parse_run_arg(s) for s in args.run]

    missing = [str(p) for p, _ in runs if not p.is_dir()]
    if missing:
        print("Error: run directories not found:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        sys.exit(1)

    gt_path = _resolve_gt_for_runs(runs, args.task, args.gt)
    print(f"Using robot GT: {gt_path}")
    print(f"Ignoring args when comparing: {sorted(ignore) or '(none)'}")
    gt_events, total_duration = load_robot_gt_param(gt_path, ignore)
    if total_duration <= 0:
        total_duration = max((e.end for e in gt_events), default=270.0)

    model_preds: dict[str, list[Event]] = {}
    for run_dir, label in runs:
        print(f"  loading {run_dir.name} ({label})...")
        preds = _collect_predictions_param(run_dir, ignore)
        model_preds[label] = preds
        if preds:
            total_duration = max(total_duration, max(e.end for e in preds))

    if not any(model_preds.values()):
        print("Error: no predictions loaded from any run.", file=sys.stderr)
        sys.exit(1)

    fig = plot_action_colored_timeline(
        gt_events, model_preds,
        total_duration=total_duration,
        title=args.title,
    )

    out = args.output
    if out is None:
        out_dir = AURA_ROOT / "figures" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_ablation_timeline_comparison_kettle.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    png_out = str(out)
    png_out = png_out[:-4] + ".png" if png_out.lower().endswith(".pdf") else png_out + ".png"
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Saved comparison timeline to:\n  {out}\n  {png_out}")

    scores: dict[str, AScore] = {}
    vanilla_exp: dict[str, TWSS] = {}
    tuned_exp: dict[str, TWSS] = {}
    for label, preds in model_preds.items():
        scores[label] = compute_a_score(gt_events, preds,
                                        total_duration=total_duration,
                                        weights=weights, tolerance=args.tolerance)
        vanilla_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta,
            gamma=DEFAULT_TWSS_GAMMA, wait_weight=DEFAULT_TWSS_WAIT_WEIGHT,
            blip_tax_delta=DEFAULT_TWSS_BLIP_TAX)
        tuned_exp[label] = compute_twss(
            gt_events, preds, total_duration=total_duration,
            alpha=args.twss_alpha, beta=args.twss_beta_tuned,
            gamma=args.twss_gamma, wait_weight=args.twss_wait_weight,
            blip_tax_delta=args.twss_blip_tax, end_weight=args.twss_end_weight)

    print("\nA-Score (Appropriateness Score) — weights "
          f"w_t={weights[0]:.3f}, w_a={weights[1]:.3f}, w_n={weights[2]:.3f}, "
          f"match tolerance={args.tolerance:.1f}s")
    print(format_score_table(scores))

    print(f"\nVanilla TWSS — α={args.twss_alpha}, β={args.twss_beta}, "
          f"γ={DEFAULT_TWSS_GAMMA}, w_wait={DEFAULT_TWSS_WAIT_WEIGHT}, "
          f"δ={DEFAULT_TWSS_BLIP_TAX} (exp wait penalty)")
    print(format_twss_table(vanilla_exp))

    print(f"\nTuned TWSS — α={args.twss_alpha}, β={args.twss_beta_tuned}, "
          f"γ={args.twss_gamma}, w_wait={args.twss_wait_weight}, "
          f"δ={args.twss_blip_tax} (exp wait penalty)")
    print(format_twss_table(tuned_exp))

    scores_out = args.scores_output or out.with_suffix(".scores.json")
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_gt": str(gt_path),
        "ignore_args": sorted(ignore),
        "weights": {"w_t": weights[0], "w_a": weights[1], "w_n": weights[2]},
        "match_tolerance_sec": args.tolerance,
        "twss_alpha": args.twss_alpha,
        "twss_vanilla": {
            "beta": args.twss_beta, "gamma": DEFAULT_TWSS_GAMMA,
            "wait_weight": DEFAULT_TWSS_WAIT_WEIGHT,
            "blip_tax_delta": DEFAULT_TWSS_BLIP_TAX,
        },
        "twss_tuned": {
            "beta": args.twss_beta_tuned, "gamma": args.twss_gamma,
            "wait_weight": args.twss_wait_weight,
            "blip_tax_delta": args.twss_blip_tax,
            "end_weight": args.twss_end_weight,
        },
        "ablations": {
            label: {
                "ascore": asdict(scores[label]),
                "twss_vanilla": asdict(vanilla_exp[label]),
                "twss_tuned": asdict(tuned_exp[label]),
            } for label in scores
        },
    }
    scores_out.write_text(json.dumps(payload, indent=2))
    print(f"Saved A-Score / TWSS JSON to:\n  {scores_out}")

    # Per-decision CSV: decision reasoning ⨉ intent reasoning ⨉ perception.
    csv_out = args.decisions_csv or (out.parent / f"{out.stem}.decisions.csv")
    n_rows = write_decisions_csv(runs, csv_out, ignore)
    print(f"Saved {n_rows} decision rows to:\n  {csv_out}")


if __name__ == "__main__":
    main()

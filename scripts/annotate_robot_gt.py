#!/usr/bin/env python3
"""Ground-truth annotator for robot interventions (``*.robot_gt.json``).

Companion to :mod:`annotate_ground_truth` (which annotates per-frame intent
state). This tool annotates the *robot* ground truth: contiguous time
intervals during which the teleoperator drove the robot through one skill,
written as schema-v1.0 ``interventions: [{id, skill, args, t_start, t_end,
rationale}, ...]`` consumed by ``scripts/eval/generate_timeline.py``.

Unlike the hand_layup robot GT (whose skills were fixed programs, so every
``args`` was ``{}``), parametric skills such as ``pick_and_place_item`` carry
arguments (``item=``, ``destination=``, ``safe=``). This tool reads each
skill's parameter schema from ``tasks/<task>/config/robot_skills.json`` and
lets you pick argument values (from ``valid_values`` / ``allowed_values``) or
type free numeric/string values.

Workflow
--------
1. Scrub to where an intervention begins, press ``[`` to set t_start.
2. Press ``k`` to pick the skill (overlay list of skill ids).
3. Press ``p`` to fill in each argument the skill declares.
4. Scrub to the end of the intervention, press ``]`` to set t_end.
5. Press ``c`` (or enter) to commit the working intervention to the list.
6. ``S`` to save. Existing files load back in so you can refine.

Controls
--------
Navigation
    a / d           step backward / forward 1 frame
    A / D (or j/l)  step backward / forward N frames (frame-skip)
    J / L           fast jump backward / forward (fast-mult x frame-skip)
    , / .           jump to prev / next intervention boundary
    g               type a frame number to jump to

Working intervention
    k               pick skill (overlay)
    p               add / edit an argument (overlay param picker, then value)
    P               clear all arguments
    [               set t_start to current frame
    ]               set t_end to current frame
    R               type a rationale string
    c / enter       commit working intervention to the list
    n               clear the working intervention (start fresh)

Committed list
    w / s           select previous / next committed intervention
    e               load selected intervention back into the working slot
    x               delete selected intervention

File
    S (shift+s)     save to disk
    q               quit (prompts if unsaved)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_scripts_dir = Path(__file__).resolve().parent
for _p in (str(_project_root), str(_scripts_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the shared GUI helpers from the intent annotator.
from annotate_ground_truth import (  # noqa: E402
    compose,
    overlay_picker,
    prompt_terminal,
    put,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("annotate_robot_gt")

SCHEMA_VERSION = "1.0"


# ─── Skill schema loading ──────────────────────────────────────────────────

@dataclass
class SkillParam:
    name: str
    type: str = "string"
    valid_values: List[str] = field(default_factory=list)
    default: Any = None
    required: bool = False


@dataclass
class Skill:
    id: str
    name: str
    params: List[SkillParam] = field(default_factory=list)


def _normalize_params(raw: Any) -> List[SkillParam]:
    """Normalize a skill's ``parameters`` into a list of :class:`SkillParam`.

    Handles both shapes seen in the repo:
      - list form (hand_layup): ``[{"name": .., "allowed_values": [..]}, ..]``
      - dict form (kettle):     ``{"item": {"valid_values": [..]}, ..}``
    """
    params: List[SkillParam] = []
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = [(p.get("name", ""), p) for p in raw if isinstance(p, dict)]
    else:
        return params

    for pname, defn in items:
        if not pname or not isinstance(defn, dict):
            continue
        vals = defn.get("valid_values") or defn.get("allowed_values") or []
        params.append(SkillParam(
            name=pname,
            type=defn.get("type", "string"),
            valid_values=[str(v) for v in vals],
            default=defn.get("default"),
            required=bool(defn.get("required", False)),
        ))
    return params


def load_skills(config_dir: Path, skills_filename: str) -> List[Skill]:
    path = config_dir / skills_filename
    data = json.loads(path.read_text(encoding="utf-8"))
    skills: List[Skill] = []
    for sk in data.get("skills", []) or []:
        sid = sk.get("id")
        if not sid:
            continue
        skills.append(Skill(
            id=sid,
            name=sk.get("name", sid),
            params=_normalize_params(sk.get("parameters")),
        ))
    return skills


# ─── Annotation model ──────────────────────────────────────────────────────

def _empty_working() -> Dict[str, Any]:
    return {"skill": "", "args": {}, "t_start": None, "t_end": None, "rationale": ""}


@dataclass
class RobotGT:
    task_id: str
    video: str
    duration_sec: float
    operator_id: str = "manual_annotation"
    description: str = ""
    notes: str = ""
    interventions: List[Dict[str, Any]] = field(default_factory=list)

    def sorted_interventions(self) -> List[Dict[str, Any]]:
        return sorted(self.interventions, key=lambda iv: float(iv.get("t_start") or 0.0))

    def to_dict(self) -> Dict[str, Any]:
        ordered = self.sorted_interventions()
        renumbered: List[Dict[str, Any]] = []
        for i, iv in enumerate(ordered, start=1):
            out = {
                "id": f"iv_{i:03d}",
                "skill": iv.get("skill", ""),
                "args": iv.get("args", {}) or {},
                "t_start": round(float(iv.get("t_start") or 0.0), 2),
                "t_end": round(float(iv.get("t_end") or 0.0), 2),
                "rationale": iv.get("rationale", "") or "",
            }
            renumbered.append(out)
        return {
            "task_id": self.task_id,
            "video": self.video,
            "duration_sec": round(float(self.duration_sec), 2),
            "operator_id": self.operator_id,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": self.description,
            "interventions": renumbered,
            "notes": self.notes,
        }


# ─── Rendering ─────────────────────────────────────────────────────────────

def _fmt_args(args: Dict[str, Any]) -> str:
    if not args:
        return "{}"
    return ", ".join(f"{k}={v}" for k, v in args.items())


def render_panel(
    height: int,
    width: int,
    skills_by_id: Dict[str, Skill],
    working: Dict[str, Any],
    interventions: List[Dict[str, Any]],
    selected_idx: int,
    frame_num: int,
    total_frames: int,
    timestamp: float,
    dirty: bool,
    message: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    put(panel, "AURA Robot GT Annotator", (12, 26), 0.7, (255, 255, 255), 2)
    dirty_tag = " *" if dirty else ""
    put(panel, f"frame {frame_num}/{total_frames - 1}  t={timestamp:.2f}s{dirty_tag}",
        (12, 50), 0.5, (180, 220, 255))

    # ── Working intervention ────────────────────────────────────────────
    y = 80
    put(panel, "WORKING intervention", (12, y), 0.5, (255, 220, 140))
    y += 22
    sk_id = working.get("skill") or "(none)"
    sk = skills_by_id.get(sk_id)
    sk_label = f"{sk_id}" + (f"  — {sk.name}" if sk else "")
    put(panel, f"skill: {sk_label}", (20, y), 0.46, (200, 230, 200)); y += 20
    put(panel, f"args:  {_fmt_args(working.get('args') or {})}", (20, y), 0.44, (200, 230, 200)); y += 20
    ts = working.get("t_start"); te = working.get("t_end")
    ts_s = f"{ts:.2f}s" if ts is not None else "—"
    te_s = f"{te:.2f}s" if te is not None else "—"
    put(panel, f"t_start: {ts_s}    t_end: {te_s}", (20, y), 0.44, (200, 230, 200)); y += 20
    rat = working.get("rationale") or ""
    if len(rat) > 60:
        rat = rat[:57] + "..."
    put(panel, f"why:   {rat or '—'}", (20, y), 0.42, (180, 200, 180)); y += 18

    # Show declared params for the chosen skill
    if sk and sk.params:
        names = ", ".join(p.name for p in sk.params)
        put(panel, f"params: {names}", (20, y), 0.4, (150, 170, 200)); y += 18
    elif sk:
        put(panel, "params: (none)", (20, y), 0.4, (150, 170, 200)); y += 18

    # ── Committed list ──────────────────────────────────────────────────
    y += 8
    put(panel, f"INTERVENTIONS ({len(interventions)})", (12, y), 0.5, (160, 200, 160))
    y += 22
    row_h = 34
    avail_bottom = height - 110
    for i, iv in enumerate(interventions):
        if y > avail_bottom:
            put(panel, "… (more below)", (12, y), 0.4, (120, 120, 120))
            break
        is_sel = (i == selected_idx)
        if is_sel:
            cv2.rectangle(panel, (6, y - 14), (width - 6, y + row_h - 14), (60, 60, 90), -1)
        color = (255, 255, 255) if is_sel else (210, 210, 210)
        skill = iv.get("skill", "")
        label = skill + (f"({_fmt_args(iv.get('args') or {})})" if iv.get("args") else "")
        if len(label) > 56:
            label = label[:53] + "..."
        put(panel, f"{i + 1}. {label}", (14, y), 0.44, color)
        put(panel, f"   {float(iv.get('t_start') or 0):.2f}s → {float(iv.get('t_end') or 0):.2f}s",
            (14, y + 16), 0.4, (170, 200, 170))
        y += row_h

    put(panel, message or "ready", (12, height - 92), 0.45, (255, 220, 140))
    hints = [
        "a/d -/+1  A/D -/+skip  J/L fast  ,/. boundary  g goto",
        "k skill  p arg  P clr-args  [ start  ] end  R why  c commit  n new",
        "w/s select  e edit  x delete  S save  q quit",
    ]
    for k, h in enumerate(hints):
        put(panel, h, (12, height - 64 + k * 18), 0.4, (170, 170, 170))
    return panel


# ─── Main ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    config_dir = _project_root / "tasks" / args.task / "config"
    if not config_dir.exists():
        logger.error("Task config not found: %s", config_dir)
        return 2

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (_project_root / video_path).resolve()
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 2

    try:
        skills = load_skills(config_dir, args.skills)
    except FileNotFoundError:
        logger.error("Skills file not found: %s", config_dir / args.skills)
        return 2
    if not skills:
        logger.error("No skills with an 'id' found in %s", args.skills)
        return 2
    skills_by_id = {s.id: s for s in skills}
    skill_ids = [s.id for s in skills]

    out_path = Path(args.output) if args.output else (
        config_dir.parent / "ground_truth" / f"{video_path.stem}.robot_gt.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return 2
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total_frames / fps) if fps > 0 else 0.0

    if out_path.exists() and not args.overwrite:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        gt = RobotGT(
            task_id=data.get("task_id", args.task),
            video=data.get("video", str(video_path)),
            duration_sec=float(data.get("duration_sec", duration) or duration),
            operator_id=data.get("operator_id", "manual_annotation"),
            description=data.get("description", ""),
            notes=data.get("notes", ""),
            interventions=data.get("interventions", []) or [],
        )
        logger.info("Loaded %d existing intervention(s) from %s",
                    len(gt.interventions), out_path)
    else:
        gt = RobotGT(
            task_id=args.task,
            video=str(video_path),
            duration_sec=duration,
            description=(
                f"Robot intervention ground truth for {video_path.name}. "
                "Each entry is a contiguous interval during which the "
                "teleoperator drove the robot through one skill. Skill ids "
                f"and arg keys mirror tasks/{args.task}/config/{args.skills}."
            ),
            notes="",
        )

    frame_num = 0
    selected_idx = -1 if not gt.interventions else 0
    dirty = False
    message = ""
    working = _empty_working()

    win = f"Robot GT — {video_path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def read_frame(n: int) -> Optional[np.ndarray]:
        n = max(0, min(n, total_frames - 1)) if total_frames > 0 else max(0, n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        return f if ok else None

    def now_sec() -> float:
        return (frame_num / fps) if fps > 0 else 0.0

    def boundaries() -> List[float]:
        bs: List[float] = []
        for iv in gt.interventions:
            bs.append(float(iv.get("t_start") or 0.0))
            bs.append(float(iv.get("t_end") or 0.0))
        return sorted(set(bs))

    def save() -> None:
        nonlocal dirty, message
        gt.duration_sec = duration
        out_path.write_text(json.dumps(gt.to_dict(), indent=2), encoding="utf-8")
        dirty = False
        message = f"saved → {out_path}"
        logger.info(message)

    def commit() -> None:
        nonlocal dirty, message, working, selected_idx
        if not working.get("skill"):
            message = "cannot commit: no skill selected (press k)"
            return
        if working.get("t_start") is None or working.get("t_end") is None:
            message = "cannot commit: set both [ t_start and ] t_end"
            return
        if float(working["t_end"]) <= float(working["t_start"]):
            message = "cannot commit: t_end must be > t_start"
            return
        # Require all 'required' params to be present.
        sk = skills_by_id.get(working["skill"])
        if sk:
            missing = [p.name for p in sk.params
                       if p.required and p.name not in (working.get("args") or {})]
            if missing:
                message = f"cannot commit: missing required arg(s): {', '.join(missing)}"
                return
        gt.interventions.append(deepcopy(working))
        gt.interventions = gt.sorted_interventions()
        # Select the just-committed one.
        selected_idx = next(
            (i for i, iv in enumerate(gt.interventions) if iv is working
             or (iv.get("skill") == working.get("skill")
                 and iv.get("t_start") == working.get("t_start"))),
            len(gt.interventions) - 1,
        )
        working = _empty_working()
        dirty = True
        message = "committed intervention"

    def edit_arg(canvas: np.ndarray) -> None:
        nonlocal dirty, message
        sk = skills_by_id.get(working.get("skill") or "")
        if not sk:
            message = "pick a skill first (k)"
            return
        if not sk.params:
            message = f"{sk.id} has no parameters"
            return
        pname = overlay_picker(canvas, win, f"Argument for {sk.id}",
                               [p.name for p in sk.params],
                               list((working.get("args") or {}).keys()))
        if pname is None:
            return
        param = next(p for p in sk.params if p.name == pname)
        if param.valid_values:
            val = overlay_picker(canvas, win, f"{pname} =", param.valid_values,
                                 [str((working.get("args") or {}).get(pname, ""))])
            if val is None:
                return
            chosen: Any = val
        else:
            default = str((working.get("args") or {}).get(pname, param.default or ""))
            raw = prompt_terminal(f"{pname} ({param.type})", default=default)
            if raw is None:
                return
            chosen = raw
            if param.type == "number":
                try:
                    chosen = float(raw)
                except ValueError:
                    message = f"{pname}: not a number"
                    return
            elif param.type == "integer":
                try:
                    chosen = int(raw)
                except ValueError:
                    message = f"{pname}: not an integer"
                    return
        working.setdefault("args", {})[pname] = chosen
        dirty = True
        message = f"{pname} = {chosen}"

    current_img = read_frame(frame_num)

    while True:
        frame_disp = current_img if current_img is not None else np.zeros((480, 640, 3), np.uint8)
        fh, fw = frame_disp.shape[:2]
        panel_w = 560
        panel_h = max(fh, 820)
        panel = render_panel(
            height=panel_h, width=panel_w, skills_by_id=skills_by_id,
            working=working, interventions=gt.interventions, selected_idx=selected_idx,
            frame_num=frame_num, total_frames=total_frames, timestamp=now_sec(),
            dirty=dirty, message=message,
        )

        # Intervention span bar over the frame.
        if total_frames > 0 and duration > 0:
            bar = frame_disp.copy()
            for i, iv in enumerate(gt.interventions):
                x0 = int((float(iv.get("t_start") or 0) / duration) * (fw - 1))
                x1 = int((float(iv.get("t_end") or 0) / duration) * (fw - 1))
                col = (80, 180, 255) if i == selected_idx else (80, 200, 120)
                cv2.rectangle(bar, (x0, fh - 12), (max(x0 + 1, x1), fh - 4), col, -1)
            cx = int((frame_num / max(1, total_frames - 1)) * (fw - 1))
            cv2.line(bar, (cx, fh - 16), (cx, fh - 1), (255, 255, 255), 1)
            frame_disp = bar

        canvas = compose(frame_disp, panel)
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        message = ""

        # ── Navigation ─────────────────────────────────────────────
        new_frame = frame_num
        if key == ord('d'):
            new_frame = frame_num + 1
        elif key == ord('a'):
            new_frame = frame_num - 1
        elif key in (ord('D'), ord('l')):
            new_frame = frame_num + args.frame_skip
        elif key in (ord('A'), ord('j')):
            new_frame = frame_num - args.frame_skip
        elif key == ord('L'):
            new_frame = frame_num + args.frame_skip * args.fast_mult
        elif key == ord('J'):
            new_frame = frame_num - args.frame_skip * args.fast_mult
        elif key == ord('.'):
            t = now_sec()
            nxt = [b for b in boundaries() if b > t + 1e-6]
            if nxt and fps > 0:
                new_frame = int(round(nxt[0] * fps))
        elif key == ord(','):
            t = now_sec()
            prv = [b for b in boundaries() if b < t - 1e-6]
            if prv and fps > 0:
                new_frame = int(round(prv[-1] * fps))
        elif key == ord('g'):
            ans = prompt_terminal("Go to frame")
            if ans and ans.strip().isdigit():
                new_frame = int(ans)

        if new_frame != frame_num:
            frame_num = max(0, min(new_frame, total_frames - 1 if total_frames > 0 else new_frame))
            current_img = read_frame(frame_num)
            continue

        # ── Working intervention edits ─────────────────────────────
        if key == ord('k'):
            pick = overlay_picker(canvas, win, "Pick skill", skill_ids,
                                  [working.get("skill", "")])
            if pick is not None:
                if pick != working.get("skill"):
                    working["skill"] = pick
                    working["args"] = {}  # args belong to a specific skill
                    # Pre-fill defaults for declared params.
                    for p in skills_by_id[pick].params:
                        if p.default is not None:
                            working["args"][p.name] = p.default
                    dirty = True
                message = f"skill = {pick}"
            continue
        if key == ord('p'):
            edit_arg(canvas)
            continue
        if key == ord('P'):
            working["args"] = {}
            dirty = True
            message = "cleared args"
            continue
        if key == ord('['):
            working["t_start"] = round(now_sec(), 2)
            dirty = True
            message = f"t_start = {working['t_start']:.2f}s"
            continue
        if key == ord(']'):
            working["t_end"] = round(now_sec(), 2)
            dirty = True
            message = f"t_end = {working['t_end']:.2f}s"
            continue
        if key == ord('R'):
            ans = prompt_terminal("Rationale", default=working.get("rationale", ""))
            if ans is not None:
                working["rationale"] = ans
                dirty = True
                message = "rationale set"
            continue
        if key in (ord('c'), 13, 10):
            commit()
            continue
        if key == ord('n'):
            working = _empty_working()
            message = "cleared working intervention"
            continue

        # ── Committed list ─────────────────────────────────────────
        if key == ord('s') and gt.interventions:
            selected_idx = (selected_idx + 1) % len(gt.interventions)
            continue
        if key == ord('w') and gt.interventions:
            selected_idx = (selected_idx - 1) % len(gt.interventions)
            continue
        if key == ord('e') and 0 <= selected_idx < len(gt.interventions):
            working = deepcopy(gt.interventions[selected_idx])
            working.setdefault("args", {})
            working.setdefault("rationale", "")
            del gt.interventions[selected_idx]
            gt.interventions = gt.sorted_interventions()
            selected_idx = min(selected_idx, len(gt.interventions) - 1)
            dirty = True
            message = "moved intervention into working slot (edit, then commit)"
            continue
        if key == ord('x') and 0 <= selected_idx < len(gt.interventions):
            removed = gt.interventions.pop(selected_idx)
            selected_idx = min(selected_idx, len(gt.interventions) - 1)
            dirty = True
            message = f"deleted {removed.get('skill', '')}"
            continue

        # ── File ops ───────────────────────────────────────────────
        if key == ord('S'):
            save()
            continue
        if key == ord('q'):
            if dirty:
                ans = prompt_terminal("Unsaved changes. Save before quit? [y/N/cancel]", default="n")
                if ans is None:
                    continue
                low = ans.strip().lower()
                if low.startswith("c"):
                    continue
                if low.startswith("y"):
                    save()
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, help="Task name (dir under tasks/)")
    p.add_argument("--video", required=True, help="Path to video (absolute or rel to project root)")
    p.add_argument("--skills", default="robot_skills.json",
                   help="Skills file under tasks/<task>/config/ (default: robot_skills.json)")
    p.add_argument("--frame-skip", type=int, default=30,
                   help="Frames to jump with A/D or j/l (default: 30)")
    p.add_argument("--fast-mult", type=int, default=10,
                   help="Multiplier for J/L fast jump (default: 10 x frame-skip)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: tasks/<task>/ground_truth/<video_stem>.robot_gt.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="Start fresh even if an annotation file exists")
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()

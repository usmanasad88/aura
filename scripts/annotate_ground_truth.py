#!/usr/bin/env python3
"""Ground-truth annotator for the intent monitor.

Produces a sparse per-video annotation file
(``<task>/ground_truth/<video_stem>.intent_gt.json``) whose keyframes
mirror the JSON an ``AURAIntentMonitor`` would emit. Annotate only the
frames where state changes; lookup fills the gaps at runtime
(see :class:`aura.monitors.intent_ground_truth.GroundTruthIntentProvider`).

Only vision-observable variables from ``state_schema.json`` are edited.
Variables with ``source: system`` / ``perception`` / ``intent_monitor``
are excluded.

Controls
--------
Navigation
    a / d           step backward / forward 1 frame
    A / D (or j/l)  step backward / forward N frames (frame-skip)
    J / L           fast jump backward / forward (fast-mult × frame-skip)
    , / .           jump to prev / next keyframe
    g               type a frame number to jump to

Editing
    w / s           select previous / next field
    ← / →           cycle enum value (or ±1 for numeric)
    space           toggle boolean (False→True→Unknown→False)

List-field editing (steps_completed / in_progress / pending)
    Select the list, then:
      +             add item (overlay picker, arrows+enter)
      -             remove item (overlay picker, arrows+enter)
      t             toggle the step highlighted in the picker
    When an item is added to steps_completed, the next pending item
    auto-promotes to steps_in_progress and the rest stay in pending.

Keyframes
    c               snapshot current state as a new keyframe at this frame
    x               delete the keyframe at this frame (if any)
    r               reload working state from the last keyframe <= current frame

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
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("annotate_gt")

_EXTERNAL_SOURCES = {"system", "perception", "intent_monitor"}
_SPECIAL_ENUM_VALUES = {"Other", "Unknown"}

_LIST_STEP_FIELDS = ("steps_completed", "steps_in_progress", "steps_pending")

# Synthetic intent-monitor fields that aren't in state_schema.
_SYNTHETIC_FIELDS: List[Tuple[str, Dict[str, Any]]] = [
    ("steps_completed", {"type": "list", "description": "DAG step ids finished"}),
    ("steps_in_progress", {"type": "list", "description": "DAG step ids currently underway"}),
    ("steps_pending", {"type": "list", "description": "DAG step ids not yet started"}),
    ("reasoning", {"type": "string", "description": "One-line rationale",
                   "valid_values": ["ground_truth", "Other", "Unknown"]}),
]


# ─── Schema / DAG loading ──────────────────────────────────────────────────

def load_schema_fields(config_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    state_path = config_dir / "state_schema.json"
    schema = json.loads(state_path.read_text(encoding="utf-8"))
    vars_dict = schema.get("state_variables", {}) or {}

    fields: List[Tuple[str, Dict[str, Any]]] = []
    for name, defn in vars_dict.items():
        if not isinstance(defn, dict):
            continue
        if defn.get("source") in _EXTERNAL_SOURCES:
            continue
        fields.append((name, defn))

    existing = {n for n, _ in fields}
    for name, defn in _SYNTHETIC_FIELDS:
        if name not in existing:
            fields.append((name, defn))
    return fields


def get_step_pool(fields: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """Shared option pool for the steps_* lists = current_action.valid_values
    minus the special 'Other'/'Unknown' tokens.
    """
    for name, defn in fields:
        if name == "current_action":
            vals = list(defn.get("valid_values") or [])
            return [v for v in vals if v not in _SPECIAL_ENUM_VALUES]
    return []


def default_state(
    fields: List[Tuple[str, Dict[str, Any]]],
    step_pool: List[str],
) -> Dict[str, Any]:
    """Initial state: completed=[], in_progress=[first step], pending=[rest]."""
    out: Dict[str, Any] = {}
    for name, defn in fields:
        t = defn.get("type", "string")
        if name == "steps_completed":
            out[name] = []
        elif name == "steps_in_progress":
            out[name] = [step_pool[0]] if step_pool else []
        elif name == "steps_pending":
            out[name] = list(step_pool[1:]) if len(step_pool) > 1 else []
        elif "default" in defn:
            out[name] = defn["default"]
        elif t == "boolean":
            out[name] = False
        elif t == "integer":
            out[name] = 0
        elif t == "number":
            out[name] = 0.0
        elif t in ("list", "array"):
            out[name] = []
        else:
            out[name] = ""
    sync_derived_fields(out)
    return out


# ─── steps_* cascade logic ─────────────────────────────────────────────────

def apply_steps_cascade(state: Dict[str, Any], step_pool: List[str]) -> None:
    """Re-normalise steps_completed/in_progress/pending given the pool.

    Rules:
      - completed: whatever is in steps_completed ∩ pool, pool order.
      - in_progress: current single-entry list intersected with pool;
        if empty, the first not-yet-completed step from the pool.
      - pending: everything else (pool - completed - in_progress).
    """
    pool_set = set(step_pool)
    completed = [s for s in step_pool if s in set(state.get("steps_completed", []) or [])]
    completed_set = set(completed)

    in_prog_raw = [
        s for s in (state.get("steps_in_progress", []) or [])
        if s in pool_set and s not in completed_set
    ]
    if in_prog_raw:
        # Keep user order but dedupe
        seen: set = set()
        in_progress: List[str] = []
        for s in in_prog_raw:
            if s not in seen:
                seen.add(s)
                in_progress.append(s)
    else:
        in_progress = []
        for s in step_pool:
            if s not in completed_set:
                in_progress = [s]
                break

    in_progress_set = set(in_progress)
    pending = [s for s in step_pool if s not in completed_set and s not in in_progress_set]

    state["steps_completed"] = completed
    state["steps_in_progress"] = in_progress
    state["steps_pending"] = pending

    sync_derived_fields(state)


def sync_derived_fields(state: Dict[str, Any]) -> None:
    """Mirror steps_in_progress[0] → current_action and
    steps_pending[0] → predicted_next_action. No-op for missing fields."""
    in_prog = state.get("steps_in_progress") or []
    if "current_action" in state and in_prog:
        state["current_action"] = in_prog[0]
    pending = state.get("steps_pending") or []
    if "predicted_next_action" in state and pending:
        state["predicted_next_action"] = pending[0]


def add_to_list(
    state: Dict[str, Any],
    field_name: str,
    item: str,
    step_pool: List[str],
) -> str:
    """Add ``item`` to ``field_name``. For step lists, cascade the others.

    Returns a short message for the status bar.
    """
    if field_name not in _LIST_STEP_FIELDS:
        lst = list(state.get(field_name, []) or [])
        if item not in lst:
            lst.append(item)
        state[field_name] = lst
        return f"{field_name} += {item}"

    # For step lists: move ``item`` into this bucket, out of the other two.
    for f in _LIST_STEP_FIELDS:
        state[f] = [s for s in (state.get(f, []) or []) if s != item]
    state.setdefault(field_name, [])
    state[field_name] = list(state[field_name]) + [item]

    if field_name == "steps_completed":
        # Auto-promote the next pool item (still pending) to in_progress,
        # only if in_progress is now empty.
        if not state.get("steps_in_progress"):
            completed_set = set(state.get("steps_completed", []))
            for s in step_pool:
                if s not in completed_set:
                    state["steps_in_progress"] = [s]
                    break

    apply_steps_cascade(state, step_pool)
    return f"{field_name} += {item}"


def remove_from_list(
    state: Dict[str, Any],
    field_name: str,
    item: str,
    step_pool: List[str],
) -> str:
    lst = list(state.get(field_name, []) or [])
    if item in lst:
        lst.remove(item)
    state[field_name] = lst
    if field_name in _LIST_STEP_FIELDS:
        apply_steps_cascade(state, step_pool)
    return f"{field_name} -= {item}"


# ─── Annotation model ──────────────────────────────────────────────────────

@dataclass
class Annotation:
    video: str
    task: str
    fps: float
    total_frames: int
    keyframes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "task": self.task,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "keyframes": sorted(self.keyframes, key=lambda k: int(k["frame_num"])),
        }

    def upsert(self, frame_num: int, timestamp_sec: float, state: Dict[str, Any]) -> None:
        for kf in self.keyframes:
            if int(kf["frame_num"]) == int(frame_num):
                kf["timestamp_sec"] = timestamp_sec
                kf["state"] = deepcopy(state)
                return
        self.keyframes.append({
            "frame_num": int(frame_num),
            "timestamp_sec": float(timestamp_sec),
            "state": deepcopy(state),
        })

    def delete_at(self, frame_num: int) -> bool:
        before = len(self.keyframes)
        self.keyframes = [k for k in self.keyframes if int(k["frame_num"]) != int(frame_num)]
        return len(self.keyframes) != before

    def state_at(self, frame_num: int) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_f = -1
        for kf in self.keyframes:
            f = int(kf["frame_num"])
            if f <= frame_num and f > best_f:
                best = kf["state"]
                best_f = f
        return deepcopy(best) if best is not None else None

    def keyframe_numbers(self) -> List[int]:
        return sorted(int(k["frame_num"]) for k in self.keyframes)

    def dedupe_adjacent(self) -> int:
        """Remove any keyframe whose state equals its predecessor's state
        (in frame-number order). Returns count removed."""
        ordered = sorted(self.keyframes, key=lambda k: int(k["frame_num"]))
        kept: List[Dict[str, Any]] = []
        removed = 0
        for kf in ordered:
            if kept and kept[-1]["state"] == kf["state"]:
                removed += 1
                continue
            kept.append(kf)
        self.keyframes = kept
        return removed


# ─── Field value editing helpers ───────────────────────────────────────────

def cycle_enum(value: Any, values: List[Any], step: int) -> Any:
    if not values:
        return value
    try:
        i = values.index(value)
    except ValueError:
        i = 0
    return values[(i + step) % len(values)]


def toggle_bool(value: Any) -> Any:
    if value is False:
        return True
    if value is True:
        return "Unknown"
    return False


def prompt_terminal(prompt: str, default: str = "") -> Optional[str]:
    try:
        sys.stdout.write(f"\n{prompt}")
        if default:
            sys.stdout.write(f" [{default}]")
        sys.stdout.write(": ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.rstrip("\n")
        return line if line else default
    except Exception:
        return None


# ─── Rendering ─────────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, pos, scale=0.5, color=(230, 230, 230), thick=1):
    cv2.putText(img, text, pos, _FONT, scale, color, thick, cv2.LINE_AA)


def render_panel(
    height: int,
    width: int,
    fields: List[Tuple[str, Dict[str, Any]]],
    state: Dict[str, Any],
    selected_idx: int,
    frame_num: int,
    total_frames: int,
    timestamp: float,
    keyframe_nums: List[int],
    is_keyframe_here: bool,
    dirty: bool,
    message: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    put(panel, "AURA Intent GT Annotator", (12, 26), 0.7, (255, 255, 255), 2)
    kf_here = " [KF]" if is_keyframe_here else ""
    dirty_tag = " *" if dirty else ""
    put(panel, f"frame {frame_num}/{total_frames - 1}  t={timestamp:.2f}s{kf_here}{dirty_tag}",
        (12, 52), 0.5, (180, 220, 255))
    put(panel, f"keyframes: {len(keyframe_nums)}", (12, 72), 0.45, (160, 200, 160))

    y = 100
    row_h_single = 22
    row_h_list = 48
    avail_bottom = height - 110

    for i, (name, defn) in enumerate(fields):
        if y > avail_bottom:
            put(panel, "… (more below)", (12, y + 10), 0.4, (120, 120, 120))
            break
        t = defn.get("type", "string")
        val = state.get(name)
        is_list = t in ("list", "array")
        is_selected = (i == selected_idx)

        row_h = row_h_list if is_list else row_h_single
        if is_selected:
            cv2.rectangle(panel, (6, y - 16), (width - 6, y + row_h - 6), (60, 60, 90), -1)
        color = (255, 255, 255) if is_selected else (210, 210, 210)
        put(panel, f"{name} ({t})", (12, y), 0.46, color)

        if is_list:
            items = list(val or [])
            # Wrap items onto two lines if long
            s = ", ".join(items) if items else "(empty)"
            if len(s) > 62:
                s = s[:59] + "..."
            put(panel, s, (22, y + 18), 0.44, (180, 230, 180))
            put(panel, f"{len(items)} items", (22, y + 36), 0.38, (140, 180, 140))
        else:
            val_str = str(val)
            if len(val_str) > 46:
                val_str = val_str[:43] + "..."
            put(panel, val_str, (22, y + 18), 0.44, (180, 230, 180))

        y += row_h + 6

    put(panel, message or "ready", (12, height - 92), 0.45, (255, 220, 140))
    hints = [
        "a/d: -/+1   A/D: -/+skip   J/L: -/+fast   ,/. KF nav   g: goto",
        "w/s: field   arrows: cycle   space: bool",
        "+/-: list add/remove (opens picker)   c: set KF   S: save   q: quit",
    ]
    for k, h in enumerate(hints):
        put(panel, h, (12, height - 64 + k * 18), 0.4, (170, 170, 170))

    return panel


def compose(frame_bgr: np.ndarray, panel: np.ndarray) -> np.ndarray:
    fh, fw = frame_bgr.shape[:2]
    ph, pw = panel.shape[:2]
    h = max(fh, ph)
    canvas = np.zeros((h, fw + pw, 3), dtype=np.uint8)
    canvas[:fh, :fw] = frame_bgr
    canvas[:ph, fw:fw + pw] = panel
    return canvas


# ─── Overlay picker (for list add/remove) ──────────────────────────────────

LEFT_KEYS = (81, 65361, 2424832, ord('h'))
RIGHT_KEYS = (83, 65363, 2555904, ord('\''))
UP_KEYS = (82, 65362, 2490368)
DOWN_KEYS = (84, 65364, 2621440)
ENTER_KEYS = (13, 10)
ESC_KEYS = (27,)


def overlay_picker(
    base_canvas: np.ndarray,
    window_name: str,
    title: str,
    options: List[str],
    currently_selected: List[str],
) -> Optional[str]:
    """Show an overlay list. Arrows move, enter picks, esc cancels.

    Returns the picked option string (always from ``options``), or None.
    """
    if not options:
        return None
    idx = 0
    selected_set = set(currently_selected)
    while True:
        overlay = base_canvas.copy()
        H, W = overlay.shape[:2]
        box_w = min(420, W - 40)
        row_h = 22
        visible_rows = min(len(options), max(6, (H - 160) // row_h))
        box_h = 60 + row_h * visible_rows + 40
        x0 = (W - box_w) // 2
        y0 = (H - box_h) // 2

        # Dim background + panel
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, base_canvas, 0.45, 0, overlay)
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 55), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 220), 1)

        put(overlay, title, (x0 + 14, y0 + 28), 0.55, (255, 255, 255), 2)
        put(overlay, "↑/↓ move   enter pick   esc cancel",
            (x0 + 14, y0 + 50), 0.4, (160, 180, 220))

        # Scroll window
        start = max(0, min(idx - visible_rows // 2, len(options) - visible_rows))
        for row, i in enumerate(range(start, start + visible_rows)):
            yy = y0 + 72 + row * row_h
            opt = options[i]
            mark = "●" if opt in selected_set else "○"
            color = (255, 255, 255) if i == idx else (200, 200, 200)
            bg_color = (80, 110, 170) if i == idx else None
            if bg_color is not None:
                cv2.rectangle(overlay, (x0 + 6, yy - 16),
                              (x0 + box_w - 6, yy + 4), bg_color, -1)
            put(overlay, f"{mark}  {opt}", (x0 + 14, yy), 0.48, color)

        cv2.imshow(window_name, overlay)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key in UP_KEYS or key == ord('w'):
            idx = (idx - 1) % len(options)
        elif key in DOWN_KEYS or key == ord('s'):
            idx = (idx + 1) % len(options)
        elif key in ENTER_KEYS or key == ord(' '):
            return options[idx]
        elif key in ESC_KEYS or key == ord('q'):
            return None


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

    fields = load_schema_fields(config_dir)
    step_pool = get_step_pool(fields)

    out_path = Path(args.output) if args.output else (
        config_dir.parent / "ground_truth" / f"{video_path.stem}.intent_gt.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return 2
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if out_path.exists() and not args.overwrite:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        ann = Annotation(
            video=data.get("video", str(video_path)),
            task=data.get("task", args.task),
            fps=float(data.get("fps", fps) or fps),
            total_frames=int(data.get("total_frames", total_frames) or total_frames),
            keyframes=data.get("keyframes", []) or [],
        )
        logger.info("Loaded %d existing keyframes from %s", len(ann.keyframes), out_path)
        removed = ann.dedupe_adjacent()
        if removed:
            logger.info("Removed %d duplicate keyframe(s) on load", removed)
    else:
        ann = Annotation(
            video=str(video_path),
            task=args.task,
            fps=fps,
            total_frames=total_frames,
        )

    def reload_working(frame_num: int) -> Dict[str, Any]:
        s = ann.state_at(frame_num)
        if s is None:
            s = default_state(fields, step_pool)
        base = default_state(fields, step_pool)
        for k, v in base.items():
            s.setdefault(k, v)
        # Ensure cascade invariants hold even if the file was hand-edited.
        apply_steps_cascade(s, step_pool)
        return s

    frame_num = 0
    selected_idx = 0
    dirty = False
    message = ""
    working_state = reload_working(frame_num)

    win = f"GT Annotator — {video_path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def read_frame(n: int) -> Optional[np.ndarray]:
        n = max(0, min(n, total_frames - 1)) if total_frames > 0 else max(0, n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        return f if ok else None

    def save() -> None:
        nonlocal dirty, message
        out_path.write_text(json.dumps(ann.to_dict(), indent=2), encoding="utf-8")
        dirty = False
        message = f"saved → {out_path}"
        logger.info(message)

    def set_kf_here() -> None:
        nonlocal dirty, message
        ts = (frame_num / fps) if fps > 0 else 0.0
        ann.upsert(frame_num, ts, working_state)
        dirty = True
        message = f"keyframe set at frame {frame_num}"

    def delete_kf_here() -> None:
        nonlocal dirty, message
        if ann.delete_at(frame_num):
            dirty = True
            message = f"keyframe deleted at frame {frame_num}"
        else:
            message = f"no keyframe at frame {frame_num}"

    current_img = read_frame(frame_num)

    while True:
        frame_disp = current_img if current_img is not None else np.zeros((480, 640, 3), np.uint8)
        fh, fw = frame_disp.shape[:2]
        panel_w = 540
        panel_h = max(fh, 760)
        panel = render_panel(
            height=panel_h, width=panel_w, fields=fields, state=working_state,
            selected_idx=selected_idx, frame_num=frame_num, total_frames=total_frames,
            timestamp=(frame_num / fps) if fps > 0 else 0.0,
            keyframe_nums=ann.keyframe_numbers(),
            is_keyframe_here=any(int(k["frame_num"]) == frame_num for k in ann.keyframes),
            dirty=dirty, message=message,
        )

        # Keyframe tick bar
        if total_frames > 0:
            bar = frame_disp.copy()
            bar_y = fh - 6
            cv2.line(bar, (0, bar_y), (fw, bar_y), (60, 60, 60), 2)
            for kf in ann.keyframe_numbers():
                x = int((kf / max(1, total_frames - 1)) * (fw - 1))
                cv2.line(bar, (x, fh - 12), (x, fh - 1), (80, 220, 80), 2)
            cx = int((frame_num / max(1, total_frames - 1)) * (fw - 1))
            cv2.line(bar, (cx, fh - 14), (cx, fh - 1), (80, 180, 255), 2)
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
            nxt = [k for k in ann.keyframe_numbers() if k > frame_num]
            if nxt:
                new_frame = nxt[0]
        elif key == ord(','):
            prv = [k for k in ann.keyframe_numbers() if k < frame_num]
            if prv:
                new_frame = prv[-1]
        elif key == ord('g'):
            ans = prompt_terminal("Go to frame")
            if ans and ans.isdigit():
                new_frame = int(ans)

        if new_frame != frame_num:
            if dirty:
                save()
            frame_num = max(0, min(new_frame, total_frames - 1 if total_frames > 0 else new_frame))
            current_img = read_frame(frame_num)
            s_here = next((k["state"] for k in ann.keyframes if int(k["frame_num"]) == frame_num), None)
            if s_here is not None:
                base = default_state(fields, step_pool)
                working_state = {**base, **deepcopy(s_here)}
                apply_steps_cascade(working_state, step_pool)
                message = f"loaded keyframe at {frame_num}"
            continue

        # ── Field selection ────────────────────────────────────────
        if key == ord('s'):
            selected_idx = (selected_idx + 1) % len(fields)
            continue
        if key == ord('w'):
            selected_idx = (selected_idx - 1) % len(fields)
            continue

        name, defn = fields[selected_idx]
        t = defn.get("type", "string")
        valid = defn.get("valid_values")
        is_list = t in ("list", "array")

        # ── Enum cycle / numeric step ──────────────────────────────
        if not is_list and (key in LEFT_KEYS or key in RIGHT_KEYS):
            step = -1 if key in LEFT_KEYS else 1
            cur = working_state.get(name)
            if valid:
                working_state[name] = cycle_enum(cur, list(valid), step)
            elif t == "integer":
                working_state[name] = int(cur or 0) + step
            elif t == "number":
                try:
                    working_state[name] = round(float(cur or 0) + step * 0.1, 3)
                except (TypeError, ValueError):
                    working_state[name] = 0.0 + step * 0.1
            elif t == "boolean":
                working_state[name] = toggle_bool(cur)
            else:
                message = f"{name}: no enum to cycle"
                continue
            dirty = True
            message = f"{name} → {working_state[name]}"
            continue

        if key == ord(' ') and not is_list:
            if t == "boolean":
                working_state[name] = toggle_bool(working_state.get(name))
                dirty = True
                message = f"{name} → {working_state[name]}"
            continue

        # ── List add / remove (picker overlay) ─────────────────────
        if is_list and key in (ord('+'), ord('=')):
            if name in _LIST_STEP_FIELDS:
                options = step_pool
            else:
                options = list(defn.get("valid_values") or [])
            if not options:
                message = f"{name}: no option pool defined"
                continue
            pick = overlay_picker(canvas, win,
                                  f"Add to {name}", options,
                                  list(working_state.get(name, []) or []))
            if pick is not None:
                message = add_to_list(working_state, name, pick, step_pool)
                dirty = True
            continue

        if is_list and key in (ord('-'), ord('_')):
            current_items = list(working_state.get(name, []) or [])
            if not current_items:
                message = f"{name}: empty"
                continue
            pick = overlay_picker(canvas, win,
                                  f"Remove from {name}", current_items,
                                  current_items)
            if pick is not None:
                message = remove_from_list(working_state, name, pick, step_pool)
                dirty = True
            continue

        # ── Keyframe ops ───────────────────────────────────────────
        if key == ord('c'):
            set_kf_here()
            continue
        if key == ord('x'):
            delete_kf_here()
            continue
        if key == ord('r'):
            working_state = reload_working(frame_num)
            message = f"reloaded state from KF ≤ {frame_num}"
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
    p.add_argument("--frame-skip", type=int, default=30,
                   help="Frames to jump with A/D or j/l (default: 30)")
    p.add_argument("--fast-mult", type=int, default=10,
                   help="Multiplier for J/L fast jump (default: 10 × frame-skip)")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: tasks/<task>/ground_truth/<video_stem>.intent_gt.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="Start fresh even if an annotation file exists")
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()

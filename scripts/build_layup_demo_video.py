#!/usr/bin/env python3
"""Build an AURA demo video for the hand_layup task.

Base video  = the source overlay video that the run processed
              (layup_gesture_demo_stationary_with_overlay.mp4, 1920x1080@30).
On top      = a bottom info panel that, at every intent/decision output point,
              briefly PAUSES the video and shows:
                INTENT MONITOR  : current action, predicted next action, reasoning
                DECISION ENGINE : robot command (decision/target/params), reasoning

Run results come from:
  logs/hand_layup_self_intent_perception/{intent_monitor,decision_engine}
Each call's meta.json carries timestamp_sec (== frame_num / 30) which maps
directly onto the source video timeline.

Output is encoded with ffmpeg (H.264, yuv420p) for broad compatibility.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path("/home/mani/Repos/aura")
RUN = ROOT / "logs/hand_layup_gt_intent_perception"
VIDEO = ROOT / "demo_data/layup_demo/layup_gesture_demo_stationary_with_overlay.mp4"
OUT_PATH = RUN / "aura_hand_layup_demo.mp4"

FONT_DIR = "/usr/share/fonts/truetype/dejavu"

# ── Layout / pacing ──────────────────────────────────────────────────
SCENE_W, SCENE_H = 1920, 1080
PANEL_H = 400
HEADER_H = 56                       # thin strip at very top of scene
OUT_W, OUT_H = SCENE_W, SCENE_H + PANEL_H   # 1920 x 1480

FPS = 25
SAMPLE_STEP_S = 0.5                 # source-video seconds between sampled scene frames
PLAY_HOLD_S = 0.12                  # output seconds each non-pause scene frame holds
PAUSE_HOLD_S = 1.8                  # seconds the video freezes at each AURA output
FADE_IN_S = 0.45                    # emphasis fade at start of a pause

# ── Colours (BGR) ────────────────────────────────────────────────────
C_PANEL_BG = (28, 24, 22)
C_INTENT = (235, 170, 70)           # blue-ish accent
C_DECISION = (90, 200, 90)
C_ACT = (90, 210, 90)
C_WAIT = (70, 170, 240)             # orange/amber
C_TEXT = (235, 235, 235)
C_DIM = (170, 170, 170)
C_WHITE = (250, 250, 250)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


F_TITLE = font("DejaVuSans-Bold.ttf", 30)
F_HEAD = font("DejaVuSans-Bold.ttf", 28)
F_LABEL = font("DejaVuSans-Bold.ttf", 23)
F_BODY = font("DejaVuSans.ttf", 22)
F_SMALL = font("DejaVuSans.ttf", 20)
F_BADGE = font("DejaVuSans-Bold.ttf", 22)


# ── Load AURA outputs ────────────────────────────────────────────────
def load_calls(kind: str):
    out = []
    for d in sorted(glob.glob(str(RUN / kind / "call_*"))):
        meta = json.load(open(os.path.join(d, "meta.json")))
        parsed = json.load(open(os.path.join(d, "response_parsed.json")))
        out.append({"ts": float(meta["timestamp_sec"]), "meta": meta, "p": parsed})
    out.sort(key=lambda r: r["ts"])
    return out


intents = load_calls("intent_monitor")
decisions = load_calls("decision_engine")
intent_ts = [r["ts"] for r in intents]
decision_ts = [r["ts"] for r in decisions]


def active(records, ts_list, ts):
    """Latest record with record.ts <= ts (None before the first)."""
    i = bisect_right(ts_list, ts) - 1
    return records[i] if i >= 0 else None


# ── Build the scene timeline from the source video ───────────────────
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = n_src / src_fps
print(f"Source video: {VIDEO.name}  {n_src} frames @ {src_fps:.2f}fps  ({dur:.1f}s)")
print(f"Intent calls: {len(intents)}  Decision calls: {len(decisions)}")

# regular sampling grid up to the last AURA output, plus the exact output times
last_ts = max(intent_ts[-1], decision_ts[-1])
end_ts = min(dur, last_ts + 2.0)
scene_set = set()
t = 0.0
while t <= end_ts:
    scene_set.add(round(t, 3))
    t += SAMPLE_STEP_S
for r in intents + decisions:
    if r["ts"] <= end_ts:
        scene_set.add(round(r["ts"], 3))
scene_times = sorted(scene_set)


# ── Mark which scene times are AURA output (pause) points ─────────────
def nearest_scene_idx(ts):
    i = bisect_right(scene_times, ts)
    if i >= len(scene_times):
        return len(scene_times) - 1
    if i > 0 and abs(scene_times[i - 1] - ts) <= abs(scene_times[i] - ts):
        return i - 1
    return i


pause_at: dict[int, set] = {}
for r in intents:
    pause_at.setdefault(nearest_scene_idx(r["ts"]), set()).add("intent")
for r in decisions:
    pause_at.setdefault(nearest_scene_idx(r["ts"]), set()).add("decision")
print(f"Scene frames: {len(scene_times)}   Distinct pause frames: {len(pause_at)}")


def read_frame_at(ts):
    """Read the source frame nearest to timestamp ts (seconds)."""
    fi = int(round(ts * src_fps))
    fi = max(0, min(fi, n_src - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, img = cap.read()
    if not ok or img is None:
        return np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8)
    if img.shape[1] != SCENE_W or img.shape[0] != SCENE_H:
        img = cv2.resize(img, (SCENE_W, SCENE_H))
    return img


# ── Text helpers ─────────────────────────────────────────────────────
def wrap(draw, text, fnt, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fmt_command(dp):
    """Human-readable robot command from a decision parsed dict."""
    dec = dp.get("decision")
    if not dec or dec == "wait":
        return "WAIT — hold position"
    tid = dp.get("target")
    s = dec + (f"  →  {tid}" if tid else "")
    params = dp.get("parameters") or {}
    if params:
        s += "  [" + ", ".join(f"{k}={v}" for k, v in params.items()) + "]"
    return s


# ── Panel renderer ───────────────────────────────────────────────────
def render_panel(scene_bgr, intent_rec, decision_rec, ts, emphasis):
    """Return a full OUT_H canvas: scene on top, info panel below."""
    canvas = np.full((OUT_H, OUT_W, 3), C_PANEL_BG, dtype=np.uint8)
    canvas[:SCENE_H, :SCENE_W] = scene_bgr

    # ── top header strip over the scene ──
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (OUT_W, HEADER_H), (18, 16, 15), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    phase = (intent_rec["p"].get("current_phase", "—") if intent_rec else "initialization")
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)

    def draw_text(dr, xy, s, fnt, c):
        dr.text(xy, s, font=fnt, fill=(c[2], c[1], c[0]))

    draw_text(d, (24, 12), "AURA  ·  Hand Layup", F_TITLE, C_WHITE)
    rt = f"t = {ts:6.1f}s     phase: {phase}"
    rw = d.textlength(rt, font=F_HEAD)
    draw_text(d, (OUT_W - rw - 24, 14), rt, F_HEAD, C_DIM)

    # ── panel columns ──
    pad = 28
    col_gap = 36
    col_w = (OUT_W - 2 * pad - col_gap) // 2
    lx = pad
    rx = pad + col_w + col_gap
    top = SCENE_H + 18

    d.line([(rx - col_gap // 2, SCENE_H + 14), (rx - col_gap // 2, OUT_H - 14)],
           fill=(70, 66, 62), width=2)

    # PIL alpha compositing needs RGBA base for the emphasis tints
    pil = pil.convert("RGBA")
    d = ImageDraw.Draw(pil)
    if emphasis["alpha"] > 0.01:
        if emphasis.get("intent"):
            ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            a = int(60 * emphasis["alpha"])
            od.rounded_rectangle([lx - 12, SCENE_H + 8, lx + col_w + 12, OUT_H - 8],
                                 radius=10, fill=(C_INTENT[2], C_INTENT[1], C_INTENT[0], a),
                                 outline=(C_INTENT[2], C_INTENT[1], C_INTENT[0], int(235 * emphasis["alpha"])),
                                 width=3)
            pil.alpha_composite(ov)
        if emphasis.get("decision"):
            ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            a = int(60 * emphasis["alpha"])
            od.rounded_rectangle([rx - 12, SCENE_H + 8, rx + col_w + 12, OUT_H - 8],
                                 radius=10, fill=(C_DECISION[2], C_DECISION[1], C_DECISION[0], a),
                                 outline=(C_DECISION[2], C_DECISION[1], C_DECISION[0], int(235 * emphasis["alpha"])),
                                 width=3)
            pil.alpha_composite(ov)
    d = ImageDraw.Draw(pil)

    def text(xy, s, fnt, c):
        d.text(xy, s, font=fnt, fill=(c[2], c[1], c[0], 255))

    def label_value(x, y, label, value, fnt, lc, vc, gap=14):
        text((x, y), label, fnt, lc)
        d.text((x + d.textlength(label, font=fnt) + gap, y), value,
               font=fnt, fill=(vc[2], vc[1], vc[0], 255))

    # ── LEFT: INTENT MONITOR ──
    y = top
    text((lx, y), "INTENT MONITOR", F_HEAD, C_INTENT)
    if emphasis.get("intent") and emphasis["alpha"] > 0.3:
        bw = d.textlength("INTENT MONITOR", font=F_HEAD)
        text((lx + bw + 16, y + 3), "● NEW", F_BADGE, C_WHITE)
    y += 44
    if intent_rec:
        ip = intent_rec["p"]
        conf = ip.get("prediction_confidence")
        conf_s = f"  ({conf:.2f})" if isinstance(conf, (int, float)) else ""
        label_value(lx, y, "Current action:", str(ip.get("current_action", "—")),
                    F_LABEL, C_DIM, C_WHITE)
        y += 34
        label_value(lx, y, "Predicted next:",
                    str(ip.get("predicted_next_action", "—")) + conf_s,
                    F_LABEL, C_DIM, C_INTENT)
        y += 38
        text((lx, y), "Reasoning", F_SMALL, C_DIM); y += 28
        for ln in wrap(d, str(ip.get("reasoning", "")), F_BODY, col_w)[:5]:
            text((lx, y), ln, F_BODY, C_TEXT); y += 28
    else:
        text((lx, y), "initializing…", F_BODY, C_DIM)

    # ── RIGHT: DECISION ENGINE ──
    y = top
    text((rx, y), "DECISION ENGINE", F_HEAD, C_DECISION)
    if emphasis.get("decision") and emphasis["alpha"] > 0.3:
        bw = d.textlength("DECISION ENGINE", font=F_HEAD)
        text((rx + bw + 16, y + 3), "● NEW", F_BADGE, C_WHITE)
    y += 44
    if decision_rec:
        dpp = decision_rec["p"]
        dec = dpp.get("decision") or "—"
        is_wait = (dec == "wait")
        dec_col = C_WAIT if is_wait else C_ACT
        dec_label = "WAIT" if is_wait else "ACT"
        text((rx, y), "Decision:", F_LABEL, C_DIM)
        d.text((rx + d.textlength("Decision:", font=F_LABEL) + 14, y - 3),
               dec_label, font=F_HEAD, fill=(dec_col[2], dec_col[1], dec_col[0], 255))
        y += 38
        text((rx, y), "Robot command:", F_LABEL, C_DIM); y += 30
        for ln in wrap(d, fmt_command(dpp), F_LABEL, col_w)[:2]:
            text((rx, y), ln, F_LABEL, C_WHITE); y += 30
        y += 8
        text((rx, y), "Reasoning", F_SMALL, C_DIM); y += 28
        for ln in wrap(d, str(dpp.get("reasoning", "")), F_BODY, col_w)[:5]:
            text((rx, y), ln, F_BODY, C_TEXT); y += 28
    else:
        text((rx, y), "initializing…", F_BODY, C_DIM)

    out = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out


# ── Encode via ffmpeg pipe ───────────────────────────────────────────
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
     "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
     "-preset", "medium", "-movflags", "+faststart", str(OUT_PATH)],
    stdin=subprocess.PIPE,
)

play_frames = max(1, round(PLAY_HOLD_S * FPS))
pause_frames = max(1, round(PAUSE_HOLD_S * FPS))
fade_frames = max(1, round(FADE_IN_S * FPS))

total_written = 0
for idx, ts in enumerate(scene_times):
    scene = read_frame_at(ts)
    irec = active(intents, intent_ts, ts + 1e-6)
    drec = active(decisions, decision_ts, ts + 1e-6)

    if idx in pause_at:
        engines = pause_at[idx]
        emph_base = {"intent": "intent" in engines, "decision": "decision" in engines}
        for k in range(pause_frames):
            alpha = min(1.0, k / fade_frames)
            emph = dict(emph_base); emph["alpha"] = alpha
            frame = render_panel(scene, irec, drec, ts, emph)
            ff.stdin.write(frame.tobytes())
            total_written += 1
    else:
        emph = {"intent": False, "decision": False, "alpha": 0.0}
        frame = render_panel(scene, irec, drec, ts, emph)
        for _ in range(play_frames):
            ff.stdin.write(frame.tobytes())
            total_written += 1

    if idx % 25 == 0:
        print(f"  scene {idx}/{len(scene_times)}  ts={ts:.1f}s  written={total_written}")

cap.release()
ff.stdin.close()
ff.wait()
out_dur = total_written / FPS
print(f"\nDone. {total_written} frames -> {out_dur:.1f}s video")
print(f"Output: {OUT_PATH}")

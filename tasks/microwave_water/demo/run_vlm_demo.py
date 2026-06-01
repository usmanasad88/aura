#!/usr/bin/env python3
"""Standalone demo: ask a local VLM where a cup is, frame-by-frame.

Queries a local VLM (via the SGLang server + :class:`LocalVLMMonitor`)
for the location of a cup in the microwave-water video, using the *same*
constrained option set as the SAM3 perception demo
(``counter_top`` / ``inside_microwave`` / ``on_top_of_microwave``).  This
lets the VLM's answers be compared directly against the geometric SAM3
results produced by ``run_perception_demo.py``.

Prerequisite — start the SGLang server in another terminal::

    ./scripts/start_sglang_server.sh                       # Qwen/Qwen3.5-0.8B
    ./scripts/start_sglang_server.sh --model Qwen/Qwen3-VL-4B-Instruct

NOTE: the SGLang server reserves most of the GPU (``mem-fraction-static``
defaults to 0.8), so run this *instead of* — not alongside — the SAM3
perception demo.  Compare offline by pointing ``--compare`` at the SAM3
demo's ``results.jsonl``.

Outputs are saved to a timestamped run folder under
``logs/microwave_water/vlm_demo/<timestamp>/``:
  - ``images/frame_NNNNN.jpg`` — annotated visualization per processed frame
  - ``results.jsonl`` — one JSON object per processed frame with the VLM
    answer, raw text, timing, and (if ``--compare``) the SAM3 location +
    agreement flag

Usage::

    # Black cup location with the default Qwen 3.5 0.8B model
    uv run python tasks/microwave_water/demo/run_vlm_demo.py \\
        --video demo_data/layup_demo/microwave_water.mp4 --frame-skip 15

    # Compare against an existing SAM3 perception run
    uv run python tasks/microwave_water/demo/run_vlm_demo.py \\
        --video demo_data/layup_demo/microwave_water.mp4 --frame-skip 15 \\
        --compare logs/microwave_water/perception_demo/<ts>/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

# Ensure project root + src are on sys.path.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
for _p in (_project_root, _project_root / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aura.monitors.local_vlm_monitor import LocalVLMMonitor, LocalVLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Location option set — must match the SAM3 perception demo ─────────
COUNTER_TOP = "counter_top"
INSIDE_MICROWAVE = "inside_microwave"
ON_TOP_OF_MICROWAVE = "on_top_of_microwave"
UNKNOWN = "unknown"
LOCATION_OPTIONS = [COUNTER_TOP, INSIDE_MICROWAVE, ON_TOP_OF_MICROWAVE]

# Canonical id ↔ human phrasing used in the question.
_CUP_WORD = {"black_cup": "black", "white_cup": "white"}

# Normalisation table: maps loose VLM phrasings → canonical labels.
_NORMALISE = {
    "counter_top": COUNTER_TOP,
    "countertop": COUNTER_TOP,
    "counter top": COUNTER_TOP,
    "counter": COUNTER_TOP,
    "kitchen counter": COUNTER_TOP,
    "on the counter": COUNTER_TOP,
    "on the counter top": COUNTER_TOP,
    "inside_microwave": INSIDE_MICROWAVE,
    "inside microwave": INSIDE_MICROWAVE,
    "inside the microwave": INSIDE_MICROWAVE,
    "in the microwave": INSIDE_MICROWAVE,
    "in microwave": INSIDE_MICROWAVE,
    "on_top_of_microwave": ON_TOP_OF_MICROWAVE,
    "on top of microwave": ON_TOP_OF_MICROWAVE,
    "on top of the microwave": ON_TOP_OF_MICROWAVE,
    "on top": ON_TOP_OF_MICROWAVE,
    "top of microwave": ON_TOP_OF_MICROWAVE,
    "top of the microwave": ON_TOP_OF_MICROWAVE,
}


def _normalise_location(answer: str) -> str:
    """Map a raw VLM answer to one of the canonical location labels.

    Picks the option phrasing that appears **last** in the text.  Reasoning
    models ("thinking" mode) enumerate every option mid-thought and only
    commit to the answer at the end, so the last mention is the conclusion;
    answer-only outputs contain a single mention, so last == only.
    """
    if not answer:
        return UNKNOWN
    a = answer.strip().strip(".\"'` ").lower()
    if a in _NORMALISE:
        return _NORMALISE[a]
    best_label = UNKNOWN
    best_pos = -1
    for phrase, label in _NORMALISE.items():
        pos = a.rfind(phrase)
        if pos > best_pos:
            best_pos = pos
            best_label = label
    return best_label


def _build_question(cup_id: str) -> str:
    """Constrained multiple-choice question for one cup's location."""
    word = _CUP_WORD.get(cup_id, "black")
    return (
        f"Look at the {word} cup in this kitchen scene with a microwave on "
        f"the counter. Where is the {word} cup right now? "
        f"Answer with EXACTLY ONE of these options and nothing else:\n"
        f"- {COUNTER_TOP} (on the kitchen counter, beside the microwave)\n"
        f"- {ON_TOP_OF_MICROWAVE} (resting on the top surface of the microwave)\n"
        f"- {INSIDE_MICROWAVE} (inside the microwave)"
    )


# ── Lightweight live JPEG server ─────────────────────────────────────

_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()

_HTML_PAGE = b"""\
<!DOCTYPE html><html><head>
<title>Microwave Water VLM Demo</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style>
</head><body>
<img id="f" src="/frame.jpg">
<script>
const img=document.getElementById('f');
setInterval(()=>{img.src='/frame.jpg?t='+Date.now()},300);
</script></body></html>
"""


class _FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML_PAGE)
        elif self.path.startswith("/frame.jpg"):
            with _jpeg_lock:
                data = _latest_jpeg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if data:
                self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress request logs.


def _publish_frame(vis_bgr: np.ndarray) -> None:
    global _latest_jpeg
    _, buf = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _jpeg_lock:
        _latest_jpeg = buf.tobytes()


def _start_server() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("0.0.0.0", port), _FrameHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port


# ── Comparison helper ────────────────────────────────────────────────

def _load_sam3_locations(path: str, cup_id: str) -> dict[int, str]:
    """Load {frame_num: location} for *cup_id* from a SAM3 results.jsonl."""
    mapping: dict[int, str] = {}
    p = Path(path)
    if not p.exists():
        logger.warning("Compare file not found: %s", path)
        return mapping
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = rec.get("frame")
            loc = rec.get("cup_locations", {}).get(cup_id)
            if frame is not None and loc is not None:
                mapping[int(frame)] = loc
    logger.info("Loaded %d SAM3 locations for %s from %s",
                len(mapping), cup_id, path)
    return mapping


# ── Visualization ────────────────────────────────────────────────────

_LOC_COLORS = {
    COUNTER_TOP: (0, 255, 0),            # green (BGR)
    ON_TOP_OF_MICROWAVE: (0, 200, 255),  # amber
    INSIDE_MICROWAVE: (255, 100, 0),     # blue
    UNKNOWN: (160, 160, 160),            # grey
}


def _annotate(
    frame: np.ndarray,
    cup_id: str,
    vlm_loc: str,
    raw: str,
    dt: float,
    sam3_loc: str | None,
    model_id: str,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"VLM: {model_id}", (255, 255, 255)))
    lines.append((f"{cup_id} -> {vlm_loc}", _LOC_COLORS.get(vlm_loc, (255, 255, 255))))
    if sam3_loc is not None:
        match = (vlm_loc == sam3_loc)
        lines.append((f"SAM3 -> {sam3_loc}", _LOC_COLORS.get(sam3_loc, (255, 255, 255))))
        lines.append(("MATCH" if match else "MISMATCH",
                      (0, 255, 0) if match else (0, 0, 255)))

    panel_h = 20 + 34 * len(lines)
    cv2.rectangle(vis, (5, 5), (560, panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (5, 5), (560, panel_h), (255, 255, 255), 1)
    y = 38
    for text, color in lines:
        cv2.putText(vis, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 34

    info = f"raw={raw[:60]!r} | {dt:.2f}s"
    cv2.putText(vis, info, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return vis


# ── Main loop ────────────────────────────────────────────────────────

def run_demo(
    video_path: str,
    cup_id: str,
    model_id: str,
    base_url: str,
    frame_skip: int,
    headless: bool,
    compare_path: str | None,
    save_path: str | None,
    thinking: bool,
    max_tokens: int | None,
) -> None:
    question = _build_question(cup_id)
    logger.info("VLM question:\n%s", question)

    # Reasoning ("thinking") models emit a long chain-of-thought before the
    # answer, so they need a large token budget; answer-only mode needs few.
    # Disabling thinking is a Qwen3 chat-template switch passed via extra_body.
    if max_tokens is None:
        max_tokens = 512 if thinking else 48
    extra = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    logger.info("Reasoning: %s | max_tokens=%d",
                "ON" if thinking else "OFF", max_tokens)

    monitor = LocalVLMMonitor(LocalVLMConfig(
        backend="sglang",
        model_id=model_id,
        sglang_base_url=base_url,
        question=question,
        structured_output=False,
        temperature=0.0,
        max_new_tokens=max_tokens,
        extra=extra,
    ))

    sam3_locs = _load_sam3_locations(compare_path, cup_id) if compare_path else {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %s  |  %dx%d @ %.1f fps  |  %d frames  |  skip=%d",
                video_path, w, h, fps, total, frame_skip)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _project_root / "logs" / "microwave_water" / "vlm_demo" / run_ts
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / "results.jsonl"
    images_dir = log_dir / "images"
    images_dir.mkdir(exist_ok=True)
    logger.info("Logging to %s", log_dir)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps / frame_skip, (w, h))
        logger.info("Saving visualisation to %s", save_path)

    show = not headless
    if show:
        port = _start_server()
        url = f"http://localhost:{port}/"
        logger.info("Live visualization: %s", url)
        import webbrowser
        webbrowser.open(url)

    frame_num = 0
    processed = 0
    n_match = 0
    n_compared = 0
    vlm_counts: dict[str, int] = {}
    results_file = open(results_path, "w")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % frame_skip != 0:
                continue

            timestamp = frame_num / fps

            t0 = time.perf_counter()
            try:
                out = asyncio.run(monitor.process_frame(frame))
                raw = out.perception.answer if out.perception else ""
            except Exception as e:
                logger.exception("VLM error on frame %d: %s", frame_num, e)
                if show:
                    _publish_frame(frame)
                continue
            dt = time.perf_counter() - t0

            processed += 1
            vlm_loc = _normalise_location(raw)
            vlm_counts[vlm_loc] = vlm_counts.get(vlm_loc, 0) + 1

            sam3_loc = sam3_locs.get(frame_num)
            match_str = ""
            if sam3_loc is not None:
                n_compared += 1
                if sam3_loc == vlm_loc:
                    n_match += 1
                    match_str = " | MATCH"
                else:
                    match_str = f" | MISMATCH (sam3={sam3_loc})"

            logger.info("Frame %5d | %6.1fs | %.2fs | %s=%s%s | raw=%r",
                        frame_num, timestamp, dt, cup_id, vlm_loc,
                        match_str, raw[:50])

            vis = _annotate(frame, cup_id, vlm_loc, raw, dt, sam3_loc, model_id)
            cv2.imwrite(str(images_dir / f"frame_{frame_num:05d}.jpg"), vis)

            record = {
                "frame": frame_num,
                "timestamp": round(timestamp, 3),
                "processing_time": round(dt, 4),
                "cup_id": cup_id,
                "model": model_id,
                "thinking": thinking,
                "vlm_location": vlm_loc,
                "vlm_raw": raw,
                "sam3_location": sam3_loc,
                "match": (sam3_loc == vlm_loc) if sam3_loc is not None else None,
            }
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

            if writer:
                writer.write(vis)
            if show:
                _publish_frame(vis)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
    finally:
        cap.release()
        results_file.close()
        if writer:
            writer.release()

    logger.info("Done. Processed %d / %d frames. Logs: %s",
                processed, frame_num, log_dir)
    logger.info("VLM location distribution: %s", vlm_counts)
    if n_compared:
        logger.info("Agreement with SAM3: %d/%d (%.1f%%)",
                    n_match, n_compared, 100.0 * n_match / n_compared)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microwave water VLM demo — ask a local VLM for a cup's location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to demo video")
    parser.add_argument(
        "--cup", default="black_cup", choices=["black_cup", "white_cup"],
        help="Which cup to localise (default: black_cup)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B",
        help="Model id served by the SGLang server (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8100/v1",
        help="SGLang OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=15,
        help="Process every N-th frame (default: 15)",
    )
    parser.add_argument(
        "--compare", default=None, metavar="RESULTS_JSONL",
        help="Path to a SAM3 perception results.jsonl to compare against",
    )
    parser.add_argument(
        "--thinking", action="store_true", default=False,
        help="Enable model reasoning/thinking mode (slower, slightly more "
             "accurate). Default off → fast answer-only via enable_thinking=False",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Generation token budget (default: 48 answer-only, 512 thinking)",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Disable visualization (log-only)",
    )
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save visualisation video to PATH",
    )
    args = parser.parse_args()
    run_demo(args.video, args.cup, args.model, args.base_url,
             args.frame_skip, args.headless, args.compare, args.save,
             args.thinking, args.max_tokens)


if __name__ == "__main__":
    main()

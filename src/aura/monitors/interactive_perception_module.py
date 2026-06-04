"""Generic interactive perception monitor backed by SAM3.

Unlike :class:`aura.monitors.perception_module.PerceptionModule` — which runs a
fixed set of *text* prompts every frame — this monitor is **interactive**: the
caller picks a pixel (a click), and SAM3's instance-interactive predictor
segments whatever object lives under that point.  It supports three prompt
modes, all on the currently-set frame:

* **point**  — one or more ``(x, y)`` clicks, each foreground (keep) or
  background (exclude).  Great for "segment the thing I just clicked".
* **box**    — an ``(x1, y1, x2, y2)`` rectangle drag.
* **text**   — an open-vocabulary phrase (e.g. ``"coffee mug"``); reuses the
  SAM3 grounding path, so it behaves like ``PerceptionModule`` for one prompt.

It also offers a lightweight **single-object tracker** for live video: seed it
with a click, and each subsequent frame is re-segmented from the previous
mask's centroid (seeded with the previous low-res mask logits for stability),
so the mask follows the object as it moves.  This is a pragmatic streaming
tracker — SAM3's full video tracker expects an offline clip — and is intended
for interactive webcam use, not long-horizon occlusion-robust tracking.

Typical usage (see ``scripts/run_interactive_perception_ui.py`` for a full
webcam + browser UI built on this)::

    monitor = InteractivePerceptionMonitor()
    monitor.set_frame(bgr_frame)                 # embeds the image once
    result = monitor.segment_at_point(x, y)      # click to segment
    mask = result.mask                           # HxW bool ndarray

    # …or track across frames:
    monitor.start_tracking(x, y)
    while True:
        result = monitor.track(next_bgr_frame)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Lazy model loading ───────────────────────────────────────────────
# SAM3 + CUDA is expensive to import/build, so load once and share.  We build
# with ``enable_inst_interactivity=True`` so the point/box (SAM1-task)
# predictor is available in addition to text grounding.
_sam3_model = None
_sam3_processor = None
_load_lock = threading.Lock()


def _load_sam3_interactive():
    """Lazy-load the SAM3 image model with instance interactivity enabled."""
    global _sam3_model, _sam3_processor
    if _sam3_model is not None and _sam3_processor is not None:
        return _sam3_model, _sam3_processor

    with _load_lock:
        if _sam3_model is not None and _sam3_processor is not None:
            return _sam3_model, _sam3_processor

        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model (instance interactivity enabled)...")
        model = build_sam3_image_model(enable_inst_interactivity=True)

        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
            logger.info("SAM3 moved to GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA not available — SAM3 will run on CPU (slow).")

        _sam3_model = model
        _sam3_processor = Sam3Processor(model)
        logger.info("SAM3 interactive model loaded.")

    return _sam3_model, _sam3_processor


# ── Result containers ────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """One segmented object on the current frame."""

    mask: np.ndarray                      # HxW bool/uint8 mask
    score: float                          # model confidence / IoU prediction
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2) of the mask
    label: str = "object"                 # semantic label (text mode) else generic
    logits: Optional[np.ndarray] = None   # low-res mask logits for mask_input reuse


@dataclass
class InteractivePerceptionConfig:
    """Configuration for :class:`InteractivePerceptionMonitor`."""

    # Confidence below which text-mode detections are dropped.
    text_confidence_threshold: float = 0.4
    # Use bf16 autocast on CUDA (matches SAM3 example notebooks).
    use_autocast: bool = True
    # Tracking: re-seed each frame with the previous mask's low-res logits.
    track_use_mask_memory: bool = True
    # Tracking: stop following once confidence drops below this (object lost).
    track_min_score: float = 0.0
    # Palette for visualizing multiple objects (BGR).
    palette: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 255, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255),
        (0, 255, 255), (255, 0, 0), (128, 0, 255), (0, 128, 255),
    ])


class InteractivePerceptionMonitor:
    """Click-to-segment / track perception monitor backed by SAM3.

    The monitor keeps the embedding of one "current" frame at a time.  Call
    :meth:`set_frame` to (re)embed, then any number of ``segment_*`` calls reuse
    that embedding cheaply.  All public methods are thread-safe with respect to
    the underlying model so a UI thread and a capture thread can share one
    instance.
    """

    def __init__(self, config: Optional[InteractivePerceptionConfig] = None) -> None:
        self.config = config or InteractivePerceptionConfig()
        self._model = None
        self._processor = None
        self._state = None              # SAM3 inference state for current frame
        self._frame_hw: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()   # serialize model access

        # Tracking state.
        self._tracking = False
        self._track_point: Optional[Tuple[float, float]] = None
        self._track_logits: Optional[np.ndarray] = None
        self._track_label = "tracked"

    # ── Model / frame management ─────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is None or self._processor is None:
            self._model, self._processor = _load_sam3_interactive()

    def _autocast(self):
        """Return a CUDA bf16 autocast context, or a no-op on CPU."""
        import torch
        import contextlib
        if self.config.use_autocast and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Embed ``frame_bgr`` (OpenCV BGR) as the current frame for prompting."""
        self._ensure_loaded()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]
        with self._lock, self._autocast():
            self._state = self._processor.set_image(image_pil)
        self._frame_hw = (h, w)

    # ── Prompt modes ─────────────────────────────────────────────────

    def segment_at_point(self, x: float, y: float, positive: bool = True) -> Optional[SegmentResult]:
        """Segment the object under a single click at pixel ``(x, y)``."""
        return self.segment_points([(x, y)], [1 if positive else 0])

    def segment_points(
        self,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
        box: Optional[Sequence[float]] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Optional[SegmentResult]:
        """Segment from point clicks (and an optional box) on the current frame.

        Args:
            points: ``(x, y)`` pixel coordinates of each click.
            labels: ``1`` for a foreground (keep) click, ``0`` for background.
            box: optional ``(x1, y1, x2, y2)`` box prompt to combine with points.
            mask_input: optional low-res logits from a previous prediction.
            multimask_output: let SAM3 return 3 candidate masks and keep the best.

        Returns the highest-scoring :class:`SegmentResult`, or ``None``.
        """
        if self._state is None:
            raise RuntimeError("Call set_frame() before segmenting.")

        pt = np.asarray(points, dtype=np.float32) if len(points) else None
        lb = np.asarray(labels, dtype=np.int32) if len(labels) else None
        bx = np.asarray(box, dtype=np.float32) if box is not None else None

        with self._lock, self._autocast():
            masks, scores, logits = self._model.predict_inst(
                self._state,
                point_coords=pt,
                point_labels=lb,
                box=bx,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )
        return self._best_result(masks, scores, logits, label="object")

    def segment_box(self, x1: float, y1: float, x2: float, y2: float) -> Optional[SegmentResult]:
        """Segment the object inside a dragged box ``(x1, y1, x2, y2)``."""
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return self.segment_points([], [], box=box, multimask_output=False)

    def segment_text(self, prompt: str) -> List[SegmentResult]:
        """Open-vocabulary segmentation of every instance matching ``prompt``."""
        if self._state is None:
            raise RuntimeError("Call set_frame() before segmenting.")

        with self._lock, self._autocast():
            self._processor.reset_all_prompts(self._state)
            out = self._processor.set_text_prompt(prompt=prompt, state=self._state)

        results: List[SegmentResult] = []
        if out.get("boxes") is None or len(out["boxes"]) == 0:
            return results

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy() if out.get("masks") is not None else None
        h, w = self._frame_hw

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < self.config.text_confidence_threshold:
                continue
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if m.ndim == 3:
                    m = m[0]
                mask = (m > 0.5).astype(np.uint8)
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            results.append(SegmentResult(
                mask=mask if mask is not None else np.zeros((h, w), np.uint8),
                score=float(score),
                bbox=(x1, y1, x2, y2),
                label=prompt,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ── Single-object tracking ───────────────────────────────────────

    def start_tracking(self, x: float, y: float, label: str = "tracked") -> Optional[SegmentResult]:
        """Seed the tracker with a click on the current frame and return the mask."""
        result = self.segment_at_point(x, y)
        if result is None:
            return None
        self._tracking = True
        self._track_label = label
        self._track_point = self._mask_centroid(result.mask) or (x, y)
        self._track_logits = result.logits
        result.label = label
        return result

    def track(self, frame_bgr: np.ndarray) -> Optional[SegmentResult]:
        """Embed ``frame_bgr`` and follow the tracked object into it.

        Re-segments from the previous mask's centroid, seeded with the previous
        low-res logits, then advances the centroid for the next frame.  Returns
        ``None`` (and stops tracking) if the object can no longer be found.
        """
        if not self._tracking or self._track_point is None:
            return None

        self.set_frame(frame_bgr)
        mask_input = self._track_logits if self.config.track_use_mask_memory else None
        if mask_input is not None and mask_input.ndim == 2:
            # predict_inst expects low-res mask logits as (1, H, W).
            mask_input = mask_input[None, :, :]
        x, y = self._track_point
        result = self.segment_points(
            [(x, y)], [1], mask_input=mask_input, multimask_output=mask_input is None
        )

        if result is None or result.score < self.config.track_min_score or result.mask.sum() == 0:
            self.stop_tracking()
            return None

        centroid = self._mask_centroid(result.mask)
        if centroid is None:
            self.stop_tracking()
            return None
        self._track_point = centroid
        self._track_logits = result.logits
        result.label = self._track_label
        return result

    def stop_tracking(self) -> None:
        """Clear tracking state."""
        self._tracking = False
        self._track_point = None
        self._track_logits = None

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    # ── Helpers ──────────────────────────────────────────────────────

    def _best_result(self, masks, scores, logits, label: str) -> Optional[SegmentResult]:
        """Pick the highest-scoring mask from a ``predict_inst`` output."""
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.size == 0 or scores.size == 0:
            return None
        # masks may be (C,H,W) for one object or (1,C,H,W)/(B,C,H,W); flatten to (C,H,W).
        if masks.ndim == 4:
            masks = masks[0]
            logits = np.asarray(logits)[0] if logits is not None else None
        best = int(np.argmax(scores))
        mask = (masks[best] > 0).astype(np.uint8)
        best_logits = np.asarray(logits)[best] if logits is not None else None
        return SegmentResult(
            mask=mask,
            score=float(scores[best]),
            bbox=self._mask_bbox(mask),
            label=label,
            logits=best_logits,
        )

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return (float(xs.mean()), float(ys.mean()))

    # ── Visualization ────────────────────────────────────────────────

    def visualize(
        self,
        frame_bgr: np.ndarray,
        results: Sequence[SegmentResult],
        points: Optional[Sequence[Tuple[float, float, int]]] = None,
    ) -> np.ndarray:
        """Overlay masks, boxes, labels (and optional click markers) on a frame.

        Args:
            frame_bgr: the frame to draw on (not modified).
            results: segmented objects to draw.
            points: optional ``(x, y, label)`` click markers (label 1=fg, 0=bg).
        """
        vis = frame_bgr.copy()
        for idx, r in enumerate(results):
            color = self.config.palette[idx % len(self.config.palette)]
            if r.mask is not None and r.mask.any():
                mask = r.mask
                if mask.shape[:2] != vis.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{r.label}: {r.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        if points:
            for (px, py, plabel) in points:
                pt_color = (0, 255, 0) if plabel else (0, 0, 255)
                cv2.drawMarker(vis, (int(px), int(py)), pt_color,
                               cv2.MARKER_STAR, 18, 2)
        return vis

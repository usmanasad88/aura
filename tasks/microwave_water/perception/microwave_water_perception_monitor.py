"""Task-specific perception monitor for the microwave water task.

Tracks a microwave (fixed reference) plus two cups (black, white) using
SAM3 segmentation with colour-specific text prompts (``"microwave"``,
``"black cup"``, ``"white cup"``).  Each cup is placed into one of three
locations *relative to the microwave* using simple front-view geometry:

* the cup mask **overlaps** the microwave mask        → ``inside_microwave``
* the cup sits **above** the microwave's top edge     → ``on_top_of_microwave``
* the cup is to the **left** of the microwave         → ``counter_top``

The video is filmed head-on from the front, so "above" and "left" in
image space map directly to physical "on top of" and "beside on the
counter".

Because SAM3 is prompted directly with ``"black cup"`` / ``"white cup"``,
cup identity needs no colour post-processing — each prompt yields its own
cup.  A cup that is not detected in a given frame (e.g. occluded behind
the bread loaf, or hidden behind the tinted microwave door) keeps its
**last-known location** so downstream consumers see a stable value rather
than a flicker to ``unknown``.  A cup that has *never* been seen reports
``unknown``.

Usage::

    monitor = MicrowaveWaterPerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)
    print(result["cup_locations"])
    # {"black_cup": "counter_top", "white_cup": "on_top_of_microwave"}
    print(result["black_cup_location"], result["white_cup_location"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────
# SAM3 text prompts.  The microwave is the spatial reference; the two cups
# are the movable objects we localise relative to it.
MICROWAVE_PROMPT = "microwave"
CUP_PROMPTS = ["black cup", "white cup"]

# Maps SAM3 cup prompt → canonical object id in initial_scene.json.
_CUP_PROMPT_TO_ID = {
    "black cup": "black_cup",
    "white cup": "white_cup",
}

# Canonical location ids — must match the ``valid_values`` lists for
# ``black_cup_location`` / ``white_cup_location`` in state_schema.json.
INSIDE_MICROWAVE = "inside_microwave"
ON_TOP_OF_MICROWAVE = "on_top_of_microwave"
COUNTER_TOP = "counter_top"
UNKNOWN = "unknown"


@dataclass
class MicrowaveWaterPerceptionConfig:
    """Configuration for the microwave water perception monitor."""

    sam3_prompts: List[str] = field(
        default_factory=lambda: [MICROWAVE_PROMPT] + CUP_PROMPTS,
    )

    confidence_threshold: float = 0.25
    # Minimum fraction of cup mask overlapping the microwave mask to call
    # the cup "inside" the microwave.
    overlap_threshold: float = 0.15
    # Vertical slack (px) added below the microwave's top edge when testing
    # whether a cup sits "on top" of the microwave.  Lets a cup whose base
    # rests just at the top surface still count as on-top.
    on_top_margin_px: int = 15
    # Horizontal slack (px) extending the microwave's x-range when deciding
    # whether an above-the-top cup is actually sitting on the microwave.
    on_top_x_margin_px: int = 40
    # Run perception every N calls (1 = every frame).
    process_every_n: int = 1
    # Max detections to keep per cup prompt.
    max_per_prompt: int = 1


class MicrowaveWaterPerceptionMonitor:
    """Perception monitor that tracks the microwave and two cups.

    Wraps ``PerceptionModule`` (composition) with SAM3 prompts and
    front-view spatial heuristics for cup→location assignment.  Maintains
    a last-known location per cup so undetected cups report their previous
    position instead of flickering to ``unknown``.
    """

    CUP_IDS = tuple(_CUP_PROMPT_TO_ID.values())  # ("black_cup", "white_cup")

    def __init__(
        self, config: Optional[MicrowaveWaterPerceptionConfig] = None
    ) -> None:
        self.config = config or MicrowaveWaterPerceptionConfig()
        self._perception = self._build_perception_module()
        self._call_count = 0

        # Tracked microwave centre — updated with EMA each frame.
        self._microwave_position: Optional[Tuple[float, float]] = None

        # Last-known location for each cup (persists across frames).
        self._last_known_locations: Dict[str, str] = {
            cid: UNKNOWN for cid in self.CUP_IDS
        }

    # ── Construction helpers ─────────────────────────────────────────

    def _build_perception_module(self):
        """Create the inner PerceptionModule with task-specific config."""
        from aura.utils.config import PerceptionConfig
        from aura.monitors.perception_module import PerceptionModule

        pcfg = PerceptionConfig(
            use_sam3=True,
            use_gemini_detection=False,
            default_prompts=self.config.sam3_prompts,
            confidence_threshold=self.config.confidence_threshold,
        )
        return PerceptionModule(config=pcfg)

    # ── Main entry point ─────────────────────────────────────────────

    async def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a BGR frame and return cup location assignments.

        Returns ``None`` when skipped (throttle) or on error.  Otherwise
        returns a dict::

            {
                "cup_locations": {"black_cup": "...", "white_cup": "..."},
                "black_cup_location": "...",
                "white_cup_location": "...",
                "microwave": <TrackedObject or None>,
                "cups": {"black_cup": <obj or None>, ...},
                "detected_this_frame": {"black_cup", ...},
                "detections": {...},
            }

        ``cup_locations`` is always a full snapshot of both cups: cups seen
        this frame report their current location, undetected cups report
        their last-known location (or ``"unknown"`` if never seen).
        """
        self._call_count += 1
        if self._call_count % self.config.process_every_n != 0:
            return None

        try:
            output = await self._perception.process_frame(frame)
        except Exception as e:
            logger.error("Perception processing failed: %s", e)
            return None

        if output is None:
            return None

        # Group detections by SAM3 prompt name.
        detections: Dict[str, list] = {p: [] for p in self.config.sam3_prompts}
        for obj in output.objects:
            if obj.name in detections:
                detections[obj.name].append(obj)

        # ── Microwave (reference) ────────────────────────────────────
        mw_candidates = sorted(
            detections.get(MICROWAVE_PROMPT, []),
            key=lambda o: o.confidence, reverse=True,
        )
        microwave = mw_candidates[0] if mw_candidates else None

        if microwave is not None and microwave.bbox is not None:
            cx, cy = microwave.bbox.center
            if self._microwave_position is None:
                self._microwave_position = (cx, cy)
            else:
                old_cx, old_cy = self._microwave_position
                alpha = 0.3
                self._microwave_position = (
                    alpha * cx + (1 - alpha) * old_cx,
                    alpha * cy + (1 - alpha) * old_cy,
                )

        mw_mask = microwave.mask if microwave is not None else None
        mw_bbox = microwave.bbox if microwave is not None else None
        logger.debug(
            "Microwave: %s",
            f"conf={microwave.confidence:.2f}" if microwave else "NOT DETECTED",
        )

        # ── Localise each cup relative to the microwave ──────────────
        cups: Dict[str, Optional[Any]] = {cid: None for cid in self.CUP_IDS}
        detected_locations: Dict[str, str] = {}

        for prompt in CUP_PROMPTS:
            cup_id = _CUP_PROMPT_TO_ID[prompt]
            candidates = sorted(
                detections.get(prompt, []),
                key=lambda o: o.confidence, reverse=True,
            )[: self.config.max_per_prompt]
            if not candidates:
                continue
            cup = candidates[0]
            cups[cup_id] = cup
            location = self._assign_cup_location(
                cup.mask, cup.bbox, mw_mask, mw_bbox,
            )
            detected_locations[cup_id] = location

        # Update last-known locations for cups resolved to a real location.
        for cup_id, location in detected_locations.items():
            if location != UNKNOWN:
                self._last_known_locations[cup_id] = location

        # Build full snapshot: detected cups use this frame's location,
        # undetected cups fall back to their last-known location.
        cup_locations: Dict[str, str] = {}
        for cup_id in self.CUP_IDS:
            if cup_id in detected_locations and detected_locations[cup_id] != UNKNOWN:
                cup_locations[cup_id] = detected_locations[cup_id]
            else:
                cup_locations[cup_id] = self._last_known_locations[cup_id]

        result: Dict[str, Any] = {
            "cup_locations": cup_locations,
            "microwave": microwave,
            "cups": cups,
            "detected_this_frame": set(detected_locations.keys()),
            "detections": {
                k: [
                    {
                        "id": o.id,
                        "confidence": o.confidence,
                        "bbox": (
                            int(o.bbox.x_min), int(o.bbox.y_min),
                            int(o.bbox.x_max), int(o.bbox.y_max),
                        ) if o.bbox else None,
                    }
                    for o in v
                ]
                for k, v in detections.items()
            },
        }

        # Expose convenience ``*_location`` keys so downstream nodes can
        # write them straight into the SSG task_state / state_schema vars.
        for cup_id in self.CUP_IDS:
            result[f"{cup_id}_location"] = cup_locations[cup_id]

        return result

    # ── Cup → location heuristic ──────────────────────────────────────

    def _assign_cup_location(
        self,
        cup_mask: Optional[np.ndarray],
        cup_bbox: Optional[Any],
        mw_mask: Optional[np.ndarray],
        mw_bbox: Optional[Any],
    ) -> str:
        """Determine a cup's location relative to the microwave.

        Front-view geometry: the three locations occupy distinct image
        regions — *above* the microwave is its top surface, *left* of it is
        the open counter, and *within* its outline is the interior.  Rules
        are evaluated in priority order:

        1. **Strong mask overlap** — the cup mask overlaps the microwave
           mask past the threshold (and is not sitting above its top edge)
           → ``inside_microwave``.  This is the explicit "if it overlaps,
           it's inside" signal.
        2. **Above the top edge** (within the microwave's horizontal span)
           → ``on_top_of_microwave``.
        3. **Left of the microwave** → ``counter_top``.
        4. **Within the microwave's horizontal span, at/below the top edge**
           → ``inside_microwave``.  Handles cups sitting in the (dark)
           cavity, which SAM3 often excludes from the microwave mask so the
           overlap test alone would miss them.

        Returns ``unknown`` only when the microwave itself is missing (no
        spatial reference) or the cup has no bbox.
        """
        if mw_bbox is None or cup_bbox is None:
            return UNKNOWN

        cup_cx, cup_cy = cup_bbox.center
        above_top = cup_cy < mw_bbox.y_min + self.config.on_top_margin_px
        within_x_span = (
            mw_bbox.x_min - self.config.on_top_x_margin_px
            <= cup_cx
            <= mw_bbox.x_max + self.config.on_top_x_margin_px
        )

        # ── 1. Strong mask overlap → inside ──────────────────────────
        # A cup resting *on top* may graze the microwave mask at its base,
        # so require the cup to also sit at/below the top edge.
        if cup_mask is not None and mw_mask is not None and not above_top:
            cup_area = int(cup_mask.sum())
            if cup_area > 0:
                m = mw_mask
                if m.shape != cup_mask.shape:
                    m = cv2.resize(
                        m, (cup_mask.shape[1], cup_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                overlap_ratio = int((cup_mask & m).sum()) / cup_area
                if overlap_ratio >= self.config.overlap_threshold:
                    return INSIDE_MICROWAVE

        # ── 2. Above the top edge → on top of the microwave ──────────
        if above_top and within_x_span:
            return ON_TOP_OF_MICROWAVE

        # ── 3. Left of the microwave → counter top ───────────────────
        if cup_cx < mw_bbox.x_min:
            return COUNTER_TOP

        # ── 4. Within the microwave outline → inside ─────────────────
        if cup_cx <= mw_bbox.x_max:
            return INSIDE_MICROWAVE

        # Anything to the right at counter level is still the open counter.
        return COUNTER_TOP

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fit_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize *mask* to (h, w) if shapes differ."""
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Visualization ────────────────────────────────────────────────

    def visualize(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
    ) -> np.ndarray:
        """Draw the microwave, cup masks/bboxes, and cup locations."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        microwave_color = (0, 200, 255)   # amber (BGR)
        cup_colors = {
            "black_cup": (0, 0, 255),     # red
            "white_cup": (255, 255, 255), # white
        }
        default_cup_color = (255, 0, 255)  # magenta fallback

        microwave = result.get("microwave")
        cups: Dict[str, Optional[Any]] = result.get("cups", {})
        cup_locations: Dict[str, str] = result.get("cup_locations", {})
        detected_this_frame = result.get("detected_this_frame", set())

        # Draw the microwave (reference region).
        if microwave is not None:
            if microwave.mask is not None:
                mask = self._fit_mask(microwave.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = microwave_color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            if microwave.bbox is not None:
                x1, y1 = int(microwave.bbox.x_min), int(microwave.bbox.y_min)
                x2, y2 = int(microwave.bbox.x_max), int(microwave.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), microwave_color, 4)
                label = f"microwave ({microwave.confidence:.2f})"
                (tw, th_), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3,
                )
                cv2.rectangle(vis, (x1, y1 - th_ - 14), (x1 + tw, y1),
                              microwave_color, -1)
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        # Draw cups.
        for cup_id, cup in cups.items():
            if cup is None:
                continue
            color = cup_colors.get(cup_id, default_cup_color)
            loc = cup_locations.get(cup_id, "?")

            if cup.mask is not None:
                mask = self._fit_mask(cup.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            if cup.bbox is not None:
                x1, y1 = int(cup.bbox.x_min), int(cup.bbox.y_min)
                x2, y2 = int(cup.bbox.x_max), int(cup.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = f"{cup_id} @ {loc} ({cup.confidence:.2f})"
                (tw, th_), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3,
                )
                txt_color = (0, 0, 0) if cup_id == "white_cup" else (255, 255, 255)
                cv2.rectangle(vis, (x1, y2), (x1 + tw, y2 + th_ + 14), color, -1)
                cv2.putText(vis, label, (x1, y2 + th_ + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, txt_color, 3)

        # Summary panel — both cups, bright if seen this frame else dimmed.
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        mw_txt = "microwave: detected" if microwave is not None else "microwave: MISSING"
        lines.append((mw_txt, microwave_color))
        for cup_id in self.CUP_IDS:
            loc = cup_locations.get(cup_id, UNKNOWN)
            if cup_id in detected_this_frame:
                color = cup_colors.get(cup_id, default_cup_color)
                lines.append((f"{cup_id}: {loc}", color))
            else:
                lines.append((f"{cup_id}: {loc} (last known)", (120, 120, 120)))

        panel_h = 42 + 44 * len(lines)
        cv2.rectangle(vis, (5, 5), (760, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (760, panel_h), (255, 255, 255), 2)
        y = 48
        for text, color in lines:
            cv2.putText(vis, text, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3)
            y += 44

        return vis

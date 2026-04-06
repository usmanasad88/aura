"""Task-specific perception monitor for the sorting task.

Tracks a table and two baskets (white, blue) plus movable objects
(balls, tools) using SAM3 segmentation with colour-specific text
prompts.  Uses spatial heuristics (mask overlap, centroid containment,
proximity) to determine which region each object is currently in.

Regions
-------
- **table**: the workspace where objects start
- **white_basket**: white basket container
- **blue_basket**: blue basket container

All movable objects start on the table.  As the human or robot moves
them, the perception monitor re-assigns each object to the region it
overlaps most.

SAM3 is prompted directly with ``"white basket"`` and ``"blue basket"``
so no programmatic colour identification is needed.

Usage::

    monitor = SortingPerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)
    print(result["object_locations"])
    # {"soccer_ball": "table", "basketball": "white_basket", ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Prompt lists ────────────────────────────────────────────────────────────
# Region prompts use colour names so SAM3 distinguishes them directly.
REGION_PROMPTS = ["table", "white basket", "blue basket"]

OBJECT_PROMPTS = [
    "soccer ball",
    "beach ball",
    "basketball",
    "blue ball",
    "purple ball",
    "green ball",
    "hole punch",
    "tape",
    "stapler",
]

# Maps SAM3 region prompt → canonical region id used in initial_scene.json.
_REGION_PROMPT_TO_ID = {
    "table": "table",
    "white basket": "white_basket",
    "blue basket": "blue_basket",
}

# Maps SAM3 object prompt → canonical object id in initial_scene.json.
_OBJECT_PROMPT_TO_ID = {
    "soccer ball": "soccer_ball",
    "beach ball": "beach_ball",
    "basketball": "basketball",
    "blue ball": "blue_ball",
    "purple ball": "purple_ball",
    "green ball": "green_ball",
    "hole punch": "punch",
    "tape": "tape",
    "stapler": "stapler",
}


@dataclass
class SortingPerceptionConfig:
    """Configuration for the sorting perception monitor."""

    sam3_prompts: List[str] = field(
        default_factory=lambda: REGION_PROMPTS + OBJECT_PROMPTS,
    )

    confidence_threshold: float = 0.25
    # Minimum fraction of object mask overlapping a region mask to assign.
    overlap_threshold: float = 0.02
    use_centroid_fallback: bool = True
    use_proximity_fallback: bool = True
    # Run perception every N calls (1 = every frame).
    process_every_n: int = 1
    # Max detections to keep per prompt.
    max_per_prompt: int = 3


class SortingPerceptionMonitor:
    """Perception monitor that tracks regions and objects for sorting.

    Wraps ``PerceptionModule`` (composition) with SAM3 prompts and
    spatial heuristics for object→region assignment.
    """

    REGION_IDS = ("table", "white_basket", "blue_basket")

    # All canonical object IDs from initial_scene.json.
    ALL_OBJECT_IDS = tuple(_OBJECT_PROMPT_TO_ID.values())

    def __init__(self, config: Optional[SortingPerceptionConfig] = None) -> None:
        self.config = config or SortingPerceptionConfig()
        self._perception = self._build_perception_module()
        self._call_count = 0

        # Tracked region centres — updated with EMA each frame.
        # Maps region id → (cx, cy).
        self._region_positions: Optional[Dict[str, Tuple[float, float]]] = None

        # Last-known location for every object (persists across frames).
        # Initialised to "unknown"; updated whenever a detection is made.
        self._last_known_locations: Dict[str, str] = {
            oid: "unknown" for oid in self.ALL_OBJECT_IDS
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
        """Process a BGR frame and return object location assignments.

        Returns ``None`` when skipped (throttle) or on error.  Otherwise
        returns a dict::

            {
                "object_locations": {"soccer_ball": "table", ...},
                "region_assignments": {"table": ..., "white_basket": ..., ...},
                "tracked_objects": [...],
                "regions": {<region_id>: [TrackedObject, ...], ...},
                "objects": {<prompt>: [TrackedObject, ...], ...},
                "detections": {...},
            }
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

        # ── Extract regions ──────────────────────────────────────────
        # Each region prompt maps to at most one detection (top confidence).
        region_objs: Dict[str, Any] = {}  # region_id → TrackedObject
        for prompt, region_id in _REGION_PROMPT_TO_ID.items():
            candidates = sorted(
                detections.get(prompt, []),
                key=lambda o: o.confidence, reverse=True,
            )
            if candidates:
                region_objs[region_id] = candidates[0]

        # Initialise / update tracked region positions (EMA).
        if self._region_positions is None:
            self._region_positions = {}
        for region_id, robj in region_objs.items():
            if robj.bbox is None:
                continue
            cx, cy = robj.bbox.center
            if region_id in self._region_positions:
                old_cx, old_cy = self._region_positions[region_id]
                alpha = 0.3
                self._region_positions[region_id] = (
                    alpha * cx + (1 - alpha) * old_cx,
                    alpha * cy + (1 - alpha) * old_cy,
                )
            else:
                self._region_positions[region_id] = (cx, cy)

        # Build mask/bbox dicts for the assignment heuristic.
        region_masks: Dict[str, Optional[np.ndarray]] = {}
        region_bboxes: Dict[str, Optional[Any]] = {}
        region_assignments: Dict[str, str] = {}
        for region_id, robj in region_objs.items():
            region_masks[region_id] = robj.mask
            region_bboxes[region_id] = robj.bbox
            region_assignments[region_id] = region_id

        logger.debug(
            "Regions detected: %s",
            {rid: f"{robj.confidence:.2f}" for rid, robj in region_objs.items()},
        )

        # ── Assign each movable object to a region ───────────────────
        object_locations: Dict[str, str] = {}
        objects_by_prompt: Dict[str, list] = {}

        for prompt in OBJECT_PROMPTS:
            objs = sorted(
                detections.get(prompt, []),
                key=lambda o: o.confidence, reverse=True,
            )[:self.config.max_per_prompt]
            objects_by_prompt[prompt] = objs

            for i, obj in enumerate(objs):
                obj_id = self._prompt_to_object_id(prompt, i)
                region = self._assign_object_to_region(
                    obj.mask, obj.bbox, region_masks, region_bboxes,
                )
                object_locations[obj_id] = region

        # Update last-known locations for detected objects.
        for obj_id, region in object_locations.items():
            if region != "unknown":
                self._last_known_locations[obj_id] = region

        # Build full location map: every known object ID gets an entry.
        # Detected objects use the current frame's assignment; undetected
        # objects use their last-known location (or "unknown").
        all_object_locations: Dict[str, str] = {}
        for obj_id in self.ALL_OBJECT_IDS:
            if obj_id in object_locations:
                all_object_locations[obj_id] = object_locations[obj_id]
            else:
                all_object_locations[obj_id] = self._last_known_locations[obj_id]

        return {
            "object_locations": all_object_locations,
            "region_assignments": region_assignments,
            "tracked_objects": output.objects,
            "regions": {rid: [robj] for rid, robj in region_objs.items()},
            "objects": objects_by_prompt,
            "detected_this_frame": set(object_locations.keys()),
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

    # ── Object → region assignment heuristic ─────────────────────────

    def _assign_object_to_region(
        self,
        obj_mask: Optional[np.ndarray],
        obj_bbox: Optional[Any],
        region_masks: Dict[str, Optional[np.ndarray]],
        region_bboxes: Dict[str, Optional[Any]],
    ) -> str:
        """Determine which region an object is in.

        Baskets sit on the table, so an object inside a basket will
        overlap both the basket mask and the table mask.  To handle this
        we prefer the *most specific* (smallest) region when multiple
        regions exceed the overlap threshold.

        Strategy (in priority order):
        1. Mask overlap — pick the smallest region above threshold.
        2. Centroid containment — prefer smallest containing bbox.
        3. Proximity — assign to nearest region centre.
        """
        # 1. Mask overlap — collect all regions above threshold, pick smallest.
        if obj_mask is not None:
            obj_area = int(obj_mask.sum())
            if obj_area > 0:
                candidates: List[Tuple[str, float, int]] = []  # (region, ratio, mask_area)
                for region, rmask in region_masks.items():
                    if rmask is None:
                        continue
                    if rmask.shape != obj_mask.shape:
                        rmask = cv2.resize(
                            rmask, (obj_mask.shape[1], obj_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    intersection = int((obj_mask & rmask).sum())
                    ratio = intersection / obj_area
                    if ratio >= self.config.overlap_threshold:
                        region_area = int(rmask.sum())
                        candidates.append((region, ratio, region_area))

                if candidates:
                    # Sort by region area ascending (smallest first) so
                    # baskets are preferred over the encompassing table.
                    candidates.sort(key=lambda c: c[2])
                    return candidates[0][0]

        # 2. Centroid containment — prefer smallest containing bbox.
        if self.config.use_centroid_fallback and obj_bbox is not None:
            cx, cy = obj_bbox.center
            containing: List[Tuple[str, int]] = []
            for region, rbbox in region_bboxes.items():
                if rbbox is None:
                    continue
                if (rbbox.x_min <= cx <= rbbox.x_max
                        and rbbox.y_min <= cy <= rbbox.y_max):
                    containing.append((region, rbbox.area))
            if containing:
                containing.sort(key=lambda c: c[1])
                return containing[0][0]

        # 3. Proximity fallback
        if self.config.use_proximity_fallback and obj_bbox is not None:
            ocx, ocy = obj_bbox.center
            best_region = None
            best_dist = float("inf")
            for region, rbbox in region_bboxes.items():
                if rbbox is None:
                    continue
                rcx, rcy = rbbox.center
                d = ((ocx - rcx) ** 2 + (ocy - rcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_region = region
            if best_region is not None:
                return best_region

        return "unknown"

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _prompt_to_object_id(prompt: str, index: int) -> str:
        """Convert a SAM3 prompt + detection index to a scene object ID.

        When only one detection is found the ``_0`` suffix is omitted so
        IDs match ``initial_scene.json``.
        """
        base = _OBJECT_PROMPT_TO_ID.get(prompt, prompt.replace(" ", "_"))
        if index == 0:
            return base
        return f"{base}_{index}"

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
        """Draw masks, bboxes, and object-region assignments on *frame*."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        region_colors = {
            "table": (200, 200, 200),       # grey (BGR)
            "white_basket": (255, 255, 255), # white
            "blue_basket": (255, 100, 0),    # blue
        }
        object_color = (0, 0, 255)  # red

        region_data = result.get("regions", {})
        region_assignments = result.get("region_assignments", {})
        object_locations = result.get("object_locations", {})

        # Draw regions.
        for region_id, robjs in region_data.items():
            color = region_colors.get(region_id, (180, 180, 180))
            for robj in robjs:
                if robj.mask is not None:
                    mask = self._fit_mask(robj.mask, fh, fw)
                    overlay = vis.copy()
                    overlay[mask > 0] = color
                    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

                if robj.bbox is not None:
                    x1, y1 = int(robj.bbox.x_min), int(robj.bbox.y_min)
                    x2, y2 = int(robj.bbox.x_max), int(robj.bbox.y_max)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                    label = f"{region_id} ({robj.confidence:.2f})"
                    (tw, th_), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
                    )
                    cv2.rectangle(
                        vis, (x1, y1 - th_ - 8), (x1 + tw, y1), color, -1,
                    )
                    cv2.putText(
                        vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                    )

        # Draw objects.
        objects_by_prompt = result.get("objects", {})
        for prompt, objs in objects_by_prompt.items():
            for i, obj in enumerate(objs):
                obj_id = self._prompt_to_object_id(prompt, i)
                loc = object_locations.get(obj_id, "?")

                if obj.mask is not None:
                    mask = self._fit_mask(obj.mask, fh, fw)
                    overlay = vis.copy()
                    overlay[mask > 0] = object_color
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                if obj.bbox is not None:
                    x1, y1 = int(obj.bbox.x_min), int(obj.bbox.y_min)
                    x2, y2 = int(obj.bbox.x_max), int(obj.bbox.y_max)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), object_color, 2)
                    label = f"{obj_id} @ {loc} ({obj.confidence:.2f})"
                    (tw, th_), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2,
                    )
                    cv2.rectangle(
                        vis, (x1, y2), (x1 + tw, y2 + th_ + 8), object_color, -1,
                    )
                    cv2.putText(
                        vis, label, (x1, y2 + th_ + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    )

        # Summary panel — list all objects (detected and undetected).
        detected_this_frame = result.get("detected_this_frame", set())
        lines: List[Tuple[str, Tuple[int, int, int]]] = []

        # Region header.
        for rid in sorted(region_assignments):
            color = region_colors.get(rid, (180, 180, 180))
            lines.append((f"[region] {rid}", color))

        # All objects — bright if detected this frame, dim if last-known.
        for oid in self.ALL_OBJECT_IDS:
            loc = object_locations.get(oid, "unknown")
            if oid in detected_this_frame:
                color = (255, 255, 255)
                lines.append((f"{oid}: {loc}", color))
            else:
                color = (120, 120, 120)
                lines.append((f"{oid}: {loc} (last known)", color))

        panel_h = 30 + 22 * len(lines)
        cv2.rectangle(vis, (5, 5), (320, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (320, panel_h), (255, 255, 255), 1)
        y = 25
        for text, color in lines:
            cv2.putText(
                vis, text, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )
            y += 22

        return vis

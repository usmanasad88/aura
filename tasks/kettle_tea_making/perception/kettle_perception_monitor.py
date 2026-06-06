"""Task-specific perception monitor for the kettle tea-making task.

Locates the key tea-making items in each frame and reports where each one
is: the **storage area** (the **blue basket** itself, padded on all sides) or
the **working area** (the table holding the kettle).  An item in neither
region is reported as ``other``.

Every item is directly identifiable by a distinctive visual prompt, so each
SAM3 text prompt maps 1:1 to a semantic item — no cross-frame identity
tracking is needed for the items:

* tea          → ``"container with a red cap"``
* paper cup    → ``"paper cup"``
* water bottle → ``"water bottle"``
* biscuits     → ``"red biscuit packet"``
* powdered milk→ ``"container with a light blue cap"``

The regions are resolved from these prompts:

* ``"blue basket"``  → the **storage area** itself, padded on all sides
* ``"table"``        → the work surfaces (up to a few detections kept)
* ``"kettle"``       → marks the **working** table among the detected tables

Region identity is registered **once** — the first frame on which the basket
(storage) and the kettle-bearing table (working) are both resolved — then
frozen and reused for the rest of the run.  Because the setup is static,
locking the geometry as well as the designation makes the regions immune to
later occlusion: the kettle covering part of the working table, or a marker
briefly vanishing, can no longer change the regions.

Location decision for each item (priority order):

1. Mask overlap / bbox containment with the **padded basket** → ``storage_area``.
2. Mask overlap / bbox containment with the **working** table → ``working_area``.
3. Otherwise                                                   → ``other``.

Usage::

    monitor = KettlePerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)
    print(result["item_locations"])
    # {"tea_bag": "storage_area", "cup": "working_area", ...}
    print(result["cup_location"])  # "working_area"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Region names ──────────────────────────────────────────────────────
# Must match the ``valid_values`` of the ``*_location`` state variables in
# ``state_schema.json``.  Note: the schema currently only declares
# ``storage_area`` / ``working_area`` / ``unknown``; we additionally emit
# ``other`` per the demo requirements (an item on neither table).
STORAGE_REGION = "storage_area"
WORKING_REGION = "working_area"
OTHER_REGION = "other"

# SAM3 prompts.
TABLE_PROMPT = "table"
STORAGE_MARKER_PROMPT = "blue basket"   # marks the storage table
WORKING_MARKER_PROMPT = "kettle"        # marks the working table
KETTLE_LID_PROMPT = "kettle lid"        # open/closed inferred vs kettle bbox

# Semantic item id → SAM3 text prompt.  The ids match the ``*_location``
# state-schema variables so downstream nodes can write them directly.
ITEM_PROMPTS: Dict[str, str] = {
    # Tea-bag container: uncomment the line matching the physical setup.
    "tea_bag": "container with a red cap",        # closed container, red lid
    # "tea_bag": "open container with no lid",    # open / lidless container
    "cup": "paper cup",
    "water_bottle": "water bottle",
    "biscuits": "red biscuit packet",
    "milk_container": "container with a light blue cap",
}


@dataclass
class KettlePerceptionConfig:
    """Configuration for the kettle perception monitor."""

    # Item id → SAM3 prompt.  One SAM3 query per distinct prompt.
    item_prompts: Dict[str, str] = field(
        default_factory=lambda: dict(ITEM_PROMPTS)
    )
    table_prompt: str = TABLE_PROMPT
    storage_marker_prompt: str = STORAGE_MARKER_PROMPT
    working_marker_prompt: str = WORKING_MARKER_PROMPT
    kettle_lid_prompt: str = KETTLE_LID_PROMPT

    # The storage area is the blue basket grown by this fraction of its bbox
    # size on every side (0.2 → +20% left/right/top/bottom).
    storage_padding_frac: float = 0.2

    confidence_threshold: float = 0.25
    # Kettle-lid open/closed: ``lid_area / kettle_bbox_area``.  The lid is
    # hinged, so opening it tilts the lid up and enlarges the kettle bbox,
    # lowering the ratio.  ratio >= threshold → CLOSED, below → OPEN.
    lid_closed_area_ratio: float = 0.10
    # Minimum fraction of an item's mask overlapping a table to assign it.
    overlap_threshold: float = 0.05
    # Minimum fraction of a marker's mask overlapping a table to bind them.
    marker_overlap_threshold: float = 0.02
    # Fall back to bbox-centroid containment when masks are unavailable.
    use_centroid_fallback: bool = True
    # Assign an item to the nearest table as a last resort (else "other").
    use_proximity_fallback: bool = False
    # Run perception every N calls (1 = every frame).
    process_every_n: int = 1

    def all_prompts(self) -> List[str]:
        """All distinct SAM3 prompts (regions + markers + items)."""
        prompts = [self.table_prompt, self.storage_marker_prompt,
                   self.working_marker_prompt, self.kettle_lid_prompt]
        prompts.extend(self.item_prompts.values())
        seen: set = set()
        out: List[str] = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out


class KettlePerceptionMonitor:
    """Perception monitor that locates tea-making items for the kettle task.

    Wraps ``PerceptionModule`` (composition) with item-specific SAM3 prompts.
    The storage area is the blue basket (padded on all sides); the working
    area is the table the kettle overlaps.  Both regions are resolved once,
    frozen, and reused; each item is assigned to whichever locked region it
    overlaps.
    """

    def __init__(self, config: Optional[KettlePerceptionConfig] = None) -> None:
        self.config = config or KettlePerceptionConfig()
        self._perception = self._build_perception_module()
        self._call_count = 0

        # Regions are registered ONCE — the first frame on which the basket
        # (storage) and the kettle-bearing table (working) both resolve — then
        # frozen and reused.  The setup is static, so locking the geometry as
        # well as the designation makes the regions immune to later occlusion
        # (e.g. the kettle covering part of the working table).  Maps region →
        # frozen ``TrackedObject``.
        self._locked_regions: Optional[Dict[str, Any]] = None

    # ── Construction helpers ─────────────────────────────────────────

    def _build_perception_module(self):
        """Create the inner PerceptionModule with task-specific config."""
        from aura.utils.config import PerceptionConfig
        from aura.monitors.perception_module import PerceptionModule

        pcfg = PerceptionConfig(
            use_sam3=True,
            use_gemini_detection=False,
            default_prompts=self.config.all_prompts(),
            confidence_threshold=self.config.confidence_threshold,
        )
        return PerceptionModule(config=pcfg)

    # ── Main entry point ─────────────────────────────────────────────

    async def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a BGR frame and return per-item location assignments.

        Returns ``None`` when skipped (throttle) or on error.  Otherwise
        returns a dict::

            {
                "item_locations": {
                    "tea_bag":        "storage_area",
                    "cup":            "working_area",
                    ...
                },
                "<item>_location": "...",     # one per item (schema-friendly)
                "regions": {"storage_area": <table obj|None>,
                            "working_area": <table obj|None>},
                "markers": {"blue basket": <obj|None>, "kettle": <obj|None>},
                "items": {"<item>": <obj|None>, ...},
                "tracked_objects": [...],
                "detections": {prompt: [...], ...},
            }

        An item not detected this frame is reported as ``"unknown"`` so the
        downstream state entry never goes stale.
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

        # Group detections by prompt name, sorted by confidence.
        detections: Dict[str, list] = {p: [] for p in self.config.all_prompts()}
        for obj in output.objects:
            if obj.name in detections:
                detections[obj.name].append(obj)
        for dets in detections.values():
            dets.sort(key=lambda o: o.confidence, reverse=True)

        # ── Reference markers + candidate tables ─────────────────────
        basket = self._top(detections.get(self.config.storage_marker_prompt))
        kettle = self._top(detections.get(self.config.working_marker_prompt))
        lid = self._top(detections.get(self.config.kettle_lid_prompt))
        tables = detections.get(self.config.table_prompt, [])[:6]

        # ── Kettle lid open/closed ───────────────────────────────────
        lid_open, lid_ratio = self._lid_state(lid, kettle)

        # ── Resolve / track the two table regions ────────────────────
        region_objs = self._resolve_tables(tables, basket, kettle)
        regions = {STORAGE_REGION: region_objs.get(STORAGE_REGION),
                   WORKING_REGION: region_objs.get(WORKING_REGION)}

        # ── Resolve each item (top detection of its prompt) ──────────
        items: Dict[str, Optional[Any]] = {}
        for item_id, prompt in self.config.item_prompts.items():
            items[item_id] = self._top(detections.get(prompt))

        # ── Classify each item onto a table ──────────────────────────
        item_locations: Dict[str, str] = {}
        for item_id, obj in items.items():
            if obj is None:
                item_locations[item_id] = "unknown"
            else:
                item_locations[item_id] = self._classify_location(obj, regions)

        # State-schema variables this monitor owns, mirrored into the SSG.
        # ``run_perception_node`` only propagates the ``task_state`` dict to
        # ``ssg.set_task_state`` — arbitrary top-level result keys are ignored —
        # so anything that must reach the decision engine goes in here.
        task_state: Dict[str, Any] = {
            f"{item_id}_location": loc
            for item_id, loc in item_locations.items()
        }
        if lid_open is not None:
            task_state["lid_open"] = lid_open

        result: Dict[str, Any] = {
            "item_locations": item_locations,
            "task_state": task_state,
            "lid_open": lid_open,
            "lid_area_ratio": lid_ratio,
            "regions": regions,
            "markers": {self.config.storage_marker_prompt: basket,
                        self.config.working_marker_prompt: kettle,
                        self.config.kettle_lid_prompt: lid},
            "items": items,
            "tracked_objects": output.objects,
            "detections": {
                k: [{"id": o.id, "confidence": o.confidence,
                     "bbox": (int(o.bbox.x_min), int(o.bbox.y_min),
                              int(o.bbox.x_max), int(o.bbox.y_max))
                     if o.bbox else None}
                    for o in v]
                for k, v in detections.items()
            },
        }

        # Schema-friendly flat keys.
        for item_id, loc in item_locations.items():
            result[f"{item_id}_location"] = loc

        return result

    # ── Table resolution & tracking ──────────────────────────────────

    def _resolve_tables(
        self,
        tables: list,
        basket: Optional[Any],
        kettle: Optional[Any],
    ) -> Dict[str, Optional[Any]]:
        """Return ``{region: obj}`` for the storage and working regions.

        The storage region is the **blue basket** grown by
        ``storage_padding_frac`` on every side; the working region is the
        table the **kettle** overlaps most.  Both are registered exactly once —
        the first frame on which they both resolve — then frozen and reused for
        the rest of the run.  This is the fix for occlusion: once the regions
        are known, the kettle covering part of the working table (or a marker
        briefly vanishing) can no longer change them.
        """
        if self._locked_regions is not None:
            return self._locked_regions

        # Storage IS the blue basket (padded), not the table under it.
        storage = self._pad_object(basket)
        # Working is the table the kettle overlaps most.
        working = self._working_table(tables, kettle)

        region_objs = {STORAGE_REGION: storage, WORKING_REGION: working}

        # Lock only once BOTH regions resolve.
        if storage is not None and working is not None:
            self._locked_regions = region_objs
            logger.info(
                "Regions registered (locked): storage_area (padded basket) "
                "bbox=%s, working_area bbox=%s",
                self._bbox_tuple(storage), self._bbox_tuple(working),
            )
            return self._locked_regions

        # Partial resolution — show what we have, keep trying next frame.
        return region_objs

    def _working_table(
        self, tables: list, kettle: Optional[Any],
    ) -> Optional[Any]:
        """The table the kettle overlaps most; the sole table as a fallback."""
        if not tables:
            return None
        scores = [self._containment_score(kettle, t) for t in tables]
        if kettle is not None and max(scores) > 0:
            return tables[int(np.argmax(scores))]
        # No kettle signal — only commit if there's a single unambiguous table.
        if len(tables) == 1:
            return tables[0]
        return None

    def _pad_object(self, obj: Optional[Any]) -> Optional[Any]:
        """Return a copy of *obj* grown by ``storage_padding_frac`` per side.

        Both the bbox and (if present) the mask are padded so downstream
        overlap and centroid-containment checks see the enlarged region.
        """
        if obj is None or obj.bbox is None:
            return obj
        from aura.core.types import BoundingBox, TrackedObject

        b = obj.bbox
        pad_x = int(round((b.x_max - b.x_min) * self.config.storage_padding_frac))
        pad_y = int(round((b.y_max - b.y_min) * self.config.storage_padding_frac))
        padded_bbox = BoundingBox(
            x_min=b.x_min - pad_x, y_min=b.y_min - pad_y,
            x_max=b.x_max + pad_x, y_max=b.y_max + pad_y,
        )

        padded_mask = obj.mask
        if obj.mask is not None and max(pad_x, pad_y) > 0:
            k = max(pad_x, pad_y)
            kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
            padded_mask = cv2.dilate(obj.mask.astype(np.uint8), kernel) > 0

        return TrackedObject(
            id=obj.id, name=STORAGE_REGION, category=obj.category,
            pose=obj.pose, bbox=padded_bbox, mask=padded_mask,
            confidence=obj.confidence, last_seen=obj.last_seen,
            velocity=obj.velocity, metadata=dict(obj.metadata),
        )

    def _containment_score(
        self, marker: Optional[Any], table: Optional[Any],
    ) -> float:
        """Fraction of *marker* overlapping *table* (mask, else centroid)."""
        if marker is None or table is None:
            return 0.0

        mmask = getattr(marker, "mask", None)
        tmask = getattr(table, "mask", None)
        if mmask is not None and tmask is not None:
            marker_area = int(mmask.sum())
            if marker_area > 0:
                if tmask.shape != mmask.shape:
                    tmask = cv2.resize(
                        tmask, (mmask.shape[1], mmask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ratio = int((mmask & tmask).sum()) / marker_area
                if ratio >= self.config.marker_overlap_threshold:
                    return ratio

        if (self.config.use_centroid_fallback
                and marker.bbox is not None and table.bbox is not None):
            cx, cy = marker.bbox.center
            tb = table.bbox
            if tb.x_min <= cx <= tb.x_max and tb.y_min <= cy <= tb.y_max:
                return 1.0
        return 0.0

    @staticmethod
    def _bbox_tuple(obj: Optional[Any]) -> Optional[Tuple[int, int, int, int]]:
        """``(x_min, y_min, x_max, y_max)`` of an object's bbox, for logging."""
        if obj is None or obj.bbox is None:
            return None
        b = obj.bbox
        return (int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max))

    # ── Kettle lid open/closed ────────────────────────────────────────

    def _lid_state(
        self, lid: Optional[Any], kettle: Optional[Any],
    ) -> Tuple[Optional[bool], float]:
        """Infer kettle-lid open/closed from ``lid_area / kettle_bbox_area``.

        The lid is hinged: opening it tilts the lid up, which enlarges the
        ``kettle`` detection's bbox and so LOWERS the ratio.  Hence
        ``ratio >= lid_closed_area_ratio`` → CLOSED, below → OPEN.

        Returns ``(is_open, ratio)``; ``is_open`` is None when the lid (or
        kettle body) isn't detected this frame.
        """
        if lid is None or kettle is None or kettle.bbox is None:
            return None, 0.0

        lmask = getattr(lid, "mask", None)
        if lmask is not None and int(lmask.sum()) > 0:
            lid_area = int(lmask.sum())
        elif lid.bbox is not None:
            lb = lid.bbox
            lid_area = max(0, (lb.x_max - lb.x_min) * (lb.y_max - lb.y_min))
        else:
            return None, 0.0

        kb = kettle.bbox
        kettle_area = max(1, (kb.x_max - kb.x_min) * (kb.y_max - kb.y_min))
        ratio = lid_area / kettle_area
        return ratio < self.config.lid_closed_area_ratio, ratio

    # ── Item location classifier ──────────────────────────────────────

    def _classify_location(
        self,
        item: Any,
        regions: Dict[str, Optional[Any]],
    ) -> str:
        """Classify an item onto the storage / working table, else other.

        Strategy: mask overlap (``|I ∩ T| / |I|``) → bbox-centroid
        containment → (optional) nearest-table proximity.
        """
        best_region: Optional[str] = None
        best_ratio = 0.0

        imask = getattr(item, "mask", None)
        item_area = int(imask.sum()) if imask is not None else 0

        for region, table in regions.items():
            if table is None:
                continue
            tmask = getattr(table, "mask", None)
            if item_area > 0 and tmask is not None:
                if tmask.shape != imask.shape:
                    tmask = cv2.resize(
                        tmask, (imask.shape[1], imask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ratio = int((imask & tmask).sum()) / item_area
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_region = region

        if best_region is not None and best_ratio >= self.config.overlap_threshold:
            return best_region

        # Centroid containment fallback.
        if self.config.use_centroid_fallback and item.bbox is not None:
            cx, cy = item.bbox.center
            for region, table in regions.items():
                if table is None or table.bbox is None:
                    continue
                tb = table.bbox
                if tb.x_min <= cx <= tb.x_max and tb.y_min <= cy <= tb.y_max:
                    return region

        # Nearest-table proximity fallback (off by default).
        if self.config.use_proximity_fallback and item.bbox is not None:
            icx, icy = item.bbox.center
            nearest, best_dist = None, float("inf")
            for region, table in regions.items():
                if table is None or table.bbox is None:
                    continue
                tcx, tcy = table.bbox.center
                d = ((icx - tcx) ** 2 + (icy - tcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, nearest = d, region
            if nearest is not None:
                return nearest

        return OTHER_REGION

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _top(dets: Optional[list]) -> Optional[Any]:
        """Return the highest-confidence detection, or None."""
        if not dets:
            return None
        return dets[0]

    @staticmethod
    def _fit_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize *mask* to (h, w) if shapes differ."""
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Visualization ────────────────────────────────────────────────

    def visualize(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw table regions, markers, item bboxes, and location labels.

        Returns a new image (does not modify the input).
        """
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        regions = result.get("regions", {})
        markers = result.get("markers", {})
        items = result.get("items", {})
        item_locations = result.get("item_locations", {})

        region_colors = {
            STORAGE_REGION: (255, 165, 0),    # orange (BGR)
            WORKING_REGION: (0, 200, 0),      # green
        }
        item_colors = {
            "tea_bag":        (0, 0, 255),       # red (red cap)
            "cup":            (255, 255, 255),  # white (paper cup)
            "water_bottle":   (255, 200, 0),    # cyan-ish
            "biscuits":       (0, 0, 255),      # red
            "milk_container": (230, 180, 80),   # light blue cap
        }
        loc_colors = {
            STORAGE_REGION: (255, 165, 0),
            WORKING_REGION: (0, 200, 0),
            OTHER_REGION:   (0, 165, 255),
            "unknown":      (128, 128, 128),
        }

        # Draw table regions first (under everything).
        for region_name, tobj in regions.items():
            if tobj is None:
                continue
            color = region_colors.get(region_name, (200, 200, 200))
            if tobj.mask is not None:
                mask = self._fit_mask(tobj.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
            if tobj.bbox is not None:
                x1, y1 = int(tobj.bbox.x_min), int(tobj.bbox.y_min)
                x2, y2 = int(tobj.bbox.x_max), int(tobj.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = f"{region_name} ({tobj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw markers as thin dashed-style outlines (basket / kettle).
        for marker_name, mobj in markers.items():
            if mobj is None or mobj.bbox is None:
                continue
            x1, y1 = int(mobj.bbox.x_min), int(mobj.bbox.y_min)
            x2, y2 = int(mobj.bbox.x_max), int(mobj.bbox.y_max)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(vis, marker_name, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Draw items.
        for item_id, obj in items.items():
            if obj is None:
                continue
            color = item_colors.get(item_id, (0, 200, 255))
            loc = item_locations.get(item_id, "unknown")
            if obj.mask is not None:
                mask = self._fit_mask(obj.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            if obj.bbox is not None:
                x1, y1 = int(obj.bbox.x_min), int(obj.bbox.y_min)
                x2, y2 = int(obj.bbox.x_max), int(obj.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{item_id} @ {loc} ({obj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y2), (x1 + tw, y2 + th + 8), color, -1)
                txt_color = (0, 0, 0) if sum(color) > 500 else (255, 255, 255)
                cv2.putText(vis, label, (x1, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

        # Summary panel (top-left).
        n_lines = len(item_locations)
        panel_h = 35 + 25 * n_lines
        cv2.rectangle(vis, (5, 5), (360, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (360, panel_h), (255, 255, 255), 1)
        cv2.putText(vis, "Item locations", (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y = 52
        for item_id in sorted(item_locations):
            loc = item_locations[item_id]
            color = loc_colors.get(loc, (200, 200, 200))
            cv2.putText(vis, f"{item_id}: {loc}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25

        return vis

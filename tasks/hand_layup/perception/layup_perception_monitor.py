"""Task-specific perception monitor for the hand layup task.

Tracks 2 tables and 2 bottles using SAM3 segmentation with generic
visual prompts (``"table"`` and ``"bottle"``), then uses spatial
heuristics to determine which table each bottle is currently on.

The two bottles are visually identical (both black), so identity is
assigned from their initial horizontal position in the frame:

* the bottle starting on the **left** of the frame  → ``resin_bottle``
* the bottle starting on the **right** of the frame → ``hardener_bottle``

After the initial assignment, each bottle is tracked by spatial
proximity (greedy nearest-neighbour against stored centroids) so that
SAM3 detection-order changes across frames do not flip the identities.

Table disambiguation uses a spatial prior derived from
``initial_scene.json``: bottles initially sit on the storage table, so
on the first frame the table whose bbox contains the most bottle
centroids is labelled ``storage_area``.

Usage::

    monitor = LayupPerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)
    print(result["bottle_locations"])
    # {"resin_bottle": "storage_area", "hardener_bottle": "storage_area"}
    print(result["resin_bottle_location"], result["hardener_bottle_location"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Semantic bottle identities, ordered left → right in the initial frame.
RESIN_BOTTLE_ID = "resin_bottle"
HARDENER_BOTTLE_ID = "hardener_bottle"
BOTTLE_IDS_LEFT_TO_RIGHT: Tuple[str, str] = (RESIN_BOTTLE_ID, HARDENER_BOTTLE_ID)

# Region names — must match IDs declared in ``initial_scene.json`` and
# the ``valid_values`` list in ``state_schema.json``.
STORAGE_REGION = "storage_area"
WORKPLACE_REGION = "workplace"


@dataclass
class LayupPerceptionConfig:
    """Configuration for the layup perception monitor."""

    # Generic SAM3 text prompts — one per visual class.
    # SAM3 may return multiple detections per prompt.
    sam3_prompts: List[str] = field(default_factory=lambda: ["table", "bottle"])

    confidence_threshold: float = 0.25
    # Minimum fraction of bottle mask overlapping a table mask to assign.
    overlap_threshold: float = 0.02
    # Use bbox centroid containment when masks are unavailable.
    use_centroid_fallback: bool = True
    # Use vertical proximity (closest table below bottle) as last resort.
    use_proximity_fallback: bool = True
    # Run perception every N calls (1 = every frame).
    process_every_n: int = 1


class LayupPerceptionMonitor:
    """Perception monitor that tracks tables and bottles for the layup task.

    Wraps ``PerceptionModule`` (composition) with generic SAM3 prompts
    and spatial heuristics for bottle→table assignment.

    On the first frame where both bottles are visible, identity is
    resolved by horizontal position (left → ``resin_bottle``, right →
    ``hardener_bottle``) and the centroids are stored.  Subsequent
    frames match detections to stored centroids by proximity so that
    identities remain stable even if SAM3 returns the bottles in a
    different detection order.

    Tables are disambiguated on the first frame using the prior that
    bottles start on the storage table, then matched spatially thereafter.
    """

    def __init__(self, config: Optional[LayupPerceptionConfig] = None) -> None:
        self.config = config or LayupPerceptionConfig()
        self._perception = self._build_perception_module()
        self._call_count = 0

        # Table identity — resolved on first frame, then spatially
        # tracked so that detection-order changes don't flip labels.
        # Maps region name → bbox center (cx, cy) for spatial matching.
        self._table_positions: Optional[Dict[str, Tuple[float, float]]] = None

        # Bottle identity — resolved on first frame with 2 bottles by
        # horizontal position, then tracked spatially thereafter.
        # Maps semantic bottle id → bbox center (cx, cy).
        self._bottle_positions: Optional[Dict[str, Tuple[float, float]]] = None

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
        """Process a BGR frame and return bottle location assignments.

        Returns ``None`` when skipped (throttle) or on error.  Otherwise
        returns a dict with semantic bottle keys::

            {
                "bottle_locations": {
                    "resin_bottle":    "storage_area",
                    "hardener_bottle": "storage_area",
                },
                "resin_bottle_location":    "storage_area",
                "hardener_bottle_location": "storage_area",
                "bottle_identities": {0: "resin_bottle", 1: "hardener_bottle"},
                "table_regions": {"table_0": "storage_area", "table_1": "workplace"},
                "tracked_objects": [TrackedObject, ...],
                "tables": [...], "bottles": [...],
                "detections": {"table": [...], "bottle": [...]},
            }

        Before both bottles have been seen simultaneously, identities
        remain unresolved and bottles are reported with the generic
        fallback ids ``bottle_0`` / ``bottle_1``.  The semantic
        ``*_location`` keys are only emitted once identities are
        resolved.
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

        # Sort each group by confidence descending — keep top 2 of each.
        tables = sorted(detections.get("table", []),
                        key=lambda o: o.confidence, reverse=True)[:2]
        bottles = sorted(detections.get("bottle", []),
                         key=lambda o: o.confidence, reverse=True)[:2]

        n_tables = len(tables)
        n_bottles = len(bottles)
        logger.debug("SAM3 raw: %d table(s), %d bottle(s)", n_tables, n_bottles)

        # ── Disambiguate tables ──────────────────────────────────────
        if self._table_positions is None and n_tables >= 2:
            idx_map = self._identify_tables(tables, bottles)
            self._table_positions = {}
            for idx, region in idx_map.items():
                tobj = tables[idx]
                if tobj.bbox is not None:
                    self._table_positions[region] = tobj.bbox.center
            logger.info("Table identity resolved: %s", self._table_positions)
        elif self._table_positions is None and n_tables == 1:
            tobj = tables[0]
            if tobj.bbox is not None:
                self._table_positions = {STORAGE_REGION: tobj.bbox.center}
            logger.info("Single table detected, assuming %s", STORAGE_REGION)

        # Match current detections to known table positions by proximity.
        table_masks: Dict[str, Optional[np.ndarray]] = {}
        table_bboxes: Dict[str, Optional[Any]] = {}
        table_regions: Dict[str, str] = {}

        if self._table_positions is not None and n_tables >= 2:
            assigned = self._match_by_position(
                tables, self._table_positions, fallback_prefix="table",
            )
            for i, region in assigned.items():
                table_regions[f"table_{i}"] = region
                table_masks[region] = tables[i].mask
                table_bboxes[region] = tables[i].bbox
                if tables[i].bbox is not None:
                    self._update_position(
                        self._table_positions, region, tables[i].bbox.center,
                    )
        elif self._table_positions is not None and n_tables == 1:
            tobj = tables[0]
            if tobj.bbox is not None:
                cx, cy = tobj.bbox.center
                best_region = min(
                    self._table_positions,
                    key=lambda r: ((cx - self._table_positions[r][0]) ** 2
                                   + (cy - self._table_positions[r][1]) ** 2),
                )
            else:
                best_region = STORAGE_REGION
            table_regions["table_0"] = best_region
            table_masks[best_region] = tobj.mask
            table_bboxes[best_region] = tobj.bbox
        else:
            for i, tobj in enumerate(tables):
                region = f"table_{i}"
                table_regions[f"table_{i}"] = region
                table_masks[region] = tobj.mask
                table_bboxes[region] = tobj.bbox

        # ── Resolve bottle identities ────────────────────────────────
        # First frame with 2 bottles: left → resin, right → hardener.
        if self._bottle_positions is None and n_bottles >= 2:
            ordered = sorted(
                [b for b in bottles if b.bbox is not None],
                key=lambda b: b.bbox.center[0],  # ascending x (left first)
            )
            if len(ordered) >= 2:
                self._bottle_positions = {
                    BOTTLE_IDS_LEFT_TO_RIGHT[0]: ordered[0].bbox.center,
                    BOTTLE_IDS_LEFT_TO_RIGHT[1]: ordered[1].bbox.center,
                }
                logger.info(
                    "Bottle identity resolved by initial x-position: "
                    "left=%s @ x=%.1f, right=%s @ x=%.1f",
                    BOTTLE_IDS_LEFT_TO_RIGHT[0], ordered[0].bbox.center[0],
                    BOTTLE_IDS_LEFT_TO_RIGHT[1], ordered[1].bbox.center[0],
                )

        # ── Assign each bottle to a table + resolve its identity ─────
        bottle_identities: Dict[int, str] = {}
        if self._bottle_positions is not None and n_bottles >= 1:
            assigned = self._match_by_position(
                bottles, self._bottle_positions, fallback_prefix="bottle",
            )
            for i, bid in assigned.items():
                bottle_identities[i] = bid
                if bottles[i].bbox is not None:
                    self._update_position(
                        self._bottle_positions, bid, bottles[i].bbox.center,
                    )

        bottle_locations: Dict[str, str] = {}
        for i, bobj in enumerate(bottles):
            bid = bottle_identities.get(i, f"bottle_{i}")
            region = self._assign_bottle_to_table(
                bobj.mask, bobj.bbox, table_masks, table_bboxes,
            )
            bottle_locations[bid] = region

        result: Dict[str, Any] = {
            "bottle_locations": bottle_locations,
            "bottle_identities": bottle_identities,
            "table_regions": table_regions,
            "tracked_objects": output.objects,
            "tables": tables,
            "bottles": bottles,
            "detections": {
                k: [{"id": o.id, "confidence": o.confidence,
                     "bbox": (int(o.bbox.x_min), int(o.bbox.y_min),
                              int(o.bbox.x_max), int(o.bbox.y_max))
                     if o.bbox else None}
                    for o in v]
                for k, v in detections.items()
            },
        }

        # Expose semantic ``*_location`` keys so downstream nodes can
        # write them directly into SSG task_state / state_schema vars.
        #
        # Once bottle identities are resolved we know which semantic ids we
        # are responsible for.  Any resolved id that SAM3 did not detect this
        # frame is reported as "unknown" rather than omitted — this prevents
        # the downstream task_state entry from going stale and retaining a
        # confident-but-outdated location value.
        if self._bottle_positions is not None:
            for bid in BOTTLE_IDS_LEFT_TO_RIGHT:
                loc = bottle_locations.get(bid, "unknown")
                result[f"{bid}_location"] = loc
                # Ensure bottle_locations is complete so detected_bottle_locations
                # in SSG task_state is also a full snapshot.
                bottle_locations[bid] = loc
            result["bottle_locations"] = bottle_locations
        else:
            for bid in BOTTLE_IDS_LEFT_TO_RIGHT:
                if bid in bottle_locations and bottle_locations[bid] != "unknown":
                    result[f"{bid}_location"] = bottle_locations[bid]

        return result

    # ── Table disambiguation ─────────────────────────────────────────

    def _identify_tables(
        self,
        tables: list,
        bottles: list,
    ) -> Dict[int, str]:
        """Identify which table is storage_area vs workplace.

        Heuristic: bottles initially sit on the storage table, so the
        table whose bbox contains the most bottle centroids is storage.
        Falls back to mask overlap, then x-position.
        """
        # Strategy 1: bbox containment of bottle centroids.
        contain_count = [0, 0]
        for tbl_idx, tobj in enumerate(tables):
            if tobj.bbox is None:
                continue
            for bobj in bottles:
                if bobj.bbox is None:
                    continue
                cx, cy = bobj.bbox.center
                if (tobj.bbox.x_min <= cx <= tobj.bbox.x_max
                        and tobj.bbox.y_min <= cy <= tobj.bbox.y_max):
                    contain_count[tbl_idx] += 1

        if contain_count[0] != contain_count[1]:
            storage_idx = 0 if contain_count[0] > contain_count[1] else 1
            other_idx = 1 - storage_idx
            return {storage_idx: STORAGE_REGION, other_idx: WORKPLACE_REGION}

        # Strategy 2: mask overlap.
        overlap_scores = [0.0, 0.0]
        for tbl_idx, tobj in enumerate(tables):
            if tobj.mask is None:
                continue
            for bobj in bottles:
                if bobj.mask is None:
                    continue
                bmask, tmask = bobj.mask, tobj.mask
                if tmask.shape != bmask.shape:
                    tmask = cv2.resize(
                        tmask, (bmask.shape[1], bmask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ba = int(bmask.sum())
                if ba > 0:
                    overlap_scores[tbl_idx] += int((bmask & tmask).sum()) / ba

        if overlap_scores[0] != overlap_scores[1]:
            storage_idx = 0 if overlap_scores[0] > overlap_scores[1] else 1
            other_idx = 1 - storage_idx
            return {storage_idx: STORAGE_REGION, other_idx: WORKPLACE_REGION}

        # Strategy 3: proximity — table closest to bottle centroids.
        dist_sums = [0.0, 0.0]
        for tbl_idx, tobj in enumerate(tables):
            if tobj.bbox is None:
                dist_sums[tbl_idx] = float("inf")
                continue
            tcx, tcy = tobj.bbox.center
            for bobj in bottles:
                if bobj.bbox is None:
                    continue
                bcx, bcy = bobj.bbox.center
                dist_sums[tbl_idx] += ((tcx - bcx) ** 2 + (tcy - bcy) ** 2) ** 0.5

        storage_idx = 0 if dist_sums[0] <= dist_sums[1] else 1
        other_idx = 1 - storage_idx
        return {storage_idx: STORAGE_REGION, other_idx: WORKPLACE_REGION}

    # ── Spatial matching (shared by tables & bottles) ────────────────

    @staticmethod
    def _match_by_position(
        detections: list,
        known_positions: Dict[str, Tuple[float, float]],
        fallback_prefix: str = "obj",
    ) -> Dict[int, str]:
        """Match detections to known-id centroids by greedy proximity.

        Returns ``{detection_idx: known_id}``.  Unmatched detections
        get a generic ``f"{fallback_prefix}_{i}"`` label derived from
        their index (e.g. ``table_2`` or ``bottle_2``).
        """
        assigned: Dict[int, str] = {}
        used_ids: set = set()

        pairs: list = []
        for i, det in enumerate(detections):
            if det.bbox is None:
                continue
            cx, cy = det.bbox.center
            for kid, (rcx, rcy) in known_positions.items():
                dist = ((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5
                pairs.append((dist, i, kid))

        pairs.sort()
        for _, i, kid in pairs:
            if i in assigned or kid in used_ids:
                continue
            assigned[i] = kid
            used_ids.add(kid)

        # Any unmatched detections get a generic fallback label.
        for i in range(len(detections)):
            if i not in assigned:
                assigned[i] = f"{fallback_prefix}_{i}"

        return assigned

    @staticmethod
    def _update_position(
        store: Dict[str, Tuple[float, float]],
        key: str,
        new_center: Tuple[float, float],
        alpha: float = 0.3,
    ) -> None:
        """Exponential moving average to smooth tracked centroids."""
        cx, cy = new_center
        if key in store:
            old_cx, old_cy = store[key]
            store[key] = (
                alpha * cx + (1 - alpha) * old_cx,
                alpha * cy + (1 - alpha) * old_cy,
            )
        else:
            store[key] = (cx, cy)

    # ── Heuristic ────────────────────────────────────────────────────

    def _assign_bottle_to_table(
        self,
        bottle_mask: Optional[np.ndarray],
        bottle_bbox: Optional[Any],
        table_masks: Dict[str, Optional[np.ndarray]],
        table_bboxes: Dict[str, Optional[Any]],
    ) -> str:
        """Determine which table a bottle is on.

        Strategy (in priority order):
        1. Mask overlap: ``|B AND T| / |B|`` for each table.
        2. Centroid containment: bottle center inside table bbox.
        3. Proximity: assign to nearest table (Euclidean between centers).
        """
        # ── 1. Mask overlap ──────────────────────────────────────────
        if bottle_mask is not None:
            bottle_area = int(bottle_mask.sum())
            if bottle_area > 0:
                best_region: Optional[str] = None
                best_ratio: float = 0.0
                for region, tmask in table_masks.items():
                    if tmask is None:
                        continue
                    if tmask.shape != bottle_mask.shape:
                        tmask = cv2.resize(
                            tmask, (bottle_mask.shape[1], bottle_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    intersection = int((bottle_mask & tmask).sum())
                    ratio = intersection / bottle_area
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_region = region

                if best_region is not None and best_ratio >= self.config.overlap_threshold:
                    return best_region

        # ── 2. Centroid containment ──────────────────────────────────
        if self.config.use_centroid_fallback and bottle_bbox is not None:
            cx, cy = bottle_bbox.center
            for region, tbbox in table_bboxes.items():
                if tbbox is None:
                    continue
                if (tbbox.x_min <= cx <= tbbox.x_max
                        and tbbox.y_min <= cy <= tbbox.y_max):
                    return region

        # ── 3. Proximity fallback ────────────────────────────────────
        if self.config.use_proximity_fallback and bottle_bbox is not None:
            bcx, bcy = bottle_bbox.center
            best_region = None
            best_dist = float("inf")
            for region, tbbox in table_bboxes.items():
                if tbbox is None:
                    continue
                tcx, tcy = tbbox.center
                d = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_region = region
            if best_region is not None:
                return best_region

        return "unknown"

    # ── Helpers ───────────────────────────────────────────────────────

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
        """Draw masks, bboxes, and bottle-table assignments on *frame*.

        Returns a new image (does not modify the input).
        """
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        tables: list = result.get("tables", [])
        bottles: list = result.get("bottles", [])
        bottle_locs = result.get("bottle_locations", {})
        table_regions = result.get("table_regions", {})
        bottle_identities: Dict[int, str] = result.get("bottle_identities", {})

        region_colors = {
            STORAGE_REGION:   (255, 165, 0),    # orange (BGR)
            WORKPLACE_REGION: (0, 255, 0),      # green
        }
        bottle_colors = {
            RESIN_BOTTLE_ID:    (0, 0, 255),    # red
            HARDENER_BOTTLE_ID: (255, 0, 255),  # magenta
        }
        default_bottle_color = (0, 200, 255)    # yellow-orange fallback

        # Draw tables.
        for i, tobj in enumerate(tables):
            tid = f"table_{i}"
            region = table_regions.get(tid, tid)
            color = region_colors.get(region, (200, 200, 200))

            if tobj.mask is not None:
                mask = self._fit_mask(tobj.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

            if tobj.bbox is not None:
                x1, y1 = int(tobj.bbox.x_min), int(tobj.bbox.y_min)
                x2, y2 = int(tobj.bbox.x_max), int(tobj.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = f"{region} ({tobj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(vis, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw bottles — colour-coded by semantic identity.
        for i, bobj in enumerate(bottles):
            bid = bottle_identities.get(i, f"bottle_{i}")
            loc = bottle_locs.get(bid, "?")
            color = bottle_colors.get(bid, default_bottle_color)

            if bobj.mask is not None:
                mask = self._fit_mask(bobj.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            if bobj.bbox is not None:
                x1, y1 = int(bobj.bbox.x_min), int(bobj.bbox.y_min)
                x2, y2 = int(bobj.bbox.x_max), int(bobj.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{bid} @ {loc} ({bobj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y2), (x1 + tw, y2 + th + 8), color, -1)
                cv2.putText(vis, label, (x1, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Summary panel (top-left).
        panel_h = 30 + 25 * (len(bottle_locs) + len(table_regions))
        cv2.rectangle(vis, (5, 5), (320, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (320, panel_h), (255, 255, 255), 1)
        y = 25
        for tid in sorted(table_regions):
            region = table_regions[tid]
            color = region_colors.get(region, (200, 200, 200))
            cv2.putText(vis, f"{tid} = {region}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25
        for bid in sorted(bottle_locs):
            loc = bottle_locs[bid]
            color = bottle_colors.get(bid, default_bottle_color)
            cv2.putText(vis, f"{bid}: {loc}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25

        return vis

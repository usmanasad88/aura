"""Task-specific perception monitor for the hand layup task.

Tracks 2 tables and 2 bottles using SAM3 segmentation with generic
visual prompts (``"table"`` and ``"bottle"``), then uses spatial
heuristics to determine which table each bottle is currently on.

Since the two bottles are visually identical (both black), this monitor
does **not** try to distinguish resin vs hardener.  In standalone mode
they are reported as ``bottle_0`` / ``bottle_1``.  When integrated with
the full AURA workflow, the decision engine maps them to semantic IDs
via action history (e.g. after ``move_resin_from_storage_to_workplace``
the bottle that moved becomes ``resin_bottle``).

Table disambiguation uses a spatial prior derived from
``initial_scene.json``: bottles initially sit on the storage table, so
on the first frame the table whose bbox contains the most bottle
centroids is labelled storage.

Usage::

    monitor = LayupPerceptionMonitor()
    result = await monitor.process_frame(bgr_frame)
    print(result["bottle_locations"])
    # {"bottle_0": "storage", "bottle_1": "storage"}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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

    Tables are disambiguated on the first frame using the prior that
    bottles start on the storage table.
    """

    def __init__(self, config: Optional[LayupPerceptionConfig] = None) -> None:
        self.config = config or LayupPerceptionConfig()
        self._perception = self._build_perception_module()
        self._call_count = 0

        # Table identity — resolved on first frame, then spatially
        # tracked so that detection-order changes don't flip labels.
        # Maps region name → bbox center (cx, cy) for spatial matching.
        self._table_positions: Optional[Dict[str, Tuple[float, float]]] = None

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
        returns a dict::

            {
                "bottle_locations": {"bottle_0": "storage", ...},
                "table_regions": {"table_0": "storage", "table_1": "workplace"},
                "tracked_objects": [TrackedObject, ...],
                "tables": [...], "bottles": [...],
                "detections": {"table": [...], "bottle": [...]},
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
            # First frame: resolve identity using bottle prior.
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
                self._table_positions = {"storage": tobj.bbox.center}
            logger.info("Single table detected, assuming storage")

        # Match current detections to known table positions by proximity.
        table_masks: Dict[str, Optional[np.ndarray]] = {}
        table_bboxes: Dict[str, Optional[Any]] = {}
        table_regions: Dict[str, str] = {}

        if self._table_positions is not None and n_tables >= 2:
            assigned = self._match_tables_by_position(tables)
            for i, region in assigned.items():
                table_regions[f"table_{i}"] = region
                table_masks[region] = tables[i].mask
                table_bboxes[region] = tables[i].bbox
                # Update stored position with exponential moving average.
                if tables[i].bbox is not None:
                    cx, cy = tables[i].bbox.center
                    if region in self._table_positions:
                        old_cx, old_cy = self._table_positions[region]
                        alpha = 0.3  # Smoothing factor.
                        self._table_positions[region] = (
                            alpha * cx + (1 - alpha) * old_cx,
                            alpha * cy + (1 - alpha) * old_cy,
                        )
                    else:
                        self._table_positions[region] = (cx, cy)
        elif self._table_positions is not None and n_tables == 1:
            # Single detection — find closest known region.
            tobj = tables[0]
            if tobj.bbox is not None:
                cx, cy = tobj.bbox.center
                best_region = min(
                    self._table_positions,
                    key=lambda r: ((cx - self._table_positions[r][0]) ** 2
                                   + (cy - self._table_positions[r][1]) ** 2),
                )
            else:
                best_region = "storage"
            table_regions["table_0"] = best_region
            table_masks[best_region] = tobj.mask
            table_bboxes[best_region] = tobj.bbox
        else:
            for i, tobj in enumerate(tables):
                region = f"table_{i}"
                table_regions[f"table_{i}"] = region
                table_masks[region] = tobj.mask
                table_bboxes[region] = tobj.bbox

        # ── Assign each bottle to a table ────────────────────────────
        bottle_locations: Dict[str, str] = {}
        for i, bobj in enumerate(bottles):
            bid = f"bottle_{i}"
            region = self._assign_bottle_to_table(
                bobj.mask, bobj.bbox, table_masks, table_bboxes,
            )
            bottle_locations[bid] = region

        return {
            "bottle_locations": bottle_locations,
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

    # ── Table disambiguation ─────────────────────────────────────────

    def _identify_tables(
        self,
        tables: list,
        bottles: list,
    ) -> Dict[int, str]:
        """Identify which table is storage vs workplace.

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
            return {storage_idx: "storage", other_idx: "workplace"}

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
            return {storage_idx: "storage", other_idx: "workplace"}

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
        return {storage_idx: "storage", other_idx: "workplace"}

    # ── Spatial table matching ──────────────────────────────────────

    def _match_tables_by_position(
        self,
        tables: list,
    ) -> Dict[int, str]:
        """Match current table detections to known positions by proximity.

        Uses greedy nearest-neighbour assignment so that detection index
        order (which may change across frames) does not affect the result.
        """
        assert self._table_positions is not None
        regions = list(self._table_positions.keys())
        assigned: Dict[int, str] = {}
        used_regions: set = set()

        # Build distance matrix and greedily assign closest pairs.
        pairs: list = []
        for i, tobj in enumerate(tables):
            if tobj.bbox is None:
                continue
            cx, cy = tobj.bbox.center
            for region in regions:
                rcx, rcy = self._table_positions[region]
                dist = ((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5
                pairs.append((dist, i, region))

        pairs.sort()
        for _, i, region in pairs:
            if i in assigned or region in used_regions:
                continue
            assigned[i] = region
            used_regions.add(region)

        # Any unmatched detections get a generic label.
        for i in range(len(tables)):
            if i not in assigned:
                assigned[i] = f"table_{i}"

        return assigned

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

        region_colors = {
            "storage": (255, 165, 0),     # orange (BGR)
            "workplace": (0, 255, 0),     # green
        }
        bottle_color = (0, 0, 255)        # red

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

        # Draw bottles.
        for i, bobj in enumerate(bottles):
            bid = f"bottle_{i}"
            loc = bottle_locs.get(bid, "?")

            if bobj.mask is not None:
                mask = self._fit_mask(bobj.mask, fh, fw)
                overlay = vis.copy()
                overlay[mask > 0] = bottle_color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            if bobj.bbox is not None:
                x1, y1 = int(bobj.bbox.x_min), int(bobj.bbox.y_min)
                x2, y2 = int(bobj.bbox.x_max), int(bobj.bbox.y_max)
                cv2.rectangle(vis, (x1, y1), (x2, y2), bottle_color, 2)
                label = f"{bid} @ {loc} ({bobj.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y2), (x1 + tw, y2 + th + 8), bottle_color, -1)
                cv2.putText(vis, label, (x1, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Summary panel (top-left).
        panel_h = 30 + 25 * (len(bottle_locs) + len(table_regions))
        cv2.rectangle(vis, (5, 5), (280, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (280, panel_h), (255, 255, 255), 1)
        y = 25
        for tid in sorted(table_regions):
            region = table_regions[tid]
            color = region_colors.get(region, (200, 200, 200))
            cv2.putText(vis, f"{tid} = {region}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25
        for bid in sorted(bottle_locs):
            loc = bottle_locs[bid]
            cv2.putText(vis, f"{bid}: {loc}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y += 25

        return vis

"""6DOF Pose Tracking Monitor for known objects using Any6D + DA3.

Wraps the Any6D 6DOF pose estimation/tracking pipeline as an AURA
monitor.  Handles:

* **Depth estimation** – Depth Anything 3 (DA3) metric depth
* **Segmentation** – SAM3 masks (from PerceptionModule or pre-generated)
* **Pose estimation** – Any6D (FoundationPose-based) register + track
* **Rendering** – nvdiffrast mesh overlay and coordinate axes

Usage example::

    from aura.monitors.pose_tracking_monitor import PoseTrackingMonitor
    from aura.utils.config import PoseTrackingConfig

    config = PoseTrackingConfig(
        mesh_map={"bottle": "demo_data/bottle/bottle.glb",
                  "scale": "demo_data/bottle/scale.glb"},
        sam3_mask_dir="results/sam3_masks",
    )
    monitor = PoseTrackingMonitor(config)
    monitor.initialise()

    # Process first frame (registers poses)
    output = await monitor.update(frame=bgr_frame_0, frame_index=0)

    # Track subsequent frames
    output = await monitor.update(frame=bgr_frame_1, frame_index=1)
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any as TypingAny, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from aura.core import (
    BoundingBox,
    MonitorType,
    ObjectPose6D,
    PoseTrackingOutput,
)
from aura.monitors.base_monitor import BaseMonitor, MonitorConfig
from aura.utils.config import PoseTrackingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy dependencies
# ---------------------------------------------------------------------------
_any6d_available: Optional[bool] = None
_da3_available: Optional[bool] = None


def _ensure_any6d_on_path(any6d_root: str) -> None:
    """Add the Any6D third-party root to ``sys.path`` so that
    ``foundationpose`` and ``estimater`` can be imported.

    Also adds ``<any6d_root>/foundationpose`` so that the compiled
    ``mycpp`` C++ extension (``mycpp.build.mycpp``) is importable.
    """
    root = str(Path(any6d_root).resolve())
    fp_dir = str(Path(any6d_root, "foundationpose").resolve())
    # foundationpose/ goes after root so that root's estimater.py wins
    if fp_dir not in sys.path:
        sys.path.append(fp_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_import_any6d(any6d_root: str):
    """Lazily import Any6D + FoundationPose modules."""
    global _any6d_available
    if _any6d_available is not None:
        return _any6d_available

    _ensure_any6d_on_path(any6d_root)
    try:
        import nvdiffrast.torch as dr  # noqa: F401
        from estimater import Any6D  # noqa: F401
        from foundationpose.Utils import (  # noqa: F401
            draw_xyz_axis,
            make_mesh_tensors,
            nvdiffrast_render,
        )
        from foundationpose.learning.training.predict_pose_refine import (
            PoseRefinePredictor,  # noqa: F401
        )
        from foundationpose.learning.training.predict_score import (
            ScorePredictor,  # noqa: F401
        )

        _any6d_available = True
    except ImportError as exc:
        logger.warning("Any6D dependencies not available: %s", exc)
        _any6d_available = False
    return _any6d_available


def _try_import_da3():
    """Lazily check Depth-Anything-3 availability."""
    global _da3_available
    if _da3_available is not None:
        return _da3_available
    try:
        from depth_anything_3.api import DepthAnything3  # noqa: F401

        _da3_available = True
    except ImportError as exc:
        logger.warning("Depth Anything 3 not available: %s", exc)
        _da3_available = False
    return _da3_available


def generate_masks_sam3(
    rgb: np.ndarray,
    prompts: List[str],
    confidence: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Run SAM3 text-prompted segmentation on a single RGB frame.

    Runs in a **subprocess** to avoid CUDA context conflicts between
    SAM3/transformers and Open3D/nvdiffrast used later in the pipeline.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) RGB image.
    prompts : list[str]
        Text prompts for objects to find (e.g. ``["bottle", "scale"]``).
    confidence : float
        Minimum confidence for SAM3 detections.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from prompt name to boolean mask (H, W).
        Only prompts with at least one detection are included.
    """
    import json
    import tempfile
    import subprocess

    # Serialise inputs to a temp directory
    with tempfile.TemporaryDirectory(prefix="sam3_masks_") as tmpdir:
        input_path = os.path.join(tmpdir, "rgb.npy")
        np.save(input_path, rgb)

        # Build the worker script inline
        worker = os.path.join(tmpdir, "sam3_worker.py")
        with open(worker, "w") as f:
            f.write(_SAM3_WORKER_SCRIPT)

        cmd = [
            sys.executable, worker,
            "--input", input_path,
            "--output_dir", tmpdir,
            "--prompts", json.dumps(prompts),
            "--confidence", str(confidence),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("SAM3 subprocess failed:\n%s", result.stderr[-2000:])
            return {}

        # Load results
        masks_out: Dict[str, np.ndarray] = {}
        manifest_path = os.path.join(tmpdir, "manifest.json")
        if not os.path.exists(manifest_path):
            logger.error("SAM3 subprocess produced no manifest")
            return {}

        with open(manifest_path) as mf:
            manifest = json.load(mf)

        for entry in manifest:
            name = entry["name"]
            mask = np.load(os.path.join(tmpdir, entry["file"]))
            masks_out[name] = mask
            logger.info(
                "SAM3: '%s' → score=%.3f, pixels=%d",
                name, entry["score"], mask.sum(),
            )

    return masks_out


_SAM3_WORKER_SCRIPT = r'''#!/usr/bin/env python
"""SAM3 mask generation worker — runs in a separate process."""
import argparse, json, os, sys
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--confidence", type=float, default=0.3)
    args = p.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    rgb = np.load(args.input)
    prompts = json.loads(args.prompts)

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    pil_image = Image.fromarray(rgb)
    state = processor.set_image(pil_image)

    manifest = []
    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        scores = state["scores"]
        if len(scores) == 0:
            continue
        det_masks = state["masks"][:, 0].cpu().float().numpy()
        det_scores = scores.cpu().float().numpy()
        best = det_scores.argmax()
        mask = (det_masks[best] > 0.5).astype(bool)
        if mask.sum() < 100:
            continue
        fname = f"mask_{prompt}.npy"
        np.save(os.path.join(args.output_dir, fname), mask)
        manifest.append({"name": prompt, "file": fname, "score": float(det_scores[best])})

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════
# Pure helper functions (no heavy imports needed)
# ═══════════════════════════════════════════════════════════════════════════


def load_glb_mesh(glb_path: str):
    """Load a GLB mesh, converting PBRMaterial → SimpleMaterial."""
    import trimesh

    mesh = trimesh.load(glb_path, force="mesh")
    if hasattr(mesh.visual, "material"):
        mat = mesh.visual.material
        if type(mat).__name__ == "PBRMaterial":
            mesh.visual.material = mat.to_simple()
    return mesh


def load_sam3_mask(
    mask_dir: str, obj_name: str, frame_idx: int
) -> Optional[np.ndarray]:
    """Load a pre-generated SAM3 mask (H, W) bool or *None*."""
    mask_path = os.path.join(mask_dir, obj_name, f"mask_{frame_idx:05d}.png")
    if not os.path.exists(mask_path):
        return None
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    return mask_img > 127


def resize_depth_to_frame(
    depth: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Bilinear-resize a depth map to *(target_h, target_w)*."""
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def downscale_for_pose(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: Optional[np.ndarray],
    K: np.ndarray,
    max_side: int = 480,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """Downscale inputs for pose estimation."""
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb, depth, mask, K, 1.0
    scale = max_side / max(H, W)
    new_W, new_H = int(W * scale), int(H * scale)
    rgb_s = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_s = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    mask_s = (
        cv2.resize(
            mask.astype(np.uint8), (new_W, new_H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if mask is not None
        else None
    )
    K_s = K.copy()
    K_s[0, :] *= scale
    K_s[1, :] *= scale
    return rgb_s, depth_s, mask_s, K_s, scale


def downscale_rgb_depth(
    rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, max_side: int = 480
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downscale rgb + depth + K for tracking."""
    H, W = rgb.shape[:2]
    if max(H, W) <= max_side:
        return rgb, depth, K
    scale = max_side / max(H, W)
    new_W, new_H = int(W * scale), int(H * scale)
    rgb_s = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_s = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    K_s = K.copy()
    K_s[0, :] *= scale
    K_s[1, :] *= scale
    return rgb_s, depth_s, K_s


def compute_da3_depth(da3_model, frames_rgb: List[np.ndarray], intrinsic=None):
    """Run DA3 metric depth.  Returns ``(depths, prediction)``."""
    pil_images = [Image.fromarray(f) for f in frames_rgb]
    extrinsics = None
    intrinsics = None
    if intrinsic is not None:
        N = len(pil_images)
        intrinsics = np.tile(intrinsic[None], (N, 1, 1)).astype(np.float32)

    prediction = da3_model.inference(
        image=pil_images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=504,
        process_res_method="upper_bound_resize",
    )
    return prediction.depth, prediction


def render_mesh_overlay(
    rgb: np.ndarray,
    mesh,
    pose: np.ndarray,
    K: np.ndarray,
    glctx,
    alpha: float = 0.55,
    mesh_tensors=None,
) -> np.ndarray:
    """Render a textured mesh on top of *rgb* with transparency."""
    import torch

    from foundationpose.Utils import make_mesh_tensors as _make_mt
    from foundationpose.Utils import nvdiffrast_render

    H, W = rgb.shape[:2]
    if mesh_tensors is None:
        mesh_tensors = _make_mt(mesh)
    ob_in_cams = torch.tensor(pose[None], device="cuda", dtype=torch.float32)

    ren_img, ren_depth, _ = nvdiffrast_render(
        K=K,
        H=H,
        W=W,
        mesh=mesh,
        ob_in_cams=ob_in_cams,
        context="cuda",
        use_light=True,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        extra={},
    )
    ren_img = (ren_img[0] * 255.0).detach().cpu().numpy().astype(np.uint8)
    ren_depth = ren_depth[0].detach().cpu().numpy()
    ren_mask = ren_depth > 0

    out = rgb.copy()
    out[ren_mask] = (
        alpha * ren_img[ren_mask].astype(np.float32)
        + (1 - alpha) * rgb[ren_mask].astype(np.float32)
    ).astype(np.uint8)
    return out


def load_intrinsics_yaml(
    yaml_path: str, vid_w: int, vid_h: int
) -> Optional[np.ndarray]:
    """Load RealSense-style intrinsics YAML and scale to *vid_w × vid_h*."""
    import yaml as _yaml

    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r") as f:
        cam_data = _yaml.load(f, Loader=_yaml.FullLoader)
    K_ref = np.array(
        [
            [cam_data["color"]["fx"], 0.0, cam_data["color"]["ppx"]],
            [0.0, cam_data["color"]["fy"], cam_data["color"]["ppy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    ref_w, ref_h = 640.0, 480.0
    K = K_ref.copy()
    K[0, :] *= vid_w / ref_w
    K[1, :] *= vid_h / ref_h
    return K


def estimate_intrinsics(w: int, h: int, fov_deg: float = 60.0) -> np.ndarray:
    """Estimate a camera matrix from a horizontal FOV."""
    fx = w / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
    return np.array(
        [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]]
    )


# ═══════════════════════════════════════════════════════════════════════════
# PoseTrackingMonitor
# ═══════════════════════════════════════════════════════════════════════════


class PoseTrackingMonitor(BaseMonitor):
    """6DOF Pose Tracking Monitor using Any6D, DA3, and SAM3.

    Lifecycle
    ---------
    1. ``__init__(config)`` – store configuration.
    2. ``initialise()``     – load DA3 depth model, Any6D scorer/refiner,
       meshes, and rasteriser context.  Call once before ``update()``.
    3. ``update(frame=..., frame_index=..., masks=...)`` – process a single
       BGR frame.  On the first call (frame_index == 0 or first unseen
       object) it registers poses; subsequent calls track.
    4. ``shutdown()``       – release GPU resources.

    The monitor produces :class:`PoseTrackingOutput` containing per-object
    6DOF poses, the depth map, intrinsics, and an optional rendered overlay.
    """

    @property
    def monitor_type(self) -> MonitorType:
        return MonitorType.POSE_TRACKING

    def __init__(self, config: Optional[PoseTrackingConfig] = None):
        if config is None:
            config = PoseTrackingConfig()
        # Build a base MonitorConfig for the parent class
        base_cfg = MonitorConfig(
            enabled=config.enabled,
            update_rate_hz=config.update_rate_hz,
            timeout_sec=config.timeout_seconds,
        )
        super().__init__(base_cfg)
        self.config: PoseTrackingConfig = config

        # Populated by initialise()
        self._da3_model = None
        self._glctx = None
        self._scorer = None
        self._refiner = None
        self._estimators: Dict[str, TypingAny] = {}   # name → Any6D
        self._poses: Dict[str, np.ndarray] = {}     # name → 4×4
        self._meshes: Dict[str, TypingAny] = {}         # name → trimesh
        self._mesh_tensors: Dict[str, TypingAny] = {}
        self._K: Optional[np.ndarray] = None
        self._frame_hw: Optional[Tuple[int, int]] = None
        self._initialised = False
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(
        self,
        frame_hw: Optional[Tuple[int, int]] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> None:
        """Load models, meshes, and build rasteriser context.

        Parameters
        ----------
        frame_hw : tuple[int, int], optional
            (H, W) of input frames.  If not given, determined from the
            first frame passed to ``update()``.
        intrinsics : np.ndarray, optional
            3×3 camera matrix.  Overrides config intrinsic_file / fov.
        """
        import torch

        cfg = self.config

        # ── Any6D ──
        if not _try_import_any6d(cfg.any6d_root):
            raise RuntimeError(
                "Any6D dependencies missing (pytorch3d/foundationpose stack). "
                "Install full Any6D dependencies in the active environment."
            )
        import nvdiffrast.torch as dr
        from foundationpose.learning.training.predict_pose_refine import (
            PoseRefinePredictor,
        )
        from foundationpose.learning.training.predict_score import ScorePredictor

        self._glctx = dr.RasterizeCudaContext()
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        logger.info("Pose tracking backend: Any6D")

        # ── DA3 ──
        if not _try_import_da3():
            raise RuntimeError(
                "Depth Anything 3 not available.  Install depth-anything-3."
            )
        from depth_anything_3.api import DepthAnything3

        model_name = f"depth-anything/{cfg.da3_model.upper()}"
        logger.info("Loading DA3 model: %s …", model_name)
        self._da3_model = DepthAnything3.from_pretrained(model_name).to("cuda")
        logger.info("DA3 model loaded.")

        # ── Intrinsics ──
        if intrinsics is not None:
            self._K = intrinsics.copy()
        elif cfg.intrinsic_file:
            if frame_hw is not None:
                H, W = frame_hw
            else:
                H, W = 480, 640  # placeholder; recalculated on first frame
            self._K = load_intrinsics_yaml(cfg.intrinsic_file, W, H)

        # ── Meshes ──
        for prefix, glb_path in cfg.mesh_map.items():
            resolved = str(Path(glb_path).resolve()) if not os.path.isabs(glb_path) else glb_path
            logger.info("Loading mesh %s → %s", prefix, resolved)
            self._meshes[prefix] = load_glb_mesh(resolved)

        self._frame_hw = frame_hw
        self._initialised = True
        os.makedirs(cfg.save_dir, exist_ok=True)
        logger.info(
            "PoseTrackingMonitor initialised: %d mesh(es), save_dir=%s",
            len(self._meshes),
            cfg.save_dir,
        )

    def shutdown(self) -> None:
        """Release GPU resources."""
        import torch

        self._da3_model = None
        self._glctx = None
        self._scorer = None
        self._refiner = None
        self._estimators.clear()
        self._mesh_tensors.clear()
        torch.cuda.empty_cache()
        self._initialised = False
        logger.info("PoseTrackingMonitor shut down.")

    # ------------------------------------------------------------------
    # BaseMonitor plumbing
    # ------------------------------------------------------------------

    async def _process(self, **kwargs) -> Optional[PoseTrackingOutput]:
        """Core processing called by ``BaseMonitor.update()``.

        Expected keyword arguments
        --------------------------
        frame : np.ndarray
            BGR image (H, W, 3).
        frame_index : int
            Monotonic frame counter (0 = first frame → register).
        masks : dict[str, np.ndarray], optional
            Pre-computed boolean masks keyed by object name.
            If *None*, pre-generated SAM3 masks from ``config.sam3_mask_dir``
            are loaded.
        depth : np.ndarray, optional
            Pre-computed metric depth (H, W) in metres.  If *None*, DA3 is
            used.
        """
        frame: np.ndarray = kwargs["frame"]
        frame_index: int = kwargs.get("frame_index", self._frame_count)
        masks: Optional[Dict[str, np.ndarray]] = kwargs.get("masks")
        depth: Optional[np.ndarray] = kwargs.get("depth")

        if not self._initialised:
            self.initialise()

        return self._process_frame(frame, frame_index, masks=masks, depth=depth)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        bgr: np.ndarray,
        frame_index: int,
        *,
        masks: Optional[Dict[str, np.ndarray]] = None,
        depth: Optional[np.ndarray] = None,
    ) -> PoseTrackingOutput:
        """Run the full pose-tracking pipeline for one frame.

        Parameters
        ----------
        bgr : np.ndarray
            BGR image (H, W, 3) – OpenCV convention.
        frame_index : int
            Current frame index (0 triggers registration).
        masks : dict[str, np.ndarray] | None
            Object masks keyed by name.  Falls back to SAM3 mask dir.
        depth : np.ndarray | None
            Pre-computed metric depth.  Falls back to DA3.
        """
        from estimater import Any6D
        from foundationpose.Utils import draw_xyz_axis, make_mesh_tensors

        cfg = self.config
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # Lazily fix intrinsics if first frame
        if self._K is None:
            if cfg.intrinsic_file:
                self._K = load_intrinsics_yaml(cfg.intrinsic_file, W, H)
            if self._K is None:
                self._K = estimate_intrinsics(W, H, cfg.fov_deg)
                logger.info("Using estimated intrinsics (fov=%.1f°)", cfg.fov_deg)
        K = self._K

        # ── 1. Depth ────────────────────────────────────────────────────
        if depth is None:
            depths, pred = compute_da3_depth(self._da3_model, [rgb], intrinsic=K)
            depth_frame = resize_depth_to_frame(depths[0], H, W) * cfg.depth_scale

            # Optionally use DA3-predicted intrinsics
            if cfg.use_da3_intrinsics and pred.intrinsics is not None:
                proc_h, proc_w = pred.depth.shape[1], pred.depth.shape[2]
                K_da3 = pred.intrinsics[0].copy()
                K_da3[0, :] *= W / proc_w
                K_da3[1, :] *= H / proc_h
                self._K = K_da3
                K = K_da3
        else:
            depth_frame = depth

        # ── 2. Resolve masks ────────────────────────────────────────────
        if masks is None:
            masks = {}
            if cfg.sam3_mask_dir and os.path.isdir(cfg.sam3_mask_dir):
                for entry in sorted(os.listdir(cfg.sam3_mask_dir)):
                    entry_path = os.path.join(cfg.sam3_mask_dir, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    m = load_sam3_mask(cfg.sam3_mask_dir, entry, frame_index)
                    if m is not None and m.sum() > 100:
                        masks[entry] = m

        # ── 3. Match masks → meshes ─────────────────────────────────────
        obj_mask_mesh: Dict[str, Tuple[np.ndarray, object]] = {}
        for track_name, mask in masks.items():
            # Find matching mesh by prefix
            for prefix, mesh in self._meshes.items():
                if track_name.startswith(prefix):
                    obj_mask_mesh[track_name] = (mask, mesh)
                    break

        # ── 4. Register / Track ─────────────────────────────────────────
        is_first = frame_index == 0
        object_poses: List[ObjectPose6D] = []
        overlay = rgb.copy()

        for obj_name, (mask, mesh) in obj_mask_mesh.items():
            # Ensure estimator exists
            if obj_name not in self._estimators:
                # First time seeing this object → create estimator & register
                obj_save = os.path.join(cfg.save_dir, obj_name)
                os.makedirs(obj_save, exist_ok=True)

                est = Any6D(
                    symmetry_tfs=None,
                    mesh=copy.deepcopy(mesh),
                    scorer=self._scorer,
                    refiner=self._refiner,
                    glctx=self._glctx,
                    debug=cfg.debug_level,
                    debug_dir=obj_save,
                )
                rgb_s, depth_s, mask_s, K_s, _ = downscale_for_pose(
                    rgb, depth_frame, mask, K, cfg.max_pose_resolution
                )
                pose = est.register_any6d(
                    K=K_s,
                    rgb=rgb_s,
                    depth=depth_s,
                    ob_mask=mask_s,
                    iteration=cfg.est_refine_iter,
                    name=obj_name,
                )
                self._estimators[obj_name] = est
                self._poses[obj_name] = pose
                self._mesh_tensors[obj_name] = make_mesh_tensors(est.mesh)
                logger.info("[%s] Registered – pose:\n%s", obj_name, pose)
            elif is_first:
                # Re-register on frame 0
                est = self._estimators[obj_name]
                rgb_s, depth_s, mask_s, K_s, _ = downscale_for_pose(
                    rgb, depth_frame, mask, K, cfg.max_pose_resolution
                )
                pose = est.register_any6d(
                    K=K_s,
                    rgb=rgb_s,
                    depth=depth_s,
                    ob_mask=mask_s,
                    iteration=cfg.est_refine_iter,
                    name=obj_name,
                )
                self._poses[obj_name] = pose
            else:
                # Tracking with per-frame mask re-registration (if mask
                # available) or pure tracking
                est = self._estimators[obj_name]
                rgb_s, depth_s, K_s = downscale_rgb_depth(
                    rgb, depth_frame, K, cfg.max_pose_resolution
                )

                if mask is not None and mask.sum() > 100:
                    # Re-register with new mask (avoids drift)
                    _, _, mask_s, K_s_reg, _ = downscale_for_pose(
                        rgb, depth_frame, mask, K, cfg.max_pose_resolution
                    )
                    pose = est.register_any6d(
                        K=K_s_reg,
                        rgb=rgb_s,
                        depth=depth_s,
                        ob_mask=mask_s,
                        iteration=cfg.est_refine_iter,
                        name=f"{obj_name}_f{frame_index}",
                        coarse_est=False,
                        refinement=False,
                    )
                else:
                    pose = est.track_one_any6d(
                        rgb=rgb_s,
                        depth=depth_s,
                        K=K_s,
                        iteration=cfg.track_refine_iter,
                    )
                self._poses[obj_name] = pose

            pose = self._poses[obj_name]

            # Build output object
            ys, xs = np.where(mask) if mask is not None else (np.array([]), np.array([]))
            bbox = None
            if len(xs) > 0:  # type: ignore[arg-type]
                bbox = BoundingBox(
                    x_min=int(xs.min()),
                    y_min=int(ys.min()),
                    x_max=int(xs.max()),
                    y_max=int(ys.max()),
                )

            object_poses.append(
                ObjectPose6D(
                    object_name=obj_name,
                    pose_4x4=pose.copy(),
                    confidence=1.0,
                    mesh_path=cfg.mesh_map.get(
                        next(
                            (p for p in cfg.mesh_map if obj_name.startswith(p)),
                            "",
                        )
                    ),
                    bbox=bbox,
                    mask=mask,
                )
            )

            # ── Render overlay ──────────────────────────────────────────
            if cfg.render_overlay:
                overlay = render_mesh_overlay(
                    overlay,
                    est.mesh,
                    pose,
                    K,
                    self._glctx,
                    alpha=cfg.overlay_alpha,
                    mesh_tensors=self._mesh_tensors.get(obj_name),
                )
                if cfg.render_axes:
                    overlay = draw_xyz_axis(
                        overlay,
                        ob_in_cam=pose,
                        scale=0.05,
                        K=K,
                        thickness=2,
                        transparency=0,
                        is_input_rgb=True,
                    )

        self._frame_count = frame_index + 1

        return PoseTrackingOutput(
            timestamp=datetime.now(),
            object_poses=object_poses,
            depth_map=depth_frame,
            intrinsics=K.copy(),
            frame_index=frame_index,
            overlay_rgb=overlay if cfg.render_overlay else None,
        )

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def visualize(
        self, frame: np.ndarray, output: PoseTrackingOutput
    ) -> np.ndarray:
        """Draw pose tracking results on a BGR frame.

        If the output already contains a rendered overlay, it is
        converted to BGR and returned.  Otherwise bounding boxes and
        object labels are drawn.
        """
        if output.overlay_rgb is not None:
            vis = cv2.cvtColor(output.overlay_rgb, cv2.COLOR_RGB2BGR)
        else:
            vis = frame.copy()

        for op in output.object_poses:
            if op.bbox is not None:
                cv2.rectangle(
                    vis,
                    (op.bbox.x_min, op.bbox.y_min),
                    (op.bbox.x_max, op.bbox.y_max),
                    (0, 255, 0),
                    2,
                )
                t = op.pose_4x4[:3, 3]
                label = f"{op.object_name} [{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]"
                cv2.putText(
                    vis,
                    label,
                    (op.bbox.x_min, op.bbox.y_min - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

        info = f"Pose Tracking: {len(output.object_poses)} obj(s), frame {output.frame_index}"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    # ------------------------------------------------------------------
    # Convenience: process a whole video file
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_path: str = "results/pose_tracked.mp4",
        skip_frames: int = 1,
    ) -> str:
        """Run the full pipeline on a video file.

        1. Extract frames (honouring *skip_frames*).
        2. On the **first frame**, run SAM3 text-prompted segmentation to
           obtain object masks for initial registration.
        3. Initialise heavy models (DA3 + Any6D).
        4. Batch-compute DA3 metric depth for all frames.
        5. Register 6DOF poses on frame 0, then track on frames 1..N.
        6. Write an output video with 3D mesh overlays.

        Returns the path to the output video.
        """
        import imageio
        import torch
        from pytorch_lightning import seed_everything

        seed_everything(0)
        cfg = self.config
        # ── Video info ──────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read video: {video_path}")
        H, W = first.shape[:2]
        cap.release()

        logger.info(
            "Processing video: %s  %d×%d @ %.1ffps, %d frames (skip=%d)",
            video_path, W, H, fps, total, skip_frames,
        )

        # ── Extract frames ──────────────────────────────────────────────
        frames_bgr: List[np.ndarray] = []
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip_frames == 0:
                frames_bgr.append(frame)
            idx += 1
        cap.release()
        N = len(frames_bgr)
        logger.info("Extracted %d frames", N)
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        # ── SAM3 masks ────────────────────────────────────────────────
        # If pre-generated SAM3 masks exist on disk, use those (matches
        # the original run_bottle_video.py workflow).  Otherwise fall back
        # to live SAM3 generation on the first frame only.
        use_pregenerated_masks = (
            cfg.sam3_mask_dir
            and os.path.isdir(cfg.sam3_mask_dir)
        )
        first_masks: Dict[str, np.ndarray] = {}

        if use_pregenerated_masks:
            logger.info("Using pre-generated SAM3 masks from: %s", cfg.sam3_mask_dir)
            for entry in sorted(os.listdir(cfg.sam3_mask_dir)):
                entry_path = os.path.join(cfg.sam3_mask_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                m = load_sam3_mask(cfg.sam3_mask_dir, entry, 0)
                if m is not None and m.sum() > 100:
                    first_masks[entry] = m
                    logger.info("  %s: mask loaded (%d pixels)", entry, m.sum())
        else:
            # Run SAM3 *before* initialise() because the Any6D imports
            # pollute sys.path and shadow the ``sam3`` package.
            prompts = cfg.sam3_prompts or list(cfg.mesh_map.keys())
            logger.info("Generating SAM3 masks for first frame: prompts=%s", prompts)
            first_masks = generate_masks_sam3(
                frames_rgb[0], prompts, confidence=cfg.sam3_confidence,
            )

        if not first_masks:
            logger.warning(
                "SAM3 found no objects on the first frame – "
                "output video will have no overlays."
            )

        # ── Initialise heavy models (DA3 + Any6D) ──────────────────────
        if not self._initialised:
            self.initialise(frame_hw=(H, W))

        # ── Batch depth (DA3) ───────────────────────────────────────────
        all_depths: List[np.ndarray] = []
        bs = cfg.da3_batch_size
        K = self._K
        if K is None:
            if cfg.intrinsic_file:
                K = load_intrinsics_yaml(cfg.intrinsic_file, W, H)
            if K is None:
                K = estimate_intrinsics(W, H, cfg.fov_deg)
            self._K = K

        for start in range(0, N, bs):
            batch = frames_rgb[start: start + bs]
            depths, pred = compute_da3_depth(self._da3_model, batch)
            if (
                cfg.use_da3_intrinsics
                and start == 0
                and pred.intrinsics is not None
            ):
                proc_h, proc_w = pred.depth.shape[1], pred.depth.shape[2]
                K_da3 = pred.intrinsics[0].copy()
                K_da3[0, :] *= W / proc_w
                K_da3[1, :] *= H / proc_h
                self._K = K_da3
                K = K_da3
            for d in depths:
                all_depths.append(
                    resize_depth_to_frame(d, H, W) * cfg.depth_scale
                )
            logger.info(
                "  DA3 batch %d/%d done", start // bs + 1, (N + bs - 1) // bs
            )

        # Free DA3 memory before pose estimation
        del self._da3_model
        self._da3_model = None
        torch.cuda.empty_cache()

        # ── Process frames ──────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_fps = fps / skip_frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (W, H))

        for i in range(N):
            if i == 0:
                # First frame: register with SAM3 masks
                masks_i = first_masks
            else:
                # Subsequent frames: let _process_frame auto-load
                # per-frame SAM3 masks from sam3_mask_dir (if available)
                # or fall back to pure tracking.
                masks_i = None

            output = self._process_frame(
                frames_bgr[i], i, masks=masks_i, depth=all_depths[i],
            )
            vis = self.visualize(frames_bgr[i], output)
            writer.write(vis)

            if i % 10 == 0 or i == N - 1:
                logger.info("  Frame %d/%d", i + 1, N)
            if i < 5 or i % 50 == 0:
                debug_path = os.path.join(
                    cfg.save_dir, f"overlay_{i:05d}.png"
                )
                if output.overlay_rgb is not None:
                    imageio.imwrite(debug_path, output.overlay_rgb)

        writer.release()
        logger.info("Output video saved: %s", output_path)
        return output_path

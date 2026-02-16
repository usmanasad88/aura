"""Tests for PoseTrackingMonitor.

Tier 1 – Pure / CPU-only tests (always run):
  - Config construction
  - Monitor instantiation
  - Helper functions (intrinsics, downscale, mask loading)
  - Visualize with synthetic data
  - Async update rejects missing frame

Tier 2 – GPU integration tests (require CUDA + heavy models):
  - Full initialise → _process_frame on real video + GLB meshes
  - process_video (first N frames)
  - Marked with ``@pytest.mark.gpu`` — skipped when no CUDA.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pytest

from aura.core import (
    BoundingBox,
    MonitorType,
    ObjectPose6D,
    PoseTrackingOutput,
)
from aura.monitors.base_monitor import MonitorConfig
from aura.utils.config import PoseTrackingConfig
from aura.monitors.pose_tracking_monitor import (
    PoseTrackingMonitor,
    estimate_intrinsics,
    downscale_for_pose,
    downscale_rgb_depth,
    load_sam3_mask,
    resize_depth_to_frame,
    load_glb_mesh,
    generate_masks_sam3,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo_data" / "layup_demo"
VIDEO_PATH = DEMO_DIR / "layup_dummy_demo_crop_1080.mp4"
BOTTLE_GLB = DEMO_DIR / "mesh3d" / "bottle.glb"
SCALE_GLB = DEMO_DIR / "mesh3d" / "scale.glb"
ANY6D_ROOT = str(Path(__file__).resolve().parents[2] / "third_party" / "any6d")
# Use a lightweight DA3 model to stay within 16 GB VRAM
DA3_MODEL = "da3-small"

_has_cuda = False
try:
    import torch
    _has_cuda = torch.cuda.is_available()
except ImportError:
    pass

gpu = pytest.mark.skipif(not _has_cuda, reason="CUDA not available")


def _make_fake_output(n_objects: int = 2) -> PoseTrackingOutput:
    """Build a synthetic PoseTrackingOutput for visualise tests."""
    poses = []
    for i in range(n_objects):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.1 * i, 0.2, 0.5]
        poses.append(
            ObjectPose6D(
                object_name=f"obj_{i}",
                pose_4x4=pose,
                confidence=0.95,
                bbox=BoundingBox(
                    x_min=100 + i * 200,
                    y_min=100,
                    x_max=300 + i * 200,
                    y_max=400,
                ),
            )
        )
    return PoseTrackingOutput(
        object_poses=poses,
        depth_map=np.random.rand(480, 640).astype(np.float32),
        intrinsics=estimate_intrinsics(640, 480),
        frame_index=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 – Pure / CPU-only tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfig:
    """PoseTrackingConfig construction and defaults."""

    def test_default_config(self):
        cfg = PoseTrackingConfig()
        assert cfg.enabled is True
        assert cfg.da3_model == "da3nested-giant-large"
        assert cfg.timeout_seconds == 60.0
        assert cfg.mesh_map == {}

    def test_config_with_meshes(self):
        cfg = PoseTrackingConfig(
            mesh_map={"bottle": str(BOTTLE_GLB), "scale": str(SCALE_GLB)},
            sam3_mask_dir="/tmp/masks",
        )
        assert "bottle" in cfg.mesh_map
        assert cfg.sam3_mask_dir == "/tmp/masks"

    def test_config_extra_fields_allowed(self):
        """Pydantic model_config extra='allow'."""
        cfg = PoseTrackingConfig(custom_field="hello")
        assert cfg.custom_field == "hello"


class TestMonitorInstantiation:
    """PoseTrackingMonitor creation (no GPU)."""

    def test_create_default(self):
        m = PoseTrackingMonitor()
        assert m.monitor_type == MonitorType.POSE_TRACKING
        assert m._initialised is False

    def test_create_with_config(self):
        cfg = PoseTrackingConfig(
            mesh_map={"bottle": str(BOTTLE_GLB)},
            render_overlay=False,
        )
        m = PoseTrackingMonitor(cfg)
        assert m.config.render_overlay is False
        assert m.config.mesh_map["bottle"] == str(BOTTLE_GLB)

    def test_base_monitor_config_compat(self):
        """BaseMonitor.update() uses timeout from PoseTrackingConfig."""
        cfg = PoseTrackingConfig(timeout_seconds=42.0)
        m = PoseTrackingMonitor(cfg)
        # The monitor stores PoseTrackingConfig as self.config
        assert m.config.timeout_seconds == 42.0


class TestHelperFunctions:
    """Pure helper functions – no GPU."""

    def test_estimate_intrinsics_shape(self):
        K = estimate_intrinsics(1920, 1080, 60.0)
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_estimate_intrinsics_principal_point(self):
        K = estimate_intrinsics(640, 480)
        assert abs(K[0, 2] - 320.0) < 1e-5
        assert abs(K[1, 2] - 240.0) < 1e-5

    def test_estimate_intrinsics_square_pixels(self):
        K = estimate_intrinsics(800, 600, 70.0)
        assert abs(K[0, 0] - K[1, 1]) < 1e-5, "fx should equal fy"

    def test_downscale_for_pose_noop(self):
        """No downscale when image ≤ max_side."""
        rgb = np.zeros((480, 480, 3), dtype=np.uint8)
        depth = np.zeros((480, 480), dtype=np.float32)
        mask = np.ones((480, 480), dtype=bool)
        K = estimate_intrinsics(480, 480)
        rgb_s, depth_s, mask_s, K_s, scale = downscale_for_pose(
            rgb, depth, mask, K, max_side=480
        )
        assert scale == 1.0
        assert rgb_s.shape == rgb.shape

    def test_downscale_for_pose_reduces(self):
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((1080, 1920), dtype=np.float32)
        mask = np.ones((1080, 1920), dtype=bool)
        K = estimate_intrinsics(1920, 1080)
        rgb_s, depth_s, mask_s, K_s, scale = downscale_for_pose(
            rgb, depth, mask, K, max_side=480
        )
        assert max(rgb_s.shape[:2]) <= 480
        assert mask_s.dtype == bool
        assert 0 < scale < 1.0

    def test_downscale_for_pose_none_mask(self):
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((1080, 1920), dtype=np.float32)
        K = estimate_intrinsics(1920, 1080)
        _, _, mask_s, _, _ = downscale_for_pose(rgb, depth, None, K, 480)
        assert mask_s is None

    def test_downscale_rgb_depth(self):
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((1080, 1920), dtype=np.float32)
        K = estimate_intrinsics(1920, 1080)
        rgb_s, depth_s, K_s = downscale_rgb_depth(rgb, depth, K, 480)
        assert max(rgb_s.shape[:2]) <= 480
        assert depth_s.shape == rgb_s.shape[:2]

    def test_resize_depth_to_frame(self):
        d = np.random.rand(250, 300).astype(np.float32)
        d2 = resize_depth_to_frame(d, 480, 640)
        assert d2.shape == (480, 640)

    def test_load_sam3_mask_missing(self):
        assert load_sam3_mask("/nonexistent", "bottle", 0) is None

    def test_load_sam3_mask_roundtrip(self):
        """Write a mask, load it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_dir = os.path.join(tmpdir, "bottle")
            os.makedirs(obj_dir)
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[100:300, 200:400] = 255
            cv2.imwrite(os.path.join(obj_dir, "mask_00000.png"), mask)

            loaded = load_sam3_mask(tmpdir, "bottle", 0)
            assert loaded is not None
            assert loaded.dtype == bool
            assert loaded.sum() == 200 * 200


class TestLoadGlbMesh:
    """Mesh loading – requires trimesh but no GPU."""

    @pytest.mark.skipif(
        not BOTTLE_GLB.exists(), reason="Demo mesh not available"
    )
    def test_load_bottle(self):
        mesh = load_glb_mesh(str(BOTTLE_GLB))
        assert mesh.vertices.shape[0] > 0
        assert mesh.vertices.shape[1] == 3
        # PBR should have been converted to Simple
        assert type(mesh.visual.material).__name__ == "SimpleMaterial"

    @pytest.mark.skipif(
        not SCALE_GLB.exists(), reason="Demo mesh not available"
    )
    def test_load_scale(self):
        mesh = load_glb_mesh(str(SCALE_GLB))
        assert mesh.vertices.shape[0] > 0


class TestVisualize:
    """PoseTrackingMonitor.visualize – CPU only."""

    def test_visualize_with_overlay(self):
        m = PoseTrackingMonitor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = _make_fake_output(2)
        # Provide a pre-rendered overlay
        output.overlay_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
        vis = m.visualize(frame, output)
        assert vis.shape == (480, 640, 3)

    def test_visualize_without_overlay(self):
        m = PoseTrackingMonitor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = _make_fake_output(2)
        output.overlay_rgb = None
        vis = m.visualize(frame, output)
        assert vis.shape == (480, 640, 3)
        # Should have green rectangles drawn
        assert vis.sum() > 0

    def test_visualize_no_objects(self):
        m = PoseTrackingMonitor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = PoseTrackingOutput(
            object_poses=[],
            frame_index=0,
        )
        vis = m.visualize(frame, output)
        assert vis.shape == (480, 640, 3)


class TestAsyncUpdate:
    """Test the async _process path rejects bad inputs."""

    @pytest.mark.asyncio
    async def test_update_without_frame_raises(self):
        m = PoseTrackingMonitor()
        # _process expects 'frame' kwarg — BaseMonitor.update catches the error
        output = await m.update()  # no frame → error → None
        assert output is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        cfg = PoseTrackingConfig(enabled=False)
        m = PoseTrackingMonitor(cfg)
        output = await m.update(frame=np.zeros((480, 640, 3), dtype=np.uint8))
        assert output is None


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 – GPU integration tests
# ═══════════════════════════════════════════════════════════════════════════


@gpu
class TestGPUInitialise:
    """Test initialise() loads models on GPU."""

    @pytest.mark.skipif(
        not BOTTLE_GLB.exists(), reason="Demo meshes not available"
    )
    def test_initialise_and_shutdown(self):
        cfg = PoseTrackingConfig(
            mesh_map={
                "bottle": str(BOTTLE_GLB),
                "scale": str(SCALE_GLB),
            },
            any6d_root=ANY6D_ROOT,
            da3_model=DA3_MODEL,
        )
        m = PoseTrackingMonitor(cfg)
        m.initialise(frame_hw=(1080, 1920))
        assert m._initialised is True
        assert m._da3_model is not None
        assert m._scorer is not None
        assert m._refiner is not None
        assert m._glctx is not None
        assert len(m._meshes) == 2
        m.shutdown()
        assert m._initialised is False
        assert m._da3_model is None


@gpu
class TestGPUProcessFrame:
    """Integration: run _process_frame on the first frame of the demo video."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        if not VIDEO_PATH.exists() or not BOTTLE_GLB.exists():
            pytest.skip("Demo data not available")

        self.cfg = PoseTrackingConfig(
            mesh_map={
                "bottle": str(BOTTLE_GLB),
                "scale": str(SCALE_GLB),
            },
            any6d_root=ANY6D_ROOT,
            da3_model=DA3_MODEL,
            save_dir=str(tmp_path / "pose_out"),
            render_overlay=True,
            render_axes=True,
            debug_level=0,
        )
        self.monitor = PoseTrackingMonitor(self.cfg)

        # Read first frame
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        ret, self.frame0 = cap.read()
        cap.release()
        assert ret, "Failed to read first frame"

    def test_process_first_frame_with_synthetic_mask(self):
        """Register on frame 0 with a hand-crafted mask covering the bottle region."""
        self.monitor.initialise(frame_hw=self.frame0.shape[:2])
        H, W = self.frame0.shape[:2]

        # Create a coarse mask in the center of the frame
        mask = np.zeros((H, W), dtype=bool)
        mask[H // 4 : 3 * H // 4, W // 3 : 2 * W // 3] = True

        output = self.monitor._process_frame(
            self.frame0, frame_index=0, masks={"bottle": mask}
        )

        assert isinstance(output, PoseTrackingOutput)
        assert output.frame_index == 0
        assert output.depth_map is not None
        assert output.depth_map.shape == (H, W)
        assert output.intrinsics is not None
        assert output.intrinsics.shape == (3, 3)
        assert len(output.object_poses) >= 1

        pose_obj = output.object_poses[0]
        assert pose_obj.object_name == "bottle"
        assert pose_obj.pose_4x4.shape == (4, 4)
        assert pose_obj.confidence > 0

        # Overlay should be rendered
        assert output.overlay_rgb is not None
        assert output.overlay_rgb.shape == (H, W, 3)

        self.monitor.shutdown()

    def test_process_two_frames_tracks(self):
        """Register on frame 0, then track on frame 1."""
        self.monitor.initialise(frame_hw=self.frame0.shape[:2])
        H, W = self.frame0.shape[:2]

        mask = np.zeros((H, W), dtype=bool)
        mask[H // 4 : 3 * H // 4, W // 3 : 2 * W // 3] = True

        out0 = self.monitor._process_frame(
            self.frame0, frame_index=0, masks={"bottle": mask}
        )
        assert len(out0.object_poses) >= 1

        # Read second frame
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        cap.read()  # skip frame 0
        ret, frame1 = cap.read()
        cap.release()
        assert ret

        # Track (no mask → pure tracking)
        out1 = self.monitor._process_frame(
            frame1, frame_index=1, masks={"bottle": None}
        )
        assert isinstance(out1, PoseTrackingOutput)
        assert out1.frame_index == 1
        assert len(out1.object_poses) >= 1

        self.monitor.shutdown()


@gpu
class TestGPUProcessVideo:
    """Integration: process_video on first few frames."""

    @pytest.mark.skipif(
        not VIDEO_PATH.exists(), reason="Demo video not available"
    )
    def test_process_video_short(self, tmp_path):
        cfg = PoseTrackingConfig(
            mesh_map={
                "bottle": str(BOTTLE_GLB),
                "scale": str(SCALE_GLB),
            },
            any6d_root=ANY6D_ROOT,
            da3_model=DA3_MODEL,
            save_dir=str(tmp_path / "vid_out"),
            render_overlay=True,
            debug_level=0,
        )
        m = PoseTrackingMonitor(cfg)
        # process_video calls initialise() internally if needed

        # Use a high skip to only process ~3 frames
        total_frames = 3371
        skip = total_frames // 3  # ~3 frames
        out_path = str(tmp_path / "tracked.mp4")

        result = m.process_video(
            str(VIDEO_PATH),
            output_path=out_path,
            skip_frames=skip,
        )
        assert os.path.isfile(result)
        # Check output video is readable
        cap = cv2.VideoCapture(result)
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 2
        cap.release()

        m.shutdown()


@gpu
class TestGPUAsyncUpdate:
    """Test the full async update() → _process() path on GPU."""

    @pytest.mark.skipif(
        not VIDEO_PATH.exists(), reason="Demo data not available"
    )
    @pytest.mark.asyncio
    async def test_async_update_first_frame(self, tmp_path):
        cfg = PoseTrackingConfig(
            mesh_map={"bottle": str(BOTTLE_GLB)},
            any6d_root=ANY6D_ROOT,
            da3_model=DA3_MODEL,
            save_dir=str(tmp_path / "async_out"),
            render_overlay=False,
        )
        m = PoseTrackingMonitor(cfg)
        m.initialise(frame_hw=(1080, 1920))

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        ret, frame = cap.read()
        cap.release()

        H, W = frame.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        mask[H // 4 : 3 * H // 4, W // 3 : 2 * W // 3] = True

        output = await m.update(
            frame=frame,
            frame_index=0,
            masks={"bottle": mask},
        )
        assert output is not None
        assert isinstance(output, PoseTrackingOutput)
        assert len(output.object_poses) >= 1

        m.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 – End-to-end: SAM3 masks → DA3 depth → Any6D pose → overlay video
# ═══════════════════════════════════════════════════════════════════════════


@gpu
class TestSAM3MaskGeneration:
    """Test live SAM3 mask generation on the first video frame."""

    @pytest.mark.skipif(
        not VIDEO_PATH.exists(), reason="Demo video not available"
    )
    def test_generate_masks_bottle_scale(self):
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        ret, bgr = cap.read()
        cap.release()
        assert ret

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = generate_masks_sam3(rgb, ["bottle", "scale"], confidence=0.3)

        assert "bottle" in masks, f"SAM3 didn't find 'bottle', got: {list(masks.keys())}"
        assert "scale" in masks, f"SAM3 didn't find 'scale', got: {list(masks.keys())}"
        H, W = rgb.shape[:2]
        assert masks["bottle"].shape == (H, W)
        assert masks["scale"].shape == (H, W)
        assert masks["bottle"].sum() > 500, "Bottle mask too small"
        assert masks["scale"].sum() > 500, "Scale mask too small"


@gpu
class TestEndToEndOverlayVideo:
    """Full pipeline: process_video with SAM3 + DA3 + Any6D → overlay output."""

    @pytest.mark.skipif(
        not VIDEO_PATH.exists() or not BOTTLE_GLB.exists(),
        reason="Demo data not available",
    )
    def test_process_video_with_overlay(self, tmp_path):
        cfg = PoseTrackingConfig(
            mesh_map={
                "bottle": str(BOTTLE_GLB),
                "scale": str(SCALE_GLB),
            },
            any6d_root=ANY6D_ROOT,
            da3_model=DA3_MODEL,
            da3_batch_size=4,
            save_dir=str(tmp_path / "e2e_out"),
            render_overlay=True,
            render_axes=True,
            debug_level=0,
            sam3_confidence=0.3,
        )
        m = PoseTrackingMonitor(cfg)

        out_path = str(tmp_path / "overlay_output.mp4")

        # Process only ~10 frames via large skip
        total_frames = 3371
        skip = total_frames // 10

        result = m.process_video(
            str(VIDEO_PATH),
            output_path=out_path,
            skip_frames=skip,
        )

        # ── Verify output video ─────────────────────────────────────
        assert os.path.isfile(result), f"Output video not created: {result}"
        cap = cv2.VideoCapture(result)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 5, f"Expected ≥5 frames, got {frame_count}"

        # Read a frame and check it's a valid colour image (not blank)
        ret, frame = cap.read()
        cap.release()
        assert ret
        assert frame.shape[2] == 3
        assert frame.mean() > 5, "Output frame appears blank"

        # ── Verify debug overlays were saved ─────────────────────────
        overlay_files = list((tmp_path / "e2e_out").glob("overlay_*.png"))
        assert len(overlay_files) >= 1, "No debug overlays saved"

        m.shutdown()

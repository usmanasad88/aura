#!/usr/bin/env python3
"""Plot 6DOF pose tracking trajectories from pose monitor results.

Reads layup_pose_monitor_summary.json and produces:
  - 3D scatter plot of object positions over time
  - XYZ component plots over frame number
  - Tracking stability metrics

Usage:
    .venv/bin/python scripts/plot_pose_trajectories.py \
        --input results/layup_pose_monitor_summary.json \
        --output figures/generated/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_pose_data(path: str) -> Dict[str, Dict]:
    """Load and organize pose data by object name."""
    with open(path) as f:
        data = json.load(f)

    objects = {}
    for sample in data["samples"]:
        frame = sample["video_frame"]
        for obj in sample["objects"]:
            name = obj["name"]
            if name not in objects:
                objects[name] = {"frames": [], "x": [], "y": [], "z": []}
            xyz = obj["translation_xyz_m"]
            objects[name]["frames"].append(frame)
            objects[name]["x"].append(xyz[0])
            objects[name]["y"].append(xyz[1])
            objects[name]["z"].append(xyz[2])

    return objects


def compute_stability(objects: Dict) -> Dict:
    """Compute tracking stability metrics per object."""
    metrics = {}
    for name, data in objects.items():
        x, y, z = np.array(data["x"]), np.array(data["y"]), np.array(data["z"])
        # Position variance (lower = more stable when stationary)
        metrics[name] = {
            "n_frames": len(data["frames"]),
            "x_std_m": round(float(x.std()), 4),
            "y_std_m": round(float(y.std()), 4),
            "z_std_m": round(float(z.std()), 4),
            "total_displacement_m": round(float(np.sqrt(
                (x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2
            )), 4),
            "mean_position_m": {
                "x": round(float(x.mean()), 4),
                "y": round(float(y.mean()), 4),
                "z": round(float(z.mean()), 4),
            },
        }
    return metrics


def plot_3d_trajectories(objects: Dict, output_dir: Path):
    """3D scatter plot of object trajectories."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"bottle_0": "#e74c3c", "bottle_1": "#3498db", "scale": "#2ecc71"}

    for name, data in objects.items():
        color = colors.get(name, "#7f8c8d")
        frames = np.array(data["frames"])
        x, y, z = np.array(data["x"]), np.array(data["y"]), np.array(data["z"])

        # Color by frame number (time)
        scatter = ax.scatter(x, y, z, c=frames, cmap="viridis",
                            s=20, alpha=0.7, label=name)
        # Connect with line
        ax.plot(x, y, z, color=color, alpha=0.3, linewidth=1)
        # Mark start and end
        ax.scatter([x[0]], [y[0]], [z[0]], marker="^", s=60, c=color,
                  edgecolors="black", zorder=5)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="s", s=60, c=color,
                  edgecolors="black", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("6DOF Object Pose Tracking — 3D Trajectories")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_pose_3d.png")
    fig.savefig(output_dir / "fig_pose_3d.pdf")
    plt.close(fig)
    print("  Generated fig_pose_3d")


def plot_xyz_over_time(objects: Dict, output_dir: Path):
    """XYZ components over frame number."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    colors = {"bottle_0": "#e74c3c", "bottle_1": "#3498db", "scale": "#2ecc71"}
    components = [("x", "X Position (m)"), ("y", "Y Position (m)"),
                  ("z", "Z Position (m)")]

    for ax, (comp, ylabel) in zip(axes, components):
        for name, data in objects.items():
            color = colors.get(name, "#7f8c8d")
            frames = data["frames"]
            vals = data[comp]
            ax.plot(frames, vals, ".-", color=color, label=name,
                    markersize=3, linewidth=1)
            # Show std band
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ax.axhspan(mean_val - std_val, mean_val + std_val,
                       color=color, alpha=0.1)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Video Frame")
    axes[0].set_title("6DOF Pose Tracking — Position Components Over Time")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_pose_xyz.png")
    fig.savefig(output_dir / "fig_pose_xyz.pdf")
    plt.close(fig)
    print("  Generated fig_pose_xyz")


def main():
    parser = argparse.ArgumentParser(description="Plot pose tracking trajectories")
    parser.add_argument("--input", type=str,
                        default="results/layup_pose_monitor_summary.json")
    parser.add_argument("--output", type=str, default="figures/generated/")
    args = parser.parse_args()

    aura_root = Path(__file__).resolve().parent.parent
    input_path = str(aura_root / args.input)
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = load_pose_data(input_path)
    print(f"Loaded {sum(len(d['frames']) for d in objects.values())} pose samples "
          f"for {len(objects)} objects")

    # Stability metrics
    stability = compute_stability(objects)
    print("\nTracking stability:")
    for name, m in stability.items():
        print(f"  {name}: σ(x)={m['x_std_m']:.4f}m  σ(y)={m['y_std_m']:.4f}m  "
              f"σ(z)={m['z_std_m']:.4f}m  displacement={m['total_displacement_m']:.4f}m")

    # Save stability metrics
    metrics_path = output_dir / "pose_stability_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stability, f, indent=2)

    print("\nGenerating figures...")
    plot_3d_trajectories(objects, output_dir)
    plot_xyz_over_time(objects, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

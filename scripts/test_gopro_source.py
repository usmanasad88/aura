#!/usr/bin/env python3
"""Quick test for GoProSource — captures frames via native photo + download and saves to disk.

Each capture triggers the GoPro shutter, downloads the .36P file, and
extracts one fisheye lens.  Expect ~4-6 seconds per frame.

Usage:
    uv run python scripts/test_gopro_source.py
    uv run python scripts/test_gopro_source.py --output-dir /tmp/gopro_test
    uv run python scripts/test_gopro_source.py --fps 0.2 --count 5
    uv run python scripts/test_gopro_source.py --lens back
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture frames from GoProSource (native photo) and save to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gopro-ip", default="172.29.170.51", metavar="IP",
        help="GoPro camera IP (default: 172.29.170.51)",
    )
    parser.add_argument(
        "--fps", type=float, default=0.3, metavar="FPS",
        help="Target capture rate (default: 0.3 — one frame every ~3s)",
    )
    parser.add_argument(
        "--lens", choices=["front", "back"], default="front",
        help="Which fisheye lens to extract (default: front)",
    )
    parser.add_argument(
        "--output-dir", default="gopro_frames", metavar="DIR",
        help="Directory to save captured frames (default: ./gopro_frames)",
    )
    parser.add_argument(
        "--count", type=int, default=None, metavar="N",
        help="Stop after N frames (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--duration", type=float, default=None, metavar="SECS",
        help="Stop after this many seconds (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from aura.sources.gopro_source import GoProSource

    source = GoProSource(
        camera_ip=args.gopro_ip,
        fps=args.fps,
        lens=args.lens,
    )

    print(f"Opening GoPro source at {args.gopro_ip} (lens={args.lens}) ...")
    try:
        source.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Saving frames to: {output_dir.resolve()}")
    if args.count:
        print(f"Will capture {args.count} frame(s).")
    elif args.duration:
        print(f"Will run for {args.duration}s.")
    else:
        print("Capturing — press Ctrl+C to stop.")
    print()

    start_wall = time.monotonic()
    saved = 0

    try:
        while True:
            if args.duration is not None and (time.monotonic() - start_wall) >= args.duration:
                print(f"\nDuration limit ({args.duration}s) reached.")
                break

            frame = source.read()
            if frame is None:
                print("WARNING: read() returned None — retrying ...")
                continue

            filename = output_dir / f"frame_{frame.frame_number:05d}_{frame.timestamp:.1f}s.jpg"
            cv2.imwrite(str(filename), frame.image)
            saved += 1

            size_kb = filename.stat().st_size / 1024
            print(
                f"  [{saved}] frame={frame.frame_number}  "
                f"t={frame.timestamp:.1f}s  "
                f"{frame.width}x{frame.height}  "
                f"{size_kb:.0f} KB  -> {filename.name}"
            )

            if args.count is not None and saved >= args.count:
                print(f"\nFrame count limit ({args.count}) reached.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        source.close()

    elapsed = time.monotonic() - start_wall
    actual_fps = saved / elapsed if elapsed > 0 else 0.0
    print(f"\nDone. Saved {saved} frame(s) in {elapsed:.1f}s ({actual_fps:.3f} fps actual).")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

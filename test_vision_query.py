#!/usr/bin/env python3
"""
Quick vision query test against the SGLang server.

Usage:
  python test_vision_query.py                          # 480p (default)
  python test_vision_query.py --original               # full resolution
  python test_vision_query.py --resolution 720          # custom height
  python test_vision_query.py --prompt "Describe this"  # custom prompt
  python test_vision_query.py --image path/to/img.png   # custom image
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

DEFAULT_IMAGE = "demo_data/layup_demo/sample_3_layup_gesture_demo_stationary_with_overlay.png"
DEFAULT_PROMPT = "Give JSON structured Yes/No response. Is the person wearing gloves? Is the person holding a bottle?"
DEFAULT_PORT = 8100
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def main():
    parser = argparse.ArgumentParser(description="Test SGLang vision API")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--original", action="store_true", help="Use original resolution (no downsampling)")
    parser.add_argument("--resolution", type=int, default=480, help="Target height in pixels (default: 480)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="SGLang server port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"✗ Image not found: {img_path}")
        return

    img = Image.open(img_path)
    orig_size = f"{img.size[0]}x{img.size[1]}"
    file_size_mb = img_path.stat().st_size / 1024 / 1024

    if args.original:
        # Send at original resolution
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        label = f"original {orig_size}"
    else:
        # Downsample
        ratio = args.resolution / img.size[1]
        new_w, new_h = int(img.size[0] * ratio), args.resolution
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        label = f"{new_w}x{new_h} (from {orig_size})"

    img_b64 = base64.b64encode(buf.getvalue()).decode()
    send_kb = len(buf.getvalue()) / 1024

    print(f"Image : {img_path.name} — {label}")
    print(f"Size  : {file_size_mb:.1f} MB on disk → {send_kb:.0f} KB sent")
    print(f"Prompt: {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}")
    print(f"Server: http://localhost:{args.port}/v1")
    print()

    t0 = time.perf_counter()
    resp = requests.post(
        f"http://localhost:{args.port}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        timeout=60,
    )
    elapsed = time.perf_counter() - t0

    result = resp.json()
    answer = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    prompt_tok = usage.get("prompt_tokens", "?")
    comp_tok = usage.get("completion_tokens", "?")

    print(f"⏱  Response time : {elapsed:.2f}s")
    print(f"📊 Tokens        : {prompt_tok} prompt + {comp_tok} completion")
    print(f"📝 Response ({len(answer)} chars):")
    print(answer)


if __name__ == "__main__":
    main()

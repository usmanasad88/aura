#!/usr/bin/env python3
"""Analyze AURA pipeline latency from intent monitor logs.

Reads meta.json files from all sessions and produces:
  - Per-component latency statistics
  - LLM inference time distribution
  - Call frequency / sampling rate analysis

Usage:
    python analyze_latency.py \
        --logs-dir logs/intent_monitor/ \
        --output results/latency/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_all_meta(logs_dir: str) -> List[Dict]:
    """Load all meta.json files from all sessions."""
    logs = Path(logs_dir)
    all_meta = []
    for session_dir in sorted(logs.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name
        for call_dir in sorted(session_dir.glob("call_*")):
            meta_path = call_dir / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            meta["session"] = session_name
            meta["call_dir"] = str(call_dir)
            all_meta.append(meta)
    return all_meta


def compute_latency_stats(all_meta: List[Dict]) -> Dict:
    """Compute latency statistics from meta records."""
    gen_times = [m["generation_time_sec"] for m in all_meta
                 if "generation_time_sec" in m and m["generation_time_sec"] > 0]

    if not gen_times:
        return {"error": "No generation time data found"}

    arr = np.array(gen_times)

    # Per-session breakdown
    sessions = {}
    for m in all_meta:
        sess = m["session"]
        if sess not in sessions:
            sessions[sess] = []
        if "generation_time_sec" in m:
            sessions[sess].append(m["generation_time_sec"])

    per_session = {}
    for sess, times in sessions.items():
        if times:
            t = np.array(times)
            per_session[sess] = {
                "n_calls": len(t),
                "mean_sec": round(float(t.mean()), 3),
                "std_sec": round(float(t.std()), 3),
                "min_sec": round(float(t.min()), 3),
                "max_sec": round(float(t.max()), 3),
            }

    # Inter-call intervals (within each session)
    intervals = []
    for m_list in [sorted(
        [m for m in all_meta if m["session"] == sess],
        key=lambda x: x.get("timestamp_sec", 0)
    ) for sess in sessions]:
        for i in range(1, len(m_list)):
            dt = m_list[i].get("timestamp_sec", 0) - m_list[i-1].get("timestamp_sec", 0)
            if dt > 0:
                intervals.append(dt)

    interval_stats = {}
    if intervals:
        iarr = np.array(intervals)
        interval_stats = {
            "mean_sec": round(float(iarr.mean()), 2),
            "std_sec": round(float(iarr.std()), 2),
            "min_sec": round(float(iarr.min()), 2),
            "max_sec": round(float(iarr.max()), 2),
            "effective_hz": round(1.0 / float(iarr.mean()), 3) if iarr.mean() > 0 else 0,
        }

    # Model breakdown
    models = {}
    for m in all_meta:
        model = m.get("model", "unknown")
        if model not in models:
            models[model] = []
        if "generation_time_sec" in m:
            models[model].append(m["generation_time_sec"])

    per_model = {}
    for model, times in models.items():
        if times:
            t = np.array(times)
            per_model[model] = {
                "n_calls": len(t),
                "mean_sec": round(float(t.mean()), 3),
                "std_sec": round(float(t.std()), 3),
                "min_sec": round(float(t.min()), 3),
                "max_sec": round(float(t.max()), 3),
                "median_sec": round(float(np.median(t)), 3),
                "p95_sec": round(float(np.percentile(t, 95)), 3),
            }

    return {
        "overall": {
            "total_calls": len(gen_times),
            "total_sessions": len(sessions),
            "mean_sec": round(float(arr.mean()), 3),
            "std_sec": round(float(arr.std()), 3),
            "min_sec": round(float(arr.min()), 3),
            "max_sec": round(float(arr.max()), 3),
            "median_sec": round(float(np.median(arr)), 3),
            "p95_sec": round(float(np.percentile(arr, 95)), 3),
            "p99_sec": round(float(np.percentile(arr, 99)), 3),
        },
        "per_session": per_session,
        "per_model": per_model,
        "inter_call_interval": interval_stats,
        "raw_generation_times": [round(float(t), 3) for t in gen_times],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze AURA pipeline latency")
    parser.add_argument("--logs-dir", type=str, default="logs/intent_monitor/")
    parser.add_argument("--output", type=str, default="results/latency/")
    args = parser.parse_args()

    aura_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = str(aura_root / args.logs_dir)
    output_dir = Path(aura_root / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_meta = load_all_meta(logs_dir)
    print(f"Loaded {len(all_meta)} meta records from {logs_dir}")

    stats = compute_latency_stats(all_meta)
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    o = stats["overall"]
    print(f"\nOverall LLM Inference Latency ({o['total_calls']} calls across {o['total_sessions']} sessions):")
    print(f"  Mean:   {o['mean_sec']:.3f}s")
    print(f"  Std:    {o['std_sec']:.3f}s")
    print(f"  Median: {o['median_sec']:.3f}s")
    print(f"  Min:    {o['min_sec']:.3f}s")
    print(f"  Max:    {o['max_sec']:.3f}s")
    print(f"  P95:    {o['p95_sec']:.3f}s")

    if stats["inter_call_interval"]:
        ic = stats["inter_call_interval"]
        print(f"\nInter-call interval:")
        print(f"  Mean: {ic['mean_sec']:.2f}s  (effective rate: {ic['effective_hz']:.3f} Hz)")
        print(f"  Std:  {ic['std_sec']:.2f}s")

    print(f"\nPer-model breakdown:")
    for model, ms in stats["per_model"].items():
        print(f"  {model}: mean={ms['mean_sec']:.3f}s, median={ms['median_sec']:.3f}s, "
              f"p95={ms['p95_sec']:.3f}s ({ms['n_calls']} calls)")

    out_path = output_dir / "latency_analysis.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

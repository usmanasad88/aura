#!/usr/bin/env python3
"""Standalone dashboard server runner - avoids heavy imports."""

import os
import sys
import json
import logging
from pathlib import Path

# Flask imports only
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Configure paths
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for dashboard
dashboard_state = {
    "is_running": False,
    "current_frame": 0,
    "total_frames": 0,
    "current_timestamp": 0.0,
    "monitor_outputs": {
        "perception": {},
        "intent": {},
        "performance": {},
        "affordance": {},
    },
    "ssg": {
        "nodes": [],
        "edges": [],
        "task_state": {},
    },
    "decision": {
        "action": None,
        "confidence": 0.0,
        "reasoning": "",
    },
    "ground_truth": {
        "action": None,
        "program": None,
    },
    "history": [],
    "results": None,
}

# Create Flask app
app = Flask(__name__, 
            static_folder=str(SCRIPT_DIR / "static"),
            template_folder=str(SCRIPT_DIR / "templates"))
CORS(app)


@app.route("/")
def index():
    """Serve main dashboard page."""
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Get current dashboard state."""
    return jsonify({
        "is_running": dashboard_state["is_running"],
        "current_frame": dashboard_state["current_frame"],
        "total_frames": dashboard_state["total_frames"],
        "current_timestamp": dashboard_state["current_timestamp"],
        "results_loaded": dashboard_state["results"] is not None,
    })


@app.route("/api/load_results", methods=["POST"])
def load_results():
    """Load results from a JSON file."""
    data = request.json or {}
    results_path = data.get("path")
    
    if not results_path:
        return jsonify({"error": "No path provided"}), 400
    
    if not Path(results_path).exists():
        return jsonify({"error": f"Results file not found: {results_path}"}), 404
    
    try:
        with open(results_path) as f:
            results = json.load(f)
        
        dashboard_state["results"] = results
        
        # Update state from results
        if results.get("frame_analyses"):
            dashboard_state["history"] = results["frame_analyses"]
            dashboard_state["total_frames"] = len(results["frame_analyses"])
        
        return jsonify({
            "status": "loaded", 
            "frames": len(results.get("frame_analyses", [])),
            "completed_programs": results.get("completed_programs", []),
            "task_complete": results.get("task_complete", False),
            "decision_accuracy": results.get("decision_accuracy", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/frame/<int:frame_idx>")
def get_frame_state(frame_idx):
    """Get state at specific frame index."""
    if not dashboard_state["history"]:
        return jsonify({"error": "No data loaded"}), 404
    
    if frame_idx < 0 or frame_idx >= len(dashboard_state["history"]):
        return jsonify({"error": "Frame index out of range"}), 400
    
    frame_data = dashboard_state["history"][frame_idx]
    return jsonify(frame_data)


@app.route("/api/timeline")
def get_timeline():
    """Get timeline data for visualization."""
    if not dashboard_state["history"]:
        return jsonify({"events": [], "decisions": []})
    
    events = []
    decisions = []
    
    for i, frame in enumerate(dashboard_state["history"]):
        timestamp = frame.get("timestamp_sec", 0)
        
        # Extract ground truth events
        if frame.get("gt_action"):
            events.append({
                "index": i,
                "timestamp": timestamp,
                "type": "ground_truth",
                "action": frame["gt_action"],
                "program": frame.get("gt_program"),
            })
        
        # Extract decisions
        if frame.get("decision_action"):
            decisions.append({
                "index": i,
                "timestamp": timestamp,
                "action": frame["decision_action"],
                "confidence": frame.get("decision_confidence", 0),
                "reasoning": frame.get("decision_reasoning", ""),
            })
    
    return jsonify({"events": events, "decisions": decisions})


@app.route("/api/summary")
def get_summary():
    """Get overall summary of results."""
    if not dashboard_state["results"]:
        return jsonify({"error": "No results loaded"}), 404
    
    results = dashboard_state["results"]
    return jsonify({
        "video_path": results.get("video_path"),
        "total_frames": results.get("total_frames_processed", 0),
        "duration_sec": results.get("total_duration_sec", 0),
        "decision_accuracy": results.get("decision_accuracy", 0),
        "completed_programs": results.get("completed_programs", []),
        "task_complete": results.get("task_complete", False),
        "final_task_state": results.get("final_task_state", {}),
    })


@app.route("/api/history")
def get_history():
    """Get all frame analyses."""
    return jsonify(dashboard_state["history"])


def main():
    """Run the dashboard server."""
    print("="*60)
    print("AURA Weigh Bottles Dashboard")
    print("="*60)
    print("Open http://localhost:5000 in your browser")
    print("="*60)
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()

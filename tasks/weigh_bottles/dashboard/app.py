"""Web-based dashboard for AURA Weigh Bottles demo.

This dashboard provides real-time visualization of:
- Monitor outputs (perception, intent, performance, affordance)
- Semantic Scene Graph state
- Decision engine reasoning
- Ground truth comparison

Usage:
    # Start the dashboard server
    python -m tasks.weigh_bottles.dashboard.app
    
    # Then open http://localhost:5000 in your browser
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict

# Flask imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Add paths
SCRIPT_DIR = Path(__file__).parent
TASK_DIR = SCRIPT_DIR.parent
AURA_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(AURA_ROOT))
sys.path.insert(0, str(AURA_ROOT / "src"))

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

if FLASK_AVAILABLE:
    app = Flask(__name__, 
                static_folder=str(SCRIPT_DIR / "static"),
                template_folder=str(SCRIPT_DIR / "templates"))
    CORS(app)
else:
    app = None


def create_app():
    """Create Flask application."""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not installed. Install with: pip install flask flask-cors")
    
    return app


# =============================================================================
# API Routes
# =============================================================================

@app.route("/")
def index():
    """Serve main dashboard page."""
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Get current dashboard state."""
    return jsonify(dashboard_state)


@app.route("/api/process", methods=["POST"])
def process_video():
    """Start processing a video file."""
    data = request.json or {}
    video_path = data.get("video_path", str(AURA_ROOT / "demo_data" / "weigh_bottles" / "video.mp4"))
    frame_skip = data.get("frame_skip", 30)
    max_frames = data.get("max_frames", None)
    
    # Start processing in background
    asyncio.run(_process_video_async(video_path, frame_skip, max_frames))
    
    return jsonify({"status": "started", "video_path": video_path})


@app.route("/api/results")
def get_results():
    """Get processing results."""
    if dashboard_state["results"]:
        return jsonify(dashboard_state["results"])
    return jsonify({"error": "No results available"})


@app.route("/api/results/<path:filename>")
def get_results_file(filename):
    """Serve saved results file."""
    results_dir = TASK_DIR / "demo" / "dashboard_outputs"
    if (results_dir / filename).exists():
        return send_from_directory(str(results_dir), filename)
    return jsonify({"error": "File not found"}), 404


@app.route("/api/load_results", methods=["POST"])
def load_results():
    """Load results from a JSON file."""
    data = request.json or {}
    results_path = data.get("path")
    
    if not results_path or not Path(results_path).exists():
        return jsonify({"error": "Results file not found"}), 404
    
    with open(results_path) as f:
        results = json.load(f)
    
    dashboard_state["results"] = results
    
    # Update state from results
    if results.get("frame_analyses"):
        dashboard_state["history"] = results["frame_analyses"]
        dashboard_state["total_frames"] = len(results["frame_analyses"])
    
    return jsonify({"status": "loaded", "frames": len(results.get("frame_analyses", []))})


@app.route("/api/frame/<int:frame_idx>")
def get_frame_state(frame_idx):
    """Get state at specific frame index."""
    if not dashboard_state["history"]:
        return jsonify({"error": "No data loaded"}), 404
    
    if frame_idx < 0 or frame_idx >= len(dashboard_state["history"]):
        return jsonify({"error": "Frame index out of range"}), 400
    
    frame_data = dashboard_state["history"][frame_idx]
    return jsonify(frame_data)


@app.route("/api/ssg")
def get_ssg():
    """Get current Semantic Scene Graph."""
    return jsonify(dashboard_state["ssg"])


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
                "correct": frame.get("decision_action") == frame.get("gt_program"),
            })
    
    return jsonify({"events": events, "decisions": decisions})


# =============================================================================
# Processing Logic
# =============================================================================

async def _process_video_async(video_path: str, frame_skip: int, max_frames: Optional[int]):
    """Process video and update dashboard state."""
    from tasks.weigh_bottles.demo.run_integrated_demo import IntegratedDemoRunner
    
    dashboard_state["is_running"] = True
    
    try:
        runner = IntegratedDemoRunner(video_path=video_path)
        results = await runner.run_direct(
            frame_skip=frame_skip,
            max_frames=max_frames,
            headless=True,
        )
        
        # Update dashboard state
        dashboard_state["results"] = {
            "video_path": results.video_path,
            "total_frames_processed": results.total_frames_processed,
            "decision_accuracy": results.decision_accuracy,
            "completed_programs": results.completed_programs,
            "task_complete": results.task_complete,
            "final_task_state": results.final_task_state,
            "frame_analyses": [asdict(fa) for fa in results.frame_analyses],
        }
        
        dashboard_state["history"] = [asdict(fa) for fa in results.frame_analyses]
        dashboard_state["total_frames"] = len(results.frame_analyses)
        
        # Save results
        output_dir = TASK_DIR / "demo" / "dashboard_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"results_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(dashboard_state["results"], f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        dashboard_state["results"] = {"error": str(e)}
    
    finally:
        dashboard_state["is_running"] = False


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the dashboard server."""
    if not FLASK_AVAILABLE:
        print("Flask not installed. Install with: pip install flask flask-cors")
        sys.exit(1)
    
    # Create template and static directories
    templates_dir = SCRIPT_DIR / "templates"
    static_dir = SCRIPT_DIR / "static"
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    # Create index.html if not exists
    index_path = templates_dir / "index.html"
    if not index_path.exists():
        with open(index_path, 'w') as f:
            f.write(get_index_html())
    
    # Create dashboard.js if not exists
    js_path = static_dir / "dashboard.js"
    if not js_path.exists():
        with open(js_path, 'w') as f:
            f.write(get_dashboard_js())
    
    # Create styles.css if not exists
    css_path = static_dir / "styles.css"
    if not css_path.exists():
        with open(css_path, 'w') as f:
            f.write(get_styles_css())
    
    print("="*60)
    print("AURA Weigh Bottles Dashboard")
    print("="*60)
    print(f"Open http://localhost:5000 in your browser")
    print("="*60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)


def get_index_html():
    """Return the HTML template for the dashboard."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURA Weigh Bottles Dashboard</title>
    <link rel="stylesheet" href="/static/styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <h1>🤖 AURA Weigh Bottles Dashboard</h1>
        <div class="controls">
            <input type="text" id="resultsPath" placeholder="Results JSON path..." />
            <button onclick="loadResults()">Load Results</button>
            <button onclick="refreshState()">Refresh</button>
        </div>
    </header>
    
    <main>
        <div class="grid">
            <!-- Timeline -->
            <section class="timeline-section">
                <h2>📊 Timeline</h2>
                <div id="timeline">
                    <input type="range" id="frameSlider" min="0" max="100" value="0" />
                    <div id="frameInfo">Frame: 0 / 0 | Time: 0.0s</div>
                </div>
                <canvas id="timelineChart"></canvas>
            </section>
            
            <!-- Monitors Grid -->
            <section class="monitors">
                <div class="monitor-card">
                    <h3>👁️ Perception</h3>
                    <div id="perceptionOutput" class="output"></div>
                </div>
                <div class="monitor-card">
                    <h3>🎯 Intent</h3>
                    <div id="intentOutput" class="output"></div>
                </div>
                <div class="monitor-card">
                    <h3>⚡ Affordance</h3>
                    <div id="affordanceOutput" class="output"></div>
                </div>
                <div class="monitor-card">
                    <h3>📈 Performance</h3>
                    <div id="performanceOutput" class="output"></div>
                </div>
            </section>
            
            <!-- SSG Visualization -->
            <section class="ssg-section">
                <h2>🌐 Semantic Scene Graph</h2>
                <div id="ssgVisualization">
                    <div id="ssgNodes"></div>
                    <div id="ssgEdges"></div>
                </div>
                <div id="taskState" class="output"></div>
            </section>
            
            <!-- Decision Engine -->
            <section class="decision-section">
                <h2>🧠 Decision Engine</h2>
                <div class="decision-display">
                    <div class="decision-action">
                        <span class="label">Action:</span>
                        <span id="decisionAction" class="value">-</span>
                    </div>
                    <div class="decision-confidence">
                        <span class="label">Confidence:</span>
                        <span id="decisionConfidence" class="value">0%</span>
                    </div>
                    <div class="decision-reasoning">
                        <span class="label">Reasoning:</span>
                        <p id="decisionReasoning">-</p>
                    </div>
                </div>
            </section>
            
            <!-- Ground Truth Comparison -->
            <section class="gt-section">
                <h2>✅ Ground Truth</h2>
                <div class="gt-display">
                    <div class="gt-action">
                        <span class="label">Expected Action:</span>
                        <span id="gtAction" class="value">-</span>
                    </div>
                    <div class="gt-program">
                        <span class="label">Expected Program:</span>
                        <span id="gtProgram" class="value">-</span>
                    </div>
                    <div id="gtMatch" class="match-indicator">-</div>
                </div>
            </section>
            
            <!-- Results Summary -->
            <section class="results-section">
                <h2>📋 Results Summary</h2>
                <div id="resultsSummary" class="output"></div>
            </section>
        </div>
    </main>
    
    <script src="/static/dashboard.js"></script>
</body>
</html>'''


def get_dashboard_js():
    """Return the JavaScript for the dashboard."""
    return '''// Dashboard state
let currentFrameIndex = 0;
let totalFrames = 0;
let frameData = [];
let timelineChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initSlider();
    initTimelineChart();
    refreshState();
});

// Initialize frame slider
function initSlider() {
    const slider = document.getElementById('frameSlider');
    slider.addEventListener('input', (e) => {
        currentFrameIndex = parseInt(e.target.value);
        updateFrameDisplay();
    });
}

// Initialize timeline chart
function initTimelineChart() {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Decision Confidence',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false,
            }, {
                label: 'Intent Confidence',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                fill: false,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    currentFrameIndex = idx;
                    document.getElementById('frameSlider').value = idx;
                    updateFrameDisplay();
                }
            }
        }
    });
}

// Load results from file
async function loadResults() {
    const pathInput = document.getElementById('resultsPath');
    const path = pathInput.value.trim();
    
    if (!path) {
        alert('Please enter a results file path');
        return;
    }
    
    try {
        const response = await fetch('/api/load_results', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: path})
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        refreshState();
        
    } catch (err) {
        console.error('Load error:', err);
        alert('Failed to load results');
    }
}

// Refresh state from server
async function refreshState() {
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        
        frameData = state.history || [];
        totalFrames = frameData.length;
        
        // Update slider
        const slider = document.getElementById('frameSlider');
        slider.max = Math.max(0, totalFrames - 1);
        
        updateFrameDisplay();
        updateTimelineChart();
        updateResultsSummary(state.results);
        
    } catch (err) {
        console.error('Refresh error:', err);
    }
}

// Update display for current frame
function updateFrameDisplay() {
    if (frameData.length === 0) {
        document.getElementById('frameInfo').textContent = 'No data loaded';
        return;
    }
    
    const frame = frameData[currentFrameIndex] || {};
    const timestamp = frame.timestamp_sec || 0;
    
    // Frame info
    document.getElementById('frameInfo').textContent = 
        `Frame: ${frame.frame_num || 0} / ${totalFrames} | Time: ${timestamp.toFixed(1)}s`;
    
    // Perception
    document.getElementById('perceptionOutput').innerHTML = formatPerception(frame);
    
    // Intent
    document.getElementById('intentOutput').innerHTML = formatIntent(frame);
    
    // Affordance
    document.getElementById('affordanceOutput').innerHTML = formatAffordance(frame);
    
    // Performance
    document.getElementById('performanceOutput').innerHTML = formatPerformance(frame);
    
    // Decision
    document.getElementById('decisionAction').textContent = frame.decision_action || '-';
    document.getElementById('decisionConfidence').textContent = 
        ((frame.decision_confidence || 0) * 100).toFixed(0) + '%';
    document.getElementById('decisionReasoning').textContent = frame.decision_reasoning || '-';
    
    // Ground truth
    document.getElementById('gtAction').textContent = frame.gt_action || '-';
    document.getElementById('gtProgram').textContent = frame.gt_program || '-';
    
    // Match indicator
    const match = document.getElementById('gtMatch');
    if (frame.gt_program && frame.decision_action) {
        if (frame.decision_action === frame.gt_program) {
            match.textContent = '✅ Match';
            match.className = 'match-indicator match';
        } else {
            match.textContent = '❌ Mismatch';
            match.className = 'match-indicator mismatch';
        }
    } else {
        match.textContent = '-';
        match.className = 'match-indicator';
    }
    
    // SSG
    updateSSGDisplay(frame);
}

// Format perception output
function formatPerception(frame) {
    const objects = frame.detected_objects || [];
    if (objects.length === 0) {
        return '<em>No objects detected</em>';
    }
    return `
        <ul>
            ${objects.map(obj => `<li>${obj}</li>`).join('')}
        </ul>
        <div class="stat">Count: ${frame.object_count || 0}</div>
    `;
}

// Format intent output
function formatIntent(frame) {
    return `
        <div class="intent-item">
            <strong>Current:</strong> ${frame.current_action || 'unknown'}
            <span class="confidence">(${((frame.current_action_confidence || 0) * 100).toFixed(0)}%)</span>
        </div>
        <div class="intent-item">
            <strong>Predicted:</strong> ${frame.predicted_next_action || 'unknown'}
            <span class="confidence">(${((frame.predicted_next_confidence || 0) * 100).toFixed(0)}%)</span>
        </div>
        ${frame.intent_reasoning ? `<div class="reasoning">${frame.intent_reasoning}</div>` : ''}
    `;
}

// Format affordance output
function formatAffordance(frame) {
    const available = frame.available_programs || [];
    const completed = frame.completed_programs || [];
    
    return `
        <div class="affordance-section">
            <strong>Available:</strong>
            <ul>
                ${available.length ? available.map(p => `<li class="available">${p}</li>`).join('') : '<li><em>None</em></li>'}
            </ul>
        </div>
        <div class="affordance-section">
            <strong>Completed:</strong>
            <ul>
                ${completed.length ? completed.map(p => `<li class="completed">${p}</li>`).join('') : '<li><em>None</em></li>'}
            </ul>
        </div>
    `;
}

// Format performance output
function formatPerformance(frame) {
    const status = frame.performance_status || 'OK';
    const statusClass = status === 'OK' ? 'ok' : (status === 'WARNING' ? 'warning' : 'error');
    
    return `
        <div class="performance-status ${statusClass}">${status}</div>
        <div class="failure-type">Failure: ${frame.failure_type || 'NONE'}</div>
        ${frame.performance_reasoning ? `<div class="reasoning">${frame.performance_reasoning}</div>` : ''}
    `;
}

// Update SSG visualization
function updateSSGDisplay(frame) {
    const nodesDiv = document.getElementById('ssgNodes');
    const taskStateDiv = document.getElementById('taskState');
    
    // Simple text representation for now
    nodesDiv.innerHTML = `
        <div class="ssg-stats">
            Nodes: ${frame.ssg_node_count || 0} | Edges: ${frame.ssg_edge_count || 0}
        </div>
    `;
    
    // Task state from frame
    const taskState = {
        current_action: frame.current_action,
        detected_objects: frame.detected_objects,
        completed_programs: frame.completed_programs,
    };
    
    taskStateDiv.innerHTML = `<pre>${JSON.stringify(taskState, null, 2)}</pre>`;
}

// Update timeline chart
function updateTimelineChart() {
    if (!timelineChart || frameData.length === 0) return;
    
    const labels = frameData.map((f, i) => f.timestamp_sec?.toFixed(1) || i);
    const decisionData = frameData.map(f => f.decision_confidence || 0);
    const intentData = frameData.map(f => f.current_action_confidence || 0);
    
    timelineChart.data.labels = labels;
    timelineChart.data.datasets[0].data = decisionData;
    timelineChart.data.datasets[1].data = intentData;
    timelineChart.update();
}

// Update results summary
function updateResultsSummary(results) {
    const summaryDiv = document.getElementById('resultsSummary');
    
    if (!results) {
        summaryDiv.innerHTML = '<em>No results available</em>';
        return;
    }
    
    summaryDiv.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <span class="label">Frames Processed:</span>
                <span class="value">${results.total_frames_processed || 0}</span>
            </div>
            <div class="summary-item">
                <span class="label">Decision Accuracy:</span>
                <span class="value">${((results.decision_accuracy || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="summary-item">
                <span class="label">Task Complete:</span>
                <span class="value">${results.task_complete ? '✅ Yes' : '❌ No'}</span>
            </div>
            <div class="summary-item">
                <span class="label">Completed Programs:</span>
                <span class="value">${(results.completed_programs || []).join(', ') || 'None'}</span>
            </div>
        </div>
        ${results.final_task_state ? `
            <h4>Final Task State:</h4>
            <pre>${JSON.stringify(results.final_task_state, null, 2)}</pre>
        ` : ''}
    `;
}
'''


def get_styles_css():
    """Return the CSS styles for the dashboard."""
    return '''/* AURA Dashboard Styles */

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
}

header {
    background: #16213e;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #0f3460;
}

header h1 {
    font-size: 1.5rem;
    color: #00d9ff;
}

.controls {
    display: flex;
    gap: 0.5rem;
}

.controls input {
    padding: 0.5rem;
    border: 1px solid #0f3460;
    border-radius: 4px;
    background: #1a1a2e;
    color: #eee;
    width: 300px;
}

.controls button {
    padding: 0.5rem 1rem;
    background: #0f3460;
    color: #00d9ff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
}

.controls button:hover {
    background: #1a4b8f;
}

main {
    padding: 1rem;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

section {
    background: #16213e;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #0f3460;
}

section h2 {
    color: #00d9ff;
    margin-bottom: 1rem;
    font-size: 1.1rem;
    border-bottom: 1px solid #0f3460;
    padding-bottom: 0.5rem;
}

section h3 {
    color: #4cc9f0;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

/* Timeline */
.timeline-section {
    grid-column: 1 / -1;
}

#timeline {
    margin-bottom: 1rem;
}

#frameSlider {
    width: 100%;
    margin-bottom: 0.5rem;
}

#frameInfo {
    text-align: center;
    color: #888;
}

#timelineChart {
    max-height: 200px;
}

/* Monitors Grid */
.monitors {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
}

.monitor-card {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #0f3460;
}

.output {
    font-size: 0.85rem;
    max-height: 200px;
    overflow-y: auto;
}

.output ul {
    list-style: none;
    padding-left: 0.5rem;
}

.output li {
    padding: 0.2rem 0;
    border-bottom: 1px dotted #0f3460;
}

.output pre {
    background: #1a1a2e;
    padding: 0.5rem;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.8rem;
}

/* Intent styling */
.intent-item {
    margin-bottom: 0.5rem;
}

.confidence {
    color: #4cc9f0;
    font-size: 0.8rem;
}

.reasoning {
    font-style: italic;
    color: #888;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

/* Affordance styling */
.affordance-section {
    margin-bottom: 0.5rem;
}

.available {
    color: #4ade80;
}

.completed {
    color: #888;
    text-decoration: line-through;
}

/* Performance styling */
.performance-status {
    font-weight: bold;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
}

.performance-status.ok {
    background: #166534;
    color: #4ade80;
}

.performance-status.warning {
    background: #854d0e;
    color: #fbbf24;
}

.performance-status.error {
    background: #991b1b;
    color: #f87171;
}

/* Decision section */
.decision-display, .gt-display {
    display: grid;
    gap: 0.5rem;
}

.decision-action, .decision-confidence, .gt-action, .gt-program {
    display: flex;
    gap: 0.5rem;
}

.label {
    color: #888;
}

.value {
    color: #4cc9f0;
    font-weight: bold;
}

/* Match indicator */
.match-indicator {
    font-weight: bold;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    text-align: center;
}

.match-indicator.match {
    background: #166534;
    color: #4ade80;
}

.match-indicator.mismatch {
    background: #991b1b;
    color: #f87171;
}

/* Results summary */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
}

.summary-item {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem;
    background: #1a1a2e;
    border-radius: 4px;
}

/* SSG */
.ssg-stats {
    color: #888;
    margin-bottom: 0.5rem;
}

/* Responsive */
@media (max-width: 1200px) {
    .monitors {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 768px) {
    .grid {
        grid-template-columns: 1fr;
    }
    
    .monitors {
        grid-template-columns: 1fr;
    }
    
    header {
        flex-direction: column;
        gap: 1rem;
    }
    
    .controls {
        flex-wrap: wrap;
    }
    
    .controls input {
        width: 100%;
    }
}
'''


if __name__ == "__main__":
    main()

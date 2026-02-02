// Dashboard state
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

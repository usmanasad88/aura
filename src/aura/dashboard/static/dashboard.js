/**
 * AURA Dashboard — Real-time SSE client.
 *
 * Connects to /api/events (Server-Sent Events) and updates
 * all UI panels as node_update events arrive from the LangGraph
 * workflow running in the backend.
 */

// ── State ───────────────────────────────────────────────────────

let state = {};
let lastUpdateTime = 0;
let frameCount = 0;
let fpsTimer = 0;
let fps = 0;

// ── SSE Connection ──────────────────────────────────────────────

function connect() {
    const evtSource = new EventSource("/api/events");

    evtSource.addEventListener("init", (e) => {
        state = JSON.parse(e.data);
        updateAll(state);
        setConnectionStatus(true);
    });

    evtSource.addEventListener("node_update", (e) => {
        const payload = JSON.parse(e.data);
        state = payload.state;
        lastUpdateTime = payload.time;
        updateAll(state);
        trackFps();
    });

    evtSource.onerror = () => {
        setConnectionStatus(false);
        // Auto-reconnect after 2s (EventSource does this anyway)
    };

    evtSource.onopen = () => {
        setConnectionStatus(true);
    };
}

function trackFps() {
    frameCount++;
    const now = performance.now();
    if (now - fpsTimer > 1000) {
        fps = frameCount;
        frameCount = 0;
        fpsTimer = now;
        document.getElementById("fps-counter").textContent = `${fps} evt/s`;
    }
}

// ── Master Update ───────────────────────────────────────────────

function updateAll(s) {
    updateStatusBadge(s);
    updateMonitorsStrip(s);
    updateFrameInfo(s);
    updatePerception(s);
    updateGesture(s);
    updateIntent(s);
    updateTaskState(s);
    updateDecision(s);
    updateActionLog(s);
    updateFooter();
}

// ── Monitors Strip ──────────────────────────────────────────────

function updateMonitorsStrip(s) {
    const el = document.getElementById("monitors-strip");
    if (!el) return;
    const active = s.active_monitors || (s.config && s.config.active_monitors) || [];
    const enableAudio = s.config && s.config.enable_audio;
    const items = ["intent", "gesture", "perception"].map(m => {
        const on = active.includes(m);
        return `<span class="mon-chip ${on ? "mon-on" : "mon-off"}">${m}</span>`;
    });
    items.push(`<span class="mon-chip ${enableAudio ? "mon-on" : "mon-off"}">audio</span>`);
    el.innerHTML = items.join("");
}

// ── Status Badge ────────────────────────────────────────────────

function updateStatusBadge(s) {
    const badge = document.getElementById("status-badge");
    const cycleEl = document.getElementById("cycle-counter");
    const launcherLink = document.getElementById("launcher-link");

    cycleEl.textContent = `Cycle: ${s.cycle_count || 0}`;

    if (s.error) {
        badge.className = "badge badge-error";
        badge.textContent = "ERROR";
    } else if (s.is_complete) {
        badge.className = "badge badge-complete";
        badge.textContent = "COMPLETE";
    } else if (s.human_requesting_help) {
        badge.className = "badge badge-help";
        badge.textContent = "HELP REQUESTED";
    } else if (s.cycle_count > 0) {
        badge.className = "badge badge-running";
        badge.textContent = "RUNNING";
    } else {
        badge.className = "badge badge-idle";
        badge.textContent = "IDLE";
    }

    // Show launcher link when workflow is done or errored
    if (launcherLink) {
        if (s.is_complete || s.error) {
            launcherLink.classList.remove("hidden");
        } else {
            launcherLink.classList.add("hidden");
        }
    }

    const stopBtn = document.getElementById("stop-workflow-btn");
    if (stopBtn) {
        if (!s.is_complete && !s.error) {
            stopBtn.classList.remove("hidden");
        } else {
            stopBtn.classList.add("hidden");
        }
    }
}

// ── Frame Info ──────────────────────────────────────────────────

function updateFrameInfo(s) {
    const el = document.getElementById("frame-info");
    const ts = (s.current_timestamp_sec || 0).toFixed(1);
    el.textContent = `Frame: ${s.current_frame_num || 0} | ${ts}s`;
}

// ── Perception Monitor ──────────────────────────────────────────

function updatePerception(s) {
    const p = s.perception || {};
    const dot = document.getElementById("perception-dot");

    // Consolidate all *_locations dicts into one object → region map.
    const locs = {};
    for (const [k, v] of Object.entries(p)) {
        if (k.endsWith("_locations") && v && typeof v === "object") {
            for (const [obj, region] of Object.entries(v)) {
                locs[obj] = region;
            }
        }
    }
    const objCount = Object.keys(locs).length;
    setText("perception-count", String(objCount));

    if (dot) dot.className = objCount > 0 ? "dot active" : "dot";

    const locEl = document.getElementById("perception-locations");
    if (locEl) {
        if (objCount > 0) {
            locEl.innerHTML = Object.keys(locs).map(obj => {
                const place = locs[obj];
                const cls = place === "unknown" ? "loc-storage" : "loc-workplace";
                return `<div class="loc-item">
                    <span class="loc-name">${escHtml(obj)}</span>
                    <span class="loc-place ${cls}">${escHtml(String(place))}</span>
                </div>`;
            }).join("");
        } else {
            locEl.innerHTML = '<span class="chip chip-pending">no detections</span>';
        }
    }

    const tsEl = document.getElementById("perception-task-state");
    if (tsEl) {
        const ts = p.task_state || {};
        const keys = Object.keys(ts).filter(k => !k.startsWith("_"));
        if (keys.length > 0) {
            tsEl.innerHTML = '<div class="state-var-grid">' +
                keys.map(k => {
                    const v = ts[k];
                    const display = typeof v === "object" ? JSON.stringify(v) : String(v);
                    return `<div class="state-var">
                        <span class="state-var-key">${escHtml(k)}</span>
                        <span class="state-var-val">${escHtml(display)}</span>
                    </div>`;
                }).join("") + '</div>';
        } else {
            tsEl.innerHTML = '<span class="chip chip-pending">--</span>';
        }
    }
}

// ── Gesture Monitor ─────────────────────────────────────────────

function updateGesture(s) {
    const g = s.gesture || {};
    const dot = document.getElementById("gesture-dot");
    const nameEl = document.getElementById("gesture-name");
    const helpEl = document.getElementById("help-requested");
    const safeEl = document.getElementById("safety-stop");
    const badgeEl = document.getElementById("gesture-badge");

    const gesture = g.dominant_gesture || "--";
    nameEl.textContent = gesture;

    // Dot color
    if (s.human_requesting_help) {
        dot.className = "dot warning";
    } else if (gesture !== "--") {
        dot.className = "dot active";
    } else {
        dot.className = "dot";
    }

    // Help badge
    helpEl.textContent = s.human_requesting_help ? "YES" : "No";
    helpEl.style.color = s.human_requesting_help ? "#ffc107" : "";

    // Safety
    const safety = g.safety_triggered || false;
    safeEl.textContent = safety ? "ACTIVE" : "No";
    safeEl.style.color = safety ? "#ff4444" : "";

    // Video overlay badge
    if (s.human_requesting_help) {
        badgeEl.className = "gesture-indicator help";
        badgeEl.textContent = `🤙 ${gesture}`;
    } else if (gesture && gesture !== "--") {
        badgeEl.className = "gesture-indicator safe";
        badgeEl.textContent = gesture;
    } else {
        badgeEl.className = "gesture-indicator hidden";
    }
}

// ── Intent Monitor ──────────────────────────────────────────────

function updateIntent(s) {
    const intent = s.intent || {};
    const dot = document.getElementById("intent-dot");

    if (intent.current_phase) {
        dot.className = "dot active";
    }

    setText("intent-phase", intent.current_phase || "--");
    setText("intent-action", intent.current_action || "--");
    setText("intent-human-state", intent.human_state || "--");

    const predicted = intent.predicted_next_action || "--";
    setText("intent-predicted", predicted);

    const conf = (intent.prediction_confidence || 0) * 100;
    document.getElementById("confidence-bar").style.width = `${conf}%`;
    setText("confidence-value", `${conf.toFixed(0)}%`);

    const genTime = intent.generation_time_sec;
    setText("intent-gen-time", genTime != null ? `${genTime.toFixed(1)}s` : "--");
    setText("intent-reasoning", intent.reasoning || "--");
}

// ── Task State ──────────────────────────────────────────────────

function updateTaskState(s) {
    // State variables
    const varsEl = document.getElementById("task-state-vars");
    const taskState = s.task_state || {};
    const keys = Object.keys(taskState).filter(k => !k.startsWith("_"));

    if (keys.length > 0) {
        varsEl.innerHTML = '<div class="state-var-grid">' +
            keys.map(k => {
                const v = taskState[k];
                const display = typeof v === "object" ? JSON.stringify(v) : String(v);
                return `<div class="state-var">
                    <span class="state-var-key">${k}</span>
                    <span class="state-var-val">${escHtml(display)}</span>
                </div>`;
            }).join("") + '</div>';
    }

    // Completed steps
    const stepsEl = document.getElementById("completed-steps");
    const completed = s.completed_steps || [];
    if (completed.length > 0) {
        stepsEl.innerHTML = completed.map(
            step => `<span class="chip chip-done">${escHtml(step)}</span>`
        ).join("");
    } else {
        stepsEl.innerHTML = '<span class="chip chip-pending">none yet</span>';
    }

    // Object locations
    const locEl = document.getElementById("object-locations");
    const locs = s.object_locations || {};
    const locKeys = Object.keys(locs);
    if (locKeys.length > 0) {
        locEl.innerHTML = locKeys.map(obj => {
            const place = locs[obj];
            const cls = place === "workplace" ? "loc-workplace" : "loc-storage";
            return `<div class="loc-item">
                <span class="loc-name">${escHtml(obj)}</span>
                <span class="loc-place ${cls}">${escHtml(place)}</span>
            </div>`;
        }).join("");
    }
}

// ── Decision Engine ─────────────────────────────────────────────

function updateDecision(s) {
    const dec = s.decision || {};
    const dot = document.getElementById("decision-dot");
    const actions = s.actions || [];

    setText("decision-mode", dec.decision_mode || "--");
    setText("pending-count", String(actions.length));
    setText("decision-reasoning", dec.reasoning || "--");

    // BT trail / branch
    const branch = dec.bt_branch || "--";
    setText("bt-branch", branch);
    const llmFlag = document.getElementById("bt-llm-flag");
    if (llmFlag) {
        if (dec.bt_llm_invoked) {
            llmFlag.classList.remove("hidden");
        } else {
            llmFlag.classList.add("hidden");
        }
    }
    const trailEl = document.getElementById("bt-trail");
    if (trailEl) {
        const trail = (dec.bt_trail || "").trim();
        if (trail && trail !== "no_branch") {
            const steps = trail.split("|").map(s => s.trim()).filter(Boolean);
            trailEl.innerHTML = steps.map(step => {
                const kind = step.split(":", 1)[0];
                return `<span class="bt-step bt-${escHtml(kind)}">${escHtml(step)}</span>`;
            }).join('<span class="bt-arrow">→</span>');
        } else {
            trailEl.innerHTML = '<span class="bt-empty">no_branch</span>';
        }
    }

    if (actions.length > 0) {
        dot.className = "dot active";
    } else {
        dot.className = "dot";
    }

    // Pending action cards
    const cardsEl = document.getElementById("pending-actions-list");
    if (actions.length > 0) {
        cardsEl.innerHTML = actions.map(a => `
            <div class="action-card">
                <span class="action-type">${escHtml(a.action_type || "?")}</span>
                <span class="action-obj">${escHtml(a.object_name || "")}</span>
                <span class="action-reason">${escHtml(a.reason || "")}</span>
            </div>
        `).join("");
    } else {
        cardsEl.innerHTML = "";
    }
}

// ── Action Log ──────────────────────────────────────────────────

function updateActionLog(s) {
    const log = s.action_log || [];
    const tbody = document.getElementById("action-log-body");

    // Only update if changed
    if (tbody.dataset.count === String(log.length)) return;
    tbody.dataset.count = String(log.length);

    // Show newest first
    const rows = log.slice().reverse().map(a => {
        const success = a.success;
        const mode = a.mode || "live";
        let statusClass, statusText;
        if (mode === "dry_run") {
            statusClass = "status-dry";
            statusText = "DRY";
        } else if (success) {
            statusClass = "status-ok";
            statusText = "OK";
        } else {
            statusClass = "status-fail";
            statusText = "FAIL";
        }
        return `<tr>
            <td>${escHtml(a.action_type || "?")}</td>
            <td>${escHtml(a.object_name || "")}</td>
            <td>${escHtml(a.trigger_step || "")}</td>
            <td class="${statusClass}">${statusText}</td>
        </tr>`;
    });

    tbody.innerHTML = rows.join("");
}

// ── Footer ──────────────────────────────────────────────────────

function setConnectionStatus(connected) {
    const el = document.getElementById("connection-status");
    if (connected) {
        el.textContent = "Connected";
        el.className = "connected";
    } else {
        el.textContent = "Disconnected — reconnecting...";
        el.className = "disconnected";
    }
}

function updateFooter() {
    const el = document.getElementById("last-update");
    el.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

// ── Helpers ─────────────────────────────────────────────────────

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function escHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Init ────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    fpsTimer = performance.now();
    connect();

    const stopBtn = document.getElementById("stop-workflow-btn");
    if (stopBtn) {
        stopBtn.addEventListener("click", async () => {
            const prev = stopBtn.textContent;
            stopBtn.textContent = "Stopping...";
            stopBtn.disabled = true;
            try {
                await fetch("/api/stop", { method: "POST" });
            } catch (e) {
                console.error("Stop failed", e);
                stopBtn.textContent = prev;
                stopBtn.disabled = false;
            }
        });
    }
});

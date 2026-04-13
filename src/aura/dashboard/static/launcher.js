/**
 * AURA Launcher — client-side logic.
 *
 * Handles:
 *  - Populating task and video dropdowns from the API
 *  - Source tab switching and conditional form logic
 *  - Preview requests
 *  - Config collection and launch POST
 */

// ── State ───────────────────────────────────────────────────────

let currentSource = "video";

// Model options per backend
const MODELS_BY_BACKEND = {
    gemini: [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-live-preview",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash",
    ],
    openai: [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o4-mini",
    ],
    sglang: [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-0.8B",
    ],
    vllm: [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-0.8B",
    ],
    ollama: [
        "llama3.2-vision",
        "llava",
        "gemma3",
    ],
    local: [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-0.8B",
    ],
};

// ── Init ────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    loadTasks();
    loadVideos();
    setupSourceTabs();
    setupSpeedSlider();
    setupRealtimeToggle();
    setupRobotMode();
    setupBackendVisibility();
    setupPreview();
    setupLaunch();
    applyLogicRules();
});

// ── Data Loading ────────────────────────────────────────────────

async function loadTasks() {
    try {
        const resp = await fetch("/api/tasks");
        const tasks = await resp.json();
        const sel = document.getElementById("task");
        sel.innerHTML = tasks.map(
            t => `<option value="${esc(t)}">${esc(t)}</option>`
        ).join("");
    } catch (e) {
        console.error("Failed to load tasks:", e);
    }
}

async function loadVideos() {
    try {
        const resp = await fetch("/api/videos");
        const videos = await resp.json();
        const sel = document.getElementById("video-path");
        sel.innerHTML = videos.map(
            v => `<option value="${esc(v)}">${esc(v)}</option>`
        ).join("");
    } catch (e) {
        console.error("Failed to load videos:", e);
    }
}

// ── Source Tabs ──────────────────────────────────────────────────

function setupSourceTabs() {
    const tabs = document.querySelectorAll(".source-tabs .tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            // Deactivate all
            tabs.forEach(t => {
                t.classList.remove("active");
                t.setAttribute("aria-selected", "false");
            });
            document.querySelectorAll(".source-panel").forEach(
                p => p.classList.remove("active")
            );

            // Activate clicked
            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");
            currentSource = tab.dataset.source;
            const panel = document.getElementById("panel-" + currentSource);
            if (panel) panel.classList.add("active");

            applyLogicRules();
            hidePreview();
        });
    });
}

// ── Speed Slider ────────────────────────────────────────────────

function setupSpeedSlider() {
    const slider = document.getElementById("speed");
    const display = document.getElementById("speed-display");
    slider.addEventListener("input", () => {
        display.textContent = parseFloat(slider.value).toFixed(1) + "x";
    });
}

// ── Realtime Toggle ─────────────────────────────────────────────

function setupRealtimeToggle() {
    const cb = document.getElementById("realtime");
    cb.addEventListener("change", applyLogicRules);
}

// ── Robot Mode ──────────────────────────────────────────────────

function setupRobotMode() {
    const radios = document.querySelectorAll('input[name="robot-mode"]');
    radios.forEach(r => {
        r.addEventListener("change", () => {
            // Update visual selection
            document.querySelectorAll(".radio-option").forEach(opt => {
                opt.classList.toggle("selected", opt.querySelector("input").checked);
            });
            // Show/hide robot URL
            const isLive = document.querySelector('input[name="robot-mode"]:checked').value === "live";
            toggleEl("robot-url-group", isLive);
        });
    });
}

// ── Backend Visibility ──────────────────────────────────────────

function setupBackendVisibility() {
    const backendEls = ["llm-backend", "intent-backend", "decision-backend"];
    backendEls.forEach(id => {
        document.getElementById(id).addEventListener("change", updateSglangVisibility);
    });

    // Update model dropdowns when backend changes
    document.getElementById("llm-backend").addEventListener("change", () => {
        populateModelSelect("model", val("llm-backend"));
    });
    document.getElementById("intent-backend").addEventListener("change", () => {
        const backend = val("intent-backend") || val("llm-backend");
        populateModelSelect("intent-model", backend, true);
    });
    document.getElementById("decision-backend").addEventListener("change", () => {
        const backend = val("decision-backend") || val("llm-backend");
        populateModelSelect("decision-model", backend, true);
    });

    // Initial population
    populateModelSelect("model", val("llm-backend"));
    populateModelSelect("intent-model", val("llm-backend"), true);
    populateModelSelect("decision-model", val("llm-backend"), true);
}

function populateModelSelect(selectId, backend, includeSharedOption) {
    const sel = document.getElementById(selectId);
    const models = MODELS_BY_BACKEND[backend] || [];
    let html = "";
    if (includeSharedOption) {
        html += '<option value="">Use shared model</option>';
    }
    html += models.map(
        m => `<option value="${esc(m)}">${esc(m)}</option>`
    ).join("");
    sel.innerHTML = html;
}

function updateSglangVisibility() {
    const shared = val("llm-backend");
    const intent = val("intent-backend");
    const decision = val("decision-backend");
    const needsSglang = [shared, intent, decision].some(
        b => ["sglang", "vllm", "openai", "local"].includes(b)
    );
    toggleEl("sglang-url-group", needsSglang);
}

// ── Logic Rules ─────────────────────────────────────────────────

function applyLogicRules() {
    const isVideo = currentSource === "video";
    const isRealtime = document.getElementById("realtime").checked;
    const realtimeLabel = document.getElementById("realtime").closest(".toggle-label");
    const timingSection = document.getElementById("timing-section");

    // Timing section: only relevant for video
    if (isVideo) {
        timingSection.classList.remove("hidden");
        realtimeLabel.classList.remove("disabled");
        document.getElementById("realtime").disabled = false;
    } else {
        // Non-video sources are always realtime
        timingSection.classList.add("hidden");
        document.getElementById("realtime").checked = true;
    }

    // Frame skip: only visible in non-realtime mode
    toggleEl("frame-skip-group", isVideo && !isRealtime);
    toggleEl("timing-hint", isVideo && !isRealtime);

    // Robot mode logic
    const forceDryRun = isVideo && !isRealtime;
    const modeGroup = document.getElementById("robot-mode-group");
    const forcedInfo = document.getElementById("robot-forced-info");

    if (forceDryRun) {
        // Force dry-run for non-realtime pre-recorded
        document.querySelector('input[name="robot-mode"][value="dry-run"]').checked = true;
        modeGroup.classList.add("disabled");
        forcedInfo.classList.remove("hidden");
        toggleEl("robot-url-group", false);
        // Update radio visual
        document.querySelectorAll(".radio-option").forEach(opt => {
            opt.classList.toggle("selected", opt.querySelector("input").checked);
        });
    } else {
        modeGroup.classList.remove("disabled");
        forcedInfo.classList.add("hidden");
        // Restore radio interactivity
        const isLive = document.querySelector('input[name="robot-mode"]:checked').value === "live";
        toggleEl("robot-url-group", isLive);
    }
}

// ── Preview ─────────────────────────────────────────────────────

function setupPreview() {
    document.getElementById("preview-btn").addEventListener("click", requestPreview);
}

async function requestPreview() {
    const btn = document.getElementById("preview-btn");
    const container = document.getElementById("preview-container");
    const img = document.getElementById("preview-img");
    const info = document.getElementById("preview-info");
    const errEl = document.getElementById("preview-error");

    btn.textContent = "Loading...";
    btn.disabled = true;
    errEl.classList.add("hidden");
    container.classList.add("hidden");

    const body = buildPreviewConfig();

    try {
        const resp = await fetch("/api/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.error || `Preview failed (${resp.status})`);
        }

        const blob = await resp.blob();
        img.src = URL.createObjectURL(blob);
        info.textContent = currentSource + " preview";
        container.classList.remove("hidden");
    } catch (e) {
        errEl.textContent = e.message;
        errEl.classList.remove("hidden");
    } finally {
        btn.textContent = "Preview Source";
        btn.disabled = false;
    }
}

function buildPreviewConfig() {
    const cfg = { source_type: currentSource };

    if (currentSource === "video") {
        cfg.video_path = val("video-path");
    } else if (currentSource === "webcam") {
        cfg.webcam_device = intVal("webcam-device");
    } else if (currentSource === "screen") {
        cfg.screen_monitor = intVal("screen-monitor");
        const l = intVal("screen-left");
        const t = intVal("screen-top");
        const w = intVal("screen-width");
        const h = intVal("screen-height");
        if (l != null && t != null && w != null && h != null) {
            cfg.screen_region = [l, t, w, h];
        }
    } else if (currentSource === "gopro") {
        cfg.gopro_ip = val("gopro-ip");
    }

    return cfg;
}

function hidePreview() {
    document.getElementById("preview-container").classList.add("hidden");
    document.getElementById("preview-error").classList.add("hidden");
}

// ── Launch ──────────────────────────────────────────────────────

function setupLaunch() {
    document.getElementById("config-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        await launchWorkflow();
    });
}

async function launchWorkflow() {
    const btn = document.getElementById("launch-btn");
    const overlay = document.getElementById("launch-overlay");

    btn.disabled = true;
    overlay.classList.remove("hidden");

    const config = collectConfig();

    try {
        const resp = await fetch("/api/launch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.error || `Launch failed (${resp.status})`);
        }

        // Redirect to monitoring dashboard after a brief delay
        setTimeout(() => {
            window.location.href = "/monitor";
        }, 1500);

    } catch (e) {
        overlay.classList.add("hidden");
        btn.disabled = false;
        alert("Launch failed: " + e.message);
    }
}

function collectConfig() {
    const isRealtime = document.getElementById("realtime").checked;
    const isDryRun = document.querySelector('input[name="robot-mode"]:checked').value === "dry-run";

    const config = {
        task: val("task"),
        source_type: currentSource,
        speed: parseFloat(document.getElementById("speed").value),
        realtime: isRealtime,
        frame_skip: intVal("frame-skip") || 30,
        dry_run: isDryRun,
        robot_url: val("robot-url"),
        model: val("model"),
        llm_backend: val("llm-backend"),
        sglang_url: val("sglang-url"),
        intent_backend: val("intent-backend") || null,
        intent_model: val("intent-model") || null,
        decision_backend: val("decision-backend") || null,
        decision_model: val("decision-model") || null,
        max_cycles: intVal("max-cycles") || null,
        use_ground_truth_robot_status: document.getElementById("ground-truth").checked,
    };

    // Source-specific fields
    if (currentSource === "video") {
        config.video_path = val("video-path");
    } else if (currentSource === "webcam") {
        config.webcam_device = intVal("webcam-device") || 0;
    } else if (currentSource === "screen") {
        config.screen_monitor = intVal("screen-monitor") || 1;
        const l = intVal("screen-left");
        const t = intVal("screen-top");
        const w = intVal("screen-width");
        const h = intVal("screen-height");
        if (l != null && t != null && w != null && h != null) {
            config.screen_region = [l, t, w, h];
        } else {
            config.screen_region = null;
        }
    } else if (currentSource === "gopro") {
        config.gopro_ip = val("gopro-ip");
        config.gopro_lens = val("gopro-lens");
    }

    return config;
}

// ── Helpers ─────────────────────────────────────────────────────

function val(id) {
    const el = document.getElementById(id);
    return el ? el.value : "";
}

function intVal(id) {
    const v = val(id);
    if (v === "" || v == null) return null;
    const n = parseInt(v, 10);
    return isNaN(n) ? null : n;
}

function toggleEl(id, show) {
    const el = document.getElementById(id);
    if (!el) return;
    if (show) {
        el.classList.remove("hidden");
    } else {
        el.classList.add("hidden");
    }
}

function esc(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

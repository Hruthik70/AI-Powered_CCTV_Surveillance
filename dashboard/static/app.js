// Sentinel-AI Dashboard State
let currentCameraId = "CAM-01";
let camerasData = {};
let currentFilter = "ALL";
let eventsHistory = [];
let ws = null;

// Clock updates
setInterval(() => {
    const now = new Date();
    document.getElementById("system-clock").innerText = now.toTimeString().split(" ")[0];
}, 1000);

// Initialize WebSocket for Real-Time Telemetry
function initWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/telemetry`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        document.getElementById("ws-status-text").innerText = "WS: Live Telemetry";
        document.getElementById("ws-status-pill").querySelector(".status-dot").className = "status-dot green";
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleTelemetryUpdate(data);
        } catch (e) {
            console.error("Telemetry parse error", e);
        }
    };
    
    ws.onclose = () => {
        document.getElementById("ws-status-text").innerText = "WS: Reconnecting...";
        document.getElementById("ws-status-pill").querySelector(".status-dot").className = "status-dot red";
        setTimeout(initWebSocket, 2000);
    };
}

function handleTelemetryUpdate(data) {
    // 1. Update Database Status
    if (data.statistics) {
        const isConnected = data.statistics.db_connected;
        document.getElementById("db-status-text").innerText = isConnected ? "MongoDB: Online" : "Database: In-Memory Safe";
    }

    // 2. Process Cameras
    if (data.cameras) {
        data.cameras.forEach(cam => {
            camerasData[cam.camera_id] = cam;
        });
        renderCameraList();
        renderActiveCameraTelemetry();
    }
}

function renderCameraList() {
    const container = document.getElementById("camera-list");
    container.innerHTML = "";

    Object.values(camerasData).forEach(cam => {
        const isActive = cam.camera_id === currentCameraId;
        const btn = document.createElement("button");
        btn.className = `cam-btn ${isActive ? "active" : ""}`;
        btn.onclick = () => selectCamera(cam.camera_id);

        btn.innerHTML = `
            <div>
                <div class="cam-info-title">${cam.camera_id} - ${cam.name}</div>
                <div class="cam-info-sub">${cam.location}</div>
            </div>
            <div class="badge ${cam.crowd_status === 'OVERCROWDED' ? 'badge-danger' : (cam.crowd_status === 'MODERATE' ? 'badge-warning' : 'badge-success')}">
                ${cam.people_count} Pax
            </div>
        `;
        container.appendChild(btn);
    });
}

function selectCamera(camId) {
    currentCameraId = camId;
    document.getElementById("active-cam-tag").innerText = camId;
    document.getElementById("main-video-stream").src = `/api/video/feed/${camId}`;
    renderCameraList();
    renderActiveCameraTelemetry();
}

function reloadStream() {
    const stream = document.getElementById("main-video-stream");
    stream.src = `/api/video/feed/${currentCameraId}?t=${Date.now()}`;
}

function renderActiveCameraTelemetry() {
    const cam = camerasData[currentCameraId];
    if (!cam) return;

    document.getElementById("current-stream-title").innerText = `${cam.camera_id} - ${cam.name} (${cam.location})`;
    document.getElementById("kpi-people-count").innerText = cam.people_count;
    document.getElementById("kpi-crowd-status").innerText = cam.crowd_status;
    document.getElementById("kpi-fps-val").innerText = cam.fps.toFixed(1);

    // Dynamic Crowd KPI Styling
    const crowdIcon = document.getElementById("crowd-icon-bg");
    if (cam.crowd_status === "OVERCROWDED") {
        crowdIcon.className = "kpi-icon red";
    } else if (cam.crowd_status === "MODERATE") {
        crowdIcon.className = "kpi-icon yellow";
    } else {
        crowdIcon.className = "kpi-icon green";
    }

    // Safety violations count in current active events
    const violations = cam.active_events ? cam.active_events.filter(e => e.event_type === "SAFETY_VIOLATION") : [];
    document.getElementById("kpi-violation-count").innerText = violations.length;

    // Render Zones
    const zonesContainer = document.getElementById("zones-list");
    zonesContainer.innerHTML = "";
    (cam.zones || []).forEach(z => {
        const item = document.createElement("div");
        item.className = `zone-item ${z.type}`;
        item.innerHTML = `
            <span><strong>${z.name}</strong></span>
            <span class="badge ${z.type === 'RESTRICTED' ? 'badge-danger' : 'badge-warning'}">${z.type}</span>
        `;
        zonesContainer.appendChild(item);
    });
}

// Polling for Alerts & Event History
async function fetchAlerts() {
    try {
        const res = await fetch("/api/alerts?status=ACTIVE&limit=25");
        const alerts = await res.json();
        renderAlerts(alerts);
    } catch (e) {
        console.error("Error fetching alerts", e);
    }
}

function renderAlerts(alerts) {
    const container = document.getElementById("alerts-feed");
    document.getElementById("active-alert-counter").innerText = alerts.length;
    container.innerHTML = "";

    if (alerts.length === 0) {
        container.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">No active alerts. System nominal.</div>`;
        return;
    }

    alerts.forEach(a => {
        const card = document.createElement("div");
        card.className = `alert-item ${a.severity}`;
        const timeStr = a.timestamp.split("T")[1]?.slice(0, 8) || a.timestamp;
        
        card.innerHTML = `
            <div class="alert-header">
                <span class="alert-type"><i class="fa-solid fa-triangle-exclamation"></i> ${a.event_type}</span>
                <span class="alert-time">${timeStr} | ${a.camera_id}</span>
            </div>
            <div class="alert-msg">${a.message}</div>
            <div class="alert-actions">
                <button class="btn btn-sm btn-resolve" onclick="resolveAlert('${a.alert_id}')">
                    <i class="fa-solid fa-check"></i> Acknowledge
                </button>
            </div>
        `;
        container.appendChild(card);
    });
}

async function resolveAlert(alertId) {
    try {
        await fetch(`/api/alerts/${alertId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: "RESOLVED" })
        });
        fetchAlerts();
    } catch (e) {
        console.error("Error resolving alert", e);
    }
}

async function fetchEventsHistory() {
    try {
        const res = await fetch("/api/events?limit=40");
        eventsHistory = await res.json();
        renderEventsTable();
    } catch (e) {
        console.error("Error fetching events", e);
    }
}

function filterEvents(type) {
    currentFilter = type;
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    event.target.classList.add("active");
    renderEventsTable();
}

function renderEventsTable() {
    const tbody = document.getElementById("events-table-body");
    tbody.innerHTML = "";

    const filtered = eventsHistory.filter(e => currentFilter === "ALL" || e.event_type === currentFilter);

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No events logged yet.</td></tr>`;
        return;
    }

    filtered.forEach(e => {
        const tr = document.createElement("tr");
        const timeStr = e.timestamp.split("T")[1]?.slice(0, 8) || e.timestamp;
        const badgeClass = e.severity === "HIGH" || e.severity === "CRITICAL" ? "badge-danger" : (e.severity === "MEDIUM" ? "badge-warning" : "badge-info");
        const tracksStr = e.track_ids && e.track_ids.length > 0 ? e.track_ids.map(id => `#${id}`).join(", ") : "N/A";

        tr.innerHTML = `
            <td>${timeStr}</td>
            <td><strong>${e.camera_id}</strong></td>
            <td>${e.event_type}</td>
            <td><span class="badge ${badgeClass}">${e.severity}</span></td>
            <td>${e.message}</td>
            <td>${tracksStr}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Initial Boot
document.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
    fetchAlerts();
    fetchEventsHistory();
    setInterval(fetchAlerts, 2500);
    setInterval(fetchEventsHistory, 3500);
});

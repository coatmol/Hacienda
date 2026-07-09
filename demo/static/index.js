/* ===================================================================
   Hacienda — Demo Frontend Logic
   Supports two modes:
     - "server"  → FastAPI backend (Docker)
     - "client"  → pure static / Netlify (client-side frame extraction,
                    direct Gemma API calls from browser)
   =================================================================== */

let mode = "url";
let rowCount = 0;
let timerInterval = null;
let backend = "client"; // "server" or "client"

// Gemma config for client mode (saved in localStorage)
let gemmaConfig = {
  baseUrl: "",
  token: "",
  model: "gemma",
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  buildKeyframeStrip();
  addRow();
  loadSettings();
  detectBackend();
});

// ---------------------------------------------------------------------------
// Settings: load baked-in config (from Netlify build) or localStorage
// ---------------------------------------------------------------------------
function loadSettings() {
  // 1. Check for build-time config (config.js sets HACIENDA_CONFIG)
  if (typeof HACIENDA_CONFIG !== "undefined" && HACIENDA_CONFIG.baseUrl && HACIENDA_CONFIG.token) {
    gemmaConfig.baseUrl = HACIENDA_CONFIG.baseUrl;
    gemmaConfig.token = HACIENDA_CONFIG.token;
    gemmaConfig.model = HACIENDA_CONFIG.model || "gemma";
    // Hide the settings panel entirely — judges don't need it
    document.getElementById("settingsBtn").style.display = "none";
    document.getElementById("settingsPanel").classList.remove("open");
    return;
  }

  // 2. Fallback: load from localStorage (local dev)
  try {
    const saved = localStorage.getItem("hacienda_gemma_config");
    if (saved) {
      gemmaConfig = { ...gemmaConfig, ...JSON.parse(saved) };
    }
  } catch {}
  document.getElementById("cfgBaseUrl").value = gemmaConfig.baseUrl;
  document.getElementById("cfgToken").value = gemmaConfig.token;
  document.getElementById("cfgModel").value = gemmaConfig.model;
}

// ---------------------------------------------------------------------------
// Backend detection
// ---------------------------------------------------------------------------
async function detectBackend() {
  const badge = document.getElementById("healthBadge");

  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      backend = "server";
      const data = await res.json();
      const gemma = data.gemma_available;
      const groq = data.groq_configured;

      if (gemma && groq) {
        badge.className = "health-badge ok";
        badge.querySelector(".health-text").textContent = "Server online";
      } else {
        badge.className = "health-badge warn";
        badge.querySelector(".health-text").textContent = "Server (partial)";
      }
      // Hide settings panel in server mode
      document.getElementById("settingsBtn").style.display = "none";
      return;
    }
  } catch {}

  // Client mode — check if Gemma credentials are configured
  backend = "client";
  updateClientHealthBadge();
}

function updateClientHealthBadge() {
  const badge = document.getElementById("healthBadge");
  if (gemmaConfig.baseUrl && gemmaConfig.token) {
    badge.className = "health-badge ok";
    badge.querySelector(".health-text").textContent = "Client mode";
  } else {
    badge.className = "health-badge err";
    badge.querySelector(".health-text").textContent = "Configure API ⟶";
  }
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------
function toggleSettings() {
  document.getElementById("settingsPanel").classList.toggle("open");
}



function saveSettings() {
  gemmaConfig.baseUrl = document.getElementById("cfgBaseUrl").value.trim();
  gemmaConfig.token = document.getElementById("cfgToken").value.trim();
  gemmaConfig.model = document.getElementById("cfgModel").value.trim() || "gemma";
  localStorage.setItem("hacienda_gemma_config", JSON.stringify(gemmaConfig));
  toggleSettings();
  updateClientHealthBadge();
  document.getElementById("status").textContent = "Settings saved.";
}

// ---------------------------------------------------------------------------
// Keyframe strip (decorative)
// ---------------------------------------------------------------------------
function buildKeyframeStrip() {
  const strip = document.getElementById("kfStrip");
  strip.innerHTML = Array.from({ length: 8 })
    .map(
      (_, i) => `
    <div class="kf" style="--i:${i}">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path d="M4 8V6a2 2 0 012-2h2M4 16v2a2 2 0 002 2h2M20 8V6a2 2 0 00-2-2h-2M20 16v2a2 2 0 01-2 2h-2" stroke="#00f0ff" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
  `
    )
    .join("");
}

// ---------------------------------------------------------------------------
// Mode switching (URL vs Upload)
// ---------------------------------------------------------------------------
function setMode(m) {
  mode = m;
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
  document.getElementById("clipRows").innerHTML = "";
  rowCount = 0;
  addRow();
}

// ---------------------------------------------------------------------------
// Row management
// ---------------------------------------------------------------------------
function addRow() {
  const id = "row" + rowCount++;
  const container = document.getElementById("clipRows");
  const row = document.createElement("div");
  row.className = "clip-row";
  row.id = id;
  row.innerHTML =
    mode === "url"
      ? `<input type="text" placeholder="https://example.com/clip.mp4" />
         <button class="remove-btn" onclick="removeRow('${id}')">✕</button>`
      : `<input type="file" accept="video/*" />
         <button class="remove-btn" onclick="removeRow('${id}')">✕</button>`;
  container.appendChild(row);
}

function removeRow(id) {
  const row = document.getElementById(id);
  if (document.getElementById("clipRows").children.length > 1) row.remove();
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
function startTimer() {
  const timerEl = document.getElementById("statusTimer");
  const start = Date.now();
  timerEl.textContent = "0:00";
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - start) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = String(elapsed % 60).padStart(2, "0");
    timerEl.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ---------------------------------------------------------------------------
// Score badge helper
// ---------------------------------------------------------------------------
function scoreBadgeHTML(scores, styleKey) {
  if (!scores || !scores[styleKey]) return "";
  const s = scores[styleKey];
  const acc = s.accuracy ?? 0;
  const style = s.style_match ?? 0;
  const avg = (acc + style) / 2;
  const tier = avg >= 0.7 ? "high" : avg >= 0.5 ? "mid" : "low";
  return `<span class="score-badge ${tier}">${(avg * 100).toFixed(0)}%</span>`;
}

// ===================================================================
// Client-side frame extraction (Canvas API)
// ===================================================================
function extractFramesFromVideo(videoSrc, numFrames = 6) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.preload = "auto";
    video.src = videoSrc;

    video.onerror = () => reject(new Error("Failed to load video. If using a URL, it may block cross-origin access — try uploading the file instead."));

    video.onloadedmetadata = () => {
      const duration = video.duration;
      if (!isFinite(duration) || duration <= 0) {
        reject(new Error("Could not determine video duration."));
        return;
      }

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const scale = Math.min(768 / video.videoWidth, 1);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);

      const count = Math.min(numFrames, Math.max(4, Math.round(duration / 3)));
      const margin = 0.15;
      const maxTs = Math.max(duration - margin, 0);
      const step = maxTs / (count - 1 || 1);
      const timestamps = Array.from({ length: count }, (_, i) =>
        Math.min(+(i * step).toFixed(2), maxTs)
      );

      const frames = [];
      let idx = 0;

      function seekNext() {
        if (idx >= timestamps.length) {
          resolve({ frames, duration: +duration.toFixed(1), timestamps });
          return;
        }
        video.currentTime = timestamps[idx];
      }

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        frames.push(canvas.toDataURL("image/jpeg", 0.65));
        idx++;
        seekNext();
      };

      seekNext();
    };
  });
}

// ===================================================================
// Gemma API client (direct from browser — client mode)
// ===================================================================
async function callGemma(systemPrompt, userContent, maxTokens = 800, temperature = 0.35) {
  let endpoint = gemmaConfig.baseUrl.replace(/\/+$/, "");
  if (!endpoint.endsWith("/chat/completions")) {
    endpoint += "/chat/completions";
  }

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${gemmaConfig.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: gemmaConfig.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      max_tokens: maxTokens,
      temperature,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gemma API error ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  return data.choices[0].message.content;
}

function extractJSON(text) {
  let cleaned = text.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  }
  try {
    return JSON.parse(cleaned);
  } catch {}
  const match = cleaned.match(/(\{[\s\S]*\})/);
  if (match) {
    try { return JSON.parse(match[1]); } catch {}
  }
  throw new Error("No valid JSON in model response");
}

// ===================================================================
// Client-mode captioning pipeline
// ===================================================================
async function runClientPipeline(videoSrc, statusCb) {
  if (!gemmaConfig.baseUrl || !gemmaConfig.token) {
    throw new Error("Gemma API not configured. Click the ⚙ button to enter your credentials.");
  }

  // 1. Extract frames
  statusCb("Extracting frames…");
  const { frames, duration, timestamps } = await extractFramesFromVideo(videoSrc, 6);

  // 2. Build multimodal content for evidence
  statusCb("Analyzing video…");
  const imageContent = frames.map((dataUrl) => ({
    type: "image_url",
    image_url: { url: dataUrl },
  }));

  const evidencePrompt =
    "Analyze these sampled frames from a video clip. Return only JSON with keys: " +
    "setting, subjects, actions, objects, visual_details, temporal_changes, mood, uncertainty, summary. " +
    "Be concrete and literal. Do not invent details that are not visible.";

  const userText = `Video duration: ${duration}s. Frame timestamps: ${timestamps.map((t) => t.toFixed(1) + "s").join(", ")}.`;

  // Call with images as multimodal content
  const evidenceRes = await callGemmaMultimodal(evidencePrompt, userText, frames, 900, 0.2);
  let evidence;
  try {
    evidence = extractJSON(evidenceRes);
  } catch {
    evidence = { summary: evidenceRes, setting: "video scene", subjects: ["visible subject"], actions: ["visible activity"] };
  }
  evidence.duration_seconds = duration;

  // 3. Generate captions
  statusCb("Generating captions…");
  const captionPrompt =
    "You are writing captions for a video-captioning benchmark. " +
    "Use only the provided visual evidence. Return only JSON where every requested style maps to one caption. " +
    "Each caption must be English, one sentence, 10-28 words, and faithful to the evidence. " +
    "Style rules: formal is objective and professional; sarcastic is dry and lightly ironic; " +
    "humorous_tech is funny with programming or technology references; " +
    "humorous_non_tech is funny with everyday humor and no tech jargon.";

  const captionInput = JSON.stringify({
    requested_styles: ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    evidence,
  });

  const captionRes = await callGemma(captionPrompt, captionInput, 700, 0.65);
  const captions = extractJSON(captionRes);

  return {
    success: true,
    captions,
    scores: null,
    duration,
  };
}

async function callGemmaMultimodal(systemPrompt, userText, frameDataUrls, maxTokens, temperature) {
  let endpoint = gemmaConfig.baseUrl.replace(/\/+$/, "");
  if (!endpoint.endsWith("/chat/completions")) {
    endpoint += "/chat/completions";
  }

  const content = [{ type: "text", text: userText }];
  for (const dataUrl of frameDataUrls) {
    content.push({
      type: "image_url",
      image_url: { url: dataUrl },
    });
  }

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${gemmaConfig.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: gemmaConfig.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content },
      ],
      max_tokens: maxTokens,
      temperature,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gemma API error ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  return data.choices[0].message.content;
}

// ===================================================================
// Generate all captions
// ===================================================================
async function generateAll() {
  const btn = document.getElementById("generateBtn");
  const statusText = document.getElementById("status");
  const results = document.getElementById("results");
  const strip = document.getElementById("kfStrip");
  const addBtn = document.getElementById("addRowBtn");

  const rows = Array.from(document.getElementById("clipRows").children);
  const inputs = rows.map((r) => r.querySelector("input")).filter(Boolean);

  // Validate
  const validInputs = inputs.filter((input) => {
    if (mode === "url") return input.value.trim().length > 0;
    return input.files && input.files.length > 0;
  });

  if (validInputs.length === 0) {
    statusText.textContent = "Please add at least one clip.";
    return;
  }

  // Lock UI
  btn.disabled = true;
  btn.classList.add("loading");
  addBtn.disabled = true;
  results.innerHTML = "";
  strip.classList.add("active");
  startTimer();

  const total = validInputs.length;
  let completed = 0;
  statusText.textContent = `Processing ${total} clip${total > 1 ? "s" : ""}…`;

  // Create placeholder sections
  const sections = validInputs.map((_, i) => {
    const section = document.createElement("div");
    section.className = "clip-result";
    section.style.animationDelay = `${i * 0.1}s`;
    section.innerHTML = `
      <div class="clip-title">Clip ${i + 1} — Processing…</div>
      <div class="grid">
        <div class="skeleton"></div>
        <div class="skeleton"></div>
        <div class="skeleton"></div>
        <div class="skeleton"></div>
      </div>
    `;
    results.appendChild(section);
    return section;
  });

  // Process each clip
  const tasks = validInputs.map(async (input, i) => {
    let data, previewSrc, label;

    try {
      if (backend === "server") {
        // --- Server mode (Docker) ---
        if (mode === "url") {
          const url = input.value.trim();
          previewSrc = url;
          label = truncateURL(url, 60);
          const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_url: url }),
          });
          data = await res.json();
        } else {
          const file = input.files[0];
          previewSrc = URL.createObjectURL(file);
          label = file.name;
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/upload", {
            method: "POST",
            body: formData,
          });
          data = await res.json();
        }
      } else {
        // --- Client mode (static / Netlify) ---
        if (mode === "url") {
          const url = input.value.trim();
          previewSrc = url;
          label = truncateURL(url, 60);
        } else {
          const file = input.files[0];
          previewSrc = URL.createObjectURL(file);
          label = file.name;
        }

        data = await runClientPipeline(previewSrc, (msg) => {
          sections[i].querySelector(".clip-title").textContent = `Clip ${i + 1} — ${msg}`;
        });
      }
    } catch (e) {
      data = { success: false, error: e.message };
      label = label || `Clip ${i + 1}`;
    }

    completed++;
    statusText.textContent = `Completed ${completed} of ${total}…`;

    renderResult(sections[i], label, previewSrc, data);
  });

  await Promise.all(tasks);

  // Unlock UI
  strip.classList.remove("active");
  stopTimer();
  statusText.textContent = `Done — ${completed} clip${completed > 1 ? "s" : ""} processed.`;
  btn.disabled = false;
  btn.classList.remove("loading");
  addBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Render a single clip result
// ---------------------------------------------------------------------------
function renderResult(section, label, previewSrc, data) {
  if (!data || !data.success) {
    section.innerHTML = `
      <div class="clip-title">${escapeHTML(label)}</div>
      <div class="error-card">
        <div class="error-label">Pipeline Error</div>
        ${escapeHTML(data?.error || "Unknown error")}
      </div>
    `;
    return;
  }

  const styles = [
    { key: "formal", label: "Formal" },
    { key: "sarcastic", label: "Sarcastic" },
    { key: "humorous_tech", label: "Humorous — Tech" },
    { key: "humorous_non_tech", label: "Humorous — Non-tech" },
  ];

  const durationBadge = data.duration
    ? `<span class="duration-badge">${data.duration}s</span>`
    : "";

  section.innerHTML = `
    <div class="clip-title">${escapeHTML(label)} ${durationBadge}</div>
    <video class="video-preview" src="${escapeAttr(previewSrc)}" controls preload="metadata"></video>
    <div class="grid">
      ${styles
        .map(
          (s) => `
        <div class="card ${s.key}">
          <div class="card-label">
            ${s.label}
            ${scoreBadgeHTML(data.scores, s.key)}
          </div>
          <div class="card-text">${escapeHTML(data.captions?.[s.key] || "—")}</div>
        </div>
      `
        )
        .join("")}
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

function escapeAttr(str) {
  return (str || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function truncateURL(url, max) {
  if (url.length <= max) return url;
  return url.slice(0, max - 3) + "…";
}

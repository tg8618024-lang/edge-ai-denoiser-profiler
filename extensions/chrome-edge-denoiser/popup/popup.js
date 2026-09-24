// Edge AI Voice Chrome Extension Popup Controller
// Follows modern async/await and Manifest V3 practices

const SERVER_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  const toggleActive = document.getElementById("toggle-active");
  const sliderIntensity = document.getElementById("slider-intensity");
  const intensityVal = document.getElementById("intensity-val");
  const statusIndicator = document.getElementById("server-status");
  const statusText = document.getElementById("status-text");
  const modeButtons = document.querySelectorAll(".mode-btn");
  const metricLatency = document.getElementById("metric-latency");
  const metricSnr = document.getElementById("metric-snr");

  // 1. Check server health
  async function checkServerHealth() {
    try {
      const resp = await fetch(`${SERVER_URL}/api/telemetry/quality`, { method: "GET" });
      if (resp.ok) {
        statusIndicator.className = "status-indicator online";
        statusText.textContent = "ONLINE";
      } else {
        statusIndicator.className = "status-indicator offline";
        statusText.textContent = "IDLE";
      }
    } catch {
      statusIndicator.className = "status-indicator offline";
      statusText.textContent = "OFFLINE";
    }
  }

  await checkServerHealth();

  // 2. Load stored user preferences
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    const data = await chrome.storage.local.get(["active", "mode", "intensity"]);
    if (data.active !== undefined) toggleActive.checked = data.active;
    if (data.intensity !== undefined) {
      sliderIntensity.value = data.intensity;
      intensityVal.textContent = `${data.intensity}%`;
    }
    if (data.mode) {
      modeButtons.forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === data.mode);
      });
      updateMetricDisplays(data.mode);
    }
  }

  function updateMetricDisplays(mode) {
    if (mode === "neural") {
      metricLatency.textContent = "0.26 ms";
      metricSnr.textContent = "+14.5 dB";
    } else if (mode === "wiener") {
      metricLatency.textContent = "0.05 ms";
      metricSnr.textContent = "+11.4 dB";
    } else if (mode === "dual") {
      metricLatency.textContent = "0.33 ms";
      metricSnr.textContent = "+13.8 dB";
    }
  }

  // 3. Save preferences & broadcast to active conferencing tabs
  async function broadcastConfig() {
    const activeModeBtn = document.querySelector(".mode-btn.active");
    const mode = activeModeBtn ? activeModeBtn.dataset.mode : "neural";
    const config = {
      active: toggleActive.checked,
      mode: mode,
      intensity: parseInt(sliderIntensity.value, 10),
    };

    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      await chrome.storage.local.set(config);
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs && tabs.length > 0) {
          await chrome.tabs.sendMessage(tabs[0].id, { type: "DENOISER_CONFIG_UPDATE", config });
        }
      } catch {
        // Tab might not have content script injected; ignored safely
      }
    }
  }

  // Event Listeners
  toggleActive.addEventListener("change", async () => {
    await broadcastConfig();
  });

  sliderIntensity.addEventListener("input", async () => {
    intensityVal.textContent = `${sliderIntensity.value}%`;
    await broadcastConfig();
  });

  modeButtons.forEach(btn => {
    btn.addEventListener("click", async () => {
      modeButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      updateMetricDisplays(btn.dataset.mode);
      await broadcastConfig();
    });
  });
});

// Edge AI Audio Denoiser - Content Script
// Injected into Google Meet, Discord, and Zoom Web to bridge WebRTC audio.

(function () {
  console.log("[Edge AI Denoiser] Content script initializing...");

  // 1. Inject webrtc_shim.js into the main page context
  const s = document.createElement("script");
  s.src = chrome.runtime.getURL("scripts/webrtc_shim.js");
  s.onload = function () {
    this.remove();
  };
  (document.head || document.documentElement).appendChild(s);

  // 2. Listen for configuration updates from the extension popup
  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === "DENOISER_CONFIG_UPDATE") {
        window.postMessage({
          source: "EDGE_AI_EXTENSION",
          type: "CONFIG_UPDATE",
          config: message.config,
        }, "*");
        sendResponse({ status: "acknowledged" });
      }
      return false;
    });
  }

  // 3. Listen for telemetry messages from the in-page WebRTC shim
  window.addEventListener("message", (event) => {
    if (event.source !== window || !event.data || event.data.source !== "EDGE_AI_WEBRTC_SHIM") {
      return;
    }
    // Future telemetry forwarding or HUD overlay
  });
})();

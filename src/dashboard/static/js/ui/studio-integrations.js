/**
 * External Studio Integrations & Audio Bridges UI Controller
 * Bridges OBS Studio IPC socket, VST3/CLAP DAW processor, WebRTC, and Chrome Extension.
 * Zero Global Scope Pollution.
 */

import { $ } from "../utils/dom.js";

export async function fetchStudioIntegrations() {
  try {
    const res = await fetch("/api/studio/integrations");
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn("Failed to fetch studio integrations:", err);
    return null;
  }
}

export async function fetchObsStatus() {
  try {
    const res = await fetch("/api/obs/status");
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn("Failed to fetch OBS status:", err);
    return null;
  }
}

export function updateObsUI(obs) {
  if (!obs) return;
  const tag = $("obsServerTag");
  const hostText = $("obsHostText");
  const portText = $("obsPortText");
  const latencyText = $("obsLatencyText");

  if (hostText && obs.host) hostText.textContent = obs.host;
  if (portText && obs.port) portText.textContent = obs.port;
  if (latencyText) {
    latencyText.textContent = obs.last_latency_ms
      ? `${obs.last_latency_ms.toFixed(2)} ms`
      : "< 1.0 ms";
  }

  if (tag) {
    if (obs.is_running) {
      tag.textContent = `LISTENING (${obs.active_clients || 0} CLIENTS)`;
      tag.style.background = "rgba(118, 185, 0, 0.25)";
      tag.style.color = "#76b900";
      tag.style.borderColor = "rgba(118, 185, 0, 0.6)";
    } else {
      tag.textContent = "STOPPED";
      tag.style.background = "rgba(248, 81, 73, 0.15)";
      tag.style.color = "#f85149";
      tag.style.borderColor = "rgba(248, 81, 73, 0.4)";
    }
  }
}

export function initStudioIntegrations() {
  const btnRefresh = $("btnRefreshStudioStatus");
  const btnStartObs = $("btnStartObsServer");
  const btnStopObs = $("btnStopObsServer");
  const btnTestWebrtc = $("btnTestWebrtcOffer");

  const refresh = async () => {
    const data = await fetchStudioIntegrations();
    if (data && data.integrations && data.integrations.obs_studio) {
      updateObsUI(data.integrations.obs_studio);
    }
  };

  if (btnRefresh) {
    btnRefresh.addEventListener("click", refresh);
  }

  if (btnStartObs) {
    btnStartObs.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/obs/start", { method: "POST" });
        if (res.ok) {
          const obs = await res.json();
          updateObsUI(obs);
        }
      } catch (e) {
        console.error("Failed to start OBS server:", e);
      }
    });
  }

  if (btnStopObs) {
    btnStopObs.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/obs/stop", { method: "POST" });
        if (res.ok) {
          const obs = await res.json();
          updateObsUI(obs);
        }
      } catch (e) {
        console.error("Failed to stop OBS server:", e);
      }
    });
  }

  if (btnTestWebrtc) {
    btnTestWebrtc.addEventListener("click", async () => {
      try {
        const mockSdp =
          "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n";
        const res = await fetch("/api/webrtc/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sdp: mockSdp, type: "offer" }),
        });
        if (res.ok) {
          const answer = await res.json();
          alert(
            `WebRTC Handshake Verified!\n\nNegotiated SDP Answer:\nCodecs: ${answer.codecs.join(
              ", "
            )}\nSample Rate: ${answer.sample_rate} Hz`
          );
        }
      } catch (e) {
        console.error("WebRTC offer test failed:", e);
      }
    });
  }

  // Initial load
  refresh();
}

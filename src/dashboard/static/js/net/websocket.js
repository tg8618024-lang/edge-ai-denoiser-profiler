/**
 * Resilient WebSocket Client for Real-Time Streaming & Telemetry
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, safeHtml, safeClass } from "../utils/dom.js";

export class WsClient {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      const connectionStatus = $("connectionStatus");
      if (connectionStatus) {
        safeHtml(connectionStatus, `<span class="pulse-dot"></span> WEBSOCKET LIVE`);
        connectionStatus.classList.remove("text-warning");
      }
      state.emit("ws_open");
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        state.emit("ws_message", msg);

        if (msg.type === "telemetry" || msg.type === "live_telemetry" || msg.type === "subtitles_update") {
          state.emit("telemetry", msg);
        } else if (msg.type === "tse_trigger_detected") {
          state.emit("tse_trigger_detected", msg);
        } else if (msg.type === "tse_status_updated") {
          if (msg.telemetry) state.emit("tse_status_updated", msg.telemetry);
        } else if (msg.type === "studio_status_updated") {
          if (msg.telemetry) state.emit("studio_status_updated", msg.telemetry);
        } else if (msg.type === "eq_updated") {
          if (msg.curve) state.emit("eq_updated", msg.curve);
        }
      } catch (err) {
        console.error("WS Parse error:", err);
      }
    };

    this.ws.onclose = () => {
      const connectionStatus = $("connectionStatus");
      if (connectionStatus) {
        safeHtml(
          connectionStatus,
          `<span class="pulse-dot" style="background:var(--warning);box-shadow:none;"></span> RECONNECTING...`
        );
        connectionStatus.classList.add("text-warning");
      }
      state.emit("ws_close");

      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => this.connect(), 1500);
    };

    this.ws.onerror = () => {
      if (this.ws) this.ws.close();
    };
  }

  send(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }
}

export const wsClient = new WsClient();

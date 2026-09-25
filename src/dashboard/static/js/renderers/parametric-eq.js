/**
 * 5-Band Studio Parametric EQ Sculptor Visualizer & Interactive Drag Logic
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, $$, safeText } from "../utils/dom.js";
import { configureEq, setEqPreset } from "../net/api.js";
import { wsClient } from "../net/websocket.js";

let canvasEqCurve = null;
let ctxEq = null;

export function freqToX(f, width) {
  const minF = 20.0;
  const maxF = 16000.0;
  const norm = (Math.log(Math.max(minF, Math.min(maxF, f))) - Math.log(minF)) / (Math.log(maxF) - Math.log(minF));
  return norm * width;
}

export function xToFreq(x, width) {
  const minF = 20.0;
  const maxF = 16000.0;
  const norm = Math.max(0.0, Math.min(1.0, x / width));
  return minF * Math.pow(maxF / minF, norm);
}

export function gainToY(g, height) {
  const minG = -18.0;
  const maxG = 18.0;
  const norm = (maxG - Math.max(minG, Math.min(maxG, g))) / (maxG - minG);
  return norm * height;
}

export function yToGain(y, height) {
  const minG = -18.0;
  const maxG = 18.0;
  const norm = Math.max(0.0, Math.min(1.0, y / height));
  return maxG - norm * (maxG - minG);
}

export function drawEqCurve(curve = state.latestEqCurve) {
  if (!canvasEqCurve) canvasEqCurve = $("canvasEqCurve");
  if (!canvasEqCurve) return;
  if (!ctxEq) ctxEq = canvasEqCurve.getContext("2d");
  if (!ctxEq) return;

  const w = canvasEqCurve.width;
  const h = canvasEqCurve.height;

  ctxEq.fillStyle = "#080C10";
  ctxEq.fillRect(0, 0, w, h);

  // Draw Grid Lines (0 dB, ±6 dB, ±12 dB)
  ctxEq.lineWidth = 1;
  [-12, -6, 0, 6, 12].forEach((db) => {
    const y = gainToY(db, h);
    ctxEq.strokeStyle = db === 0 ? "rgba(255, 255, 255, 0.25)" : "rgba(255, 255, 255, 0.08)";
    ctxEq.beginPath();
    ctxEq.moveTo(0, y);
    ctxEq.lineTo(w, y);
    ctxEq.stroke();

    ctxEq.fillStyle = "rgba(139, 148, 158, 0.6)";
    ctxEq.font = "9px monospace";
    ctxEq.fillText(`${db > 0 ? "+" : ""}${db}dB`, 4, y - 3);
  });

  // Draw Frequency Vertical Grid Lines (100Hz, 1kHz, 10kHz)
  [100, 1000, 10000].forEach((f) => {
    const x = freqToX(f, w);
    ctxEq.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctxEq.beginPath();
    ctxEq.moveTo(x, 0);
    ctxEq.lineTo(x, h);
    ctxEq.stroke();
  });

  // Draw Magnitude Response Curve
  if (curve && curve.freqs_hz && curve.gain_db && curve.freqs_hz.length > 0) {
    const freqs = curve.freqs_hz;
    const gains = curve.gain_db;

    const grad = ctxEq.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(118, 185, 0, 0.35)");
    grad.addColorStop(0.5, "rgba(118, 185, 0, 0.12)");
    grad.addColorStop(1, "rgba(118, 185, 0, 0.0)");

    ctxEq.beginPath();
    for (let i = 0; i < freqs.length; i++) {
      const x = freqToX(freqs[i], w);
      const y = gainToY(gains[i], h);
      if (i === 0) ctxEq.moveTo(x, y);
      else ctxEq.lineTo(x, y);
    }
    ctxEq.lineTo(w, gainToY(0, h));
    ctxEq.lineTo(0, gainToY(0, h));
    ctxEq.closePath();
    ctxEq.fillStyle = grad;
    ctxEq.fill();

    ctxEq.beginPath();
    for (let i = 0; i < freqs.length; i++) {
      const x = freqToX(freqs[i], w);
      const y = gainToY(gains[i], h);
      if (i === 0) ctxEq.moveTo(x, y);
      else ctxEq.lineTo(x, y);
    }
    ctxEq.strokeStyle = "#76B900";
    ctxEq.lineWidth = 2.5;
    ctxEq.shadowColor = "rgba(118, 185, 0, 0.6)";
    ctxEq.shadowBlur = 8;
    ctxEq.stroke();
    ctxEq.shadowBlur = 0;
  }

  // Draw Draggable Filter Node Handles
  state.eqBands.forEach((band, i) => {
    const nx = freqToX(band.freq, w);
    const ny = gainToY(band.gain, h);

    if (i === state.eqActiveBand) {
      ctxEq.beginPath();
      ctxEq.arc(nx, ny, 14, 0, 2 * Math.PI);
      ctxEq.fillStyle = "rgba(118, 185, 0, 0.25)";
      ctxEq.fill();
    }

    ctxEq.beginPath();
    ctxEq.arc(nx, ny, 7, 0, 2 * Math.PI);
    ctxEq.fillStyle = band.color || "#76B900";
    ctxEq.fill();
    ctxEq.lineWidth = 2;
    ctxEq.strokeStyle = "#FFFFFF";
    ctxEq.stroke();

    ctxEq.fillStyle = "#FFFFFF";
    ctxEq.font = "bold 9px monospace";
    ctxEq.textAlign = "center";
    ctxEq.textBaseline = "middle";
    ctxEq.fillText(`${i + 1}`, nx, ny);
  });
}

export function activateEqUI(enabled = true) {
  const toggleEqMaster = $("toggleEqMaster");
  const badgeEqState = $("badgeEqState");
  if (toggleEqMaster) {
    toggleEqMaster.checked = enabled;
  }
  if (badgeEqState) {
    badgeEqState.textContent = enabled ? "ACTIVE" : "BYPASS";
    badgeEqState.style.background = enabled ? "#76B900" : "";
    badgeEqState.style.color = enabled ? "#000" : "";
  }
}

export function updateEqCurveUI(curve) {
  if (state.eqIsDragging) return;
  state.latestEqCurve = curve;

  if (curve.bands && Array.isArray(curve.bands)) {
    curve.bands.forEach((b, i) => {
      if (state.eqBands[i]) {
        state.eqBands[i].freq = b.freq_hz;
        state.eqBands[i].gain = b.gain_db;
        state.eqBands[i].q = b.q;
        state.eqBands[i].type = b.type;
      }
      const elF = $(`eqBand${i}Freq`);
      const elG = $(`eqBand${i}Gain`);
      const elQ = $(`eqBand${i}Q`);
      if (elF) elF.textContent = b.freq_hz >= 1000 ? `${(b.freq_hz / 1000).toFixed(1)} kHz` : `${Math.round(b.freq_hz)} Hz`;
      if (elG) elG.textContent = `${b.gain_db > 0 ? "+" : ""}${b.gain_db.toFixed(1)} dB`;
      if (elQ) elQ.textContent = b.q.toFixed(2);
    });
  }

  const eqPresetBtns = $$(".btn-eq-preset");
  if (curve.preset) {
    eqPresetBtns.forEach((btn) => {
      if (btn.dataset.preset === curve.preset) btn.classList.add("active");
      else btn.classList.remove("active");
    });
  }

  if (curve.enabled !== undefined) {
    activateEqUI(curve.enabled);
  }

  drawEqCurve(curve);
}

export function initParametricEq() {
  canvasEqCurve = $("canvasEqCurve");
  if (canvasEqCurve) {
    ctxEq = canvasEqCurve.getContext("2d");
  }

  const toggleEqMaster = $("toggleEqMaster");
  const badgeEqState = $("badgeEqState");
  const eqPresetBtns = $$(".btn-eq-preset");
  const eqBandCards = $$(".eq-band-card");

  if (canvasEqCurve) {
    canvasEqCurve.addEventListener("mousedown", (e) => {
      const rect = canvasEqCurve.getBoundingClientRect();
      const scaleX = canvasEqCurve.width / rect.width;
      const scaleY = canvasEqCurve.height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      state.eqActiveBand = -1;
      let minD = 24.0;
      state.eqBands.forEach((band, i) => {
        const bx = freqToX(band.freq, canvasEqCurve.width);
        const by = gainToY(band.gain, canvasEqCurve.height);
        const d = Math.hypot(mx - bx, my - by);
        if (d < minD) {
          minD = d;
          state.eqActiveBand = i;
        }
      });

      if (state.eqActiveBand >= 0) {
        state.eqIsDragging = true;
        activateEqUI(true);
        eqBandCards.forEach((c, i) => {
          if (i === state.eqActiveBand) c.classList.add("selected");
          else c.classList.remove("selected");
        });
        drawEqCurve(state.latestEqCurve);
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.eqIsDragging || state.eqActiveBand < 0 || !canvasEqCurve) return;
      const rect = canvasEqCurve.getBoundingClientRect();
      const scaleX = canvasEqCurve.width / rect.width;
      const scaleY = canvasEqCurve.height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      const newF = Math.round(xToFreq(mx, canvasEqCurve.width));
      const newG = Math.round(yToGain(my, canvasEqCurve.height) * 10) / 10;

      state.eqBands[state.eqActiveBand].freq = newF;
      if (state.eqBands[state.eqActiveBand].type !== "high_pass") {
        state.eqBands[state.eqActiveBand].gain = newG;
      }

      const elF = $(`eqBand${state.eqActiveBand}Freq`);
      const elG = $(`eqBand${state.eqActiveBand}Gain`);
      if (elF) elF.textContent = newF >= 1000 ? `${(newF / 1000).toFixed(1)} kHz` : `${newF} Hz`;
      if (elG) elG.textContent = `${newG > 0 ? "+" : ""}${newG.toFixed(1)} dB`;

      drawEqCurve(state.latestEqCurve);

      wsClient.send({
        type: "set_eq_band",
        band_index: state.eqActiveBand,
        freq_hz: newF,
        gain_db: state.eqBands[state.eqActiveBand].gain,
      });
    });

    window.addEventListener("mouseup", () => {
      if (state.eqIsDragging) {
        state.eqIsDragging = false;
        if (state.eqActiveBand >= 0) {
          activateEqUI(true);
          configureEq({
            band_index: state.eqActiveBand,
            freq_hz: state.eqBands[state.eqActiveBand].freq,
            gain_db: state.eqBands[state.eqActiveBand].gain,
          })
            .then((data) => {
              if (data && data.curve) updateEqCurveUI(data.curve);
            })
            .catch(console.error);
        }
      }
    });
  }

  eqPresetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.dataset.preset;
      eqPresetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const isFlat = preset === "flat";
      activateEqUI(!isFlat);
      wsClient.send({ type: "set_eq_preset", preset });
      setEqPreset(preset)
        .then((data) => {
          if (data && data.curve) updateEqCurveUI(data.curve);
        })
        .catch(console.error);
    });
  });

  if (toggleEqMaster) {
    toggleEqMaster.addEventListener("change", () => {
      const en = toggleEqMaster.checked;
      if (badgeEqState) {
        badgeEqState.textContent = en ? "ACTIVE" : "BYPASS";
        badgeEqState.style.background = en ? "#76B900" : "";
        badgeEqState.style.color = en ? "#000" : "";
      }
      wsClient.send({ type: "set_eq_enabled", enabled: en });
      configureEq({ enabled: en }).catch(console.error);
    });
  }

  state.on("eq_updated", (curve) => {
    updateEqCurveUI(curve);
  });
}

/**
 * 60 FPS Real-Time HiDPI Oscilloscope Renderer (Overlay, Split, Difference Shading)
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { CONFIG } from "../config.js";
import { $, $$ } from "../utils/dom.js";

let canvasOsc = null;
let ctxOsc = null;

export function initOscilloscope() {
  canvasOsc = $("canvasOscilloscope");
  if (canvasOsc) {
    ctxOsc = canvasOsc.getContext("2d");
  }

  const btnOscOverlay = $("btnOscOverlay");
  const btnOscSplit = $("btnOscSplit");
  const btnOscDiff = $("btnOscDiff");
  const tbButtons = $$(".btn-tb");
  const pillChanIn = $("pillChanIn");
  const pillChanOut = $("pillChanOut");
  const pillChanSub = $("pillChanSub");
  const tagChanIn = $("tagChanIn");
  const tagChanOut = $("tagChanOut");
  const tagChanSub = $("tagChanSub");

  if (btnOscOverlay) {
    btnOscOverlay.addEventListener("click", () => {
      btnOscOverlay.classList.add("active");
      if (btnOscSplit) btnOscSplit.classList.remove("active");
      state.oscMode = "overlay";
    });
  }

  if (btnOscSplit) {
    btnOscSplit.addEventListener("click", () => {
      btnOscSplit.classList.add("active");
      if (btnOscOverlay) btnOscOverlay.classList.remove("active");
      state.oscMode = "split";
    });
  }

  if (btnOscDiff) {
    btnOscDiff.addEventListener("click", () => {
      state.diffShading = !state.diffShading;
      btnOscDiff.classList.toggle("active", state.diffShading);
    });
  }

  function setupChannelToggle(pill, tag, key) {
    if (!pill) return;
    pill.addEventListener("click", () => {
      state.chanVisible[key] = !state.chanVisible[key];
      pill.classList.toggle("inactive", !state.chanVisible[key]);
      if (tag) tag.textContent = state.chanVisible[key] ? "ON" : "OFF";
    });
  }

  setupChannelToggle(pillChanIn, tagChanIn, "in");
  setupChannelToggle(pillChanOut, tagChanOut, "out");
  setupChannelToggle(pillChanSub, tagChanSub, "sub");

  tbButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tbButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.oscWindowSamples = parseInt(btn.dataset.window, 10);
    });
  });
}

export function drawReticleGrid(w, h, centerY, ampHeight, ctx = ctxOsc) {
  if (!ctx) return;
  ctx.save();
  ctx.strokeStyle = "rgba(0, 229, 255, 0.08)";
  ctx.lineWidth = 1;

  // Horizontal grid lines (+1.0, +0.5, 0.0, -0.5, -1.0)
  const divisions = [-1.0, -0.5, 0.5, 1.0];
  divisions.forEach((d) => {
    const y = centerY - d * ampHeight;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  });

  // Zero-axis glowing reference line
  ctx.strokeStyle = "rgba(118, 185, 0, 0.32)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(0, centerY);
  ctx.lineTo(w, centerY);
  ctx.stroke();

  // Vertical time division tick marks
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  const numDivs = 8;
  for (let i = 1; i < numDivs; i++) {
    const x = (i / numDivs) * w;
    ctx.beginPath();
    ctx.moveTo(x, centerY - ampHeight);
    ctx.lineTo(x, centerY + ampHeight);
    ctx.stroke();
  }
  ctx.restore();
}

export function drawDifferenceArea(bufNoisy, bufClean, startIdx, numSamples, w, centerY, ampHeight, ctx = ctxOsc) {
  if (!ctx) return;
  ctx.save();
  ctx.beginPath();
  const step = w / (numSamples - 1);

  // Forward along Noisy Input
  for (let i = 0; i < numSamples; i++) {
    const s = bufNoisy[startIdx + i];
    const x = i * step;
    const y = centerY - s * ampHeight;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  // Backward along Clean Output
  for (let i = numSamples - 1; i >= 0; i--) {
    const s = bufClean[startIdx + i];
    const x = i * step;
    const y = centerY - s * ampHeight;
    ctx.lineTo(x, y);
  }

  ctx.closePath();
  ctx.fillStyle = "rgba(255, 82, 82, 0.22)";
  ctx.fill();
  ctx.restore();
}

export function drawWaveformPath(buf, startIdx, numSamples, w, centerY, ampHeight, color, glowColor, lineWidth, ctx = ctxOsc) {
  if (!ctx) return;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = 8;
  ctx.lineJoin = "round";

  const step = w / (numSamples - 1);
  for (let i = 0; i < numSamples; i++) {
    const s = buf[startIdx + i];
    const y = centerY - s * ampHeight;
    const x = i * step;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function renderOverlayOscilloscope(w, h, startIdx, numSamples) {
  const centerY = h / 2;
  const ampHeight = (h / 2) * 0.88;

  drawReticleGrid(w, h, centerY, ampHeight, ctxOsc);

  // 0. Difference Area Shading
  if (state.diffShading && state.chanVisible.in && state.chanVisible.out) {
    drawDifferenceArea(state.oscBufNoisy, state.oscBufClean, startIdx, numSamples, w, centerY, ampHeight, ctxOsc);
  }

  // 1. Subtracted Noise Delta
  if (state.chanVisible.sub) {
    drawWaveformPath(
      state.oscBufNoise,
      startIdx,
      numSamples,
      w,
      centerY,
      ampHeight,
      "#b388ff",
      "rgba(179, 136, 255, 0.55)",
      1.6,
      ctxOsc
    );
  }

  // 2. Chaotic Noisy Input
  if (state.chanVisible.in) {
    drawWaveformPath(
      state.oscBufNoisy,
      startIdx,
      numSamples,
      w,
      centerY,
      ampHeight,
      "#ff5252",
      "rgba(255, 82, 82, 0.65)",
      2.0,
      ctxOsc
    );
  }

  // 3. Clean Crystal Denoised Output
  if (state.chanVisible.out) {
    drawWaveformPath(
      state.oscBufClean,
      startIdx,
      numSamples,
      w,
      centerY,
      ampHeight,
      "#76b900",
      "rgba(118, 185, 0, 0.9)",
      2.5,
      ctxOsc
    );
  }

  // Channel indicator tags
  ctxOsc.font = "700 11px 'JetBrains Mono', monospace";
  let tagX = 18;
  if (state.chanVisible.in) {
    ctxOsc.fillStyle = "rgba(255, 82, 82, 0.9)";
    ctxOsc.fillText("● x[n] NOISY INPUT", tagX, 24);
    tagX += 165;
  }
  if (state.chanVisible.out) {
    ctxOsc.fillStyle = "rgba(118, 185, 0, 0.95)";
    ctxOsc.fillText("● y[n] DENOISED VOICE", tagX, 24);
    tagX += 180;
  }
  if (state.chanVisible.sub) {
    ctxOsc.fillStyle = "rgba(179, 136, 255, 0.9)";
    ctxOsc.fillText("● Δ SUBTRACTED NOISE", tagX, 24);
  }
}

function renderSplitOscilloscope(w, h, startIdx, numSamples) {
  const trackH = h / 3;

  // Track 1: Noisy Input
  if (state.chanVisible.in) {
    ctxOsc.save();
    ctxOsc.beginPath();
    ctxOsc.rect(0, 0, w, trackH);
    ctxOsc.clip();
    drawReticleGrid(w, trackH, trackH * 0.5, trackH * 0.42, ctxOsc);
    drawWaveformPath(
      state.oscBufNoisy,
      startIdx,
      numSamples,
      w,
      trackH * 0.5,
      trackH * 0.42,
      "#ff5252",
      "rgba(255, 82, 82, 0.65)",
      1.8,
      ctxOsc
    );
    ctxOsc.font = "700 10px 'JetBrains Mono', monospace";
    ctxOsc.fillStyle = "#ff5252";
    ctxOsc.fillText("CH 1: NOISY INPUT x[n] (MUDDY CHAOS)", 16, 16);
    ctxOsc.restore();
  }

  // Track 2: Denoised Voice
  if (state.chanVisible.out) {
    ctxOsc.save();
    ctxOsc.beginPath();
    ctxOsc.rect(0, trackH, w, trackH);
    ctxOsc.clip();
    drawReticleGrid(w, trackH, trackH * 1.5, trackH * 0.42, ctxOsc);
    drawWaveformPath(
      state.oscBufClean,
      startIdx,
      numSamples,
      w,
      trackH * 1.5,
      trackH * 0.42,
      "#76b900",
      "rgba(118, 185, 0, 0.9)",
      2.0,
      ctxOsc
    );
    ctxOsc.font = "700 10px 'JetBrains Mono', monospace";
    ctxOsc.fillStyle = "#76b900";
    ctxOsc.fillText("CH 2: DENOISED SPEECH y[n] (CRYSTAL VOICE)", 16, trackH + 16);
    ctxOsc.restore();
  }

  // Track 3: Subtracted Noise Delta
  if (state.chanVisible.sub) {
    ctxOsc.save();
    ctxOsc.beginPath();
    ctxOsc.rect(0, trackH * 2, w, trackH);
    ctxOsc.clip();
    drawReticleGrid(w, trackH, trackH * 2.5, trackH * 0.42, ctxOsc);
    drawWaveformPath(
      state.oscBufNoise,
      startIdx,
      numSamples,
      w,
      trackH * 2.5,
      trackH * 0.42,
      "#b388ff",
      "rgba(179, 136, 255, 0.65)",
      1.8,
      ctxOsc
    );
    ctxOsc.font = "700 10px 'JetBrains Mono', monospace";
    ctxOsc.fillStyle = "#b388ff";
    ctxOsc.fillText("CH 3: SUBTRACTED NOISE DELTA x[n] - y[n] (GARBAGE REMOVED)", 16, trackH * 2 + 16);
    ctxOsc.restore();
  }

  // Track separator lines
  ctxOsc.save();
  ctxOsc.strokeStyle = "rgba(35, 48, 62, 0.8)";
  ctxOsc.lineWidth = 1;
  ctxOsc.beginPath();
  ctxOsc.moveTo(0, trackH);
  ctxOsc.lineTo(w, trackH);
  ctxOsc.moveTo(0, trackH * 2);
  ctxOsc.lineTo(w, trackH * 2);
  ctxOsc.stroke();
  ctxOsc.restore();
}

export function renderOscilloscope() {
  if (!canvasOsc || !ctxOsc) {
    canvasOsc = $("canvasOscilloscope");
    if (canvasOsc) ctxOsc = canvasOsc.getContext("2d");
    else return;
  }

  const dpr = window.devicePixelRatio || 1;
  const rect = canvasOsc.getBoundingClientRect();
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);
  if (targetW > 0 && targetH > 0 && (canvasOsc.width !== targetW || canvasOsc.height !== targetH)) {
    canvasOsc.width = targetW;
    canvasOsc.height = targetH;
  }

  const w = canvasOsc.width;
  const h = canvasOsc.height;
  ctxOsc.clearRect(0, 0, w, h);

  const numSamples = state.oscWindowSamples;
  const startIdx = CONFIG.OSC_BUF_LEN - numSamples;

  if (state.oscMode === "overlay") {
    renderOverlayOscilloscope(w, h, startIdx, numSamples);
  } else {
    renderSplitOscilloscope(w, h, startIdx, numSamples);
  }
}

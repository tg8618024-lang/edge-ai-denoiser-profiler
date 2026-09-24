/**
 * 60 FPS Dual STFT Waterfall & 3D Isometric Spectrogram Visualizer
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, safeText } from "../utils/dom.js";

let canvasBefore = null;
let ctxBefore = null;
let canvasAfter = null;
let ctxAfter = null;

export function initSpectrograms() {
  canvasBefore = $("canvasBefore");
  if (canvasBefore) {
    ctxBefore = canvasBefore.getContext("2d", { willReadFrequently: true });
  }
  canvasAfter = $("canvasAfter");
  if (canvasAfter) {
    ctxAfter = canvasAfter.getContext("2d", { willReadFrequently: true });
  }

  const btnSpec3D = $("btnSpec3D");
  const specModeBadge = $("specModeBadge");

  if (btnSpec3D) {
    btnSpec3D.addEventListener("click", () => {
      state.isSpec3D = !state.isSpec3D;
      btnSpec3D.classList.toggle("active", state.isSpec3D);
      if (specModeBadge) {
        safeText(specModeBadge, state.isSpec3D ? "3D RIDGES 60 FPS" : "CANVAS 60 FPS");
      }
      if (ctxBefore && canvasBefore) {
        ctxBefore.clearRect(0, 0, canvasBefore.width, canvasBefore.height);
      }
      if (ctxAfter && canvasAfter) {
        ctxAfter.clearRect(0, 0, canvasAfter.width, canvasAfter.height);
      }
    });
  }
}

export function getCyberpunkColor(val) {
  const v = Math.min(1.0, Math.max(0.0, val));
  if (v < 0.15) {
    const t = v / 0.15;
    return [Math.floor(4 * t), Math.floor(18 * t), Math.floor(8 * t)];
  } else if (v < 0.45) {
    const t = (v - 0.15) / 0.30;
    return [
      Math.floor(4 + (118 - 4) * t),
      Math.floor(18 + (185 - 18) * t),
      Math.floor(8 * (1 - t)),
    ];
  } else if (v < 0.80) {
    const t = (v - 0.45) / 0.35;
    return [
      Math.floor(118 + (210 - 118) * t),
      Math.floor(185 + (255 - 185) * t),
      Math.floor(20 * t),
    ];
  } else {
    const t = (v - 0.80) / 0.20;
    return [
      Math.floor(210 + (255 - 210) * t),
      255,
      Math.floor(20 + (255 - 20) * t),
    ];
  }
}

export function drawWaterfall(ctx, canvas, spec) {
  if (!ctx || !canvas || !spec) return;
  const w = canvas.width;
  const h = canvas.height;
  const shift = 2; // Shift down 2 pixels

  ctx.drawImage(canvas, 0, 0, w, h - shift, 0, shift, w, h - shift);

  const imgData = ctx.createImageData(w, shift);
  const data = imgData.data;
  const numBins = spec.length;

  for (let x = 0; x < w; x++) {
    const binIdx = Math.floor((x / w) * numBins);
    const mag = spec[binIdx] || 0.0;
    const norm = Math.min(1.0, Math.max(0.0, (Math.log10(mag + 1e-4) + 3.0) / 3.2));
    const [r, g, b] = getCyberpunkColor(norm);

    for (let y = 0; y < shift; y++) {
      const idx = (y * w + x) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

export function draw3DSpectrogram(ctx, canvas, history, isClean) {
  if (!ctx || !canvas) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
  bgGrad.addColorStop(0, "#040608");
  bgGrad.addColorStop(1, "#0a0f15");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  if (!history || history.length === 0) return;

  const total = history.length;
  for (let i = total - 1; i >= 0; i--) {
    const slice = history[i];
    const z = i / Math.max(1, total - 1); // 0 = front, 1 = back
    const scale = 1.0 - z * 0.40;
    const sliceW = w * 0.90 * scale;
    const startX = (w - sliceW) / 2;
    const baseY = h * 0.88 - z * (h * 0.65);
    const numBins = slice.length;
    const stepX = sliceW / Math.max(1, numBins - 1);

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(startX, baseY + 6);

    const points = [];
    for (let b = 0; b < numBins; b++) {
      const mag = slice[b] || 0.0;
      const norm = Math.min(1.0, Math.max(0.0, (Math.log10(mag + 1e-4) + 3.0) / 3.2));
      const elev = norm * (h * 0.32) * scale;
      const px = startX + b * stepX;
      const py = baseY - elev;
      points.push({ x: px, y: py });
      ctx.lineTo(px, py);
    }
    ctx.lineTo(startX + sliceW, baseY + 6);
    ctx.closePath();

    ctx.fillStyle = `rgba(5, 8, 12, ${0.85 - z * 0.15})`;
    ctx.fill();

    ctx.beginPath();
    for (let b = 0; b < points.length; b++) {
      if (b === 0) ctx.moveTo(points[b].x, points[b].y);
      else ctx.lineTo(points[b].x, points[b].y);
    }
    const alpha = 1.0 - z * 0.55;
    if (isClean) {
      ctx.strokeStyle = `rgba(118, 185, 0, ${alpha})`;
      ctx.shadowColor = "rgba(118, 185, 0, 0.4)";
    } else {
      ctx.strokeStyle = `rgba(255, 128, 0, ${alpha})`;
      ctx.shadowColor = "rgba(255, 128, 0, 0.4)";
    }
    ctx.shadowBlur = 4;
    ctx.lineWidth = 1.6 * scale;
    ctx.stroke();
    ctx.restore();
  }
}

export function renderSpectrograms() {
  if (state.isSpec3D) {
    draw3DSpectrogram(ctxBefore, canvasBefore, state.specHistoryIn, false);
    draw3DSpectrogram(ctxAfter, canvasAfter, state.specHistoryOut, true);
  } else if (state.hasNewSpecFrame) {
    drawWaterfall(ctxBefore, canvasBefore, state.lastSpecIn);
    drawWaterfall(ctxAfter, canvasAfter, state.lastSpecOut);
    state.hasNewSpecFrame = false;
  }
}

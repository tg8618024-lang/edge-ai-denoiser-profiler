/**
 * Polar Lissajous Vectorscope Radar & Phase Correlation Analyzer
 * Zero Global Scope Pollution
 */

import { $, safeText } from "../utils/dom.js";

let canvasVectorscope = null;
let ctxVectorscope = null;

export function initVectorscope() {
  canvasVectorscope = $("canvasVectorscope");
  if (canvasVectorscope) {
    ctxVectorscope = canvasVectorscope.getContext("2d");
  }
}

export function drawVectorscope(vData) {
  if (!canvasVectorscope) canvasVectorscope = $("canvasVectorscope");
  if (!canvasVectorscope) return;
  if (!ctxVectorscope) ctxVectorscope = canvasVectorscope.getContext("2d");
  if (!ctxVectorscope) return;

  const w = canvasVectorscope.width;
  const h = canvasVectorscope.height;
  const cx = w / 2;
  const cy = h / 2;
  const R = w * 0.44;

  ctxVectorscope.fillStyle = "#06090E";
  ctxVectorscope.fillRect(0, 0, w, h);

  // Polar Radar Graticules
  ctxVectorscope.strokeStyle = "rgba(0, 229, 255, 0.20)";
  ctxVectorscope.lineWidth = 1;
  [0.33, 0.66, 1.0].forEach((rFrac) => {
    ctxVectorscope.beginPath();
    ctxVectorscope.arc(cx, cy, R * rFrac, 0, 2 * Math.PI);
    ctxVectorscope.stroke();
  });

  // 45-degree Diagonal Mid/Side Reticle Axes
  ctxVectorscope.strokeStyle = "rgba(0, 229, 255, 0.15)";
  ctxVectorscope.beginPath();
  // Vertical Mid axis (M)
  ctxVectorscope.moveTo(cx, cy - R);
  ctxVectorscope.lineTo(cx, cy + R);
  // Horizontal Side axis (S)
  ctxVectorscope.moveTo(cx - R, cy);
  ctxVectorscope.lineTo(cx + R, cy);
  // 45-deg Left (L) and Right (R) diagonals
  const dR = R * 0.7071;
  ctxVectorscope.moveTo(cx - dR, cy - dR);
  ctxVectorscope.lineTo(cx + dR, cy + dR);
  ctxVectorscope.moveTo(cx - dR, cy + dR);
  ctxVectorscope.lineTo(cx + dR, cy - dR);
  ctxVectorscope.stroke();

  // Plot Lissajous Orbit Points
  if (vData && vData.orbit_x && vData.orbit_y && vData.orbit_x.length > 0) {
    const ox = vData.orbit_x;
    const oy = vData.orbit_y;
    const corr = vData.phase_correlation !== undefined ? vData.phase_correlation : 1.0;

    let orbitColor = "#00E5FF";
    if (corr > 0.65) orbitColor = "#76B900"; // Solid in-phase
    else if (corr > 0.20) orbitColor = "#00E5FF"; // Normal stereo
    else if (corr > -0.50) orbitColor = "#FFB300"; // Wide stereo
    else orbitColor = "#FF5252"; // Phase cancellation

    ctxVectorscope.strokeStyle = orbitColor;
    ctxVectorscope.lineWidth = 1.8;
    ctxVectorscope.shadowColor = orbitColor;
    ctxVectorscope.shadowBlur = 6;

    ctxVectorscope.beginPath();
    for (let i = 0; i < ox.length; i++) {
      const px = cx + ox[i] * R * 0.90;
      const py = cy - oy[i] * R * 0.90;
      if (i === 0) ctxVectorscope.moveTo(px, py);
      else ctxVectorscope.lineTo(px, py);
    }
    ctxVectorscope.stroke();

    ctxVectorscope.fillStyle = "#FFFFFF";
    for (let i = 0; i < ox.length; i += 4) {
      const px = cx + ox[i] * R * 0.90;
      const py = cy - oy[i] * R * 0.90;
      ctxVectorscope.fillRect(px - 1, py - 1, 2, 2);
    }
    ctxVectorscope.shadowBlur = 0;
  }
}

export function updateVectorscopeUI(vData) {
  if (!vData) return;
  drawVectorscope(vData);

  const phaseCorrVal = $("phaseCorrVal");
  const phaseBarFill = $("phaseBarFill");
  const monoCompatVal = $("monoCompatVal");
  const monoCompatBar = $("monoCompatBar");
  const stereoWidthVal = $("stereoWidthVal");
  const spatialFieldVal = $("spatialFieldVal");
  const vectorscopeStatusBadge = $("vectorscopeStatusBadge");

  const corr = vData.phase_correlation !== undefined ? vData.phase_correlation : 1.0;
  const monoPct = vData.mono_compatibility_pct !== undefined ? vData.mono_compatibility_pct : 100.0;
  const width = vData.stereo_width !== undefined ? vData.stereo_width : 0.0;

  if (phaseCorrVal) {
    safeText(phaseCorrVal, `${corr >= 0 ? "+" : ""}${corr.toFixed(2)}`);
  }
  if (phaseBarFill) {
    const pct = Math.min(100, Math.max(0, ((corr + 1.0) / 2.0) * 100));
    phaseBarFill.style.width = `${pct}%`;
  }

  if (monoCompatVal) {
    safeText(monoCompatVal, `${monoPct.toFixed(1)}%`);
  }
  if (monoCompatBar) {
    monoCompatBar.style.width = `${Math.min(100, Math.max(0, monoPct))}%`;
  }

  if (stereoWidthVal) {
    safeText(stereoWidthVal, width.toFixed(2));
  }
  if (spatialFieldVal) {
    if (width < 0.25) safeText(spatialFieldVal, "Focused Center");
    else if (width < 0.75) safeText(spatialFieldVal, "Natural Spatial");
    else safeText(spatialFieldVal, "Wide Ambient");
  }

  if (vectorscopeStatusBadge) {
    if (corr > 0.65) {
      vectorscopeStatusBadge.textContent = "🟢 ROCK SOLID PHASE (+1.0)";
      vectorscopeStatusBadge.className = "status-tag tag-good";
    } else if (corr > 0.20) {
      vectorscopeStatusBadge.textContent = "🟢 ACCEPTABLE PHASE";
      vectorscopeStatusBadge.className = "status-tag tag-good";
    } else if (corr > -0.50) {
      vectorscopeStatusBadge.textContent = "🟡 STEREO WIDE (0.0)";
      vectorscopeStatusBadge.className = "status-tag tag-warning";
    } else {
      vectorscopeStatusBadge.textContent = "🔴 PHASE INVERTED / CANCELLATION";
      vectorscopeStatusBadge.className = "status-tag tag-bad";
    }
  }
}

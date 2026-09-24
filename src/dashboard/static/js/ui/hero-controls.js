/**
 * Hero Storyboard Controls, Magic Denoise & Purity Dial
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { CONFIG } from "../config.js";
import { $, safeText } from "../utils/dom.js";
import { audioManager } from "../audio/audio-manager.js";

export function updatePurityDial(purityPct, noiseErased) {
  const purityPctText = $("purityPctText");
  const noiseErasedText = $("noiseErasedText");
  const purityGaugeBar = $("purityGaugeBar");

  if (purityPctText) safeText(purityPctText, `${purityPct.toFixed(1)}%`);
  if (noiseErasedText) safeText(noiseErasedText, `Noise Erased: ${noiseErased.toFixed(1)}%`);

  const purityCircumference = CONFIG.CIRCUMFERENCE_PURITY;
  const purityOffset = purityCircumference * (1.0 - Math.min(100.0, Math.max(0.0, purityPct)) / 100.0);
  if (purityGaugeBar) purityGaugeBar.style.strokeDashoffset = purityOffset;
}

export function updateVuMeters(rmsIn = 0.245, rmsOut = 0.128, rmsNoise = 0.182) {
  const vuIn = $("vuIn");
  const vuOut = $("vuOut");
  const vuNoise = $("vuNoise");

  if (vuIn) vuIn.style.width = `${Math.min(100, Math.max(10, rmsIn * 220)).toFixed(0)}%`;
  if (vuOut) vuOut.style.width = `${Math.min(100, Math.max(10, rmsOut * 220)).toFixed(0)}%`;
  if (vuNoise) vuNoise.style.width = `${Math.min(100, Math.max(10, rmsNoise * 220)).toFixed(0)}%`;
}

export function syncABUi(cleanSelected) {
  const labelA = $("labelA");
  const labelB = $("labelB");
  const abToggleCheckbox = $("abToggleCheckbox");
  const magicDenoiseBtn = $("magicDenoiseBtn");
  const magicStateText = $("magicStateText");
  const magicSubText = $("magicSubText");
  const crossfadePill = $("crossfadePill");
  const cfNoisyLabel = $("cfNoisyLabel");
  const cfCleanLabel = $("cfCleanLabel");
  const purityStatusBadge = $("purityStatusBadge");

  if (cleanSelected) {
    if (labelB) labelB.classList.add("active");
    if (labelA) labelA.classList.remove("active");
    if (abToggleCheckbox) abToggleCheckbox.checked = true;

    if (magicDenoiseBtn) {
      magicDenoiseBtn.classList.remove("bypassed");
      magicDenoiseBtn.classList.add("active");
    }
    if (magicStateText) magicStateText.textContent = "MAGIC AI DENOISE: ACTIVE";
    if (magicSubText) magicSubText.textContent = "Listening to 100% Clean Crystal Voice";

    if (crossfadePill) {
      crossfadePill.classList.remove("bypassed");
      crossfadePill.style.left = "calc(100% - 28px)";
    }
    if (cfCleanLabel) cfCleanLabel.classList.add("active");
    if (cfNoisyLabel) cfNoisyLabel.classList.remove("active");

    if (purityStatusBadge) {
      purityStatusBadge.innerHTML = "&#128142; CRYSTAL BROADCAST VOICE";
      purityStatusBadge.className = "purity-status-badge text-accent";
    }
  } else {
    if (labelA) labelA.classList.add("active");
    if (labelB) labelB.classList.remove("active");
    if (abToggleCheckbox) abToggleCheckbox.checked = false;

    if (magicDenoiseBtn) {
      magicDenoiseBtn.classList.remove("active");
      magicDenoiseBtn.classList.add("bypassed");
    }
    if (magicStateText) magicStateText.textContent = "RAW DIRTY AUDIO (UNFILTERED)";
    if (magicSubText) magicSubText.textContent = "Warning: Listening to Loud Background Mud & Noise";

    if (crossfadePill) {
      crossfadePill.classList.add("bypassed");
      crossfadePill.style.left = "2px";
    }
    if (cfNoisyLabel) cfNoisyLabel.classList.add("active");
    if (cfCleanLabel) cfCleanLabel.classList.remove("active");

    if (purityStatusBadge) {
      purityStatusBadge.innerHTML = "&#9888;&#65039; RAW NOISY INPUT (BYPASS)";
      purityStatusBadge.className = "purity-status-badge text-warning";
    }
  }
}

export function initHeroControls({ onStartStreamIfNeeded }) {
  const labelA = $("labelA");
  const labelB = $("labelB");
  const cfNoisyLabel = $("cfNoisyLabel");
  const cfCleanLabel = $("cfCleanLabel");
  const crossfadeTrack = $("crossfadeTrack");
  const magicDenoiseBtn = $("magicDenoiseBtn");
  const abToggleCheckbox = $("abToggleCheckbox");
  const volumeSlider = $("volumeSlider");
  const volumeValue = $("volumeValue");

  function setABMode(cleanSelected) {
    audioManager.init();
    audioManager.setABMode(cleanSelected);
    syncABUi(cleanSelected);
  }

  if (labelA) labelA.addEventListener("click", () => setABMode(false));
  if (labelB) labelB.addEventListener("click", () => setABMode(true));
  if (cfNoisyLabel) cfNoisyLabel.addEventListener("click", () => setABMode(false));
  if (cfCleanLabel) cfCleanLabel.addEventListener("click", () => setABMode(true));
  if (crossfadeTrack) crossfadeTrack.addEventListener("click", () => setABMode(!state.isCleanActive));

  if (magicDenoiseBtn) {
    magicDenoiseBtn.addEventListener("click", () => {
      if (!state.isStreaming && !state.micActive) {
        if (onStartStreamIfNeeded) onStartStreamIfNeeded();
        setABMode(true);
      } else {
        setABMode(!state.isCleanActive);
      }
    });
  }

  if (abToggleCheckbox) {
    abToggleCheckbox.addEventListener("change", (e) => {
      setABMode(e.target.checked);
    });
  }

  if (volumeSlider) {
    volumeSlider.addEventListener("input", (e) => {
      audioManager.init();
      const val = parseInt(e.target.value, 10);
      if (volumeValue) volumeValue.textContent = `${val}%`;
      audioManager.setVolume(val);
    });
  }

  state.on("ab_mode_changed", (cleanSelected) => {
    syncABUi(cleanSelected);
  });
}

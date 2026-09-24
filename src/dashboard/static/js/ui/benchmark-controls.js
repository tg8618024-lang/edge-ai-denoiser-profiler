/**
 * Benchmark Controls, Precision Switcher, Presets, Dual-Model & WAV Upload
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { $, $$, safeText } from "../utils/dom.js";
import { audioManager } from "../audio/audio-manager.js";
import { wsClient } from "../net/websocket.js";
import { updateSettings, uploadWav } from "../net/api.js";
import { updatePurityDial } from "./hero-controls.js";

let ctxUploadPreview = null;
let canvasUploadPreview = null;

export function setStreamingState(active, micStreamer = null) {
  audioManager.init();
  if (audioManager.audioCtx && audioManager.audioCtx.state === "suspended") {
    audioManager.audioCtx.resume().catch(() => {});
  }

  if (active && state.micActive && micStreamer) {
    micStreamer.stopMicrophone();
  }

  state.isStreaming = active;
  const btnStreamToggle = $("btnStreamToggle");
  const btnHeroStream = $("btnHeroStream");
  const heroStreamIcon = $("heroStreamIcon");
  const heroStreamText = $("heroStreamText");

  if (state.isStreaming) {
    if (btnStreamToggle) {
      btnStreamToggle.innerHTML = `<span class="icon">&#9632;</span> STOP BENCHMARK STREAM`;
      btnStreamToggle.classList.add("btn-warning");
    }
    if (btnHeroStream) {
      if (heroStreamIcon) heroStreamIcon.innerHTML = "&#9632;";
      if (heroStreamText) heroStreamText.textContent = "STOP SOUND DEMO";
      btnHeroStream.classList.add("btn-warning");
    }
    wsClient.send({
      type: "start_benchmark",
      preset: state.activePreset,
      target_snr: state.activeTargetSnr,
    });
  } else {
    if (btnStreamToggle) {
      btnStreamToggle.innerHTML = `<span class="icon">&#9658;</span> START BENCHMARK STREAM`;
      btnStreamToggle.classList.remove("btn-warning");
    }
    if (btnHeroStream) {
      if (heroStreamIcon) heroStreamIcon.innerHTML = "&#9658;";
      if (heroStreamText) heroStreamText.textContent = "START SOUND DEMO";
      btnHeroStream.classList.remove("btn-warning");
    }
    wsClient.send({ type: "stop_benchmark" });
  }
}

export function updateIntensity(val) {
  state.currentIntensity = Math.min(1.0, Math.max(0.0, val));
  const pct = Math.round(state.currentIntensity * 100);

  const suppressionSlider = $("suppressionSlider");
  const suppressionValue = $("suppressionValue");
  const intensityChips = $$(".btn-chip");

  if (suppressionSlider) suppressionSlider.value = pct;

  let desc = "CUSTOM";
  if (pct <= 25) desc = "NATURAL AMBIENCE";
  else if (pct <= 50) desc = "BALANCED FILTER";
  else if (pct <= 75) desc = "STUDIO ISOLATION";
  else desc = "MAXIMUM ISOLATION";

  if (suppressionValue) {
    safeText(suppressionValue, `${pct}% (${desc})`);
  }

  intensityChips.forEach((chip) => {
    const chipVal = parseFloat(chip.dataset.intensity);
    if (Math.abs(chipVal - state.currentIntensity) < 0.05) {
      chip.classList.add("active");
    } else {
      chip.classList.remove("active");
    }
  });

  updateSettings({ intensity: state.currentIntensity }).catch(() => {});
  wsClient.send({ type: "set_intensity", intensity: state.currentIntensity });
}

export function renderUploadPreview(noisyArr, cleanArr) {
  if (!canvasUploadPreview) canvasUploadPreview = $("canvasUploadPreview");
  if (!canvasUploadPreview) return;
  if (!ctxUploadPreview) ctxUploadPreview = canvasUploadPreview.getContext("2d");
  if (!ctxUploadPreview) return;

  const w = canvasUploadPreview.width;
  const h = canvasUploadPreview.height;
  ctxUploadPreview.fillStyle = "#06090d";
  ctxUploadPreview.fillRect(0, 0, w, h);

  const mid = h / 2;
  ctxUploadPreview.strokeStyle = "rgba(35, 48, 62, 0.6)";
  ctxUploadPreview.lineWidth = 1;
  ctxUploadPreview.beginPath();
  ctxUploadPreview.moveTo(0, mid);
  ctxUploadPreview.lineTo(w, mid);
  ctxUploadPreview.stroke();

  const len = noisyArr.length;
  ctxUploadPreview.strokeStyle = "rgba(255, 82, 82, 0.6)";
  ctxUploadPreview.lineWidth = 1.5;
  ctxUploadPreview.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (i / (len - 1)) * w;
    const y = mid - noisyArr[i] * (h * 0.44);
    if (i === 0) ctxUploadPreview.moveTo(x, y);
    else ctxUploadPreview.lineTo(x, y);
  }
  ctxUploadPreview.stroke();

  ctxUploadPreview.strokeStyle = "#88d400";
  ctxUploadPreview.lineWidth = 1.5;
  ctxUploadPreview.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (i / (len - 1)) * w;
    const y = mid - cleanArr[i] * (h * 0.44);
    if (i === 0) ctxUploadPreview.moveTo(x, y);
    else ctxUploadPreview.lineTo(x, y);
  }
  ctxUploadPreview.stroke();
}

function playUploadedChannel(channel) {
  if (!state.uploadedAudioId) return;
  if (state.uploadedAudioPlayer) {
    state.uploadedAudioPlayer.pause();
    state.uploadedAudioPlayer = null;
  }
  state.uploadedAudioPlayer = new Audio(`/api/audio/download/${state.uploadedAudioId}?channel=${channel}`);
  state.uploadedAudioPlayer.play().catch(() => {});
}

async function handleFileUpload(file) {
  if (!file.name.toLowerCase().endsWith(".wav")) {
    alert("Please select a valid .wav audio file.");
    return;
  }

  const uploadDropzone = $("uploadDropzone");
  const uploadStatusCard = $("uploadStatusCard");
  const uploadFileName = $("uploadFileName");
  const uploadFileStatus = $("uploadFileStatus");
  const uploadMetaDuration = $("uploadMetaDuration");
  const uploadMetaSamples = $("uploadMetaSamples");
  const uploadMetaPurity = $("uploadMetaPurity");
  const uploadMetaSnr = $("uploadMetaSnr");

  if (uploadDropzone) uploadDropzone.classList.add("hidden");
  if (uploadStatusCard) uploadStatusCard.classList.remove("hidden");
  if (uploadFileName) uploadFileName.textContent = file.name;
  if (uploadFileStatus) {
    uploadFileStatus.textContent = "AI PROFILING...";
    uploadFileStatus.className = "status-tag tag-ai";
  }

  try {
    const data = await uploadWav(file, state.currentIntensity);
    state.uploadedAudioId = data.audio_id;
    if (uploadFileStatus) {
      uploadFileStatus.textContent = "PROCESSED & READY";
      uploadFileStatus.className = "status-tag tag-good";
    }
    if (uploadMetaDuration) uploadMetaDuration.textContent = `${data.duration_sec}s`;
    if (uploadMetaSamples) uploadMetaSamples.textContent = `${data.num_samples.toLocaleString()}`;
    if (uploadMetaPurity) uploadMetaPurity.textContent = `${data.metrics.purity_pct}%`;
    if (uploadMetaSnr) uploadMetaSnr.textContent = `+${data.metrics.snr_delta_db} dB`;

    updatePurityDial(data.metrics.purity_pct, data.metrics.noise_erased_pct);

    if (data.preview && data.preview.noisy && data.preview.denoised) {
      renderUploadPreview(data.preview.noisy, data.preview.denoised);
    }
  } catch (err) {
    if (uploadFileStatus) {
      uploadFileStatus.textContent = "ERROR: " + err.message;
      uploadFileStatus.className = "status-tag tag-bad";
    }
    if (uploadDropzone) uploadDropzone.classList.remove("hidden");
  }
}

export function initBenchmarkControls(micStreamer = null) {
  const btnStreamToggle = $("btnStreamToggle");
  const btnHeroStream = $("btnHeroStream");
  const presetButtons = $$(".btn-preset");
  const precButtons = $$(".btn-prec");
  const suppressionSlider = $("suppressionSlider");
  const intensityChips = $$(".btn-chip");

  const btnSourceBenchmark = $("btnSourceBenchmark");
  const btnSourceUpload = $("btnSourceUpload");
  const btnSourceMic = $("btnSourceMic");
  const benchmarkControls = $("benchmarkControls");
  const uploadControls = $("uploadControls");
  const micControls = $("micControls");
  const btnMicToggle = $("btnMicToggle");

  const uploadDropzone = $("uploadDropzone");
  const wavFileInput = $("wavFileInput");
  const btnPlayUploadedClean = $("btnPlayUploadedClean");
  const btnPlayUploadedNoisy = $("btnPlayUploadedNoisy");

  const toggleDualModel = $("toggleDualModel");
  const badgeDualState = $("badgeDualState");
  const sliderCrossfade = $("sliderCrossfade");
  const crossBlendPct = $("crossBlendPct");

  if (btnStreamToggle) {
    btnStreamToggle.addEventListener("click", () => setStreamingState(!state.isStreaming, micStreamer));
  }
  if (btnHeroStream) {
    btnHeroStream.addEventListener("click", () => setStreamingState(!state.isStreaming, micStreamer));
  }

  // Presets
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activePreset = btn.dataset.preset;
      state.activeTargetSnr = parseFloat(btn.dataset.snr || "0");

      if (state.isStreaming) {
        wsClient.send({
          type: "start_benchmark",
          preset: state.activePreset,
          target_snr: state.activeTargetSnr,
        });
      }
    });
  });

  // Precision Switcher
  const precModeVal = $("precModeVal");
  const aiChipMode = $("aiChipMode");
  const precMemVal = $("precMemVal");
  const precCompVal = $("precCompVal");
  const precSqnrVal = $("precSqnrVal");

  function setPrecisionMode(mode) {
    precButtons.forEach((b) => b.classList.remove("active"));
    const activeBtn = document.querySelector(`.btn-prec[data-prec="${mode}"]`);
    if (activeBtn) activeBtn.classList.add("active");

    state.activePrecision = mode;
    if (precModeVal) precModeVal.textContent = mode;
    if (aiChipMode) aiChipMode.textContent = `${mode} TENSOR`;

    if (mode === "FP32") {
      if (precMemVal) precMemVal.textContent = "162.0 KB";
      if (precCompVal) precCompVal.textContent = "0.0%";
      if (precSqnrVal) precSqnrVal.textContent = "∞ dB";
    } else if (mode === "FP16") {
      if (precMemVal) precMemVal.textContent = "81.0 KB";
      if (precCompVal) precCompVal.textContent = "50.0%";
      if (precSqnrVal) precSqnrVal.textContent = "73.6 dB";
    } else if (mode === "INT8") {
      if (precMemVal) precMemVal.textContent = "41.7 KB";
      if (precCompVal) precCompVal.textContent = "74.3%";
      if (precSqnrVal) precSqnrVal.textContent = "39.9 dB";
    }

    wsClient.send({ type: "set_precision", precision: mode });
  }

  precButtons.forEach((btn) => {
    btn.addEventListener("click", () => setPrecisionMode(btn.dataset.prec));
  });

  // Source Navigation
  if (btnSourceBenchmark) {
    btnSourceBenchmark.addEventListener("click", () => {
      btnSourceBenchmark.classList.add("active");
      if (btnSourceUpload) btnSourceUpload.classList.remove("active");
      if (btnSourceMic) btnSourceMic.classList.remove("active");
      if (benchmarkControls) benchmarkControls.classList.remove("hidden");
      if (uploadControls) uploadControls.classList.add("hidden");
      if (micControls) micControls.classList.add("hidden");
      if (state.micActive && micStreamer) micStreamer.stopMicrophone();
    });
  }

  if (btnSourceUpload) {
    btnSourceUpload.addEventListener("click", () => {
      btnSourceUpload.classList.add("active");
      if (btnSourceBenchmark) btnSourceBenchmark.classList.remove("active");
      if (btnSourceMic) btnSourceMic.classList.remove("active");
      if (uploadControls) uploadControls.classList.remove("hidden");
      if (benchmarkControls) benchmarkControls.classList.add("hidden");
      if (micControls) micControls.classList.add("hidden");
      if (state.micActive && micStreamer) micStreamer.stopMicrophone();
      if (state.isStreaming) setStreamingState(false, micStreamer);
    });
  }

  if (btnSourceMic) {
    btnSourceMic.addEventListener("click", () => {
      btnSourceMic.classList.add("active");
      if (btnSourceBenchmark) btnSourceBenchmark.classList.remove("active");
      if (btnSourceUpload) btnSourceUpload.classList.remove("active");
      if (micControls) micControls.classList.remove("hidden");
      if (benchmarkControls) benchmarkControls.classList.add("hidden");
      if (uploadControls) uploadControls.classList.add("hidden");
      if (state.isStreaming) setStreamingState(false, micStreamer);
    });
  }

  if (btnMicToggle && micStreamer) {
    btnMicToggle.addEventListener("click", async () => {
      if (!state.micActive) {
        if (state.isStreaming) setStreamingState(false, micStreamer);
        await micStreamer.startMicrophone();
      } else {
        micStreamer.stopMicrophone();
      }
    });
  }

  // Suppression slider & chips
  if (suppressionSlider) {
    suppressionSlider.addEventListener("input", (e) => {
      updateIntensity(parseFloat(e.target.value) / 100.0);
    });
  }
  intensityChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      updateIntensity(parseFloat(chip.dataset.intensity));
    });
  });

  // Drag and drop WAV
  if (uploadDropzone && wavFileInput) {
    uploadDropzone.addEventListener("click", () => wavFileInput.click());
    uploadDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadDropzone.classList.add("dragover");
    });
    uploadDropzone.addEventListener("dragleave", () => {
      uploadDropzone.classList.remove("dragover");
    });
    uploadDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
    wavFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  if (btnPlayUploadedClean) {
    btnPlayUploadedClean.addEventListener("click", () => playUploadedChannel("denoised"));
  }
  if (btnPlayUploadedNoisy) {
    btnPlayUploadedNoisy.addEventListener("click", () => playUploadedChannel("noisy"));
  }

  // Dual Model Controls
  if (toggleDualModel) {
    toggleDualModel.addEventListener("change", () => {
      state.isDualModel = toggleDualModel.checked;
      if (badgeDualState) {
        badgeDualState.textContent = state.isDualModel ? "CONCURRENT LIVE" : "OFFLINE";
        badgeDualState.style.background = state.isDualModel ? "#76B900" : "";
        badgeDualState.style.color = state.isDualModel ? "#000" : "";
      }
      wsClient.send({
        type: "set_dual_mode",
        enabled: state.isDualModel,
        crossfade: (sliderCrossfade ? parseInt(sliderCrossfade.value, 10) : 50) / 100.0,
      });
    });
  }

  if (sliderCrossfade) {
    sliderCrossfade.addEventListener("input", () => {
      const val = parseInt(sliderCrossfade.value, 10);
      state.crossfadeAlpha = val / 100.0;
      if (crossBlendPct) {
        crossBlendPct.textContent = `${100 - val}% / ${val}%`;
      }
      wsClient.send({
        type: "set_crossfade",
        alpha: state.crossfadeAlpha,
      });
    });
  }

  return { setPrecisionMode };
}

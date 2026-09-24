/**
 * Edge AI Real-Time Neural Audio Denoiser & Latency Profiler
 * NVIDIA RTX Broadcast & Studio Testbench Client — ES Module Bootstrap
 * Zero Global Scope Pollution
 */

import { state } from "./state.js";
import { $, safeText } from "./utils/dom.js";
import { audioManager } from "./audio/audio-manager.js";
import { MicStreamer } from "./audio/mic-streamer.js";
import { wsClient } from "./net/websocket.js";
import { fetchStatus, fetchClassifierStatus, fetchEqResponse } from "./net/api.js";

import { initOscilloscope, renderOscilloscope } from "./renderers/oscilloscope.js";
import { initSpectrograms, renderSpectrograms } from "./renderers/spectrogram.js";
import { initParametricEq, updateEqCurveUI } from "./renderers/parametric-eq.js";
import { initVectorscope } from "./renderers/vectorscope.js";

import { initHeaderControls } from "./ui/header-controls.js";
import { initHeroControls, syncABUi } from "./ui/hero-controls.js";
import { initWorkspaceTabs } from "./ui/workspace-tabs.js";
import { initBenchmarkControls, setStreamingState } from "./ui/benchmark-controls.js";
import { handleTelemetryMessage, initTelemetryReports } from "./ui/telemetry-display.js";
import { initTranslationSubtitles, updateSubtitlesUI } from "./ui/translation-subtitles.js";
import { initStudioRecorder, accumulateRecordingData, startRecordingSession } from "./ui/studio-recorder.js";
import { initStudioSuite, updateTseUI, handleTseTriggerDetected, updateStudioUI, updateAutoAdaptUI } from "./ui/studio-suite.js";
import { initBatchBenchmark } from "./ui/batch-benchmark.js";

let micStreamer = null;

export function initApp() {
  micStreamer = new MicStreamer(wsClient);

  // Initialize Audio & Renderers
  initOscilloscope();
  initSpectrograms();
  initParametricEq();
  initVectorscope();

  // Helper callbacks
  const onToggleStream = () => setStreamingState(!state.isStreaming, micStreamer);
  const onToggleAB = () => {
    audioManager.init();
    audioManager.setABMode(!state.isCleanActive);
    syncABUi(state.isCleanActive);
  };
  const onStartStreamIfNeeded = () => {
    if (!state.isStreaming && !state.micActive) {
      setStreamingState(true, micStreamer);
    }
  };

  // Initialize UI components
  const benchmarkController = initBenchmarkControls(micStreamer);
  initHeaderControls({
    onToggleStream,
    onToggleAB,
    onSetPrecision: (mode) => benchmarkController.setPrecisionMode(mode),
  });
  initHeroControls({ onStartStreamIfNeeded });
  initWorkspaceTabs();
  initTelemetryReports();
  initTranslationSubtitles(micStreamer);
  initStudioRecorder({ onStartStreamIfNeeded });
  initStudioSuite();
  initBatchBenchmark();

  // Wire WebSocket Events
  state.on("telemetry", (msg) => {
    handleTelemetryMessage(msg, {
      onUpdateSubtitles: updateSubtitlesUI,
      onUpdateTse: updateTseUI,
      onUpdateStudio: updateStudioUI,
      onAccumulateRecording: accumulateRecordingData,
    });
  });

  state.on("tse_trigger_detected", handleTseTriggerDetected);
  state.on("tse_status_updated", updateTseUI);
  state.on("studio_status_updated", updateStudioUI);

  // Initial fetch of system status
  fetchStatus()
    .then((data) => {
      if (data.precision) {
        const precModeVal = $("precModeVal");
        const aiChipMode = $("aiChipMode");
        const precMemVal = $("precMemVal");
        const precCompVal = $("precCompVal");
        if (precModeVal) precModeVal.textContent = data.precision;
        if (aiChipMode) aiChipMode.textContent = `${data.precision} TENSOR`;
        if (precMemVal) precMemVal.textContent = `${data.model_size_kb} KB`;
        if (precCompVal) precCompVal.textContent = `${data.compression_pct}%`;
      }
      if (data.process_rss_mb) {
        const rssDisplay = $("rssDisplay");
        if (rssDisplay) rssDisplay.textContent = `${data.process_rss_mb} MB`;
      }
      if (data.simd) {
        const simdBadge = $("simdBadge");
        const simdHardwareVal = $("simdHardwareVal");
        const tierName = data.simd.name || (data.simd.tier === 1 ? "Native C (AVX2/NEON)" : data.simd.tier === 2 ? "Numba LLVM JIT (AVX2/VNNI)" : "NumPy Fallback");
        if (simdBadge) simdBadge.textContent = `⚡ TIER ${data.simd.tier}: ${tierName.toUpperCase()}`;
        if (simdHardwareVal) simdHardwareVal.textContent = `Tier ${data.simd.tier}: ${tierName.split(' ')[0]}`;
      }
    })
    .catch(() => {});

  // Initial fetch of EQ curve
  fetchEqResponse()
    .then((data) => {
      if (data) updateEqCurveUI(data);
    })
    .catch(() => {});

  // Initial fetch of classifier status
  fetchClassifierStatus()
    .then((data) => {
      if (data && data.auto_adapt_enabled !== undefined) {
        updateAutoAdaptUI(data.auto_adapt_enabled);
      }
      if (data && data.noise_type && data.label) {
        const fpName = $("fpName");
        const fpIcon = $("fpIcon");
        if (fpName) fpName.textContent = data.label;
        if (fpIcon && data.icon) fpIcon.textContent = data.icon;
      }
    })
    .catch(() => {});

  // Start 60 FPS Render Loop
  function renderLoop() {
    renderOscilloscope();
    renderSpectrograms();
    requestAnimationFrame(renderLoop);
  }
  requestAnimationFrame(renderLoop);

  // Connect WebSocket
  wsClient.connect();
}

// Auto-bootstrap when loaded directly in browser
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
}

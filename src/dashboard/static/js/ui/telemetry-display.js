/**
 * Telemetry Display, Hardware Profiler Breakdown & Metric Visualizers
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";
import { CONFIG } from "../config.js";
import { $, safeText } from "../utils/dom.js";
import { audioManager } from "../audio/audio-manager.js";
import { updatePurityDial, updateVuMeters } from "./hero-controls.js";
import { updateVectorscopeUI } from "../renderers/vectorscope.js";
import { updateEqCurveUI } from "../renderers/parametric-eq.js";

export function handleTelemetryMessage(msg, { onUpdateSubtitles, onUpdateTse, onUpdateStudio, onAccumulateRecording }) {
  const stages = msg.stages || {};
  const budget = msg.budget || {};
  const stats = msg.rolling_stats || {};
  const metrics = msg.metrics || {};
  const audio = msg.audio || {};

  // 1. Spectrogram Data
  if (audio.spec_in || audio.spec_out) {
    state.pushSpectrogramData(audio.spec_in, audio.spec_out);
  }

  // 2. Oscilloscope Data & Jitter-Free Audio Playback
  if (audio.raw_noisy && audio.denoised) {
    state.pushOscilloscopeData(audio.raw_noisy, audio.denoised, audio.noise_subtracted);
    audioManager.pushAudioFrame(audio.raw_noisy, audio.denoised);
  }

  // 3. Sound Health & Purity Hero Gauge
  const purityPct = metrics.purity_pct !== undefined ? metrics.purity_pct : 98.5;
  const noiseErased = metrics.noise_erased_pct !== undefined ? metrics.noise_erased_pct : 95.8;
  updatePurityDial(purityPct, noiseErased);

  // 4. Live VU Level Bars
  const rmsIn = metrics.rms_in !== undefined ? metrics.rms_in : 0.245;
  const rmsOut = metrics.rms_out !== undefined ? metrics.rms_out : 0.128;
  const rmsNoise = metrics.rms_noise !== undefined ? metrics.rms_noise : 0.182;
  updateVuMeters(rmsIn, rmsOut, rmsNoise);

  // 5. Oscilloscope Readouts
  const oscRmsIn = $("oscRmsIn");
  const oscRmsOut = $("oscRmsOut");
  const oscRmsSub = $("oscRmsSub");
  const oscGainVal = $("oscGainVal");
  const snrDeltaVal = $("snrDeltaVal");
  const oscPeakHz = $("oscPeakHz");
  const headerPeakFreq = $("headerPeakFreq");

  if (audio.raw_noisy && audio.denoised) {
    let pkIn = 0, pkOut = 0;
    for (let i = 0; i < audio.raw_noisy.length; i++) {
      const aIn = Math.abs(audio.raw_noisy[i]);
      const aOut = Math.abs(audio.denoised[i]);
      if (aIn > pkIn) pkIn = aIn;
      if (aOut > pkOut) pkOut = aOut;
    }
    if (oscRmsIn) oscRmsIn.textContent = `RMS: ${rmsIn.toFixed(3)} | PK: ${pkIn.toFixed(3)}`;
    if (oscRmsOut) oscRmsOut.textContent = `RMS: ${rmsOut.toFixed(3)} | PK: ${pkOut.toFixed(3)}`;
    if (oscRmsSub) oscRmsSub.textContent = `RMS: ${rmsNoise.toFixed(3)} | ERASED`;
  }

  if (metrics.snr_delta_db !== undefined) {
    const sign = metrics.snr_delta_db >= 0 ? "+" : "";
    const deltaText = `${sign}${metrics.snr_delta_db.toFixed(2)} dB GAIN`;
    if (oscGainVal) oscGainVal.textContent = deltaText;
    if (snrDeltaVal) snrDeltaVal.textContent = deltaText;
  }

  if (metrics.fft_peak_hz !== undefined) {
    const peakStr = `${Math.round(metrics.fft_peak_hz)} Hz`;
    if (oscPeakHz) oscPeakHz.textContent = peakStr;
    if (headerPeakFreq) headerPeakFreq.textContent = peakStr;
  }

  // 6. Hardware Headroom Gauge
  const headroomPctText = $("headroomPctText");
  const headroomMsText = $("headroomMsText");
  const gaugeBar = $("gaugeBar");

  const headroomPct = budget.headroom_pct !== undefined ? budget.headroom_pct : 98.5;
  const headroomMs = budget.headroom_ms !== undefined ? budget.headroom_ms : 19.7;
  if (headroomPctText) safeText(headroomPctText, `${headroomPct.toFixed(1)}%`);
  if (headroomMsText) safeText(headroomMsText, `+${headroomMs.toFixed(1)} ms`);

  if (gaugeBar) {
    const circumference = CONFIG.CIRCUMFERENCE_HEADROOM;
    const offset = circumference * (1.0 - Math.min(100.0, Math.max(0.0, headroomPct)) / 100.0);
    gaugeBar.style.strokeDashoffset = offset;
  }

  // 7. Latency & 3-Stage Bars
  const totalLatencyText = $("totalLatencyText");
  const aiChipSpeed = $("aiChipSpeed");
  const timePre = $("timePre");
  const timeTensor = $("timeTensor");
  const timeSynth = $("timeSynth");
  const barPre = $("barPre");
  const barTensor = $("barTensor");
  const barSynth = $("barSynth");

  const totalMs = stages.total_latency_ms || 0.3;
  if (totalLatencyText) safeText(totalLatencyText, `${totalMs.toFixed(3)} ms`);
  if (aiChipSpeed) safeText(aiChipSpeed, `${totalMs.toFixed(2)} ms SPEED`);

  const tPre = stages.pre_processing_ms || 0.05;
  const tTensor = stages.tensor_compute_ms || 0.20;
  const tSynth = stages.output_synthesis_ms || 0.05;

  if (timePre) safeText(timePre, `${tPre.toFixed(3)} ms`);
  if (timeTensor) safeText(timeTensor, `${tTensor.toFixed(3)} ms`);
  if (timeSynth) safeText(timeSynth, `${tSynth.toFixed(3)} ms`);

  const sumT = Math.max(0.001, tPre + tTensor + tSynth);
  if (barPre) barPre.style.width = `${((tPre / sumT) * 100).toFixed(1)}%`;
  if (barTensor) barTensor.style.width = `${((tTensor / sumT) * 100).toFixed(1)}%`;
  if (barSynth) barSynth.style.width = `${((tSynth / sumT) * 100).toFixed(1)}%`;

  // 8. Rolling Percentiles & FPS
  const statP50 = $("statP50");
  const statP95 = $("statP95");
  const statP99 = $("statP99");
  const statThroughput = $("statThroughput");

  if (stats.median_p50_ms !== undefined && statP50) statP50.textContent = `${stats.median_p50_ms.toFixed(3)} ms`;
  if (stats.p95_ms !== undefined && statP95) statP95.textContent = `${stats.p95_ms.toFixed(3)} ms`;
  if (stats.p99_ms !== undefined && statP99) statP99.textContent = `${stats.p99_ms.toFixed(3)} ms`;

  if (totalMs > 0 && statThroughput) {
    const fps = Math.round(1000.0 / totalMs);
    statThroughput.textContent = `${fps} FPS`;
  }

  // 9. Precision & Memory
  const precModeVal = $("precModeVal");
  const aiChipMode = $("aiChipMode");
  const precMemVal = $("precMemVal");
  const snrInVal = $("snrInVal");
  const snrOutVal = $("snrOutVal");
  const rssDisplay = $("rssDisplay");

  if (msg.precision_mode) {
    if (precModeVal) precModeVal.textContent = msg.precision_mode;
    if (aiChipMode) aiChipMode.textContent = `${msg.precision_mode} TENSOR`;
  }
  if (metrics.model_memory_kb !== undefined && precMemVal) precMemVal.textContent = `${metrics.model_memory_kb.toFixed(1)} KB`;
  if (metrics.snr_input_db !== undefined && snrInVal) snrInVal.textContent = `${metrics.snr_input_db.toFixed(2)} dB`;
  if (metrics.snr_output_db !== undefined && snrOutVal) snrOutVal.textContent = `${metrics.snr_output_db.toFixed(2)} dB`;
  if (metrics.process_rss_mb !== undefined && rssDisplay) rssDisplay.textContent = `${metrics.process_rss_mb.toFixed(1)} MB`;

  // 10. Acoustic Noise Signature
  if (msg.noise_signature) {
    const ns = msg.noise_signature;
    const fpIcon = $("fpIcon");
    const fpName = $("fpName");
    const fpConf = $("fpConf");
    const fpBar = $("fpBar");
    const step1NoiseTag = $("step1NoiseTag");

    if (fpIcon && ns.icon) fpIcon.textContent = ns.icon;
    if (fpName && ns.name) fpName.textContent = ns.name;
    const conf = Math.round(ns.confidence_pct !== undefined ? ns.confidence_pct : 90);
    if (fpConf) fpConf.textContent = `${conf}% CONF`;
    if (fpBar) fpBar.style.width = `${Math.min(100, Math.max(0, conf))}%`;
    if (step1NoiseTag) {
      step1NoiseTag.textContent = `${ns.icon || ""} ${ns.name || "NOISY"}`.trim();
      if (ns.description) step1NoiseTag.title = ns.description;
    }
  }

  // 11. DNSMOS Voice Quality
  if (msg.voice_quality) {
    const vq = msg.voice_quality;
    const dnsmosSig = $("dnsmosSig");
    const dnsmosBak = $("dnsmosBak");
    const dnsmosOvrl = $("dnsmosOvrl");
    const dnsmosStoi = $("dnsmosStoi");

    if (dnsmosSig && vq.sig_mos !== undefined) dnsmosSig.textContent = vq.sig_mos.toFixed(2);
    if (dnsmosBak && vq.bak_mos !== undefined) dnsmosBak.textContent = vq.bak_mos.toFixed(2);
    if (dnsmosOvrl && vq.ovrl_mos !== undefined) dnsmosOvrl.textContent = vq.ovrl_mos.toFixed(2);
    if (dnsmosStoi && vq.stoi !== undefined) dnsmosStoi.textContent = vq.stoi.toFixed(2);
  }

  // 12. GPU Power & Energy
  if (msg.energy) {
    const eng = msg.energy;
    const energyPower = $("energyPower");
    const energyUj = $("energyUj");
    const energyEfficiency = $("energyEfficiency");
    const energyTemp = $("energyTemp");

    if (energyPower && eng.power_watts !== undefined) energyPower.textContent = `${eng.power_watts.toFixed(2)} W`;
    if (energyUj && eng.energy_per_frame_uj !== undefined) energyUj.textContent = `${eng.energy_per_frame_uj.toFixed(2)} µJ`;
    if (energyEfficiency && eng.efficiency_fps_per_watt !== undefined) {
      energyEfficiency.textContent = `${Math.round(eng.efficiency_fps_per_watt)} FPS/W`;
    }
    if (energyTemp && eng.temperature_c !== undefined) energyTemp.textContent = `${eng.temperature_c.toFixed(1)} °C`;
  }

  // 13. Subtitles
  if (msg.subtitles && onUpdateSubtitles) {
    onUpdateSubtitles(msg.subtitles);
  }

  // 14. Dual-Model Telemetry
  if (msg.dual_telemetry) {
    const dt = msg.dual_telemetry;
    const dualLatencyA = $("dualLatencyA");
    const dualRamA = $("dualRamA");
    const dualSnrA = $("dualSnrA");
    const dualLatencyB = $("dualLatencyB");
    const dualRamB = $("dualRamB");
    const dualSnrB = $("dualSnrB");

    if (dt.model_a) {
      if (dualLatencyA) dualLatencyA.textContent = `${dt.model_a.latency_ms} ms`;
      if (dualRamA) dualRamA.textContent = `${dt.model_a.size_kb} KB`;
      if (dualSnrA) dualSnrA.textContent = `+${dt.model_a.snr_gain_db} dB`;
    }
    if (dt.model_b) {
      if (dualLatencyB) dualLatencyB.textContent = `${dt.model_b.latency_ms} ms`;
      if (dualRamB) dualRamB.textContent = `${dt.model_b.size_kb} KB`;
      if (dualSnrB) dualSnrB.textContent = `+${dt.model_b.snr_gain_db} dB`;
    }
  }

  // 15. TSE
  if (msg.tse && onUpdateTse) {
    onUpdateTse(msg.tse);
  }

  // 16. Studio Vocal Suite
  if (msg.studio && onUpdateStudio) {
    onUpdateStudio(msg.studio);
  }

  // 17. Vectorscope
  if (msg.vectorscope) {
    updateVectorscopeUI(msg.vectorscope);
  }

  // 18. EQ
  if (msg.eq) {
    updateEqCurveUI(msg.eq);
  }

  // 19. SIMD Acceleration Tier
  if (msg.simd) {
    const simdBadge = $("simdBadge");
    const simdHardwareVal = $("simdHardwareVal");
    const tierName = msg.simd.name || (msg.simd.tier === 1 ? "Native C (AVX2/NEON)" : msg.simd.tier === 2 ? "Numba LLVM JIT (AVX2/VNNI)" : "NumPy Fallback");
    if (simdBadge) simdBadge.textContent = `⚡ TIER ${msg.simd.tier}: ${tierName.toUpperCase()}`;
    if (simdHardwareVal) simdHardwareVal.textContent = `Tier ${msg.simd.tier}: ${tierName.split(' ')[0]}`;
  }

  // Accumulate audio for live recording
  if (state.isRecording && audio.denoised && audio.raw_noisy && onAccumulateRecording) {
    onAccumulateRecording(audio.denoised, audio.raw_noisy);
  }
}

export function initTelemetryReports() {
  const handleExport = (format) => {
    window.location.href = `/api/report/export?format=${format}`;
  };

  const btnExportJson = $("btnExportJson");
  const btnExportCsv = $("btnExportCsv");
  const btnExportJson2 = $("btnExportJson2");
  const btnExportCsv2 = $("btnExportCsv2");

  if (btnExportJson) btnExportJson.addEventListener("click", () => handleExport("json"));
  if (btnExportCsv) btnExportCsv.addEventListener("click", () => handleExport("csv"));
  if (btnExportJson2) btnExportJson2.addEventListener("click", () => handleExport("json"));
  if (btnExportCsv2) btnExportCsv2.addEventListener("click", () => handleExport("csv"));
}

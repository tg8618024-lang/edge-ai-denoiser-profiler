/**
 * Central Reactive Application State Store & Event Bus
 * Zero Global Scope Pollution
 */

import { CONFIG, DEFAULT_EQ_BANDS } from "./config.js";

class AppState {
  constructor() {
    // Audio graph & playback state
    this.isCleanActive = true;
    this.volume = 80;
    this.micActive = false;

    // Benchmark streaming & preset state
    this.isStreaming = false;
    this.activePreset = "white";
    this.activeTargetSnr = 0.0;
    this.activePrecision = "FP32";
    this.currentIntensity = 1.0;
    this.autoAdaptEnabled = false;

    // Dual-model state
    this.isDualModel = false;
    this.crossfadeAlpha = 0.5;

    // Oscilloscope buffers (2048 capacity)
    this.oscBufNoisy = new Float32Array(CONFIG.OSC_BUF_LEN);
    this.oscBufClean = new Float32Array(CONFIG.OSC_BUF_LEN);
    this.oscBufNoise = new Float32Array(CONFIG.OSC_BUF_LEN);
    this.oscMode = "overlay"; // "overlay" | "split"
    this.oscWindowSamples = 512; // 512 | 1024
    this.diffShading = true;
    this.chanVisible = { in: true, out: true, sub: true };

    // Spectrogram buffers & queues
    this.lastSpecIn = new Float32Array(CONFIG.SPEC_BINS);
    this.lastSpecOut = new Float32Array(CONFIG.SPEC_BINS);
    this.hasNewSpecFrame = false;
    this.isSpec3D = false;
    this.specHistoryIn = [];
    this.specHistoryOut = [];

    // Live Recording state
    this.isRecording = false;
    this.recStartTime = 0;
    this.recTimerInterval = null;
    this.recCleanPcm = [];
    this.recNoisyPcm = [];
    this.lastRecordedSessionId = null;
    this.lastTranscriptText = "";
    this.lastTranslatedText = "";

    // Uploaded Audio state
    this.uploadedAudioId = null;
    this.uploadedAudioPlayer = null;

    // EQ Sculptor state
    this.eqBands = JSON.parse(JSON.stringify(DEFAULT_EQ_BANDS));
    this.eqActiveBand = -1;
    this.eqIsDragging = false;
    this.latestEqCurve = null;

    // Batch Benchmark state
    this.latestBatchReport = null;

    // Pub/Sub listeners
    this._listeners = new Map();
  }

  on(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this._listeners.has(event)) {
      this._listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this._listeners.has(event)) {
      this._listeners.get(event).forEach((cb) => {
        try {
          cb(data);
        } catch (err) {
          console.error(`Error in listener for event "${event}":`, err);
        }
      });
    }
  }

  pushOscilloscopeData(noisy, clean, noise) {
    const n = noisy.length;
    this.oscBufNoisy.copyWithin(0, n);
    this.oscBufClean.copyWithin(0, n);
    this.oscBufNoise.copyWithin(0, n);

    const startIdx = CONFIG.OSC_BUF_LEN - n;
    for (let i = 0; i < n; i++) {
      this.oscBufNoisy[startIdx + i] = noisy[i];
      this.oscBufClean[startIdx + i] = clean[i];
      this.oscBufNoise[startIdx + i] = noise ? noise[i] : (noisy[i] - clean[i]);
    }
  }

  pushSpectrogramData(specIn, specOut) {
    if (specIn) {
      this.lastSpecIn = specIn;
      this.hasNewSpecFrame = true;
      this.specHistoryIn.unshift(specIn);
      if (this.specHistoryIn.length > CONFIG.SPEC_3D_SLICES) {
        this.specHistoryIn.pop();
      }
    }
    if (specOut) {
      this.lastSpecOut = specOut;
      this.hasNewSpecFrame = true;
      this.specHistoryOut.unshift(specOut);
      if (this.specHistoryOut.length > CONFIG.SPEC_3D_SLICES) {
        this.specHistoryOut.pop();
      }
    }
  }
}

export const state = new AppState();

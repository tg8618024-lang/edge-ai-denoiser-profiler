/**
 * Web Audio Graph Manager & AudioWorklet Bridge
 * Zero Global Scope Pollution
 */

import { state } from "../state.js";

export class AudioManager {
  constructor() {
    this.audioCtx = null;
    this.masterGain = null;
    this.noisyGain = null;
    this.cleanGain = null;
    this.workletNode = null;
    this.isWorkletReady = false;
    this.nextFallbackPlayTime = 0;
  }

  async init() {
    if (this.audioCtx) {
      if (this.audioCtx.state === "suspended") {
        await this.audioCtx.resume();
      }
      return;
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;

    try {
      this.audioCtx = new AudioContextClass({ latencyHint: "interactive" });
    } catch (e) {
      this.audioCtx = new AudioContextClass();
    }

    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume().catch(() => {});
    }

    this.masterGain = this.audioCtx.createGain();
    this.masterGain.gain.setValueAtTime(state.volume / 100.0, this.audioCtx.currentTime);
    this.masterGain.connect(this.audioCtx.destination);

    this.noisyGain = this.audioCtx.createGain();
    this.cleanGain = this.audioCtx.createGain();

    if (state.isCleanActive) {
      this.noisyGain.gain.setValueAtTime(0.0, this.audioCtx.currentTime);
      this.cleanGain.gain.setValueAtTime(1.0, this.audioCtx.currentTime);
    } else {
      this.noisyGain.gain.setValueAtTime(1.0, this.audioCtx.currentTime);
      this.cleanGain.gain.setValueAtTime(0.0, this.audioCtx.currentTime);
    }

    this.noisyGain.connect(this.masterGain);
    this.cleanGain.connect(this.masterGain);

    // Initialize AudioWorklet
    if (this.audioCtx.audioWorklet) {
      try {
        await this.audioCtx.audioWorklet.addModule("/static/js/audio/audio-worklet-processor.js");
        this.workletNode = new AudioWorkletNode(this.audioCtx, "jitter-buffer-processor", {
          numberOfInputs: 0,
          numberOfOutputs: 2,
          outputChannelCount: [1, 1],
          processorOptions: { inputSampleRate: 16000 },
        });

        // Output 0 -> Noisy Gain, Output 1 -> Clean Gain
        this.workletNode.connect(this.noisyGain, 0, 0);
        this.workletNode.connect(this.cleanGain, 1, 0);
        this.isWorkletReady = true;
      } catch (err) {
        console.warn("AudioWorklet initialization fallback to buffer scheduler:", err);
        this.isWorkletReady = false;
        this.nextFallbackPlayTime = this.audioCtx.currentTime + 0.05;
      }
    } else {
      this.isWorkletReady = false;
      this.nextFallbackPlayTime = this.audioCtx.currentTime + 0.05;
    }
  }

  pushAudioFrame(noisyArr, cleanArr) {
    if (!this.audioCtx || this.audioCtx.state === "suspended") return;

    if (this.isWorkletReady && this.workletNode) {
      const noisyF32 = new Float32Array(noisyArr);
      const cleanF32 = new Float32Array(cleanArr);
      this.workletNode.port.postMessage(
        { type: "push_audio", noisy: noisyF32, clean: cleanF32 },
        [noisyF32.buffer, cleanF32.buffer]
      );
    } else {
      // Graceful fallback for environments where AudioWorklet is unavailable
      this._scheduleFallbackPlayback(noisyArr, cleanArr);
    }
  }

  _scheduleFallbackPlayback(noisyPcm, cleanPcm) {
    const frameLen = noisyPcm.length;
    if (frameLen === 0) return;

    const bufNoisy = this.audioCtx.createBuffer(1, frameLen, 16000);
    const bufClean = this.audioCtx.createBuffer(1, frameLen, 16000);

    bufNoisy.getChannelData(0).set(noisyPcm);
    bufClean.getChannelData(0).set(cleanPcm);

    const srcN = this.audioCtx.createBufferSource();
    const srcC = this.audioCtx.createBufferSource();

    srcN.buffer = bufNoisy;
    srcC.buffer = bufClean;

    srcN.connect(this.noisyGain);
    srcC.connect(this.cleanGain);

    const now = this.audioCtx.currentTime;
    if (this.nextFallbackPlayTime < now) {
      this.nextFallbackPlayTime = now + 0.030;
    } else if (this.nextFallbackPlayTime > now + 0.25) {
      this.nextFallbackPlayTime = now + 0.040;
    }

    srcN.start(this.nextFallbackPlayTime);
    srcC.start(this.nextFallbackPlayTime);
    this.nextFallbackPlayTime += frameLen / 16000.0;
  }

  setABMode(cleanSelected) {
    state.isCleanActive = cleanSelected;
    if (!this.audioCtx) return;

    const now = this.audioCtx.currentTime;
    const rampTime = 0.020; // 20ms click-free cross-fade

    this.noisyGain.gain.cancelScheduledValues(now);
    this.cleanGain.gain.cancelScheduledValues(now);
    this.noisyGain.gain.setValueAtTime(this.noisyGain.gain.value, now);
    this.cleanGain.gain.setValueAtTime(this.cleanGain.gain.value, now);

    if (cleanSelected) {
      this.noisyGain.gain.linearRampToValueAtTime(0.0, now + rampTime);
      this.cleanGain.gain.linearRampToValueAtTime(1.0, now + rampTime);
    } else {
      this.cleanGain.gain.linearRampToValueAtTime(0.0, now + rampTime);
      this.noisyGain.gain.linearRampToValueAtTime(1.0, now + rampTime);
    }

    state.emit("ab_mode_changed", cleanSelected);
  }

  setVolume(valPct) {
    state.volume = valPct;
    if (this.masterGain && this.audioCtx) {
      this.masterGain.gain.setValueAtTime(valPct / 100.0, this.audioCtx.currentTime);
    }
  }

  reset() {
    if (this.isWorkletReady && this.workletNode) {
      this.workletNode.port.postMessage({ type: "reset" });
    }
    if (this.audioCtx) {
      this.nextFallbackPlayTime = this.audioCtx.currentTime + 0.03;
    }
  }
}

export const audioManager = new AudioManager();

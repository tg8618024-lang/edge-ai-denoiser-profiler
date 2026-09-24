/**
 * Real-Time Edge AI Jitter-Free Audio Ring Buffer Processor
 *
 * Runs exclusively on the browser's dedicated high-priority Audio Rendering Thread.
 * Fully immune to main-thread UI stalls, garbage collection pauses, and browser tab background throttling.
 *
 * Backed by a 16,384-sample dual-channel circular ring buffer with a 512-sample pre-roll cushion
 * and pitch-accurate linear resampling (16kHz source -> hardware sampleRate e.g. 48kHz).
 */

class JitterBufferProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Capacity: 16,384 samples (1.024s at 16kHz)
    this.capacity = 16384;
    this.noisyRing = new Float32Array(this.capacity);
    this.cleanRing = new Float32Array(this.capacity);

    this.writeHead = 0;
    this.readHead = 0;
    this.available = 0;

    // Pre-roll jitter cushion: 512 samples (32ms at 16kHz)
    this.preRoll = 512;
    this.isPlaying = false;

    // Resampling ratio: input sample rate is 16,000 Hz from neural pipeline
    const procOpts = (options && options.processorOptions) || {};
    this.inputSampleRate = procOpts.inputSampleRate || 16000;
    // Global sampleRate provided by AudioWorkletGlobalScope
    this.ratio = this.inputSampleRate / sampleRate;
    this.resamplePhase = 0.0;

    this.port.onmessage = this.handleMessage.bind(this);
  }

  handleMessage(event) {
    const msg = event.data;
    if (!msg) return;

    if (msg.type === "push_audio") {
      const noisy = msg.noisy;
      const clean = msg.clean;
      if (!noisy || !clean) return;

      const len = noisy.length;
      if (len === 0) return;

      // Handle buffer overrun by advancing read pointer (discarding oldest audio)
      if (this.available + len > this.capacity) {
        const drop = (this.available + len) - this.capacity;
        this.readHead = (this.readHead + drop) % this.capacity;
        this.available -= drop;
      }

      // Write incoming samples into circular ring buffer
      for (let i = 0; i < len; i++) {
        const idx = (this.writeHead + i) % this.capacity;
        this.noisyRing[idx] = noisy[i];
        this.cleanRing[idx] = clean[i];
      }
      this.writeHead = (this.writeHead + len) % this.capacity;
      this.available += len;

      // Engage playback once pre-roll cushion threshold is reached
      if (!this.isPlaying && this.available >= this.preRoll) {
        this.isPlaying = true;
      }
    } else if (msg.type === "reset") {
      this.writeHead = 0;
      this.readHead = 0;
      this.available = 0;
      this.isPlaying = false;
      this.resamplePhase = 0.0;
      this.noisyRing.fill(0);
      this.cleanRing.fill(0);
    }
  }

  _catmullRom(ring, idxM1, idx0, idx1, idx2, mu) {
    const ym1 = ring[idxM1];
    const y0 = ring[idx0];
    const y1 = ring[idx1];
    const y2 = ring[idx2];

    const c0 = y0;
    const c1 = 0.5 * (y1 - ym1);
    const c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    const c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);

    return ((c3 * mu + c2) * mu + c1) * mu + c0;
  }

  process(inputs, outputs, parameters) {
    // outputs[0] = Noisy Channel (Mono)
    // outputs[1] = Clean Channel (Mono)
    const outNoisy = outputs[0] ? outputs[0][0] : null;
    const outClean = outputs[1] ? outputs[1][0] : null;

    if (!outNoisy || !outClean) return true;
    const quantum = outNoisy.length; // Always 128 samples in Web Audio API

    // Under-run or initial pre-roll state: output clean digital silence
    if (!this.isPlaying || this.available < 2) {
      outNoisy.fill(0);
      outClean.fill(0);
      if (this.available === 0) {
        this.isPlaying = false;
      }
      return true;
    }

    if (Math.abs(this.ratio - 1.0) < 1e-4) {
      // 1:1 Sample Rate (AudioContext natively operating at 16 kHz)
      const samplesToRead = Math.min(quantum, this.available);
      for (let i = 0; i < samplesToRead; i++) {
        const idx = (this.readHead + i) % this.capacity;
        outNoisy[i] = this.noisyRing[idx];
        outClean[i] = this.cleanRing[idx];
      }

      if (samplesToRead < quantum) {
        // Buffer starvation during quantum: zero remainder to avoid noise clicks
        outNoisy.fill(0, samplesToRead);
        outClean.fill(0, samplesToRead);
        this.isPlaying = false;
      }

      this.readHead = (this.readHead + samplesToRead) % this.capacity;
      this.available -= samplesToRead;
    } else {
      // High-Fidelity 4-Point Catmull-Rom Cubic Hermite Spline Resampling (16kHz -> hardware e.g. 48kHz)
      for (let i = 0; i < quantum; i++) {
        if (this.available < 2) {
          outNoisy.fill(0, i);
          outClean.fill(0, i);
          this.isPlaying = false;
          break;
        }

        const frac = this.resamplePhase;

        if (this.available >= 4) {
          const idxM1 = (this.readHead - 1 + this.capacity) % this.capacity;
          const idx0 = this.readHead;
          const idx1 = (this.readHead + 1) % this.capacity;
          const idx2 = (this.readHead + 2) % this.capacity;

          outNoisy[i] = this._catmullRom(this.noisyRing, idxM1, idx0, idx1, idx2, frac);
          outClean[i] = this._catmullRom(this.cleanRing, idxM1, idx0, idx1, idx2, frac);
        } else {
          // Fallback to linear interpolation when boundary buffer is nearly exhausted
          const idx0 = this.readHead;
          const idx1 = (this.readHead + 1) % this.capacity;
          outNoisy[i] = this.noisyRing[idx0] * (1.0 - frac) + this.noisyRing[idx1] * frac;
          outClean[i] = this.cleanRing[idx0] * (1.0 - frac) + this.cleanRing[idx1] * frac;
        }

        this.resamplePhase += this.ratio;
        while (this.resamplePhase >= 1.0) {
          this.resamplePhase -= 1.0;
          this.readHead = (this.readHead + 1) % this.capacity;
          this.available -= 1;
        }
      }
    }

    return true;
  }
}

registerProcessor("jitter-buffer-processor", JitterBufferProcessor);

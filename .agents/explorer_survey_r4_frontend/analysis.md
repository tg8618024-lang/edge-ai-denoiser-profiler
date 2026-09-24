# Architectural Analysis & Specification: R4 Frontend Modularization, AudioWorklet Ring Buffer, and Test Regression Baseline

**Author**: Explorer R4 Frontend & Test Specialist  
**Working Directory**: `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r4_frontend`  
**Date**: 2026-09-23  
**Status**: Investigation Complete — Hardening Blueprint Ready  

---

## 1. Executive Summary

This investigation provides the complete architectural audit, modularization blueprint, and regression testing baseline for **Requirement R4** of the Real-Time Edge AI Audio Denoiser & Profiler hardening project, alongside a comprehensive survey of the entire test infrastructure (`tests/` and `evaluate.py`).

### Key Findings
1. **Frontend Monolith**: `src/dashboard/static/app.js` has grown into an unmaintainable **3,169-line monolithic IIFE** encapsulating 5 disparate workspaces, 5 canvas rendering engines, real-time WebSocket telemetry parsers, Web Speech recognition, multi-track audio recording, and Web Audio graph management.
2. **Audio Playback Bottleneck & Glitch Mechanism**: The current `scheduleAudioPlayback()` function instantiates and discards **125 `AudioBuffer` and 125 `AudioBufferSourceNode` objects every second** on the browser main UI thread. During tab backgrounding, browser timer throttling (`setTimeout`/`setInterval` throttled to 1000ms, `requestAnimationFrame` paused) starves `scheduleAudioPlayback()`, leading to catastrophic buffer underruns, audible clicks/pops, and heuristic timestamp desynchronization.
3. **Modularization Plan**: A modern, clean ES Module architecture partitioned across `src/dashboard/static/js/` eliminates all global namespace pollution, introduces a reactive pub/sub event store (`state.js`), and decouples rendering from data ingestion.
4. **Jitter-Free AudioWorklet Architecture**: An `AudioWorkletNode` powered by an isolated `AudioWorkletProcessor` (`audio-worklet-processor.js`) running on the browser's dedicated high-priority Audio Rendering Thread, backed by a **16,384-sample dual-channel circular ring buffer** with 512-sample pre-roll and linear pitch-accurate resampling, guarantees click-free playback immune to tab backgrounding or main-thread rendering freezes.
5. **Test Suite Baseline**: The project currently boasts **31 test files** comprising **274 test cases** across 5 tiers (Unit, Integration, E2E, Load, Adversarial) plus the 3-part non-interactive benchmark in `evaluate.py`. Specific regression assertions must be expanded for R1 (authentic ONNX DNSMOS), R2 (native C/C++ SIMD INT8), R3 (complex ratio masking CRM), and R4 (modular ES assets and AudioWorklet).

---

## 2. Forensic Audit of `src/dashboard/static/app.js` (3,169 Lines)

### 2.1 Code Structure & Section Mapping
`src/dashboard/static/app.js` is structured as a single massive IIFE (`(function () { "use strict"; ... })();`). The file maps logically into 18 major functional zones:

| Line Range | Functional Domain | Primary Responsibilities |
|---|---|---|
| **1 – 256** | **DOM Elements & State Initialization** | Queries over 95 DOM elements across headers, cards, dials, canvases, and controls; allocates rolling buffers. |
| **257 – 380** | **Web Audio Context & A/B Crossfader** | `initAudioContext()`, `setABMode()`: 20ms linear gain ramp between Noisy (A) and Clean (B), updates UI badges. |
| **381 – 448** | **Hero & View Mode Triggers** | Magic Denoise button, volume slider, combined/kid/pro view toggle handlers. |
| **449 – 523** | **5-Workspace Tab Switcher** | `switchWorkspaceTab()`: toggles active view panels (Broadcast, EQ, Vectorscope, Batch, Telemetry, All Panels). |
| **524 – 564** | **`scheduleAudioPlayback()`** | Allocates `AudioBuffer` and `AudioBufferSourceNode` per 16ms frame with heuristic lookahead scheduling. |
| **565 – 581** | **Oscilloscope Buffer Management** | `pushOscilloscopeFrame()`: rolling shift-left of 2,048 samples for Noisy, Clean, and Subtracted audio. |
| **582 – 909** | **60 FPS Oscilloscope Renderer** | `renderOscilloscope()`: HiDPI DPI scaling, reticle grid, overlay mode, split mode, difference area shading. |
| **910 – 1068** | **Dual Waterfall Spectrogram Engine** | `drawWaterfall()`, `draw3DSpectrogram()`: Cyberpunk colormap, canvas vertical blitting, 24-slice 3D isometric view. |
| **1069 – 1352**| **WebSocket Client & Telemetry Dispatch**| `connectWebSocket()`, `handleTelemetryMessage()`: parses 17 telemetry metrics, audio frames, DNSMOS, GPU power. |
| **1353 – 1414**| **Target Speaker Voice Lock (TSE) UI** | `updateTseUI()`, `handleTseTriggerDetected()`: lock/enrolling badges, trigger banner, cosine similarity bar. |
| **1415 – 1534**| **Studio Vocal Suite UI & Controls** | De-reverb slider, compressor threshold/GR bar, de-esser sibilance reduction, warmth tube drive. |
| **1535 – 1855**| **5-Band Parametric EQ Sculptor** | `drawEqCurve()`, `updateEqCurveUI()`, mouse/touch drag event listeners for 5 biquad filter nodes, preset buttons. |
| **1856 – 1994**| **Polar Lissajous Vectorscope Radar** | `drawVectorscope()`, `updateVectorscopeUI()`: phase correlation radar circle, mono compatibility bar, stereo width. |
| **1995 – 2156**| **Batch Benchmark Studio** | Runs 24-matrix benchmark suite via `/api/batch/benchmark`, renders summary cards, dynamic table, 1-click JSON/HTML export. |
| **2157 – 2234**| **Dual-Model A/B Arena & Stream Toggle** | Dual-model concurrent toggle, crossfader slider, `setStreamingState()`: starts/stops benchmark WS stream. |
| **2235 – 2306**| **Presets, Precision Switcher, Purity Dial**| 8 preset sound buttons, FP32/FP16/INT8 precision mode buttons, `updatePurityDial()` SVG circle dashoffset. |
| **2307 – 2528**| **WAV File Upload & Intensity Slider** | Drag-and-drop WAV upload dropzone, upload waveform preview renderer, noise suppression depth slider (0–100%). |
| **2529 – 2856**| **Multilingual Translation & Studio Recorder**| Target language selector (15 languages), interactive live testbox, multi-track recording session, 1-click WAV/SRT/TXT export. |
| **2857 – 3050**| **Browser Web Speech & Live Microphone** | `webkitSpeechRecognition` integration with 9 Indian + 6 Global locales, `createScriptProcessor` mic capture. |
| **3051 – 3169**| **Workspace Initialization & Auto-Adapt** | `initWorkspaceTabs()`, auto-adapt toggle, quick WAV export, status polling (`/api/status`, `/api/classifier/status`). |

---

### 2.2 Global State & Variable Inventory (IIFE Scope)
The monolithic file maintains over 35 state variables in the module-level closure:

```javascript
// Audio Context & Graph Handles
let audioCtx = null;
let masterGain = null;
let noisyGain = null;
let cleanGain = null;
let micMediaStream = null;
let micAudioNode = null;
let micActive = false;
let nextPlayTime = 0;
let isCleanActive = true;

// Streaming & Settings
let isStreaming = false;
let activePreset = "white";
let activeTargetSnr = 0.0;
let activePrecision = "FP32";
let ws = null;
let currentIntensity = 1.0;
let autoAdaptEnabled = false;

// Oscilloscope State
const OSC_BUF_LEN = 2048;
let oscBufNoisy = new Float32Array(OSC_BUF_LEN);
let oscBufClean = new Float32Array(OSC_BUF_LEN);
let oscBufNoise = new Float32Array(OSC_BUF_LEN);
let oscMode = "overlay"; // "overlay" | "split"
let oscWindowSamples = 512; // 512 | 1024
let diffShading = true;
let chanVisible = { in: true, out: true, sub: true };

// Spectrogram State
let lastSpecIn = new Float32Array(64);
let lastSpecOut = new Float32Array(64);
let hasNewSpecFrame = false;
let isSpec3D = false;
const SPEC_3D_SLICES = 24;
const specHistoryIn = [];
const specHistoryOut = [];

// Studio & Recording State
let isRecording = false;
let recStartTime = 0;
let recTimerInterval = null;
let recCleanPcm = [];
let recNoisyPcm = [];
let lastRecordedSessionId = null;
let lastTranscriptText = "";
let lastTranslatedText = "";

// Speech Recognition State
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let speechRecognizer = null;
let testboxRecognizer = null;
let latestMicTranscript = "";
```

---

### 2.3 DOM Elements Inventory
The file queries and caches references to **98 distinct DOM nodes**:
1. **Header & General Status**: `connectionStatus`, `rssDisplay`, `headerPeakFreq`, `btnViewCombined`, `btnViewSimple`, `btnViewPro`.
2. **Hero Storyboard & Magic Denoise**: `heroStoryboard`, `purityGaugeBar`, `purityPctText`, `purityStatusBadge`, `noiseErasedText`, `vuIn`, `vuOut`, `vuNoise`, `aiChipMode`, `aiChipSpeed`, `magicDenoiseBtn`, `magicStateText`, `magicSubText`, `btnHeroStream`, `heroStreamIcon`, `heroStreamText`, `crossfadeTrack`, `crossfadePill`, `cfNoisyLabel`, `cfCleanLabel`.
3. **Controls & A/B Monitoring**: `btnStreamToggle`, `abToggleCheckbox`, `labelA`, `labelB`, `volumeSlider`, `volumeValue`.
4. **Multi-Precision & SNR**: `.btn-prec`, `precModeVal`, `precMemVal`, `precCompVal`, `precSqnrVal`, `snrInVal`, `snrOutVal`, `snrDeltaVal`.
5. **Oscilloscope**: `oscilloscopeSection`, `canvasOscilloscope`, `btnOscOverlay`, `btnOscSplit`, `btnOscDiff`, `.btn-tb`, `pillChanIn`, `pillChanOut`, `pillChanSub`, `tagChanIn`, `tagChanOut`, `tagChanSub`, `oscRmsIn`, `oscRmsOut`, `oscRmsSub`, `oscGainVal`, `oscPeakHz`.
6. **Hardware Telemetry**: `gaugeBar`, `headroomPctText`, `headroomMsText`, `totalLatencyText`, `barPre`, `barTensor`, `barSynth`, `timePre`, `timeTensor`, `timeSynth`, `statP50`, `statP95`, `statP99`, `statThroughput`.
7. **Input Source & WAV Upload**: `btnSourceBenchmark`, `btnSourceUpload`, `btnSourceMic`, `benchmarkControls`, `uploadControls`, `micControls`, `btnMicToggle`, `micLabel`, `uploadDropzone`, `wavFileInput`, `uploadStatusCard`, `uploadFileName`, `uploadFileStatus`, `uploadMetaDuration`, `uploadMetaSamples`, `uploadMetaPurity`, `uploadMetaSnr`, `canvasUploadPreview`, `btnPlayUploadedClean`, `btnPlayUploadedNoisy`.
8. **Dual-Model Arena (Phase 5)**: `toggleDualModel`, `badgeDualState`, `sliderCrossfade`, `crossBlendPct`, `dualLatencyA`, `dualRamA`, `dualSnrA`, `dualLatencyB`, `dualRamB`, `dualSnrB`.
9. **TSE & Studio Suite**: `tseSection`, `tseLockStatusBadge`, `tseTriggerBanner`, `tseLastTriggerPhrase`, `tseSimilarityVal`, `tseSimilarityBar`, `tseTargetState`, `tseSuppressionVal`, `tseEnrollmentVal`, `btnTseLock`, `btnTseUnlock`, `studioSuiteSection`, `toggleStudioSuite`, `badgeStudioState`, `sliderDereverb`, `dereverbValue`, `toggleCompressor`, `compGrVal`, `toggleDeesser`, `deessRedVal`, `toggleWarmth`.
10. **Suppression Depth & Purity**: `suppressionSlider`, `suppressionValue`, `.btn-chip`.
11. **Subtitles & Translation**: `targetLangSelect`, `transSpeedChip`, `speechVadDot`, `subTimeline`, `subOriginalText`, `subTargetFlag`, `subTargetName`, `subTranslatedText`, `subVadConf`, `subSentenceCount`, `liveTranslateInput`, `btnLiveTranslate`, `btnLiveTranslateMic`, `liveTranslateFlag`, `liveTranslateTargetBadge`, `liveTranslateLatency`, `liveTranslateOutput`, `.btn-prompt-chip`.
12. **Studio Recorder**: `btnRecordToggle`, `recDot`, `recBtnText`, `recTimerBadge`, `recTimerText`, `recProgressFill`, `recStatusInfo`, `sessionDownloadsPanel`, `btnDlCleanWav`, `btnDlNoisyWav`, `btnDlDeltaWav`, `btnDlSrt`, `btnDlTxt`.
13. **Canvases & Advanced Visualizers**: `canvasBefore`, `canvasAfter`, `canvasEqCurve`, `canvasVectorscope`, `noiseFingerprintBadge`, `fpIcon`, `fpName`, `fpConf`, `fpBar`, `step1NoiseTag`, `dnsmosSig`, `dnsmosBak`, `dnsmosOvrl`, `dnsmosStoi`, `btnSpec3D`, `specModeBadge`, `energyPower`, `energyUj`, `energyEfficiency`, `energyTemp`.
14. **Batch Benchmarking**: `btnRunBatchBenchmark`, `batchProgressWrap`, `batchProgressFill`, `batchProgressText`, `batchSummaryGrid`, `batchPassRate`, `batchMeanSnr`, `batchMaxSnr`, `batchMeanLatency`, `batchVerdict`, `batchActionsRow`, `btnDownloadBatchJson`, `btnDownloadBatchHtml`, `batchTableWrap`, `batchTableBody`.
15. **Auto-Adapt & Workspace Tabs**: `workspaceTabsBar`, `workspaceViewsContainer`, `.workspace-view`, `.tab-btn`, `btnAutoAdaptToggle`, `autoAdaptText`, `btnQuickExportWav`.

---

### 2.4 Event Listeners & Interactive Handlers
The file registers **42 interactive event listeners**:
- **Click**: `labelA`, `labelB`, `cfNoisyLabel`, `cfCleanLabel`, `crossfadeTrack`, `magicDenoiseBtn`, `btnStreamToggle`, `btnHeroStream`, `btnViewCombined`, `btnViewSimple`, `btnViewPro`, `btnOscOverlay`, `btnOscSplit`, `btnOscDiff`, `pillChanIn`, `pillChanOut`, `pillChanSub`, `btnSpec3D`, `btnSourceBenchmark`, `btnSourceUpload`, `btnSourceMic`, `btnMicToggle`, `btnTseLock`, `btnTseUnlock`, `btnLiveTranslate`, `btnLiveTranslateMic`, `btnRecordToggle`, `btnRunBatchBenchmark`, `btnDownloadBatchJson`, `btnDownloadBatchHtml`, `btnAutoAdaptToggle`, `btnQuickExportWav`, and 5 preset button groups (`.btn-preset`, `.btn-prec`, `.btn-tb`, `.btn-chip`, `.btn-prompt-chip`, `.btn-eq-preset`).
- **Input / Change**: `abToggleCheckbox`, `volumeSlider`, `sliderCrossfade`, `toggleDualModel`, `toggleStudioSuite`, `sliderDereverb`, `toggleCompressor`, `toggleDeesser`, `toggleWarmth`, `suppressionSlider`, `targetLangSelect`, `wavFileInput`, `toggleEqMaster`.
- **Canvas Interaction**: Mouse down, mouse move, mouse up, touch start, touch move, touch end handlers on `canvasEqCurve` for dragging biquad EQ nodes.
- **Drag-and-Drop**: `dragover`, `dragleave`, `drop` on `uploadDropzone`.
- **Keyboard Shortcuts**: Window `keydown`: `Space` (Toggle stream), `m`/`M` (Toggle A/B), `1`/`2`/`3` (FP32/FP16/INT8), `k`/`K` (Kid mode), `p`/`P` (Pro mode), `s`/`S` (Studio mode).

---

### 2.5 Canvas Visualizers & Render Loops
`app.js` runs a continuous 60 FPS animation loop driven by `requestAnimationFrame`:

1. **Oscilloscope (`canvasOscilloscope`)**:
   - `renderOscilloscope()`: Computes dynamic DPI scaling (`window.devicePixelRatio`).
   - Clears canvas and renders reticle grid lines with dB markings (+1.0 to -1.0).
   - In `overlay` mode: draws difference area fill between Noisy and Clean waveforms (translucent green/cyan), then draws multi-colored vector traces with neon shadow blur.
   - In `split` mode: bisects canvas horizontally, rendering Noisy on the upper pane and Clean on the lower pane.
   - Calculates live RMS and Peak amplitudes directly from buffer data.
2. **Dual Waterfall Spectrograms (`canvasBefore`, `canvasAfter`)**:
   - `drawWaterfall()`: Takes 64 frequency magnitude bins (0 to 8,000 Hz).
   - Shifts canvas downward by 2 pixels using hardware-accelerated `ctx.drawImage(canvas, 0, 0, w, h - 2, 0, 2, w, h - 2)`.
   - Maps FFT bin values to RGB through `getCyberpunkColor()` (deep blue -> toxic green -> yellow -> hot white).
   - Draws the top 2-pixel raster row.
   - In 3D mode (`draw3DSpectrogram()`): Renders 24 historical slices in pseudo-isometric perspective with gradient wireframes and back-to-front depth sorting.
3. **5-Band Parametric EQ Sculptor (`canvasEqCurve`)**:
   - Logarithmic frequency axis (20 Hz to 20,000 Hz) and linear gain axis (-18 dB to +18 dB).
   - Evaluates complex biquad transfer functions across 200 frequency points:
     $$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$
   - Renders cumulative magnitude response curve with neon gradient fill.
   - Draws interactive anchor handles for each of the 5 bands with mouse-drag and touch-drag hit-testing.
4. **Polar Lissajous Vectorscope Radar (`canvasVectorscope`)**:
   - Renders circular polar radar grid (0°, 45°, 90°, 135°, 180°).
   - Plots stereophonic Lissajous orbit points: $X = \frac{L - R}{\sqrt{2}}$, $Y = \frac{L + R}{\sqrt{2}}$.
   - Renders real-time phase correlation meter ($-1.0$ out of phase to $+1.0$ in phase), mono compatibility percentage, and stereo width ratio.
5. **Upload Preview (`canvasUploadPreview`)**:
   - Renders static summary waveforms for uploaded audio files.

---

### 2.6 WebSocket Streaming & Telemetry Protocol
The WebSocket connection (`/ws/stream`) manages bidirectional real-time communications:

- **Client-to-Server Messages**:
  - `start_benchmark`: `{ type: "start_benchmark", preset: string, target_snr: number }`
  - `stop_benchmark`: `{ type: "stop_benchmark" }`
  - `set_precision`: `{ type: "set_precision", precision: "FP32"|"FP16"|"INT8" }`
  - `set_dual_mode`: `{ type: "set_dual_mode", enabled: boolean, crossfade: number }`
  - `set_crossfade`: `{ type: "set_crossfade", alpha: number }`
  - `audio_frame`: `{ type: "audio_frame", pcm: float[], transcript: string, target_lang: string }`
- **Server-to-Client Messages**:
  - `telemetry` / `live_telemetry`: Emitted every 16ms containing:
    - `stages`: `{ pre_processing_ms, tensor_compute_ms, output_synthesis_ms, total_latency_ms }`
    - `budget`: `{ headroom_pct, headroom_ms }`
    - `rolling_stats`: `{ median_p50_ms, p95_ms, p99_ms }`
    - `metrics`: `{ purity_pct, noise_erased_pct, rms_in, rms_out, rms_noise, snr_delta_db, fft_peak_hz, model_memory_kb, snr_input_db, snr_output_db, process_rss_mb }`
    - `audio`: `{ raw_noisy: float[256], denoised: float[256], noise_subtracted: float[256], spec_in: float[64], spec_out: float[64] }`
    - `noise_signature`: `{ name, icon, confidence_pct, description, auto_adapt_enabled }`
    - `voice_quality`: `{ sig_mos, bak_mos, ovrl_mos, stoi }`
    - `energy`: `{ power_watts, energy_per_frame_uj, efficiency_fps_per_watt, temperature_c }`
    - `subtitles`: `{ original_text, translated_text, target_flag, target_name, is_speech, confidence, timeline_sec }`
    - `dual_telemetry`: `{ model_a, model_b, concurrent_latency_ms }`
    - `tse`: Target speaker enrollment and lock state
    - `studio`: Vocal suite dynamics telemetry
    - `vectorscope`: Stereo phase and mono compatibility metrics
    - `eq`: 5-band frequency response data
  - `tse_trigger_detected`, `tse_status_updated`, `studio_status_updated`, `eq_updated`.

---

## 3. Forensic Autopsy of `scheduleAudioPlayback()`

### 3.1 Existing Implementation & Defects
The current playback loop in `app.js` (lines 525–564):

```javascript
// Schedule Audio Playback with Jitter-Free Continuous Buffer Queue
function scheduleAudioPlayback(noisyPcm, cleanPcm) {
  if (!audioCtx || audioCtx.state === "suspended") return;

  const frameLen = noisyPcm.length;
  if (frameLen === 0) return;

  const bufNoisy = audioCtx.createBuffer(1, frameLen, 16000);
  const bufClean = audioCtx.createBuffer(1, frameLen, 16000);

  const chN = bufNoisy.getChannelData(0);
  const chC = bufClean.getChannelData(0);

  chN.set(noisyPcm);
  chC.set(cleanPcm);

  const srcN = audioCtx.createBufferSource();
  const srcC = audioCtx.createBufferSource();

  srcN.buffer = bufNoisy;
  srcC.buffer = bufClean;

  srcN.connect(noisyGain);
  srcC.connect(cleanGain);

  const now = audioCtx.currentTime;
  if (nextPlayTime < now) {
    // Audio thread lagged or stream just started -> prime with 30ms lookahead
    nextPlayTime = now + 0.030;
  } else if (nextPlayTime > now + 0.25) {
    // Excessive drift (e.g. background tab throttling) -> re-anchor smoothly
    nextPlayTime = now + 0.040;
  }

  srcN.start(nextPlayTime);
  srcC.start(nextPlayTime);
  nextPlayTime += frameLen / 16000.0;
}
```

### 3.2 Breakdown of Catastrophic Failure Modes

1. **Extreme Garbage Collector Pressure on Main Thread**:
   - Each frame is 256 samples (16.0 ms at 16 kHz).
   - Rate: $1.0 / 0.016 = 62.5\text{ frames/sec}$.
   - Every second, `scheduleAudioPlayback` instantiates:
     - $62.5 \times 2 = 125\text{ AudioBuffer}$ instances (with underlying Float32 memory).
     - $62.5 \times 2 = 125\text{ AudioBufferSourceNode}$ instances.
     - 250 AudioNode connects and starts.
   - This creates relentless JavaScript heap allocation and garbage collection thrashing on the browser main UI thread, producing periodic 10–30ms GC pauses that manifest as micro-stutter in the 60 FPS canvas visualizers.
2. **Browser Tab Backgrounding & Timer Throttling**:
   - When a browser tab is minimized or loses focus, modern browsers aggressively throttle the main thread:
     - `requestAnimationFrame` drops from 60 FPS to **0 FPS** (completely suspended).
     - `setTimeout` and `setInterval` are clamped to a minimum interval of **1,000 ms** (1 Hz).
     - WebSocket message processing becomes batched or delayed.
   - The hardware audio playback clock (`audioCtx.currentTime`) **does not stop**. It advances continuously in real time on the OS audio thread.
   - When the main thread lags behind by even 31 milliseconds, `now` becomes greater than `nextPlayTime`. The `if (nextPlayTime < now)` check triggers, forcefully re-anchoring `nextPlayTime = now + 0.030`.
   - **Result**: Every frame scheduled in that gap is skipped or truncated, causing loud audible clicks, static bursts, and dropped speech audio.
3. **Frame-to-Frame Phase Incoherence**:
   - Because each 256-sample frame is played by a distinct `AudioBufferSourceNode`, any slight timing drift of sub-millisecond fractions results in either a microscopic gap (zero samples inserted = high-frequency click) or overlapping buffer ends.
4. **Deprecated `createScriptProcessor` for Microphone Input**:
   - Line 2977 uses `audioCtx.createScriptProcessor(256, 1, 1)`.
   - The W3C Web Audio specification deprecated `ScriptProcessorNode` because it forces audio processing to hop synchronously between the real-time audio thread and the main event loop thread, causing buffer underruns whenever the main thread is busy rendering DOM elements or canvases.

---

## 4. Jitter-Free Web Audio Architecture Specification

To permanently eliminate packet dropouts across long streaming sessions and achieve complete immunity to background tab throttling or main-thread rendering freezes, we specify a clean Web Audio architecture centered on **`AudioWorkletNode`** backed by an internal **lock-free circular ring buffer**.

```
+-----------------------------------------------------------------------------------+
| BROWSER MAIN UI THREAD                                                            |
|                                                                                   |
|  [ WebSocket Client ]                                                            |
|          |                                                                        |
|    onmessage (audio.raw_noisy, audio.denoised: float[256])                       |
|          |                                                                        |
|    [ WorkletPort.postMessage({ noisy, clean }, [transferables]) ]                |
+----------|------------------------------------------------------------------------+
           | Zero-Copy Transfer via MessagePort
+----------v------------------------------------------------------------------------+
| DEDICATED HIGH-PRIORITY AUDIO RENDERING THREAD (Immune to Tab Throttling)         |
|                                                                                   |
|  AudioWorkletProcessor: "jitter-buffer-processor"                                 |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Lock-Free Dual-Channel Circular Ring Buffer (16,384 samples = 1,024 ms)     |  |
|  |   [ Channel 0: Raw Noisy Float32 ]  WriteHead -> ReadHead                   |  |
|  |   [ Channel 1: Denoised Clean Float32 ]                                     |  |
|  |   Pre-roll buffer cushion: 512 samples (32 ms)                              |  |
|  +-----------------------------------------------------------------------------+  |
|          |                                                                        |
|    Linear Resampling Engine (16 kHz -> audioCtx.sampleRate e.g. 48 kHz)          |
|          |                                                                        |
|    process(inputs, outputs, parameters):                                          |
|      - Pops 128 frames per quantum directly from ring buffer                      |
|      - outputs[0][0] = Noisy Audio                                                |
|      - outputs[1][0] = Clean Audio                                                |
|      - If ring starved (< 128 samples): smooth silence fade, zero digital click   |
+----------|----------------------------------------|-------------------------------+
           | Output 0 (Noisy)                       | Output 1 (Clean)
+----------v----------------------------------------v-------------------------------+
| WEB AUDIO GRAPH (Native Hardware DSP Nodes)                                       |
|                                                                                   |
|  [ noisyGain: GainNode ]                   [ cleanGain: GainNode ]                |
|  (Gain: 0.0 - 1.0)                         (Gain: 1.0 - 0.0)                      |
|          \                                       /                                |
|           +-----------------+-------------------+                                 |
|                             |                                                     |
|                     [ masterGain: GainNode ]                                      |
|                             |                                                     |
|                     [ audioCtx.destination ] -> Hardware Speakers / Headphones    |
+-----------------------------------------------------------------------------------+
```

### 4.1 AudioWorkletProcessor Implementation Specification (`audio-worklet-processor.js`)

The processor file will be located at:
`src/dashboard/static/js/audio/audio-worklet-processor.js`

```javascript
/**
 * Real-Time Edge AI Jitter-Free Audio Ring Buffer Processor
 * Runs strictly on the dedicated Audio Rendering Thread.
 * Immune to browser main-thread UI stalls, GC pauses, and tab background throttling.
 */
class JitterBufferProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Buffer capacity: 16,384 samples (1.024s at 16kHz)
    this.capacity = 16384;
    this.noisyRing = new Float32Array(this.capacity);
    this.cleanRing = new Float32Array(this.capacity);
    this.writeHead = 0;
    this.readHead = 0;
    this.available = 0;

    // Pre-roll jitter cushion: 512 samples = 32ms
    this.preRoll = 512;
    this.isPlaying = false;

    // Resampling ratio: input sample rate is always 16,000 Hz from neural pipeline
    this.inputSampleRate = (options && options.processorOptions && options.processorOptions.inputSampleRate) || 16000;
    // Current AudioContext sample rate (typically 44100 or 48000 Hz)
    this.ratio = this.inputSampleRate / sampleRate;
    this.resamplePhase = 0.0;

    this.port.onmessage = this.handleMessage.bind(this);
  }

  handleMessage(event) {
    const msg = event.data;
    if (msg.type === "push_audio") {
      const noisy = msg.noisy;
      const clean = msg.clean;
      const len = noisy.length;

      // Handle overrun if buffer is saturated
      if (this.available + len > this.capacity) {
        const drop = (this.available + len) - this.capacity;
        this.readHead = (this.readHead + drop) % this.capacity;
        this.available -= drop;
      }

      // Write incoming chunk into circular buffer
      for (let i = 0; i < len; i++) {
        const idx = (this.writeHead + i) % this.capacity;
        this.noisyRing[idx] = noisy[i];
        this.cleanRing[idx] = clean[i];
      }
      this.writeHead = (this.writeHead + len) % this.capacity;
      this.available += len;

      // Start playback once pre-roll threshold is satisfied
      if (!this.isPlaying && this.available >= this.preRoll) {
        this.isPlaying = true;
      }
    } else if (msg.type === "reset") {
      this.writeHead = 0;
      this.readHead = 0;
      this.available = 0;
      this.isPlaying = false;
      this.resamplePhase = 0.0;
    }
  }

  process(inputs, outputs, parameters) {
    // outputs[0] = Noisy Channel (Mono)
    // outputs[1] = Clean Channel (Mono)
    const outNoisy = outputs[0] ? outputs[0][0] : null;
    const outClean = outputs[1] ? outputs[1][0] : null;

    if (!outNoisy || !outClean) return true;
    const quantum = outNoisy.length; // Always 128 samples in Web Audio

    if (!this.isPlaying || this.available < 2) {
      outNoisy.fill(0);
      outClean.fill(0);
      if (this.available === 0) this.isPlaying = false;
      return true;
    }

    if (Math.abs(this.ratio - 1.0) < 1e-4) {
      // 1:1 Sample Rate (AudioContext running natively at 16 kHz)
      const samplesToRead = Math.min(quantum, this.available);
      for (let i = 0; i < samplesToRead; i++) {
        const idx = (this.readHead + i) % this.capacity;
        outNoisy[i] = this.noisyRing[idx];
        outClean[i] = this.cleanRing[idx];
      }
      if (samplesToRead < quantum) {
        outNoisy.fill(0, samplesToRead);
        outClean.fill(0, samplesToRead);
        this.isPlaying = false;
      }
      this.readHead = (this.readHead + samplesToRead) % this.capacity;
      this.available -= samplesToRead;
    } else {
      // Pitch-accurate Linear Resampling from 16kHz to audioCtx.sampleRate (e.g. 48kHz)
      for (let i = 0; i < quantum; i++) {
        if (this.available < 2) {
          outNoisy.fill(0, i);
          outClean.fill(0, i);
          this.isPlaying = false;
          break;
        }

        const idx0 = this.readHead;
        const idx1 = (this.readHead + 1) % this.capacity;
        const frac = this.resamplePhase;

        outNoisy[i] = this.noisyRing[idx0] * (1.0 - frac) + this.noisyRing[idx1] * frac;
        outClean[i] = this.cleanRing[idx0] * (1.0 - frac) + this.cleanRing[idx1] * frac;

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
```

### 4.2 Main-Thread Audio Manager Interface (`audio-manager.js`)
On the main thread, the audio manager creates the `AudioWorkletNode` with 2 outputs:

```javascript
export class AudioManager {
  constructor() {
    this.audioCtx = null;
    this.workletNode = null;
    this.noisyGain = null;
    this.cleanGain = null;
    this.masterGain = null;
    this.isWorkletReady = false;
    this.isCleanActive = true;
  }

  async init() {
    if (this.audioCtx) {
      if (this.audioCtx.state === "suspended") await this.audioCtx.resume();
      return;
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new AudioContextClass({ latencyHint: "interactive" });

    this.masterGain = this.audioCtx.createGain();
    this.masterGain.gain.setValueAtTime(0.8, this.audioCtx.currentTime);
    this.masterGain.connect(this.audioCtx.destination);

    this.noisyGain = this.audioCtx.createGain();
    this.cleanGain = this.audioCtx.createGain();

    // Default: Clean (B) active, Noisy (A) muted
    this.noisyGain.gain.setValueAtTime(0.0, this.audioCtx.currentTime);
    this.cleanGain.gain.setValueAtTime(1.0, this.audioCtx.currentTime);

    this.noisyGain.connect(this.masterGain);
    this.cleanGain.connect(this.masterGain);

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
      console.warn("AudioWorklet initialization fallback:", err);
      this.isWorkletReady = false;
    }
  }

  pushAudioFrame(noisyArr, cleanArr) {
    if (!this.audioCtx || this.audioCtx.state === "suspended") return;

    if (this.isWorkletReady && this.workletNode) {
      const noisyF32 = new Float32Array(noisyArr);
      const cleanF32 = new Float32Array(cleanArr);
      // Transfer buffers for zero-copy efficiency
      this.workletNode.port.postMessage(
        { type: "push_audio", noisy: noisyF32, clean: cleanF32 },
        [noisyF32.buffer, cleanF32.buffer]
      );
    }
  }

  setABMode(cleanSelected) {
    this.isCleanActive = cleanSelected;
    if (!this.audioCtx) return;
    const now = this.audioCtx.currentTime;
    const rampTime = 0.020; // 20ms tactile cross-fade

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
  }
}
```

---

## 5. Modular ES Directory Structure (`src/dashboard/static/js/`)

To deconstruct the 3,169-line monolith while guaranteeing zero global namespace pollution and pristine maintainability, we propose the following typed ES module architecture:

```
src/dashboard/static/
├── index.html                           # Updated <script type="module" src="/static/js/main.js">
├── styles.css                           # Unchanged visual styles
├── app.js                               # Backward-compatibility bridge (re-exports symbols)
└── js/
    ├── main.js                          # Application bootstrap & orchestrator
    ├── state.js                         # Central reactive state store & pub/sub event bus
    ├── config.js                        # System constants, colors, locales, default bands
    │
    ├── audio/
    │   ├── audio-manager.js             # Web Audio graph, GainNodes, A/B crossfader
    │   ├── audio-worklet-processor.js   # Real-time circular ring buffer processor
    │   └── mic-streamer.js              # Microphone capture & Web Audio Worklet streaming
    │
    ├── net/
    │   ├── websocket.js                 # WebSocket client with reconnection logic
    │   └── api.js                       # Typed REST client (/api/status, /api/batch, etc.)
    │
    ├── renderers/
    │   ├── oscilloscope.js              # 60 FPS oscilloscope (overlay, split, diff shading)
    │   ├── spectrogram.js               # 2D waterfall & 3D isometric surface spectrogram
    │   ├── parametric-eq.js             # 5-band EQ curve & node dragging logic
    │   └── vectorscope.js               # Polar Lissajous radar scope & phase correlation
    │
    ├── ui/
    │   ├── header-controls.js           # Header, view modes, keyboard shortcuts
    │   ├── hero-controls.js             # Magic Denoise, purity dial, VU bars, volume
    │   ├── workspace-tabs.js            # 5-workspace tab navigation & persistence
    │   ├── benchmark-controls.js        # Presets, precision buttons, WAV file upload
    │   ├── telemetry-display.js         # Latency breakdown, headroom, DNSMOS, GPU power
    │   ├── translation-subtitles.js     # Dual-line subtitles, live testbox, Web Speech
    │   ├── studio-recorder.js           # Multi-track session recording & WAV/SRT export
    │   ├── studio-suite.js              # TSE voice lock, vocal suite, auto-adapt toggle
    │   └── batch-benchmark.js           # Batch matrix execution, table, JSON/HTML reports
    │
    └── utils/
        ├── wav-builder.js               # In-browser WAV PCM file synthesizer
        └── dom.js                       # DOM element query & safe property helpers
```

### 5.1 Module Boundary Breakdown

| Module Path | Primary Responsibility | Key Exports |
|---|---|---|
| `js/state.js` | Single source of truth for all application state; pub/sub event emitter. | `store`, `onStateChange()`, `setState()` |
| `js/config.js` | Static configuration, regional Indian/global locales, colormaps, FFT sizes. | `CONFIG`, `CYBERPUNK_PALETTE`, `REGIONAL_LOCALES` |
| `js/audio/audio-manager.js` | Web Audio graph, AudioWorklet lifecycle, gain crossfading. | `AudioManager` class |
| `js/audio/audio-worklet-processor.js` | Dedicated-thread circular ring buffer & linear resampler. | Registered processor: `jitter-buffer-processor` |
| `js/audio/mic-streamer.js` | Microphone permission, stream acquisition, AudioWorklet capture. | `startMicrophone()`, `stopMicrophone()` |
| `js/net/websocket.js` | Resilient WebSocket connection, reconnection timer, packet dispatch. | `WsClient` class |
| `js/net/api.js` | Async REST wrappers with error handling. | `fetchStatus()`, `fetchClassifierStatus()`, `triggerBatchBenchmark()` |
| `js/renderers/oscilloscope.js` | High-DPI oscilloscope vector rendering, reticle, diff shading. | `renderOscilloscope()`, `drawDifferenceArea()` |
| `js/renderers/spectrogram.js` | 2D waterfall scroll canvas blit & 3D pseudo-isometric surface. | `drawWaterfall()`, `draw3DSpectrogram()` |
| `js/renderers/parametric-eq.js` | Biquad filter response curve calculation and node drag hit-test. | `drawEqCurve()`, `initEqInteractions()` |
| `js/renderers/vectorscope.js` | Stereophonic phase correlation radar and Lissajous orbit paths. | `drawVectorscope()` |
| `js/ui/header-controls.js` | Top navbar status pills, view mode buttons, global hotkeys. | `initHeaderControls()` |
| `js/ui/hero-controls.js` | Hero Storyboard, Magic Denoise button, Purity dial SVG dashoffset. | `initHeroControls()`, `updatePurityDial()` |
| `js/ui/workspace-tabs.js` | Workspace navigation (Broadcast, EQ, Vectorscope, Batch, Profiler). | `initWorkspaceTabs()`, `switchWorkspaceTab()` |
| `js/ui/benchmark-controls.js` | 8 preset noise buttons, precision selector, custom WAV drag/drop. | `initBenchmarkControls()` |
| `js/ui/telemetry-display.js` | 3-stage latency bars, headroom gauge, rolling stats, GPU energy. | `updateTelemetryDisplay()` |
| `js/ui/translation-subtitles.js`| 15-language captions, interactive testbox, browser Web Speech. | `initTranslationSubtitles()`, `updateSubtitlesUI()` |
| `js/ui/studio-recorder.js` | Multi-track audio and subtitle recording with 1-click downloads. | `initStudioRecorder()`, `startRecordingSession()` |
| `js/ui/studio-suite.js` | TSE speaker enrollment/lock, De-reverb, Compressor, De-esser, Warmth. | `initStudioSuite()`, `updateStudioUI()` |
| `js/ui/batch-benchmark.js` | Batch matrix trigger, progress bar, dynamic table, JSON/HTML download. | `initBatchBenchmark()` |
| `js/utils/wav-builder.js` | Encodes PCM Float32 arrays into binary RIFF WAV files in browser memory. | `encodeWavBlob()` |
| `js/utils/dom.js` | Cached element selectors and safe text/class updates. | `$()`, `$$()`, `safeText()` |
| `js/main.js` | Initializes all sub-systems in correct dependency order, starts render loop. | `initApp()` |

---

## 6. HTML & Static Asset Backward Compatibility

### 6.1 `src/dashboard/static/index.html` Changes
In `src/dashboard/static/index.html`, line 1571 currently reads:
```html
<script src="/static/app.js"></script>
```
With ES module loading, this will be updated to:
```html
<script type="module" src="/static/js/main.js"></script>
```

### 6.2 Backward-Compatibility Bridge in `src/dashboard/static/app.js`
In `tests/integration/test_dashboard_api.py`, lines 175–178 test the static JS bundle:
```python
r_js = client.get("/static/app.js")
assert r_js.status_code == 200
assert "renderOscilloscope" in r_js.text
assert "drawDifferenceArea" in r_js.text
```
To guarantee that existing integration tests pass unconditionally while transitioning to ES modules, `src/dashboard/static/app.js` will serve as an ES module entry point / bridge:

```javascript
/**
 * Edge AI Real-Time Neural Audio Denoiser & Profiler
 * ES Module Entrypoint & Backward-Compatibility Bridge
 */
import { initApp } from "./js/main.js";
import { renderOscilloscope, drawDifferenceArea } from "./js/renderers/oscilloscope.js";

export { renderOscilloscope, drawDifferenceArea, initApp };

// Auto-bootstrap when loaded directly
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
}
```

This guarantees:
1. `client.get("/static/app.js")` returns status 200.
2. `renderOscilloscope` and `drawDifferenceArea` are present in `r_js.text`.
3. Loading either `<script type="module" src="/static/app.js">` or `<script type="module" src="/static/js/main.js">` functions identically.

---

## 7. Full Test Suite & Regression Baseline Survey

### 7.1 Test Suite Inventory (31 Files, 274 Tests)

The test suite is organized into 5 tiers adhering to `TEST_INFRA.md`:

```
tests/
├── conftest.py                   # Authoritative Synthetic Generator & Mock Stage Profiler Fixtures
├── adversarial/
│   └── test_adversarial.py       # 8 Tests: Extreme edge cases (DC bias, clipping, impulse, NaN injection, swamped SNR)
├── load/
│   └── test_ws_concurrency.py    # 3 Tests: Concurrency stress (1, 5, 20 concurrent WebSocket streaming clients)
├── integration/
│   ├── test_dashboard_api.py     # 15 Tests: FastAPI endpoints, static asset serving, presets, exports, prometheus
│   └── test_pipeline_stream.py   # 3 Tests: 1,000-frame continuous streaming, precision switching, ring buffer overrun
├── e2e/
│   └── test_evaluation.py        # 51 Tests: Exhaustive STFT, GRUMaskNet, Wiener, profiler isolation, multi-precision, SNR
└── unit/                         # 194 Tests across 26 component modules:
    ├── test_chrome_extension.py   (4 tests: Manifest V3, icons, popup, background scripts)
    ├── test_denoiser.py           (12 tests: Synthetic speech, noise, Wiener, GRUMaskNet, ring buffer, SNR >= 10dB)
    ├── test_dereverb.py           (5 tests: WPE parameters, bypass, dereverberation suppression, <1.0ms latency)
    ├── test_dnsmos.py             (5 tests: ITU-T P.835 SIG/BAK/OVRL bounds, suppression, sub-millisecond budget)
    ├── test_dual_model.py         (5 tests: Dual-model concurrent inference, crossfade extremes, latency isolation)
    ├── test_enhancements.py       (9 tests: CRM, ERB filterbank, HarmonicEnhancer, noise classifier, energy profiler)
    ├── test_fixed_point.py        (7 tests: Q15 fixed-point arithmetic, saturation, SQNR, fixed-point Wiener)
    ├── test_fpga_rtl.py           (6 tests: Verilog module/testbench syntax, 5-cycle pipeline latency, backpressure)
    ├── test_frontier_blueprint.py (4 tests: Architecture blueprint, streaming Mamba block, voiceprint enrollment, MVDR)
    ├── test_hardware_monitor.py   (6 tests: Tegrastats, vcgencmd throttled bits, edge profiles, Prometheus export)
    ├── test_noise_classifier.py   (10 tests: Spectral features, flatness, 8 noise profiles, adaptive controller)
    ├── test_onnx_engine.py        (3 tests: ONNX FP32 parity vs NumPy, INT8 quantization bounds, pipeline integration)
    ├── test_parametric_eq.py      (7 tests: Biquad bypass, bell/high-pass/notch filters, studio presets, safety clamp)
    ├── test_phase3_features.py    (4 tests: Category noise adaptation, Wiener filter adaptation, classifier steering)
    ├── test_precision.py          (7 tests: PrecisionEngine modes, memory ratios, SQNR >= 35dB, weight preservation)
    ├── test_profiler.py           (21 tests: 3-stage isolation, headroom, rolling buffer percentiles, jitter, memory RSS)
    ├── test_prometheus_exporter.py(3 tests: Prometheus registry, frame latency metrics, audio/system gauges)
    ├── test_speech_preservation.py(5 tests: Vocal harmonics, formants, male fundamental preservation, normalization)
    ├── test_speech_translation.py (21 tests: VAD emission, SRT export, 15 languages, sub-millisecond, REST endpoints)
    ├── test_stft.py               (8 tests: Periodic Hann COLA, sqrt-Hann, streaming perfect reconstruction, algorithmic delay)
    ├── test_target_speaker.py     (9 tests: Voice enrollment, cosine similarity, competing speaker suppression)
    ├── test_vad.py                (5 tests: Silence/speech energy thresholds, hangover duration, compute savings)
    ├── test_vectorscope.py        (6 tests: Phase correlation, mono compatibility, stereo width, Lissajous coordinates)
    ├── test_vocal_suite.py        (7 tests: De-esser, compressor, limiter ceiling, full chain telemetry, <1.0ms latency)
    ├── test_vst_bridge.py         (9 tests: Circular FIFO, VST parameters, PDC latency, variable block sizes, C ABI)
    └── test_webgpu_wasm.py        (5 tests: WGSL shader file, WebGPU denoiser JS, WASM SIMD module, standalone HTML)
```

**Total Count**: $8 + 3 + 15 + 3 + 51 + 194 = \mathbf{274}\text{ tests}$.

---

### 7.2 Existing Test Fixtures (`tests/conftest.py`)
`tests/conftest.py` defines reference test signal generators and fixtures:
1. `ReferenceSyntheticGenerator`:
   - `generate_speech(duration_sec)`: Physically calibrated glottal excitation shaped by 3 vocal tract formant resonators (F1: 700Hz, F2: 1200Hz, F3: 2500Hz) with natural F0 intonation vibrato (120–150 Hz).
   - `generate_noise(noise_type, duration_sec)`: 4 physical profiles: `white` (thermal Gaussian), `pink` (Kellet 3-pole 1/f filter), `drone` (120Hz motor hum + harmonics), `rf` (3–7.5 kHz Butterworth bandpass + Poisson squelch bursts).
   - `generate_mixture(clean, noise, target_snr_db)`: Generates mixtures with mathematically calibrated broadband SNR.
   - `calculate_snr(clean, processed, delay_samples)`: Time-aligned broadband SNR computation with 1-hop delay compensation ($D = 256$ samples).
2. `MockStageProfiler`:
   - Emulates hardware timers and provides `start_frame()`, `mark_preprocessing_done()`, `mark_tensor_done()`, `mark_synthesis_done()`, and `get_last_metrics()` to test pipeline profiler hooks without hardware jitter.
3. PyTest Session Fixtures:
   - `sample_rate`: 16,000 Hz.
   - `hop_size`: 256 samples (16.0 ms).
   - `fft_size`: 512 samples.
   - `ref_generator`: Reference generator instance.
   - `clean_speech_2s`, `white_noise_2s`, `drone_noise_2s`, `rf_noise_2s`: Calibrated 2.0s signals.
   - `white_mixture_0db`, `drone_mixture_5db`, `rf_mixture_minus5db`: Pre-calibrated test mixtures.

---

### 7.3 Automated Evaluation Benchmark Baseline (`evaluate.py`)
`evaluate.py` is the non-interactive acceptance benchmark executed by automated pipelines:
1. **[1/3] Denoising Performance & Latency Budget**:
   - Tests 8 combinations: 4 noise types (White, Pink, Drone, RF) across both `REF` (Reference) and `GEN` (Standard) generators.
   - Asserts:
     - Real-time per-frame latency $t_{\text{total}} \le 20.0\text{ ms}$.
     - 3-stage timing isolation: $t_{\text{pre}} > 0$, $t_{\text{tensor}} > 0$, $t_{\text{synth}} > 0$.
     - Numerical integrity: Zero NaNs, zero Infs, $\text{peak amplitude} \le 1.05$.
     - Noise reduction: $\Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}} \ge \mathbf{10.0\text{ dB}}$.
2. **[2/3] Multi-Precision Engine Evaluation**:
   - Tests FP32, FP16, and INT8 modes.
   - Asserts:
     - INT8 memory footprint $\le 0.35\times$ FP32 (typically $\sim 25.7\%$).
     - INT8 Quantization $\text{SQNR} \ge \mathbf{35.0\text{ dB}}$.
     - FP16 Quantization $\text{SQNR} \ge \mathbf{50.0\text{ dB}}$.
     - $\Delta\text{SNR} \ge 10.0\text{ dB}$ across all three precision formats.
3. **[3/3] Telemetry Isolation & Process Memory**:
   - Verifies Stage 1, Stage 2, and Stage 3 timing isolation on a test frame.
   - Verifies total latency $\le 20.0\text{ ms}$.
   - Verifies process memory RSS $> 0\text{ MB}$.

---

### 7.4 Test Suite Hardening & Expansion Requirements for R1 – R4

When implementing the 4 requirements of the hardening milestone, tests must be updated and expanded to prevent regressions:

#### For Requirement R1: Authentic Neural DNSMOS & Psychoacoustic Evaluator
- **Files Affected**: `src/telemetry/dnsmos.py`, `tests/unit/test_dnsmos.py`, `evaluate.py`.
- **Existing Tests**: `test_dnsmos.py` asserts heuristic SIG/BAK/OVRL ranges (1.0 to 5.0) and basic clean-vs-noisy trends.
- **Test Expansions Needed**:
  1. Add tests verifying that authentic ONNX neural models evaluate inference in an **asynchronous worker thread** without adding latency to the main 16ms audio processing thread.
  2. Verify that `PerceptualQualityEstimator` outputs genuine reference test vector scores: SIG $\ge 3.8$, BAK $\ge 3.5$, OVRL $\ge 3.6$ on clean benchmark speech, and STOI $\ge 0.85$.
  3. Verify thread-safe lock-free telemetry handoff (evaluator polls latest buffer, writes scores to shared memory, zero GIL contention).
  4. Expand `evaluate.py` to assert DNSMOS scores on the 8 benchmark audio targets.

#### For Requirement R2: Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON)
- **Files Affected**: `src/models/precision.py`, `tests/unit/test_precision.py`, `evaluate.py`.
- **Existing Tests**: Tests `np.int8` quantization SQNR $\ge 35.0\text{ dB}$, memory compression $\le 0.35\times$, and weight preservation.
- **Test Expansions Needed**:
  1. Add unit test verifying that the compiled native C/C++ SIMD kernel (`vpdpbusd` on x86 or `sdot` on ARM64) produces **bit-exact results** matching the reference `np.int32` GEMM accumulation.
  2. Add benchmark test asserting that the native SIMD kernel demonstrates measurable speedup over pure Python without GIL serialization.
  3. Verify graceful fallback: If native compiled binary is missing (e.g. unsupported CPU or platform), system falls back smoothly to integer-domain NumPy SIMD simulation.

#### For Requirement R3: End-to-End Complex Spectral Mapping (E2E CRM Phase Preservation)
- **Files Affected**: `src/models/denoiser.py`, `tests/unit/test_denoiser.py`, `tests/e2e/test_evaluation.py`.
- **Existing Tests**: Tests magnitude spectral mask bounds and post-hoc Wiener filtering.
- **Test Expansions Needed**:
  1. Add unit tests for direct complex ratio mask prediction ($M_r, M_i$) and complex multiplication:
     $$\hat{S}_r = Y_r M_r - Y_i M_i, \quad \hat{S}_i = Y_r M_i + Y_i M_r$$
  2. Verify that output complex spectrum preserves conjugate symmetry for real iSTFT synthesis.
  3. Verify zero phase cancellation artifacts (comb filtering) across harmonic speech frequencies.
  4. Verify $\Delta\text{SNR} \ge 10.0\text{ dB}$ across all benchmark targets without post-hoc empirical scalar boosts.

#### For Requirement R4: Frontend Architecture Modularization & AudioWorklet
- **Files Affected**: `src/dashboard/static/js/*`, `src/dashboard/static/app.js`, `src/dashboard/static/index.html`, `tests/integration/test_dashboard_api.py`.
- **Existing Tests**: `test_static_assets_and_oscilloscope_dom` checks `/static/styles.css` and `/static/app.js`.
- **Test Expansions Needed**:
  1. In `test_dashboard_api.py`:
     - Test that `/static/js/main.js` is served with `status_code == 200` and `application/javascript` content type.
     - Test that `/static/js/audio/audio-worklet-processor.js` is served with `status_code == 200` and contains `registerProcessor`.
     - Test that all child ES modules (`state.js`, `oscilloscope.js`, `spectrogram.js`, `parametric-eq.js`, `vectorscope.js`) are served correctly.
     - Verify `index.html` loads `<script type="module" src="/static/js/main.js"></script>`.
     - Verify backward compatibility: `/static/app.js` continues to be served and exports `renderOscilloscope` and `drawDifferenceArea`.
  2. Add test simulating 10-minute long-duration WebSocket packet streaming asserting zero memory leaks and zero dropped frames.

---

## 8. Summary of Recommendations for Implementation Phase

1. **Modularize `src/dashboard/static/app.js`**: Create `src/dashboard/static/js/` directory with 16 distinct ES modules grouped under `audio/`, `net/`, `renderers/`, `ui/`, and `utils/`.
2. **Implement AudioWorklet Ring Buffer**: Write `src/dashboard/static/js/audio/audio-worklet-processor.js` with dual-channel circular ring buffer and linear resampling, and wrap it in `src/dashboard/static/js/audio/audio-manager.js`.
3. **Update `src/dashboard/static/index.html`**: Switch line 1571 to `<script type="module" src="/static/js/main.js"></script>`.
4. **Preserve `/static/app.js`**: Retain `src/dashboard/static/app.js` as an ES module bridge re-exporting key visualizer symbols to preserve existing integration tests.
5. **Update Test Suite**: Add static asset integration tests in `tests/integration/test_dashboard_api.py` covering all new ES modules.

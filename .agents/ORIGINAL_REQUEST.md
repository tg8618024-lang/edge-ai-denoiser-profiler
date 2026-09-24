# Original User Request

## Initial Request — 2026-09-06T17:42:09Z

<USER_REQUEST>
A real-time audio and signal testbench inspired by NVIDIA RTX Voice / Broadcast. It takes noisy microphone or RF audio, runs it through a lightweight deep neural network (or DSP spectral filter), removes background noise in real-time, and profiles edge inference metrics (latency breakdown across pre-processing, tensor compute, output synthesis, and precision trade-offs across FP32, FP16, and INT8) with interactive visualization.

Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Integrity mode: development

## Requirements

### R1. Real-Time Neural Audio Denoising Pipeline
Implement a low-latency audio processing pipeline that takes incoming audio streams in small frame buffers (e.g. 10–20ms chunks) and applies a lightweight neural denoising model or spectral subtraction filter to suppress background acoustic noise and RF static while preserving speech and signal clarity. Support both live microphone streaming and pre-recorded benchmark audio files (e.g. noisy speech, RF static, fan/drone humming).

### R2. NVIDIA-Style Latency & Pipeline Profiler
Instrument the audio and inference pipeline with fine-grained per-frame telemetry. The profiler must measure and log elapsed time (in milliseconds) partitioned across three distinct stages: Pre-processing (STFT, framing, windowing), Tensor Compute (neural network or DSP filtering inference), and Output Synthesis (iSTFT, overlap-add). Compute rolling statistics including median, P95, and P99 latency against real-time frame budget constraints.

### R3. Multi-Precision Comparison Mode (FP32, FP16, INT8)
Support running inference across FP32, FP16, and INT8 quantized representations. Measure and report the trade-offs between precision formats in terms of latency speedup, memory footprint, and output signal quality (e.g. Signal-to-Noise Ratio [SNR] improvement).

### R4. Interactive Visual Dashboard & Waterfall Spectrogram
Provide a modern web dashboard featuring:
- A real-time dual Before/After STFT waterfall spectrogram visualizer.
- An interactive A/B audio toggle to switch between noisy input and cleaned output seamlessly during playback.
- An NVIDIA-inspired latency breakdown visualization displaying real-time stage millisecond graphs and budget headroom indicators.
- Precision mode selectors (FP32 / FP16 / INT8) dynamically reflecting performance and quality metrics.

## Acceptance Criteria

### Denoising Quality & Real-Time Constraints
- [ ] Denoising pipeline achieves at least 10 dB SNR improvement on noisy benchmark audio clips without audible clipping or severe speech degradation.
- [ ] Per-frame processing latency operates reliably within the real-time frame budget (e.g., <= 20ms per frame) during continuous streaming.

### Profiling & Multi-Precision Telemetry
- [ ] Latency profiler accurately isolates and reports millisecond metrics for all three stages: Pre-processing, Tensor Compute, and Output Synthesis.
- [ ] Precision comparison mode runs and logs performance and memory deltas for FP32, FP16, and INT8 models/configurations.

### Testbench Functionality & UI
- [ ] Modern web UI launches and connects to the backend, rendering real-time dual before/after waterfall spectrograms.
- [ ] Audio controls allow playback, pausing, live mic capture, and instant A/B switching between original noisy audio and denoised audio.

### Independent Verification
- [ ] An automated, non-interactive evaluation script (e.g., `python -m pytest` or `python evaluate.py`) executes end-to-end on bundled benchmark test audio, verifying noise reduction metrics (SNR >= 10 dB), latency profiling assertions, and precision switching, exiting with code 0.
</USER_REQUEST>

## Follow-up — 2026-09-19T14:50:17Z

<USER_REQUEST>
This is a single self-contained fix; keep it small and focused.

Upgrade the Real-Time Speech-to-Text Transcriber and Multilingual Translation Engine to top-tier studio quality: resolve streaming timeline wraparound and token fallback bugs, achieve complete natural translation across all 9 Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi) and 6 Global languages (Spanish, French, German, Japanese, Chinese, Italian), integrate browser Web Speech recognition for live mic input, and elevate the translation and subtitle suite to a prominent Top-of-Page Hero Suite directly below the header.

Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Integrity mode: development

## Requirements

### R1. Robust Streaming & Live Speech-to-Text Engine
Fix timestamp wraparound and VAD emission in streaming ASR so continuous benchmark playback never stalls or freezes words when audio loops. Integrate live browser Web Speech recognition (`webkitSpeechRecognition` with regional Indian locales like `hi-IN`, `ta-IN`, `te-IN`, etc.) in microphone mode, passing transcribed speech directly to the backend.

### R2. Top-Tier Multilingual Translation Engine (Indian & Global Languages)
Eliminate broken partial-token translations (no mixed-language strings). Implement full natural phrase mappings and comprehensive vocabulary translation dictionaries covering all benchmark utterances, greetings, tech commands, and system states across all 9 Indian languages and 6 Global languages. Guarantee sub-millisecond execution (<0.1 ms) with native script Unicode typography.

### R3. Top-Level Hero Suite UI & Interactive Controls
Relocate the Live Subtitles and Multilingual Translation card to the top level of the dashboard as a prominent Hero Suite directly below the header. Provide:
- Glowing dual-line caption display (Original English + Target Native Script).
- Interactive Live Translate test box allowing users to type or speak any custom text and see instant translations.
- Target language dropdown prominently featuring the 🇮🇳 Indian Languages Suite at the top.
- Integrated multi-track session recorder and 1-click downloads (Clean WAV, Raw WAV, Delta WAV, SRT Subtitles, Transcript TXT).

## Acceptance Criteria

### Translation Quality & Zero Broken Strings
- [ ] No partial or mixed-language fallback output (e.g., no "Listening for आवाज़ stream..."); full natural sentence translation across all 15 supported languages.
- [ ] Translation inference executes in under 1.0 ms per frame without cloud API costs or network latency.

### Streaming Transcription & Continuous Playback
- [ ] ASR engine maintains continuous streaming and word emission without stalling or freezing when audio loops or repeats.
- [ ] Live microphone input streams real-time recognized speech directly into the translation engine.

### Studio UI Prominence & Verification
- [ ] Translation and subtitle suite is positioned at the top level of the visual dashboard with dual-line native script display.
- [ ] All automated tests (`pytest`) pass 100%, and `evaluate.py` exits with code 0.
</USER_REQUEST>

## Follow-up — 2026-09-22T18:36:21Z

<USER_REQUEST>
Systematically resolve all roasted forensic defects across the Edge AI Audio Denoiser & Profiler testbench with pinpoint accuracy: expand pitch autocorrelation history to 512 samples to prevent false harmonic comb teeth on transient noises, ensure the 5-band parametric EQ is active and reactive to canvas node dragging, upgrade INT8 quantization to authentic integer-domain matrix multiplication simulation (int8 operands with int32 accumulation), expand multilingual translation vocabulary to prevent mixed-language fallback fragments, and verify the entire UI and test suite via browser and automated test runners.

Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Integrity mode: development

## Requirements

### R1. Robust Pitch Detection & Harmonic Comb Filtering
Eliminate false pitch hallucinations caused by 56-sample autocorrelation truncation in `HarmonicEnhancer`. Implement a circular 512-sample history window so autocorrelation across the full lag range (40--200 samples) maintains at least 312 samples of continuous overlap, preventing noise spikes or desk bumps from falsely triggering harmonic comb teeth.

### R2. Responsive Parametric EQ & Canvas Node Synchronization
Ensure the 5-band parametric equalizer is enabled and reactive by default upon user interaction, guaranteeing that dragging biquad filter nodes or selecting studio presets instantly alters the live audio processing stream.

### R3. Authentic Integer-Domain INT8 Quantization Simulation
Refactor `src/models/precision.py` to eliminate the float32 cast-and-multiply pseudo-quantization. Implement authentic integer arithmetic simulation: quantize weights and inputs to `np.int8`, perform GEMM accumulation using 32-bit integers (`np.int32`), and apply output scaling and bias addition to match real edge NPU/DSP hardware execution.

### R4. Complete Multilingual Translation Grammar & Phrasing
Expand the translation engine dictionary and tokenizer in `src/models/translator.py` to ensure complete, natural sentences across all 9 Indian languages and 6 Global languages without leftover untranslated English fragments.

### R5. Comprehensive Browser & Automated Regression Verification
Audit the running dashboard (http://127.0.0.1:8000) using browser inspection to ensure all 5 workspaces, canvases (Oscilloscope, Waterfall Spectrogram, Parametric EQ, Polar Vectorscope), and audio controls load and operate with zero JavaScript console errors or dropped Web Audio buffers. Ensure 100% of tests pass in `pytest -q` and `evaluate.py` exits with code 0.

## Acceptance Criteria

### Audio DSP & Quantization Correctness
- [ ] `HarmonicEnhancer` autocorrelation maintains >= 300 samples of overlap across all pitch lags (40--200 samples), with zero comb distortion on noisy transient inputs.
- [ ] Parametric EQ directly processes streaming audio upon node movement without requiring manual hidden toggles.
- [ ] INT8 inference performs accumulation in 32-bit integers (`np.int32`) from 8-bit integer operands (`np.int8`), matching hardware SIMD integer dot-product arithmetic.

### Translation & Stability
- [ ] Benchmark utterances translate into 100% fluent native script across all 15 languages without remaining English fragment words.
- [ ] 100% of tests pass in `pytest -q`, and `evaluate.py` exits with code 0 achieving >= 10.0 dB SNR gain across all benchmark profiles.

### Visual & Browser Audit
- [ ] Live dashboard (http://127.0.0.1:8000) loads all 5 workspaces without JavaScript console errors or dropped Web Audio buffers.
</USER_REQUEST>

## Follow-up — 2026-09-23T11:37:17Z

<USER_REQUEST>
Execute a ruthless, uncompromising forensic audit, stress-test, and architectural hardening of the Real-Time Edge AI Audio Denoiser & Inference Profiler (`edge_ai_denoiser_profiler`), exposing and eliminating all remaining heuristic shortcuts, analytical approximations, and architectural bottlenecks to achieve genuine NVIDIA Broadcast / RTX Voice production engineering parity.

Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Integrity mode: benchmark

## Requirements

### R1. Authentic Neural DNSMOS & Psychoacoustic Evaluator Replacement
Eliminate the analytical sigmoid curve-fit formulas currently masquerading as ITU-T P.835 DNSMOS, PESQ, and STOI in `src/telemetry/dnsmos.py`. Integrate genuine lightweight ONNX evaluation graphs (e.g. Microsoft DNSMOS P.835 lightweight ONNX model or authentic pyroomacoustics/pystoi 1/3-octave band correlation) running in an asynchronous background profiler thread without impacting the 16.0 ms audio processing budget.

### R2. Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON)
Upgrade the Python-level `np.int8` matrix multiplication in `src/models/precision.py` by compiling and linking a native C/C++ or ctypes extension executing true hardware SIMD instructions (`vpdpbusd` for x86_64 / `sdot` for ARM64), eliminating Python bytecode and GIL overhead during INT8 inference benchmarking.

### R3. End-to-End Complex Spectral Mapping (E2E CRM Phase Preservation)
Refactor `GRUMaskNet` from a magnitude-only gain mask network with post-hoc CRM heuristics into an authentic complex spectral mapping architecture (predicting complex real $M_r$ and imaginary $M_i$ ratio masks directly or dual-path complex recurrent units), eliminating phase cancellation artifacts without relying on empirical boost scalars.

### R4. Frontend Architecture Modularization & Jitter-Free Audio Ring Buffer
Deconstruct the 3,169-line monolithic `src/dashboard/static/app.js` into clean, typed ES modules. Replace the naive `scheduleAudioPlayback()` packet feeding with a Web Audio API `AudioWorkletNode` backed by a shared circular ring buffer, guaranteeing click-free, glitch-free audio playback immune to browser tab backgrounding or main-thread rendering hiccups.

## Acceptance Criteria

### Psychoacoustic & Metric Authenticity
- [ ] DNSMOS (SIG, BAK, OVRL), STOI, and PESQ scores are generated by real neural/psychoacoustic inference models, verified against reference audio test vectors with zero analytical formula fallbacks.
- [ ] Profiling overhead of DNSMOS evaluation executes out-of-band in an async worker thread without adding latency to the real-time audio pipeline.

### Silicon & Compute Rigor
- [ ] INT8 inference executes via native compiled SIMD instructions (`vpdpbusd` or equivalent), demonstrating measurable speedup over FP32 without Python interpreter GIL serialization.
- [ ] Quantization SQNR >= 35.0 dB and memory footprint <= 30% of FP32 maintained.

### Architectural & UI Integrity
- [ ] `app.js` decomposed into modular, maintainable units with zero global variable namespace pollution.
- [ ] AudioWorklet playback eliminates packet dropouts across 10-minute continuous streaming sessions.
- [ ] Full regression suite passes with 100% of tests passing in `pytest` and `evaluate.py` exit code 0.
</USER_REQUEST>


# Project: Edge AI Audio Denoiser & Profiler (Hardening & Parity Phase)

## Architecture
The Edge AI Audio Denoiser & Profiler is an edge audio testbench inspired by NVIDIA RTX Voice / Broadcast.
In this hardening phase, the architecture eliminates heuristic shortcuts, analytical approximations, and main-thread bottlenecks:
1. **Authentic ITU-T P.835 Neural DNSMOS & STOI**: Replaces heuristic sigmoid curve-fits with genuine neural ONNX evaluation graphs and 1/3-octave band correlation STOI running in an asynchronous background thread (`AsyncQualityEvaluatorWorker`) with 0.000 ms impact on the 16.0 ms audio processing budget.
2. **Native C/C++ SIMD Integer-Domain Kernel**: Implements true hardware vector instructions (`vpdpbusd` for AVX-VNNI, `vpmaddubsw`/`vpmaddwd` for AVX2) via ctypes with GIL release and a zero-compiler Numba LLVM JIT vectorization layer, achieving true silicon speedup over FP32 while maintaining SQNR >= 35.0 dB and memory <= 30.0% of FP32.
3. **End-to-End Complex Spectral Mapping (E2E CRM)**: Replaces magnitude-only gain masking and empirical harmonic boost kludges with authentic complex ratio masks ($M_r, M_i$) and Cartesian complex multiplication $S = Y \cdot M = (Y_r M_r - Y_i M_i) + j(Y_r M_i + Y_i M_r)$, preserving speech phase.
4. **Modular ES Frontend & AudioWorklet Ring Buffer**: Deconstructs monolithic `app.js` into clean, typed ES modules under `src/dashboard/static/js/` with zero global namespace pollution. Replaces main-thread audio packet scheduling with a dedicated `AudioWorkletNode` backed by a 16,384-sample circular ring buffer for glitch-free playback immune to background tab throttling.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Low-latency Framing & Windowing | 10-20ms frames (256 hop, 512 FFT @ 16kHz) with square-root Hann window | M3 | R1 |
| 2 | STFT / iSTFT Engine | Machine-precision overlap-add reconstruction ($<10^{-14}$ error), 1-hop delay | M3 | R1 |
| 3 | E2E Complex Spectral Mapping GRUMaskNet | Genuine real ($M_r$) and imaginary ($M_i$) mask prediction with complex multiplication | M3 | R3 |
| 4 | Clean Phase Reconstruction | Elimination of empirical harmonic boost scalars and phase gradient heuristics | M3 | R3 |
| 5 | Decision-Directed Wiener Filter | DSP spectral gain computation preventing musical noise | M3 | R1 |
| 6 | Synthetic Benchmark Audio Generator | Glottal pulse harmonic speech + vowel formants mixed with 4 noise types | M5 | R1 |
| 7 | Benchmark SNR Gain Verification | Guarantee >= 10 dB SNR improvement across all noise benchmark clips | M3, M5 | AC |
| 8 | 3-Stage Hardware Latency Telemetry | Sub-millisecond isolation of Pre-processing, Tensor Compute, Output Synthesis | M2 | R2 |
| 9 | Multi-Precision Engine (FP32) | Baseline 32-bit floating point inference kernel | M2 | R3 |
| 10 | Multi-Precision Engine (FP16) | Half-precision 16-bit float computation with 50% parameter memory reduction | M2 | R3 |
| 11 | Multi-Precision Engine (INT8 SIMD) | Native C/C++ SIMD (`vpdpbusd`/AVX2) + Numba JIT integer dot product with INT32 accum | M2 | R2 |
| 12 | Quantization SQNR & Memory Guard | SQNR >= 35.0 dB and memory footprint <= 30.0% of FP32 baseline | M2 | R2, AC |
| 13 | Authentic Neural DNSMOS Engine | ITU-T P.835 Neural ONNX model graph predicting SIG, BAK, OVRL | M1 | R1 |
| 14 | Psychoacoustic STOI Evaluator | Authentic 1/3-octave band correlation across 15 spectral bands | M1 | R1 |
| 15 | Asynchronous Profiler Worker Thread | Non-blocking ring buffer queue running DNSMOS/STOI out-of-band at 0.000 ms audio overhead | M1 | R1 |
| 16 | FastAPI WebSocket Telemetry Server | Async `/ws/stream` broadcasting audio frames and profiler telemetry at 50 FPS | M4 | R4 |
| 17 | Modular ES Frontend Architecture | Deconstruction of `app.js` into 16 clean ES modules with central reactive state store | M4 | R4 |
| 18 | Jitter-Free AudioWorklet Ring Buffer | Dedicated AudioWorklet thread with 16,384-sample circular ring buffer and pre-roll cushion | M4 | R4 |
| 19 | Dual Waterfall Spectrogram & Canvases | Real-time STFT Before/After canvas blitter, Oscilloscope, EQ, and Vectorscope | M4 | R4 |
| 20 | Non-Interactive `evaluate.py` & PyTest | Standalone CLI script testing SNR, latency, and precision with exit code 0 | M5 | AC |
| 21 | Forensic Audit & Integrity Verification | Zero tolerance for heuristic shortcuts, curve-fits, or fake evaluations | M5 | AC |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Authentic Neural DNSMOS & Async Profiler | ONNX P.835 DNSMOS, 1/3-octave STOI, `AsyncQualityEvaluatorWorker` in `src/telemetry/` | none | PLANNED |
| M2 | Native C/C++ SIMD Integer Kernel | AVX2 / AVX-VNNI `vpdpbusd` SIMD kernel, ctypes GIL release, Numba JIT fallback in `src/models/precision.py` | none | PLANNED |
| M3 | End-to-End Complex Spectral Mapping (E2E CRM) | Refactor `GRUMaskNet` and `ComplexRatioMasker` for direct $M_r, M_i$ prediction and $S=Y \cdot M$ in `src/models/` | none | PLANNED |
| M4 | Frontend Modularization & AudioWorklet | ES module structure (`src/dashboard/static/js/`), `AudioWorkletNode` ring buffer, and `index.html` update | none | PLANNED |
| M5 | Comprehensive Regression & Forensic Gate | Full test suite execution (`pytest`, `evaluate.py`), adversarial verification, and forensic audit | M1, M2, M3, M4 | PLANNED |

---

## Interface Contracts

### M1: DNSMOS ↔ Audio Pipeline Contract
- **Module**: `src.telemetry.dnsmos.AsyncQualityEvaluatorWorker` & `src.audio.pipeline.AudioDenoisingPipeline`
- **Interface**:
  ```python
  class AsyncQualityEvaluatorWorker:
      def enqueue_frame(self, noisy_pcm: np.ndarray, denoised_pcm: np.ndarray, vad_active: bool) -> None:
          """Lock-free enqueue from real-time audio thread. Overhead < 1 us."""
          ...
      def get_latest_scores(self) -> PerceptualQualityScore:
          """Non-blocking atomic read of latest evaluated scores."""
          ...
  ```
- **Invariants**:
  - `enqueue_frame` takes $< 5\ \mu\text{s}$ and never blocks on locks or I/O.
  - Neural DNSMOS evaluation runs in daemon thread every 500 ms.
  - Zero analytical sigmoid curve-fits in score computation.

### M2: Native SIMD Kernel Contract
- **Module**: `src.models.precision.PrecisionEngine` & `src.models.kernels.simd_dispatch`
- **Interface**:
  ```python
  def int8_gemv(x_int8: np.ndarray, w_int8: np.ndarray, bias_int32: np.ndarray) -> np.ndarray:
      """Executes INT8 x INT8 -> INT32 GEMV via native SIMD (vpdpbusd / AVX2 / Numba JIT) with GIL released."""
      ...
  ```
- **Invariants**:
  - Quantization SQNR $\ge 35.0$ dB.
  - INT8 parameter memory footprint $\le 30.0\%$ of FP32 baseline.
  - GIL is released during matrix multiplication.
  - Measurable speedup over FP32 without Python bytecode loop serialization.

### M3: E2E CRM Denoiser Contract
- **Module**: `src.models.denoiser.GRUMaskNet` & `src.models.crm.ComplexRatioMasker`
- **Interface**:
  ```python
  class GRUMaskNet:
      def forward_crm(self, spec_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
          """Returns (real_mask M_r, imag_mask M_i) each bounded in [-1.5, 1.5]."""
          ...
      def apply_crm(self, spec_complex: np.ndarray, M_r: np.ndarray, M_i: np.ndarray) -> np.ndarray:
          """Computes S = Y * M = (Y_r * M_r - Y_i * M_i) + 1j * (Y_r * M_i + Y_i * M_r)."""
          ...
  ```
- **Invariants**:
  - Direct algebraic complex ratio mask multiplication.
  - SNR improvement $\ge 10.0$ dB across benchmark profiles.
  - Zero empirical harmonic boost kludges or phase gradient heuristics.

### M4: Frontend ES Modules & AudioWorklet Contract
- **Module**: `src/dashboard/static/js/main.js` & `audio/audio-worklet-processor.js`
- **Interface**:
  - `AudioWorkletNode` receives Float32Array PCM frames over MessagePort from WebSocket.
  - Ring buffer capacity: 16,384 samples with 512-sample pre-roll.
  - `/static/app.js` maintained as an ES module entry point / bridge for backward compatibility.
  - Zero global variable namespace pollution (`window` object clean).

---

## Code Layout
```
edge_ai_denoiser_profiler/
├── ORIGINAL_REQUEST.md
├── PROJECT.md
├── TEST_INFRA.md
├── evaluate.py
├── run_dashboard.py
├── requirements.txt
├── src/
│   ├── audio/
│   │   ├── dataset.py
│   │   ├── harmonics.py      # Cleaned of empirical 0.32 boost
│   │   ├── pipeline.py       # Decoupled async telemetry queue
│   │   ├── stft.py
│   │   └── stream.py
│   ├── models/
│   │   ├── crm.py            # Algebraic complex ratio masking (Mr, Mi)
│   │   ├── denoiser.py       # GRUMaskNet with E2E CRM architecture
│   │   ├── precision.py      # SIMD INT8 engine dispatch
│   │   └── kernels/
│   │       ├── edge_simd_int8.c  # Native C SIMD (vpdpbusd / AVX2)
│   │       └── simd_dispatch.py  # ctypes & Numba JIT loader
│   ├── telemetry/
│   │   ├── dnsmos.py         # Neural ONNX P.835 & STOI evaluator
│   │   ├── dnsmos_p835.onnx  # ITU-T P.835 ONNX model graph
│   │   ├── memory.py
│   │   ├── profiler.py
│   │   ├── ring_buffer.py
│   │   └── schema.py
│   └── dashboard/
│       ├── app.py
│       └── static/
│           ├── app.js        # ES module bridge
│           ├── index.html    # <script type="module" src="/static/js/main.js">
│           ├── styles.css
│           └── js/
│               ├── main.js
│               ├── state.js
│               ├── audio/
│               │   ├── audio-manager.js
│               │   └── audio-worklet-processor.js
│               ├── net/
│               │   └── websocket-client.js
│               ├── renderers/
│               │   ├── oscilloscope.js
│               │   ├── spectrogram-2d.js
│               │   ├── spectrogram-3d.js
│               │   ├── parametric-eq.js
│               │   └── vectorscope.js
│               └── ui/
│                   ├── controls.js
│                   ├── telemetry-panel.js
│                   └── translation-hero.js
└── tests/
    ├── unit/
    ├── integration/
    ├── e2e/
    └── adversarial/
```

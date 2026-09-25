# Real-Time Edge AI & Audio DSP Engineering Rules

These rules govern all digital signal processing (DSP), neural network inference, and real-time audio pipeline development within the `edge_ai_denoiser_profiler` project.

---

## 1. Anti-Simulation Invariant (Authentic Physics & Math)
- **Zero Placeholders**: Never substitute recurrent or deep connections with zeroed matrices ($W_{\text{rec}} = 0$) or passive pass-throughs. Recurrent hidden states must propagate temporal context across sequential frame hops with non-zero Frobenius norm weights calibrated via real BPTT or trained state-space updates.
- **True INT8 Arithmetic**: INT8 precision modes must execute authentic integer-domain matrix multiplication:
  $$\mathbf{x}_{\text{int8}} \in [-128, 127], \quad \mathbf{W}_{\text{int8}} \in [-128, 127]$$
  $$\mathbf{z}_{\text{int32}} = \mathbf{x}_{\text{int8}} \cdot \mathbf{W}_{\text{int8}} + \mathbf{b}_{\text{int32}}$$
  $$\mathbf{a}_{\text{float32}} = \text{Dequantize}(\mathbf{z}_{\text{int32}}, s_x \cdot s_w)$$
  Floating-point clipping or simulated quantization noise is strictly forbidden when claiming INT8 engine support.
- **Perceptual Metrics**: ITU-T P.835 (SIG/BAK/OVRL MOS), STOI, and PESQ telemetry must be derived from legitimate psychoacoustic spectral distortion and bark-scale sub-band SNR models, not synthetic random numbers or linear interpolations.

---

## 2. Real-Time Streaming Audio DSP Guardrails
- **Streaming Boundary Windowing**: Never apply independent raised-cosine (Hann/Hamming) windowing to raw PCM frames at streaming boundaries without matching synthesis windowing and overlap-add ($H = N/2$). Doing so creates amplitude modulation at the frame rate ($f_s / H = 62.5\text{ Hz}$).
- **IIR & Biquad Nyquist Bounds**: Center and cutoff frequencies for all IIR biquad filters (Parametric EQ, high-pass, low-shelf, notch) must be bounded strictly below the Nyquist limit:
  $$f_c < \frac{f_s}{2} - 100\text{ Hz}$$
  to prevent bilinear transform warping singularities and unstable poles.
- **$C^1$ Continuous Dynamic Range Limiting**: All output master limiting and saturation transfer functions must maintain first-derivative continuity ($C^1$) using hyperbolic tangent or polynomial soft-knees to prevent digital hard clipping and odd-harmonic splattering.
- **Selective Harmonic Enhancement**: Pitch-synchronous harmonic enhancement must only boost voiced speech formants. When the acoustic scene classifier identifies stationary tonal interference (e.g. drone motor hum, HVAC harmonics), harmonic enhancement must be selectively throttled to prevent amplifying acoustic noise peaks.

---

## 3. Model Parity & Downstream Runtime Synchronization
- Any re-training, weight re-calibration, or architecture modification on in-memory models (NumPy/PyTorch) must immediately trigger an automated re-export to edge deployment runtimes (`src/models/export_onnx.py`).
- Pre-processing steps (e.g. log-magnitude epsilon, STFT window normalization, rumble attenuation) must match identically between Python inference pipelines and compiled ONNX execution graphs.

---

## 4. Engineering Workflow State Machine
All code development and refactoring must strictly execute the 9-stage sequence:
`Task → Inspect → Plan → Implement → Test → Review diff → Benchmark → Commit → Next task`.
Never commit code without running automated tests, reviewing exact diffs, and verifying `evaluate.py` benchmarks (SNR gain >= 10.0 dB, latency <= 20.0 ms).

---

## 5. Web Audio Ingress Safety & Resampling
- **Mute Gain Node**: Microphone capture nodes (`ScriptProcessorNode` or `AudioWorkletNode`) must connect to `audioCtx.destination` only via a 0-gain mute node (`gain = 0.0`) to prevent acoustic feedback screeching while maintaining browser processing clocks.
- **Hardware Sample-Rate Downsampling**: When hardware audio runs at 44.1 kHz or 48 kHz, ingress audio must be resampled to 16,000 Hz using Catmull-Rom cubic interpolation with continuous fractional phase tracking across chunk boundaries, delivering exact 256-sample chunks to the backend.

---

## 6. Concurrency & Per-Client Session Isolation
All stateful telemetry estimators (such as `PerceptualQualityEstimator` maintaining moving-average DNSMOS state) must be instantiated per WebSocket client connection (`websocket_stream`). Never mutate shared global metric singletons from concurrent stream handlers.

---

## 7. High-Throughput Binary Audio Streaming & Vectorization
- Audio streaming between client and server must prioritize zero-copy binary framing (`ADEN` protocol, 1,040-byte ingress / 3,872-byte egress) over JSON text payloads.
- High-frequency array serialization in frame-rate loops must use vectorized NumPy transformations (`np.round().tolist()`, `.reshape().mean()`) rather than per-element Python list comprehensions to prevent GIL bottlenecks under multi-client concurrency.

---

## 8. Pitch Autocorrelation & Transient Immunity
- Pitch autocorrelation across the full lag range (40--200 samples) must maintain a minimum 512-sample continuous rolling history buffer to guarantee $\ge 312$ samples of continuous overlap.
- Valid sample counts must be tracked explicitly; never use `np.count_nonzero` heuristics to verify audio buffers.
- Pitch candidates must satisfy interior local maximum constraints ($nccf[i] > nccf[i-1]$ and $nccf[i] > nccf[i+1]$ with peak threshold $\ge 0.40$). Aperiodic transient spikes (desk bumps, clicks) exhibit monotonically decaying correlations from boundary lags and must be rejected to prevent harmonic comb distortion.

---

## 9. Biquad Pole Stability & Reactive Equalizer Control
- **Schur-Cohn / Jury Pole Stability Invariant**: Every 2nd-order IIR biquad filter in the cascade must verify that denominator polynomial roots satisfy $|p_{1,2}| \le 0.9995$. If boundary cutoffs, extreme $Q$ ($Q \ge 10.0$), or numeric roundoff push pole radii toward the unit circle, contract the poles radially ($\rho = 0.999 / r_{max}$) to guarantee strict bounded-input bounded-output (BIBO) stability without infinite resonance under all conditions.
- **Immediate UI & DSP Reactivity**: When a user configures a filter band or selects an audio preset, the equalizer must automatically transition from bypass to active (`enabled = True`), and the visual interface (`toggleEqMaster`, `badgeEqState`) must synchronize immediately with positive visual and acoustic feedback.



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

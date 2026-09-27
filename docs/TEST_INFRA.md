# E2E Test Infra: Edge AI Audio Denoiser & Profiler

## Test Philosophy
- **Opaque-box, Requirement-driven**: Derived directly from `ORIGINAL_REQUEST.md`, testing user-facing requirements (real-time processing, noise suppression SNR >= 10 dB, 3-stage latency telemetry, multi-precision speedup/memory trade-offs, interactive dashboard streaming).
- **Zero Heavy External Dependencies**: Built entirely on NumPy, SciPy, FastAPI, and PyTest; fully runnable headless in CI/CD without physical audio hardware.
- **Hierarchical Tiered Testing**: 5 distinct tiers covering feature isolation, boundary conditions, combinatorial pairs, real-world acoustic scenarios, and adversarial stress-testing.

---

## Feature Inventory Coverage
| # | Feature | Target Requirement | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|:---:|:---:|:---:|:---:|
| F1 | Framing & STFT/iSTFT Reconstruction | R1, AC | 5 | 5 | ✓ | ✓ |
| F2 | GRU-MaskNet & Wiener Spectral Filtering | R1, AC | 5 | 5 | ✓ | ✓ |
| F3 | Synthetic Audio & Noise Generation | R1, AC | 5 | 5 | ✓ | ✓ |
| F4 | Streaming Pipeline & Frame Buffering | R1, AC | 5 | 5 | ✓ | ✓ |
| F5 | 3-Stage Latency Profiler Isolation | R2, AC | 5 | 5 | ✓ | ✓ |
| F6 | Rolling Percentile Stats (P50/P95/P99) | R2, AC | 5 | 5 | ✓ | ✓ |
| F7 | Multi-Precision Engine (FP32/FP16/INT8)| R3, AC | 5 | 5 | ✓ | ✓ |
| F8 | Process Memory Footprint Telemetry | R3, AC | 5 | 5 | ✓ | ✓ |
| F9 | Dashboard WebSocket Telemetry Stream | R4, AC | 5 | 5 | ✓ | ✓ |
| F10| Dual Waterfall & A/B Audio Switcher | R4, AC | 5 | 5 | ✓ | ✓ |

---

## Test Architecture
- **Test Runner**:
  - Automated Non-Interactive Evaluation: `python evaluate.py --benchmark all`
  - Test Suite: `python -m pytest tests/ -v`
- **Pass / Fail Semantics**:
  - Exit code `0` on success. Non-zero on any failure.
  - Zero tolerance for NaN / Inf in audio signals.
  - All assertions must pass unconditionally.
- **Directory Layout**:
  - `tests/unit/`: Component-level tests (STFT, Wiener, Profiler, Quantization).
  - `tests/integration/`: Pipeline streaming and WebSocket endpoint contracts.
  - `tests/e2e/`: End-to-end evaluation benchmark asserting $\Delta\text{SNR} \ge 10.0\text{ dB}$, continuous latency $\le 20.0\text{ ms}$, and multi-precision scaling.
  - `tests/adversarial/`: Tier 5 robustness tests (DC offset, clipping at $\pm 1.0$, silence inputs, impulse bursts, buffer jitter).

---

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Metric |
|---|---|---|---|
| S1 | Synthetic Speech + White Acoustic Noise (0 dB SNR) | F1, F2, F3, F4, F5 | $\Delta\text{SNR} \ge 10.0\text{ dB}$, Latency $\le 20\text{ ms}$ |
| S2 | Drone / HVAC Low-Frequency Humming (5 dB SNR) | F1, F2, F3, F6 | Harmonic notch suppression $\ge 12\text{ dB}$ |
| S3 | RF Static Bursts & Crackle (-5 dB SNR) | F1, F2, F3, F7 | Transient suppression $\ge 10\text{ dB}$ |
| S4 | Continuous 1000-frame Streaming Run | F4, F5, F6, F8 | Zero buffer underrun, P99 $\le 20\text{ ms}$, zero memory leak |
| S5 | Multi-Precision Precision Comparison Mode | F5, F7, F8 | FP32 vs FP16 vs INT8 latency, memory $\le 0.35\times$, SQNR $\ge 35\text{ dB}$ |

---

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: $\ge 5$ test cases per feature covering nominal conditions.
- **Tier 2 (Boundary & Corner Cases)**: $\ge 5$ test cases covering extremes (zero amplitude, maximum amplitude $\pm 1.0$, high frequencies, tiny buffer sizes, single-sample boundaries).
- **Tier 3 (Cross-Feature Combinations)**: Pairwise integration across audio framing, neural masking, profiling, and precision modes.
- **Tier 4 (Real-World Scenarios)**: $\ge 5$ application scenarios exercising the full end-to-end pipeline.
- **Tier 5 (Adversarial Robustness)**: Extreme edge cases designed to stress numerical stability, audio buffer overflow/underflow, and rapid mode switching.

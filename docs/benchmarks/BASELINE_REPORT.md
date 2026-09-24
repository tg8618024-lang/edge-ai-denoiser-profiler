# Baseline Telemetry & Benchmark Report (Phase 0)

**Captured**: 2026-09-20 16:43:51 UTC | **Python**: 3.13.7

This report captures the pre-enhancement performance baseline of the **Real-Time Edge AI Audio Denoiser & Profiler** pipeline before subsequent Phase 1–5 optimizations. All future enhancements are evaluated against these numbers.

## 1. 3-Stage Hardware Latency Breakdown & Memory

| Pipeline Stage | Latency | Real-Time Frame Budget | Headroom |
|:---|:---|:---|:---|
| **Stage 1: Pre-processing (STFT, Windowing)** | `0.043 ms` | 20.000 ms | - |
| **Stage 2: Tensor Compute (GRUMaskNet / Wiener)** | `0.167 ms` | 20.000 ms | - |
| **Stage 3: Output Synthesis (iSTFT, Overlap-Add)** | `0.036 ms` | 20.000 ms | - |
| **Total Frame Processing Time** | **`0.258 ms`** | **20.000 ms** | **`19.742 ms` (98.7%)** |

- **Process Memory (RSS)**: `106.06 MB`
- **Nominal Frame Hop**: `16.0 ms` (`256 samples @ 16 kHz`)
- **Algorithmic Delay**: `16.0 ms` (`256 samples` deterministic delay)

## 2. Noise Suppression Performance (SNR Gain)

| Noise Profile | Suite | In SNR (dB) | Out SNR (dB) | Delta SNR Gain (dB) | P50 Latency | P95 Latency | Status |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `white` | REF | `-0.00` | `13.62` | **`+13.62 dB`** | `0.441 ms` | `0.651 ms` | PASS |
| `pink` | REF | `-0.00` | `11.41` | **`+11.41 dB`** | `0.436 ms` | `0.645 ms` | PASS |
| `drone` | REF | `5.00` | `16.12` | **`+11.12 dB`** | `0.431 ms` | `0.523 ms` | PASS |
| `rf` | REF | `-5.00` | `21.76` | **`+26.76 dB`** | `0.438 ms` | `0.713 ms` | PASS |
| `white` | GEN | `0.00` | `11.91` | **`+11.91 dB`** | `0.461 ms` | `0.897 ms` | PASS |
| `pink` | GEN | `0.00` | `11.69` | **`+11.69 dB`** | `0.448 ms` | `0.748 ms` | PASS |
| `drone` | GEN | `0.00` | `10.96` | **`+10.96 dB`** | `0.450 ms` | `0.873 ms` | PASS |
| `rf_static` | GEN | `0.00` | `12.18` | **`+12.18 dB`** | `0.440 ms` | `1.146 ms` | PASS |

## 3. Multi-Precision Engine (FP32 vs FP16 vs INT8)

| Mode | Model Size | RAM Savings | SQNR (dB) | Denoising SNR Gain | Validation |
|:---|:---|:---|:---|:---|:---|
| **FP32** | `165,892 B` | `0.0%` | `100.0 dB` | `+14.49 dB` | PASS |
| **FP16** | `82,946 B` | `50.0%` | `73.6 dB` | `+14.49 dB` | PASS |
| **INT8** | `42,656 B` | `74.3%` | `39.9 dB` | `+14.49 dB` | PASS |

---
*Report generated automatically by `benchmark_baseline.py`.* 

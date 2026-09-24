# Worker M2 Implementation Task

## Objective
Implement Milestone 2: 3-Stage Hardware Latency Profiler & Multi-Precision Quantization Engine, and apply the SNR filter tuning:

### 1. Telemetry Subsystem
- `src/telemetry/__init__.py`: Export `StageProfiler`, `RollingMetricsBuffer`, `get_process_memory`, `TelemetryPayload`, `FrameMetrics`.
- `src/telemetry/profiler.py`: Implement `StageProfiler` with dual API (`start_stage`/`end_stage` and `start_frame`/`mark_preprocessing_done`/`mark_tensor_done`/`mark_synthesis_done`), hardware timing with `perf_counter_ns` clamped to >= 1.0 us, and `get_last_metrics()` returning `FrameMetrics` (subclass of `dict` with property accessors).
- `src/telemetry/ring_buffer.py`: Implement `RollingMetricsBuffer` with fixed NumPy preallocation, computing `window_size`, `total_frames_processed`, `p50_total_ms`, `p95_total_ms`, `p99_total_ms`, `jitter_ms`, `overrun_count`.
- `src/telemetry/memory.py`: Implement `get_process_memory()` returning `ProcessMemorySnapshot(rss_mb=..., private_mb=...)` using 64-bit `ctypes` on Windows (`psapi.GetProcessMemoryInfo`) and `resource` on POSIX without third-party dependencies.
- `src/telemetry/schema.py`: Implement `TelemetryPayload` matching M2 <-> M3 WebSocket schema.
- `tests/unit/test_profiler.py`: Implement unit tests for `StageProfiler`, `RollingMetricsBuffer`, and `get_process_memory`.

### 2. Multi-Precision Quantization Engine
- `src/models/precision.py`: Implement `PrecisionEngine` supporting FP32, FP16, and symmetric affine INT8 quantization with scale, zero-point (0), clipping in [-128, 127], INT32 accumulation GEMM, memory reduction <= 0.35x, and SQNR >= 35.0 dB.
- `tests/unit/test_precision.py`: Implement unit tests for FP32/FP16/INT8 modes, weight quantization SQNR, memory reduction ratios, and numerical fidelity.

### 3. Audio Pipeline & Denoiser Updates
- `src/models/denoiser.py`: Apply the calibrated Wiener filter tuning (two-rate alpha 0.92/0.96, Ephraim-Malah gain, noise PSD tracking, sub-100Hz rumble notch, restored recurrent link) and update `src/models/default_weights.npz` so that SNR gain >= 10.0 dB across all noise benchmarks.
- `src/audio/pipeline.py`: Support `pipeline.set_precision(mode)` and ensure seamless profiler stage timing.

### 4. Verification
- Run `pytest tests/unit/ -v` (verify 100% pass including new test_profiler.py and test_precision.py).
- Run `pytest tests/adversarial/ -v` (verify 100% pass).
- Run `pytest tests/e2e/ -v` (verify all unblocked tests pass, achieving 100% pass across Tiers 1-4).
- Write `handoff.md` with all commands and verified test results.

## 2026-09-06T19:41:25Z
You are Worker M2 tasked with implementing Milestone 2 (3-Stage Hardware Latency Profiler, Rolling Stats Ring Buffer, Native Memory Telemetry, Multi-Precision Quantization Engine, and Denoising Filter Calibration).

Your working directory is:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2

Mandatory Inputs:
1. Read the authoritative user request at:
   C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
2. Read C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md and TEST_INFRA.md
3. Read your task assignment at:
   C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2\DISPATCH.md
4. Read the 3 Explorer handoff reports for complete code blueprints and mathematical formulas:
   - Profiler & Telemetry blueprint:
     C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_profiler\handoff.md
   - Precision Engine & Quantization blueprint:
     C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_precision\handoff.md
   - SNR Tuning & Filter Calibration blueprint:
     C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning\handoff.md


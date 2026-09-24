# BRIEFING — 2026-09-06T18:02:00Z

## Mission
Survey and design the NVIDIA-Style Latency Profiler & Multi-Precision Comparison Engine (Requirements R2, R3) for real-time audio denoiser testbench.

## 🔒 My Identity
- Archetype: explorer
- Roles: profiler_specialist, quantization_specialist, survey_analyst
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_profiler_quant
- Original parent: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Milestone: Milestone 0 (Survey & Architecture Specification)

## 🔒 Key Constraints
- Read-only investigation of code/system — do NOT implement production source code directly.
- All proposals, schemas, algorithms, and architectures documented in report.md and handoff.md.
- Stage telemetry MUST measure exactly 3 stages: Pre-processing, Tensor Compute, and Output Synthesis.
- High-resolution timing with `time.perf_counter_ns()`.
- Rolling statistics: circular buffer tracking median, P95, P99 latency against <= 20ms frame budget, computing headroom.
- Multi-precision support: FP32, FP16, INT8 (dynamic quantization / affine scale-zero quantization / INT8 GEMM, memory footprint RSS / weight size in MB/KB, latency speedup, SNR impact).
- Telemetry schema: JSON/dataclass data contracts for per-frame & aggregated stats.

## Current Parent
- Conversation ID: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Updated: 2026-09-06T17:48:00Z

## Investigation State
- **Explored paths**:
  - ORIGINAL_REQUEST.md
  - .agents/orchestrator_1/BRIEFING.md
  - .agents/explorer_survey_dsp_model/DISPATCH.md
  - .agents/explorer_survey_ui_e2e/report.md
  - System environment: Python 3.13.7, numpy 2.5.2, scipy 1.18.1, numba 0.67.0, soundfile, sounddevice, PyQt6, pyqtgraph.
- **Key findings**:
  - `time.perf_counter_ns()` overhead is 109.39 ns/call (~0.00065 ms per frame across 3 stages, < 0.004% of 20 ms budget).
  - Preallocated circular ring buffer in NumPy computes rolling Median, P95, P99 in ~140 microseconds with zero allocations during streaming.
  - Symmetric INT8 affine quantization with INT32 accumulation achieves 38.09 dB SQNR with < 0.15 dB SNR degradation, preserving > 12 dB overall SNR gain.
  - INT8 provides 75% parameter memory reduction (4x compression); FP16 provides 50% parameter reduction.
  - Zero-dependency system RSS memory reader implemented via 64-bit Win32 `psapi.GetProcessMemoryInfo` (and `resource` on POSIX).
  - Telemetry wire contracts aligned 100% with Explorer 3's WebSocket UI schema.
- **Unexplored areas**:
  - None. Full scope of profiler, circular buffer, multi-precision quantization, memory telemetry, and wire schema surveyed and architected.

## Key Decisions Made
- Implement zero-allocation `StageProfiler` using `perf_counter_ns()`.
- Preallocate 2D NumPy array for `RollingMetricsBuffer` with sliding window percentile calculations.
- Support dual-engine quantization: PyTorch dynamic quantization + pure NumPy/Numba vectorized INT8 GEMM with INT32 accumulator.
- Use native ctypes Win32 `GetProcessMemoryInfo` for host process RSS to avoid `psutil` dependency.

## Artifact Index
- report.md — Comprehensive survey and architecture specification for Profiler and Multi-Precision Engine.
- handoff.md — 5-component handoff report for Orchestrator and Worker agents.
- progress.md — Liveness heartbeat.

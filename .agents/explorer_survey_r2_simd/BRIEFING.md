# BRIEFING — 2026-09-23T11:55:00Z

## Mission
Investigate and produce a detailed architectural analysis and implementation specification for R2: Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON) for INT8 matrix multiplication and quantization in edge_ai_denoiser_profiler.

## 🔒 My Identity
- Archetype: explorer
- Roles: Teamwork explorer, read-only investigation, architectural analysis, SIMD specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd
- Original parent: 3581185e-376e-4294-ac05-fca0ad7593e2
- Milestone: hardening_survey_r2

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Audit src/models/precision.py and benchmark np.int8 / np.int32 accumulation
- Investigate execution environment on Windows (C/C++ compilers, ctypes/setuptools DLL build capabilities)
- Design native C/C++ SIMD kernel (AVX-512 VNNI, AVX2 maddubs/madd, SSE4.1, scalar fallback)
- Guarantee GIL release during execution
- Verify quantization constraints: SQNR >= 35.0 dB, footprint <= 30% FP32, measurable speedup

## Current Parent
- Conversation ID: 3581185e-376e-4294-ac05-fca0ad7593e2
- Updated: 2026-09-23T11:55:00Z

## Investigation State
- **Explored paths**:
  - `src/models/precision.py` lines 1-276: Model dimensions, weight caching, `_gemm_int8()` casting bottleneck.
  - Host environment: Windows 11 AMD64, Intel Core i5-12450H (Alder Lake with AVX2 & AVX-VNNI `vpdpbusd`), Python 3.13.
  - Host toolchains: No `cl.exe`, `gcc.exe`, `clang.exe` in PATH or system folders. `numba 0.67.0` + `llvmlite 0.49.0` and `cffi` installed.
  - Benchmarks: FP32 = 0.0265 ms, INT8 = 0.1659 ms (6.26x slower than FP32). INT8 footprint = 42,656 B (25.71% of FP32).
- **Key findings**:
  - Root cause of INT8 slow-down: Dynamic per-frame `w_int8.astype(np.int32)` allocations (>164 KB/frame) and lack of SIMD int32 BLAS in NumPy.
  - Recommended architecture: 3-tier SIMD dispatch: Tier 1 (Native C DLL with AVX-VNNI/AVX2 loaded via ctypes releasing GIL), Tier 2 (Numba LLVM JIT with `nogil=True`), Tier 3 (Zero-allocation NumPy fallback).
- **Unexplored areas**: None for R2 SIMD survey scope. Ready for implementation by builder agent.

## Key Decisions Made
- Use transposed weight layout $(M, K)$ for contiguous SIMD row streaming.
- Precompute constant bias folding term $C_j = 128 \sum W_{i, j}$ to enable zero-runtime-overhead unsigned/signed `vpdpbusd` and `vpmaddubsw` arithmetic.
- Adopt 3-tier fallback architecture to guarantee test pass and high performance across any edge deployment environment.

## Artifact Index
- DISPATCH.md — Initial dispatch record
- BRIEFING.md — Working memory & state
- progress.md — Liveness heartbeat
- benchmark_simd.py — Reproducible benchmarking harness
- analysis.md — Full architectural analysis and implementation specification
- handoff.md — 5-component handoff report

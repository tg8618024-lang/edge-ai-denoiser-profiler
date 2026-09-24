# BRIEFING — 2026-09-23T12:31:01Z

## Mission
Implement native SIMD integer-domain matrix multiplication (AVX-VNNI / AVX2 / SSE4.1 / Scalar C kernel, Numba LLVM JIT vectorization, zero-allocation NumPy fallback) with zero per-frame heap allocations, >= 35.0 dB SQNR, <= 30.0% FP32 memory footprint, and measurable speedup over FP32 without GIL serialization.

## 🔒 My Identity
- Archetype: worker_m2_simd_gen2
- Roles: implementer, qa, specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd_gen2
- Original parent: 3581185e-376e-4294-ac05-fca0ad7593e2
- Milestone: M2 Native SIMD INT8 Kernel & Precision Engine Hardening

## 🔒 Key Constraints
- Genuine implementation only, no cheating, no hardcoded test outputs or dummy facades.
- Exclusively own:
  - src/models/precision.py
  - src/models/kernels/
  - tests/unit/test_precision.py
- .agents/ holds only agent metadata, NEVER place source code or tests there.
- Guarantee INT8 Quantization SQNR >= 35.0 dB and memory footprint <= 30.0% of FP32 baseline.
- 3-tier dispatch architecture: Tier 1 ctypes C DLL, Tier 2 Numba LLVM JIT (@njit(fastmath=True, nogil=True)), Tier 3 cached zero-allocation NumPy fallback.
- Release GIL in native/JIT execution.
- Pass `pytest tests/unit/test_precision.py`.

## Current Parent
- Conversation ID: 3581185e-376e-4294-ac05-fca0ad7593e2
- Updated: not yet

## Task Summary
- **What to build**: Native C SIMD kernel (`src/models/kernels/edge_simd_int8.c`, `edge_simd_int8.h`), 3-tier dispatch manager (`src/models/kernels/simd_dispatch.py`), refactored `PrecisionEngine` (`src/models/precision.py`) with pre-transposed weights and zero-allocation per-frame execution, and comprehensive unit tests (`tests/unit/test_precision.py`).
- **Success criteria**: INT8 inference faster than FP32 baseline without GIL serialization, zero per-frame heap allocations, SQNR >= 35.0 dB, memory <= 30% of FP32, all precision tests passing.
- **Interface contracts**: `PrecisionEngine` API compatible with pipeline and evaluation suite.
- **Code layout**: `src/models/kernels/`, `src/models/precision.py`, `tests/unit/test_precision.py`.

## Key Decisions Made
- Architecture: 3-tier hybrid SIMD engine.
- Layout: Pre-transpose weights to $(M, K)$ row-major layout for streaming continuous memory access.
- Pre-quantize and fold static weight offsets and biases during weight cache construction to avoid runtime bias quantization.

## Change Tracker
- **Files modified**: None yet
- **Build status**: Pending
- **Pending issues**: Initial implementation

## Quality Status
- **Build/test result**: Not yet run
- **Lint status**: Pending
- **Tests added/modified**: Pending

## Loaded Skills
- None required for this phase.

## Artifact Index
- `context.md` — Worker context
- `DISPATCH.md` — Dispatch requirements

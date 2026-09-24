# BRIEFING — 2026-09-23T12:06:00Z

## Mission
Implement native SIMD INT8 GEMV/GEMM kernel with multi-tier dispatch (C DLL with AVX-VNNI/AVX2/SSE4.1, Numba LLVM JIT, zero-alloc cached NumPy), eliminate per-frame allocations, and verify SQNR >= 35.0 dB and memory <= 30.0% of FP32 baseline.

## 🔒 My Identity
- Archetype: implementer, qa, specialist
- Roles: implementer, qa, specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd
- Original parent: 3581185e-376e-4294-ac05-fca0ad7593e2
- Milestone: M2 SIMD INT8 Kernel Hardening

## 🔒 Key Constraints
- DO NOT CHEAT. All implementations must be genuine.
- DO NOT hardcode test results, expected outputs, or verification strings.
- Exclusively own:
  * src/models/precision.py
  * src/models/kernels/
  * tests/unit/test_precision.py
- .agents/ holds only metadata. Never place source code or tests there.
- Guarantee INT8 Quantization SQNR >= 35.0 dB.
- Guarantee memory footprint <= 30.0% of FP32 baseline.
- Eliminate per-frame heap allocations during INT8 inference (buffer reuse/pre-allocated outputs).
- Support Tier 1 (C DLL ctypes), Tier 2 (Numba LLVM JIT), Tier 3 (NumPy fallback).
- Releasing GIL where applicable.

## Current Parent
- Conversation ID: 3581185e-376e-4294-ac05-fca0ad7593e2
- Updated: 2026-09-23T12:06:00Z

## Task Summary
- **What to build**: Native SIMD INT8 GEMV/GEMM kernels (`edge_simd_int8.c`), 3-tier dispatch (`simd_dispatch.py`), update `precision.py` for zero-allocation INT8 inference, and rigorous unit tests (`test_precision.py`).
- **Success criteria**: All tests in `tests/unit/test_precision.py` pass; SQNR >= 35dB; INT8 memory <= 30% FP32; speedup over FP32 without GIL serialization.
- **Interface contracts**: `src/models/precision.py` API compatibility with existing Denoiser models.

## Key Decisions Made
- [Pending initial inspection]

## Artifact Index
- `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd\DISPATCH.md` — Assignment instructions
- `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd\context.md` — Worker context
- `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd\analysis.md` — Explorer analysis

## Change Tracker
- **Files modified**: None yet
- **Build status**: Untested
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pending
- **Lint status**: Pending
- **Tests added/modified**: Pending

## Loaded Skills
- None requested in prompt

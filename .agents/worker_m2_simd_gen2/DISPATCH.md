## 2026-09-23T12:31:01Z

You are Worker M2 SIMD INT8 Kernel (Gen 2) for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd_gen2
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context and survey findings at:
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_simd_gen2\context.md
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd\analysis.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership:
You exclusively own:
- src/models/precision.py
- src/models/kernels/ (new directory for SIMD C/C++ kernel and dispatch)
- tests/unit/test_precision.py

Your Task:
1. Implement native SIMD integer-domain matrix multiplication to replace the unvectorized heap-allocating `_gemm_int8` loop in `src/models/precision.py`.
2. Implement native C kernel `src/models/kernels/edge_simd_int8.c` implementing `int8_gemv` using VEX-encoded `vpdpbusd` (AVX-VNNI), `vpmaddubsw`/`vpmaddwd` (AVX2), SSE4.1, and Scalar fallbacks.
3. Implement `src/models/kernels/simd_dispatch.py` providing:
   - Tier 1: `ctypes` loader for native DLL if compiled/available, releasing the GIL.
   - Tier 2: Numba LLVM JIT vectorization kernel (`@njit(fastmath=True, nogil=True)`) compiling SIMD integer instructions directly into native machine code on this machine and releasing the GIL.
   - Tier 3: Zero-allocation cached NumPy fallback with pre-transposed $(M, K)$ weights and pre-quantized biases.
4. Eliminate per-frame heap allocations during INT8 inference.
5. Guarantee INT8 Quantization SQNR >= 35.0 dB and memory footprint <= 30.0% of FP32 baseline (e.g. ~42,656 bytes vs 165,892 bytes = 25.7%).
6. Demonstrate measurable speedup over FP32 without Python GIL serialization.
7. Run `pytest tests/unit/test_precision.py` and verify all tests pass.

Deliverables:
- Write `handoff.md` with: Observation, Logic Chain, Caveats, Conclusion, Verification Method (including test commands and output).
- Send completion message to parent.

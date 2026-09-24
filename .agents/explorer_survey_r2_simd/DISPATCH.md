## 2026-09-23T11:45:00Z

<USER_REQUEST>
You are Explorer R2 SIMD Specialist for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd\context.md

Your Task:
Investigate and produce a detailed architectural analysis and implementation specification for:
R2: Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON):
1. Audit `src/models/precision.py` and benchmark how `np.int8` matrix multiplication and `np.int32` accumulation currently operate.
2. Investigate the execution environment on this Windows machine:
   - Check available C/C++ compilers (`cl.exe`, `gcc`, `clang`, etc.) or whether Python `setuptools`/`distutils`/ctypes can compile a shared DLL (`.dll`).
3. Design a native C/C++ SIMD kernel:
   - Implement INT8 matrix multiplication: 8-bit signed/unsigned inputs with 32-bit accumulation.
   - Leverage hardware SIMD instructions where available:
     * AVX-512 VNNI / AVX-VNNI: `_mm256_dpbusd_epi32` (`vpdpbusd`)
     * AVX2: `_mm256_maddubs_epi16` followed by `_mm256_madd_epi16`
     * SSE4.1 / Scalar fallback if advanced vector extensions are not supported on the host CPU.
   - Guarantee GIL release during execution (`ctypes` or C extension release GIL) so CPU inference does not serialize Python threads.
4. Verify quantization constraints:
   - Maintain INT8 Quantization SQNR >= 35.0 dB and memory footprint <= 30% of FP32.
   - Demonstrate measurable speedup over FP32 without Python interpreter overhead.

Deliverables:
- Write full findings to `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd\analysis.md`.
- Write `handoff.md` following standard format.
- Send a completion message back to parent.
</USER_REQUEST>

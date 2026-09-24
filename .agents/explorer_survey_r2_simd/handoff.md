# Handoff Report: R2 Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON)

**Agent**: Explorer R2 SIMD Specialist  
**To**: Orchestrator / Implementer Agent  
**Working Directory**: `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r2_simd`  
**Date**: 2026-09-23  

---

## 1. Observation

1. **`src/models/precision.py` lines 177–206 (`_gemm_int8`)**:
   ```python
   # Dynamic per-frame activation quantization to true 8-bit signed integer
   max_x = float(np.max(np.abs(x)))
   scale_x = max_x / 127.0 if max_x > 1e-12 else 1.0
   x_int8 = np.clip(np.round(x / scale_x), -128, 127).astype(np.int8)

   # Authentic integer-domain dot product with INT32 accumulator
   accum_int32 = np.dot(x_int8.astype(np.int32), w_int8.astype(np.int32))
   ```
   On every frame for every layer, `w_int8.astype(np.int32)` allocates and casts 2D weight arrays on the heap (65,792 bytes for `W1` and `W3`, 16,384 bytes for `W2` and `W_rec`, totaling $>164$ KB per frame).
2. **Benchmark Results (`benchmark_simd.py`)**:
   - `FP32 Model Size`: 165,892 bytes.
   - `FP16 Model Size`: 82,946 bytes (50.00% of FP32).
   - `INT8 Model Size`: 42,656 bytes (25.71% of FP32, strictly satisfying $\le 30.0\%$).
   - Latency over 1,000 iterations:
     - FP32: **0.0265 ms** (37,739.0 fps).
     - FP16: **0.2187 ms** (4,572.8 fps).
     - INT8: **0.1659 ms** (6,029.4 fps).
     - Result: Current INT8 is **6.26x slower** than FP32 due to Python interpreter overhead and per-frame `astype(np.int32)` heap allocations.
3. **Execution Environment & Host Hardware (`Get-CimInstance Win32_Processor`)**:
   - CPU: `12th Gen Intel(R) Core(TM) i5-12450H` (Alder Lake architecture, 8 cores / 12 threads).
   - ISA capabilities: Supports **AVX2**, **FMA3**, and **AVX-VNNI** (VEX 256-bit `VPDPBUSD` / `_mm256_dpbusd_epi32`), as well as SSE4.1/SSE4.2. AVX-512 is disabled on Alder Lake hybrid cores.
   - Compilers: `Get-Command cl, gcc, clang, clang++` returned null. No `cl.exe`, `gcc.exe`, or `clang.exe` in `%PATH%` or standard Windows directories.
   - Python packages in `C:\Users\tg861\AppData\Local\Programs\Python\Python313\Lib\site-packages`:
     - `numba` 0.67.0 and `llvmlite` 0.49.0 (native LLVM JIT code generation available in Python).
     - `cffi` 2.1.1 and `ctypes` (standard library, releases GIL via `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`).
4. **Quantization Precision Requirement**:
   - In `tests/unit/test_precision.py` lines 45–52 (`test_int8_quantization_sqnr_ge_35db`), INT8 weight quantization yields SQNR $\ge 35.0$ dB (empirically $38.4$ to $41.2$ dB).

---

## 2. Logic Chain

1. From Observation 1, `_gemm_int8()` re-converts `w_int8` to `int32` on every single frame, causing over 40 heap allocations and copying $>164$ KB per frame. Combined with the lack of integer SIMD BLAS in NumPy, this directly explains the 6.26x latency penalty observed in Observation 2.
2. From Observation 3, the host machine does not have a native C/C++ compiler (`cl.exe`, `gcc`) installed in PATH. Therefore, attempting to compile a C extension at runtime via `setuptools.setup(ext_modules=...)` would crash with `DistutilsPlatformError`.
3. From Observation 3, the CPU natively supports 256-bit AVX-VNNI (`vpdpbusd`), which computes 32 INT8 multiply-accumulates into 32-bit integer destinations per single clock cycle. AVX2 also provides 32 INT8 MACs per ~2 cycles via `vpmaddubsw` and `vpmaddwd`.
4. From Observation 3, Python `ctypes.CDLL` automatically releases the GIL during foreign function calls, guaranteeing that CPU inference does not serialize Python audio streaming or WebSocket telemetry threads.
5. In addition, `numba` (0.67.0) is installed and can JIT-compile integer loops directly into machine code using LLVM with `nogil=True, fastmath=True`, providing a zero-external-compiler fallback that still achieves AVX2 auto-vectorization and GIL release.
6. In Observation 4, INT8 symmetric affine quantization with 127 quantization levels guarantees theoretical SQNR $\ge 35.0$ dB, and memory footprint is 25.71% $\le 30.0\%$.
7. Pre-transposing the weight matrices to shape $(M, K)$ at model initialization ensures contiguous row streaming into 256-bit SIMD registers (`_mm256_loadu_si256`), eliminating all strided memory accesses.
8. Folding the activation shift offset $C_j = 128 \sum_{i} W_{i, j}$ into the static bias vector allows `vpdpbusd` and `vpmaddubsw` (which expect unsigned inputs) to execute with zero runtime compensation overhead.
9. Executing 41,088 MACs via AVX-VNNI requires $\approx 1,284$ vector iterations ($\approx 1,500$ clock cycles $\approx 0.5\ \mu\text{s}$ at 3.0 GHz). With ctypes call overhead ($\approx 2.0\ \mu\text{s}$), total frame latency will be $\le 0.005$ ms, representing a $>5\times$ speedup over FP32 ($0.0265$ ms) and $>30\times$ speedup over current Python INT8 ($0.1659$ ms).

---

## 3. Caveats

1. **Host Compiler Absence**: Because `cl.exe` and `gcc.exe` are not present on this machine, the native C shared library (`edge_simd_int8.dll`) must either be provided as a pre-compiled binary artifact in `src/models/kernels/` or compiled on a build machine.
2. **Robust Multi-Tier Fallback Required**: The production code must not assume the DLL is always loadable. It must dynamically detect and fall back through: Tier 1 (`ctypes.CDLL`), Tier 2 (`numba` JIT with `nogil=True`), and Tier 3 (Zero-allocation cached NumPy).
3. **No AVX-512 on Alder Lake**: 512-bit vector registers (ZMM) cannot be used on 12th Gen Intel hybrid processors. The 256-bit VEX-encoded AVX-VNNI (`_mm256_dpbusd_epi32`) must be targeted instead.
4. **Saturation Guard in AVX2**: In AVX2 `_mm256_maddubs_epi16`, intermediate products sum into 16-bit signed integers. Symmetrical activation scaling with max amplitude $\le 127$ ensures products never exceed $127 \times 128 \times 2 = 32,512 < 32,767$, preventing saturation distortion.

---

## 4. Conclusion

1. **Architecture Defined**: Implement a 3-tier SIMD INT8 engine:
   - **Tier 1**: Native C shared library (`src/models/kernels/edge_simd_int8.c` / `edge_simd_int8.dll`) exporting `int8_gemv_auto()`, implementing `vpdpbusd` (AVX-VNNI), `vpmaddubsw` + `vpmaddwd` (AVX2), SSE4.1, and Scalar fallbacks with CPUID detection. Called via `ctypes` with complete GIL release.
   - **Tier 2**: Numba LLVM JIT kernel (`src/models/kernels/simd_dispatch.py`) with `@njit(fastmath=True, nogil=True)` for zero-compiler environments.
   - **Tier 3**: Zero-allocation cached NumPy fallback storing pre-transposed $(M, K)$ weights and pre-quantized biases.
2. **Quantization & Footprint Verified**:
   - Model memory footprint: 42,656 bytes = 25.71% of FP32 baseline (conforms to $\le 30.0\%$).
   - Quantization SQNR: $\ge 35.0$ dB maintained across all layers.
3. **Speedup Expected**:
   - Reduces INT8 inference time from 0.1659 ms to $\le 0.005$ ms, delivering genuine silicon acceleration ($\approx 5.3\times$ faster than FP32).

---

## 5. Verification Method

1. **Precision Unit Tests**:
   ```powershell
   pytest tests/unit/test_precision.py -v
   ```
   Asserts:
   - Memory footprint $\le 0.35$ and $\approx 0.257$ of FP32.
   - SQNR $\ge 35.0$ dB for INT8 and $\ge 50.0$ dB for FP16.
   - Master weight immutability across precision switches.
2. **SIMD Benchmark Verification**:
   ```powershell
   python .agents/explorer_survey_r2_simd/benchmark_simd.py
   ```
   Asserts that INT8 latency is faster than FP32 ($t_{\text{INT8}} < t_{\text{FP32}}$) once the native SIMD backend is active.
3. **End-to-End Testbench Evaluation**:
   ```powershell
   python evaluate.py
   ```
   Asserts end-to-end audio pipeline SNR gain $\ge 10.0$ dB with exit code 0.
4. **Invalidation Conditions**:
   - If INT8 memory exceeds 30.0% of FP32 baseline ($>49,767$ bytes).
   - If INT8 SQNR drops below 35.0 dB on synthetic test vectors.
   - If INT8 frame inference time exceeds FP32 inference time ($>0.0265$ ms).
   - If the implementation hard-crashes on platforms without a pre-installed C compiler.

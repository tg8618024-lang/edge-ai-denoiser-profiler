## 2026-09-23T01:28:11Z

You are Worker 2 implementing R3 (Authentic Integer-Domain INT8 Quantization Simulation).
Your working directory is C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_int8_precision.

MANDATORY FIRST STEP: Read the authoritative requirements at:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
and read the detailed explorer findings in:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_2_gen2\report.md
and C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_2_gen2\handoff.md.

EXCLUSIVE FILE OWNERSHIP:
You have exclusive write access to:
- src/models/precision.py
- tests/unit/test_precision.py
DO NOT edit any other files.

Requirements:
1. Refactor src/models/precision.py to eliminate float32 cast-and-multiply pseudo-quantization.
2. Implement authentic integer arithmetic simulation:
   - Quantize weights and inputs to np.int8 (with symmetric/affine scaling and zero-point).
   - Perform GEMM matrix multiplication accumulation using 32-bit integers (np.int32).
   - Add 32-bit integer bias (b_int32 in np.int32) directly inside accumulator registers.
   - Implement integer-domain ReLU: np.maximum(z_int32, 0).
   - Apply output dequantization / scaling at layer output to match real edge NPU/DSP hardware execution.
3. Ensure parameter memory footprint <= 0.35x FP32, SQNR >= 35.0 dB, and SNR delta >= 10.0 dB.
4. Verification: Run tests using run_command:
   pytest tests/unit/test_precision.py
   python evaluate.py --benchmark all
   Ensure all tests pass.
5. Write detailed report to your directory and self-contained handoff.md, then send a message back.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

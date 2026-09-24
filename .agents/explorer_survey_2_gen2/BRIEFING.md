# BRIEFING — 2026-09-22T19:16:00Z

## Mission
Investigate R3 (Authentic Integer-Domain INT8 Quantization Simulation) in edge_ai_denoiser_profiler, detail current pseudo-quantization vs authentic integer arithmetic design, and formulate complete implementation specifications and verification methods.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesis
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_2_gen2
- Original parent: 37613f20-d735-4d4c-af6d-73777c5e0e97
- Milestone: Investigation and Architectural Design for R3 (INT8 Quantization Simulation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Do NOT write or modify source code files
- Write findings to report.md and handoff.md in working directory
- Communicate via send_message to parent (ID: 37613f20-d735-4d4c-af6d-73777c5e0e97)

## Current Parent
- Conversation ID: 37613f20-d735-4d4c-af6d-73777c5e0e97
- Updated: 2026-09-22T19:16:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (specifically Follow-up — 2026-09-22T18:36:21Z R3)
  - `PROJECT.md` (Features 13-16, Interface Contracts, Code Layout)
  - `src/models/precision.py` (PrecisionEngine, _gemm_int8, _build_quantized_caches, forward_frame)
  - `src/models/denoiser.py` (GRUMaskNet, HybridDenoiser, forward_frame delegation)
  - `src/audio/pipeline.py` (set_precision, process_frame stage 2 tensor compute)
  - `evaluate.py` (precision evaluation suite, memory reduction <= 0.35x, SQNR >= 35 dB, delta SNR >= 10 dB)
  - `tests/unit/test_precision.py` (unit tests for PrecisionEngine)
  - `tests/e2e/test_evaluation.py` (Tier 1, Tier 3, Tier 4 test contracts)
  - `calibrate_model.py` (training and weights layout)
  - Previous explorer handoff `.agents/explorer_m2_precision/handoff.md`
- **Key findings**:
  - Current implementation in `src/models/precision.py` uses "float32 cast-and-multiply pseudo-quantization":
    1. In `_gemm_int8`: upcasts `x_int8` and `w_int8` to `int32` before dot product, immediately dequantizes to `float32` via `accum_int32.astype(np.float32) * (scale_x * scale_w)`, and adds unquantized float32 bias `y += self.fp32_weights[bias_key]`.
    2. Biases in `_build_quantized_caches` are stored as `float32`, never quantized to `int32`.
    3. `forward_frame` converts activations back to `float32` after every single layer, runs ReLU in `float32`, re-quantizes activations dynamically from `float32`, and adds recurrent state in `float32`.
    4. Missing affine quantization with zero-point support.
  - Authentic integer design formulated:
    1. Authentic 8-bit integer operands (`np.int8`) for both weights and activations.
    2. 32-bit integer accumulation (`np.int32`) with zero overflow risk (max accumulator 4.2M << 2.14B).
    3. 32-bit integer bias addition `b_int32` directly inside the integer accumulator before any dequantization.
    4. Symmetric and affine zero-point compensation ($Z_x, Z_W$).
    5. Integer ReLU $\max(z_{int32}, 0)$.
    6. Exact memory footprint maintained at 42,656 bytes (0.257x ratio <= 0.35x target).
    7. SQNR maintained at ~38.1 to 40.0 dB (>= 35.0 dB target), SNR delta >= 10.4 dB (>= 10.0 dB target).
- **Unexplored areas**: None, all critical paths analyzed.

## Key Decisions Made
- Structure comprehensive report into `report.md` covering all 4 survey aspects.
- Structure self-contained 5-component handoff into `handoff.md` with concrete verification commands.

## Artifact Index
- DISPATCH.md — Dispatch log
- BRIEFING.md — Persistent agent state
- progress.md — Liveness tracker
- report.md — Comprehensive investigation report
- handoff.md — 5-component handoff report

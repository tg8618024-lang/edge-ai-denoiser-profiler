# BRIEFING — 2026-09-23T01:43:00Z

## Mission
Authentic Integer-Domain INT8 Quantization Simulation (Requirement R3)

## 🔒 My Identity
- Archetype: worker
- Roles: implementer, qa, specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_int8_precision
- Original parent: 37613f20-d735-4d4c-af6d-73777c5e0e97
- Milestone: Requirement R3 (Authentic Integer-Domain INT8 Quantization Simulation)

## 🔒 Key Constraints
- Exclusive write access: src/models/precision.py and tests/unit/test_precision.py ONLY.
- Eliminate float32 cast-and-multiply pseudo-quantization.
- Authentic integer arithmetic simulation:
  - Quantize weights and inputs to np.int8 (with symmetric/affine scaling and zero-point).
  - Perform GEMM matrix multiplication accumulation using 32-bit integers (np.int32).
  - Add 32-bit integer bias (b_int32 in np.int32) directly inside accumulator registers.
  - Implement integer-domain ReLU: np.maximum(z_int32, 0).
  - Apply output dequantization / scaling at layer output to match real edge NPU/DSP hardware execution.
- Maintain parameter memory footprint <= 0.35x FP32, SQNR >= 35.0 dB, and SNR delta >= 10.0 dB.
- Integrity: DO NOT hardcode test results, dummy/facade implementations, or circumvent intended tasks.

## Current Parent
- Conversation ID: 37613f20-d735-4d4c-af6d-73777c5e0e97
- Updated: not yet

## Task Summary
- **What to build**: Full authentic integer-domain quantization simulation in `src/models/precision.py` and comprehensive test coverage in `tests/unit/test_precision.py`.
- **Success criteria**:
  - `pytest tests/unit/test_precision.py` passes 100%.
  - `python evaluate.py --benchmark all` passes with exit code 0.
  - Memory ratio <= 0.35x, SQNR >= 35.0 dB, SNR delta >= 10.0 dB.
- **Interface contracts**: PROJECT.md, ORIGINAL_REQUEST.md.
- **Code layout**: src/models/precision.py, tests/unit/test_precision.py.

## Change Tracker
- **Files modified**: pending
- **Build status**: pending
- **Pending issues**: none

## Quality Status
- **Build/test result**: pending
- **Lint status**: 0 violations
- **Tests added/modified**: pending

## Loaded Skills
- None

## Key Decisions Made
- Use symmetric int8 for weights and zero-centered inputs, affine int8 with zero-point for post-ReLU asymmetric activations.
- Compute dot product with np.int8 operands and np.int32 accumulation.
- Add integer bias b_int32 in np.int32 directly to accumulator.
- Perform integer ReLU np.maximum(accum_int32, 0) before dequantization.
- Apply float scaling only at layer exit.

## Artifact Index
- DISPATCH.md — Worker instructions from parent
- BRIEFING.md — Persistent memory
- progress.md — Liveness heartbeat
- handoff.md — 5-component completion report

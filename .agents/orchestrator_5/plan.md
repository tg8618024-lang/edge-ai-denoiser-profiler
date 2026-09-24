# Implementation Plan — Orchestrator 5 Hardening

## Overview
Forensic audit, stress-testing, and architectural hardening of `edge_ai_denoiser_profiler` across four critical tracks:
1. R1: Authentic Neural DNSMOS & Psychoacoustic Evaluator Replacement (`src/telemetry/dnsmos.py`, async background worker).
2. R2: Native C/C++ SIMD Integer-Domain Kernel (AVX2 / VNNI / NEON, `vpdpbusd`/`sdot`, eliminating Python GIL overhead).
3. R3: End-to-End Complex Spectral Mapping (E2E CRM Phase Preservation in `GRUMaskNet`, direct complex ratio mask prediction).
4. R4: Frontend Architecture Modularization (ES modules deconstruction of `app.js`) & Jitter-Free AudioWorklet circular ring buffer.

## Execution Strategy

### Step 0: Survey & Codebase Mapping (3 Explorers in Parallel)
- Explorer 1 (`.agents/explorer_survey_r1_r3`):
  - Audit `src/telemetry/dnsmos.py` to identify all sigmoid curve-fit formulas and dependencies. Determine ONNX / lightweight neural models or authentic STOI/PESQ implementations and async thread architecture.
  - Audit `src/models/denoiser.py` and `GRUMaskNet` to inspect how magnitude masks vs CRM heuristics are currently structured and how to refactor to genuine complex ratio masking ($M_r, M_i$).
- Explorer 2 (`.agents/explorer_survey_r2_simd`):
  - Audit `src/models/precision.py` and determine current INT8 matrix multiplication simulation.
  - Investigate C/C++ compiler availability on Windows (MSVC `cl.exe`, `gcc`/MinGW, or clang) and design native SIMD kernel (`vpdpbusd` / AVX2 / fallback SSE4.1) exposed via ctypes or C extension.
- Explorer 3 (`.agents/explorer_survey_r4_frontend`):
  - Audit `src/dashboard/static/app.js` (3,169 lines), identify module boundaries (audio, canvases, websocket, subtitles, controls), and structure into ES modules without global pollution.
  - Inspect `scheduleAudioPlayback()` and design Web Audio API `AudioWorkletNode` + circular ring buffer.
  - Review test suite (`pytest`, `evaluate.py`) and existing test coverage.

### Step 1: Synthesis & PROJECT.md Update
- Merge explorer findings into Feature Inventory and update milestone contracts in `PROJECT.md`.

### Step 2: Implementation & Verification
- Dispatch specialized Workers per milestone with mandatory integrity warning.
- Workers run builds, unit tests, and performance benchmarks.

### Step 3: Dual Verification & Adversarial Stress Testing
- Dispatch Reviewers and Challengers.
- Execute automated test suite and evaluate.py.

### Step 4: Forensic Audit & Integrity Gate
- Dispatch `teamwork_preview_auditor` for binary veto verification.
- Review GATE_STATUS.md.

### Step 5: Final Delivery
- Deliver handoff report and notify Sentinel / Parent.

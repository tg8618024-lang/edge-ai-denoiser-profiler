# Dispatch: Test Writer (E2E Testing Track)

## Context
Project: Edge AI Audio Denoiser & Profiler
Target Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your Working Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\test_writer_e2e
Authoritative User Request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Project Decomposition & Interfaces: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md
E2E Test Infrastructure Spec: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\TEST_INFRA.md
Survey Specification: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e\report.md

## Write Ownership
You exclusively own and will create:
- `tests/conftest.py`
- `tests/e2e/test_evaluation.py`
- `tests/adversarial/test_adversarial.py`
- `TEST_READY.md` (publish when suite is complete)
DO NOT modify implementation code under `src/`.

## Mission & Implementation Details
Design and implement the E2E test suite and non-interactive automated test runner per `TEST_INFRA.md`:
1. `tests/conftest.py`:
   - Shared fixtures for sample rates, synthetic audio fixtures, profiler mock/dummy fixtures, temporary directories.
2. `tests/e2e/test_evaluation.py`:
   - Tests covering Tiers 1-4:
     - Tier 1: Feature isolation (framing, STFT, neural denoiser, 3-stage latency, multi-precision FP32/FP16/INT8).
     - Tier 2: Boundary conditions (zero-amplitude silence, max-amplitude $\pm 1.0$, tiny audio clips, single frame).
     - Tier 3: Combinatorial pairs (denoiser with profiler active, precision switching during streaming).
     - Tier 4: Real-world benchmark scenarios (Speech + White noise $\ge 10\text{ dB SNR}$, Speech + Drone hum, Speech + RF static, continuous 500-frame streaming without memory leak).
3. `tests/adversarial/test_adversarial.py`:
   - Tier 5 adversarial tests: DC bias offset, hard clipping, high-frequency transients, NaN/Inf input handling, rapid mode switching.
4. Verify by running tests (or verifying importability/dry-run), and create `TEST_READY.md` at project root with the coverage summary.

## 2026-09-06T18:02:27Z
You are Test Writer for the E2E Testing Track.
Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\test_writer_e2e
Authoritative request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Project plan: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md
E2E Test Infrastructure Spec: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\TEST_INFRA.md
Survey Specification: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e\report.md
Your dispatch instructions: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\test_writer_e2e\DISPATCH.md

Write Ownership:
You exclusively own:
- tests/conftest.py
- tests/e2e/test_evaluation.py
- tests/adversarial/test_adversarial.py
- TEST_READY.md (publish when suite is complete)
DO NOT modify implementation code under src/.

Mission:
Build the E2E test suite and test harness covering Tiers 1-4 in test_evaluation.py and Tier 5 in test_adversarial.py:
1. Feature coverage (framing, STFT, Wiener/GRU denoiser, latency profiler, FP32/FP16/INT8).
2. Boundary & corner cases (silence, clipping, tiny audio, single frame).
3. Cross-feature combinations (profiler + denoiser, precision switching during streaming).
4. Real-world benchmark scenarios (Speech + White noise >= 10 dB SNR, Drone hum, RF static, 500-frame streaming without memory leak).
5. Tier 5 adversarial stress tests (DC offset, clipping at +-1.0, NaN/Inf handling, buffer jitter).
6. Run tests (or verify syntax and mock execution) and write TEST_READY.md at project root with the coverage summary.
7. Write your handoff report to C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\test_writer_e2e\handoff.md and send a message to parent orchestrator.

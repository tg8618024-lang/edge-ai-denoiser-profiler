# Task Assignment — Worker M2 Gen3

## Mission
Complete Milestone 2: Finalize noise PSD scaling and drone harmonic notch tracking in `src/models/denoiser.py`, verify all unit, adversarial, and E2E tests pass, and report verified test outputs.

## Working Directory
`C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_gen3`

## Authoritative Inputs
1. `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md`
2. `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md`
3. `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\TEST_INFRA.md`
4. Explorer reports:
   - `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_profiler\handoff.md`
   - `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_precision\handoff.md`
   - `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning\handoff.md`

## Current Codebase State
- Telemetry modules (`src/telemetry/profiler.py`, `ring_buffer.py`, `memory.py`, `schema.py`) pass 19/19 tests in `tests/unit/test_profiler.py`.
- Precision engine (`src/models/precision.py`) and `tests/unit/test_precision.py` pass 7/7 tests.
- Audio pipeline (`src/audio/pipeline.py`) has precision switching.

## Instructions
1. Inspect `src/models/denoiser.py`:
   - In `DecisionDirectedWienerFilter.compute_gain`:
     Scale the 15th percentile noise PSD by `* 5.5` (`noise_psd = np.percentile(self.mag_history, 15, axis=0) * 5.5 + 1e-8`).
   - For stationary / harmonic noise (specifically drone hum harmonics around 120Hz, 240Hz, 360Hz -> FFT bins 3-4, 7-8, 11-12):
     Ensure the filter applies sufficient attenuation so delta SNR >= 10.0 dB across all noise benchmarks (white, pink, drone, RF static) in both `test_denoiser.py` and `test_evaluation.py` (test_scenario_1, test_scenario_2, test_scenario_3, test_scenario_4).
   - Ensure the hybrid gain fusion preserves speech formants and maintains musical noise protection.
2. Execute tests using `.venv\Scripts\python.exe`:
   - `.venv\Scripts\python.exe -m pytest tests/unit/ -v`
   - `.venv\Scripts\python.exe -m pytest tests/adversarial/ -v`
   - `.venv\Scripts\python.exe -m pytest tests/e2e/ -v`
3. Document test commands, pass/fail counts, and execution logs in `handoff.md`.
4. Send a completion message to your parent orchestrator.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

# Worker M2 Replacement Task (from interruption point)

## Context
Worker M2 (gen1) created the telemetry modules and precision engine, and passed 19/19 profiler unit tests and 6/7 precision unit tests before a network timeout interrupted it.

## Remaining Work Items
1. In `tests/unit/test_precision.py` and `src/models/precision.py`:
   - Ensure `W_rec` (shape 64x64, float32) is included in model weights dictionary so model byte size is exactly 165892 bytes (or matches `get_model_size_bytes()`), and all 7 tests in `tests/unit/test_precision.py` pass.
2. In `src/models/denoiser.py`:
   - Apply the calibrated noise PSD scaling (`* 5.5` on the 15th percentile in `DecisionDirectedWienerFilter.compute_gain`) which was empirically verified to produce 11.46 dB SNR gain on white noise and 19.44 dB on RF static.
   - For drone hum (120 Hz, 240 Hz, 360 Hz harmonics): apply harmonic notch attenuation on bins 3-4, 7-8, 11-12 to ensure delta SNR >= 10.0 dB in `test_scenario_2` and `test_denoiser.py`.
   - Ensure `compute_gain` in `HybridDenoiser` fuses neural and Wiener gains smoothly without attenuating speech formants.
3. Test Execution & Verification:
   - Run: `.venv\Scripts\python.exe -m pytest tests/unit/ -v` (Expect: 100% pass across test_stft.py, test_profiler.py, test_precision.py, test_denoiser.py).
   - Run: `.venv\Scripts\python.exe -m pytest tests/adversarial/ -v` (Expect: 8/8 pass).
   - Run: `.venv\Scripts\python.exe -m pytest tests/e2e/ -v` (Expect: all Tier 1-4 tests pass).
4. Write handoff report to:
   `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m2_gen2\handoff.md`
5. Send completion message to parent orchestrator.

## 2026-09-07T01:58:30Z
Error: The stream was interrupted. Please continue the task you were working on.

# Progress Tracker — Worker M2 Gen 2

Last visited: 2026-09-07T01:59:30Z

## Status
- Verified `tests/unit/test_precision.py` (7/7 passed).
- Now updating `src/models/denoiser.py`:
  1. `DecisionDirectedWienerFilter.compute_gain`: apply `* 5.5` scaling to the 15th percentile noise PSD estimate (`noise_psd = np.percentile(self.mag_history, 15, axis=0) * 5.5 + 1e-8`).
  2. For drone hum (120 Hz, 240 Hz, 360 Hz harmonics): apply harmonic notch attenuation on bins 3-4, 7-8, 11-12 to ensure delta SNR >= 10.0 dB in both `test_scenario_2` and `test_denoiser.py`.
  3. Ensure `compute_gain` in `HybridDenoiser` fuses neural and Wiener gains smoothly without attenuating speech formants.

## Next Steps
- Apply changes to `src/models/denoiser.py`.
- Run tests and verify.

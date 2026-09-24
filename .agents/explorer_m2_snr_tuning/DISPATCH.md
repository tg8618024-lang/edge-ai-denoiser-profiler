# Explorer M2 SNR Tuning Task

## Objective
Investigate and resolve the SNR gain failures identified by worker_m1_verify:
1. Review `tests/unit/test_denoiser.py:310` where pink noise reached 9.76 dB (target >= 10.0 dB) and drone hum reached 9.68 dB on `SyntheticAudioGenerator`.
2. Review `tests/e2e/test_evaluation.py:941, 971, 1000` where real-world benchmarks on `ReferenceSyntheticGenerator` reached only ~5.1 dB in default hybrid mode.
3. Investigate the Decision-Directed Wiener Filter and GRU-MaskNet implementations in `src/models/denoiser.py` and `src/audio/pipeline.py`:
   - Decision-Directed a priori SNR smoothing factor $\alpha$ (e.g. 0.96 vs 0.98).
   - Noise PSD tracking adaptation speed and minimum tracking floor.
   - Spectral floor / attenuation limiter $\beta$ (e.g. -18 dB to -24 dB).
   - Hybrid fusion weighting $\rho$ between GRU-MaskNet and Wiener filter.
   - Formant alignment between speech generators and neural/Wiener filter bands.
4. Provide a concrete, mathematically sound recommendation for `src/models/denoiser.py` and `src/audio/pipeline.py` so that:
   - All 4 noise types (white, pink, drone, rf) achieve >= 10.0 dB SNR improvement on 0 dB mixtures in `test_denoiser.py`.
   - All Tier 4 real-world benchmarks in `test_evaluation.py` achieve >= 10.0 dB SNR improvement.
   - All 8 adversarial tests continue passing 100%.
5. Provide detailed diagnosis, tuning formula, and code recommendations in `handoff.md`.

## 2026-09-06T19:19:00Z
You are an Explorer focusing on investigating and tuning the SNR improvement and noise suppression in Milestone 1/2.
Your working directory is:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning

Mandatory inputs:
1. Read the authoritative user request at:
   C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
2. Read C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md and TEST_INFRA.md
3. Read your task description at:
   C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning\DISPATCH.md
4. Read the verification findings in:
   C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1_verify\handoff.md
5. Inspect `src/models/denoiser.py`, `src/audio/pipeline.py`, `src/audio/dataset.py`, `tests/unit/test_denoiser.py`, and `tests/e2e/test_evaluation.py`.
6. Diagnose why pink noise and drone hum fell slightly short of 10.0 dB in `test_denoiser.py` and why `test_evaluation.py` Tier 4 scenarios scored ~5.1 dB in hybrid mode.
7. Devise an exact parameter tuning and filter calibration solution so that SNR gain >= 10.0 dB across all 4 noise types on both speech generators, while preserving speech clarity and passing all 8 adversarial tests.

Output:
Write a comprehensive diagnosis and solution recommendation in:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning\handoff.md
Then send a completion message to your parent.

# BRIEFING — 2026-09-06T18:03:00Z

## Mission
Implement Milestone 1: Audio Core & Denoising Pipeline (Streaming STFT/iSTFT, GRU-MaskNet & Wiener filter, synthetic dataset generator, ring buffer audio streaming, and AudioDenoisingPipeline with profiler hooks).

## 🔒 My Identity
- Archetype: implementer / qa / specialist
- Roles: implementer, qa, specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1
- Original parent: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Milestone: M1 (Audio Core & Denoising Pipeline)

## 🔒 Key Constraints
- Exclusively own and write only to:
  - `src/__init__.py`
  - `src/audio/__init__.py`
  - `src/audio/stft.py`
  - `src/audio/stream.py`
  - `src/audio/dataset.py`
  - `src/audio/pipeline.py`
  - `src/models/__init__.py`
  - `src/models/denoiser.py`
  - `tests/unit/test_stft.py`
  - `tests/unit/test_denoiser.py`
- DO NOT CHEAT: Genuine logic only, no hardcoded test results, no dummy facades.
- Machine-precision overlap-add reconstruction (< 1e-14 error) and 1-hop algorithmic delay.
- Continuous spectral gain mask G(f) in [0, 1] preserving formants and suppressing noise.
- Broadband SNR >= 10 dB improvement on 0 dB mixtures without clipping or distortion.
- AudioDenoisingPipeline with hook points for 3-stage latency profiling.
- Pure NumPy / SciPy implementation (fast, zero heavy framework dependencies).

## Current Parent
- Conversation ID: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Updated: 2026-09-06T18:03:00Z

## Task Summary
- **What to build**: Pure NumPy/SciPy streaming STFT/iSTFT engine, GRU-MaskNet recurrent denoiser + Decision-Directed Wiener spectral filter, synthetic speech + 4 noise generators + broadband SNR evaluation, ring buffer streaming chunker, and AudioDenoisingPipeline with profiler stage hooks.
- **Success criteria**: Perfect reconstruction error < 1e-14; >= 10 dB SNR improvement across white, pink, drone, and RF noise on 0 dB mixtures; all unit tests in test_stft.py and test_denoiser.py pass.
- **Interface contracts**: `PROJECT.md` § Interface Contracts (M1 <-> M2 `AudioDenoisingPipeline.process_frame(frame_pcm, profiler=None)`).
- **Code layout**: `PROJECT.md` § Code Layout.

## Key Decisions Made
- Use periodic Hann window of length N=512, hop H=256, with sqrt(w) for both analysis and synthesis to ensure COLA condition sum(w_a * w_s) = 1.0.
- State-space ring buffer for streaming STFT with 1-hop algorithmic delay (256 samples = 16ms @ 16kHz).
- GRU-MaskNet with 64 hidden units (~58k parameters) and calibrated initial weights, coupled with Decision-Directed Wiener filter with dynamic noise PSD estimation (percentile tracking) for clean formant preservation and musical noise suppression.
- Support parameter save/load in NumPy format (.npz or dict of arrays).
- Synthetic speech with 120-220 Hz pitch glottal pulses, 3 resonant IIR formant filters (500, 1500, 2500 Hz), and syllabic AM envelope.
- 4 calibrated noise generators: White Gaussian, Pink (1/f), Drone motor harmonic hum (60/120/240 Hz), and RF static crackle bursts.

## Artifact Index
- `handoff.md` — Handoff report upon milestone completion.
- `progress.md` — Heartbeat progress tracking.

## Change Tracker
- **Files modified**: None yet.
- **Build status**: Untested.
- **Pending issues**: Initial implementation.

## Quality Status
- **Build/test result**: Not run yet.
- **Lint status**: Clean.
- **Tests added/modified**: `tests/unit/test_stft.py`, `tests/unit/test_denoiser.py` (planned).

## Loaded Skills
- None required.

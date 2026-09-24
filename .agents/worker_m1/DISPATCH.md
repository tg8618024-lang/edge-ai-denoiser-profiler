# Dispatch: Worker 1 (Milestone 1 - Audio Core & Denoising Pipeline)

## Context
Project: Edge AI Audio Denoiser & Profiler
Target Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your Working Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1
Authoritative User Request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Project Decomposition & Interfaces: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md
Architecture & Math Survey: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model\report.md

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You exclusively own and will create/modify:
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
DO NOT edit any other files.

## Mission & Implementation Details
Implement Milestone 1 using pure NumPy/SciPy (which are already installed in the environment; no heavy PyTorch required):
1. `src/audio/stft.py`:
   - Streaming STFT/iSTFT class with $N=512$, $H=256$ (16ms frame @ 16kHz).
   - Square-root periodic Hann analysis and synthesis windows guaranteeing exact machine-precision overlap-add reconstruction ($<10^{-14}$ error) and exactly 1 hop algorithmic delay without boundary pops.
2. `src/models/denoiser.py`:
   - Decision-Directed Wiener spectral filter and lightweight GRU recurrent mask estimator.
   - Computes a continuous spectral gain mask $G(f) \in [0, 1]$ applied to STFT magnitude bins, preserving speech formants while suppressing background noise.
   - Supports parameter saving/loading in NumPy format.
3. `src/audio/dataset.py`:
   - Synthetic speech generator (glottal harmonic excitation at 120-220 Hz + 3 resonant formant bandpass filters at 500, 1500, 2500 Hz + syllabic amplitude envelope).
   - 4 calibrated noise generators: White Gaussian noise, Pink noise ($1/f$ filter), Drone/HVAC humming (60Hz/120Hz/240Hz harmonics), and RF static bursts (Poisson impulsivity).
   - Mixture generator supporting target input SNR (e.g. 0 dB, 5 dB, 10 dB).
   - Broadband SNR calculation function: $\text{SNR} = 10 \log_{10} \frac{\sum s^2}{\sum (\hat{s} - s)^2}$.
   - Must guarantee >= 10 dB SNR improvement on 0 dB mixtures without clipping or distortion!
4. `src/audio/stream.py`:
   - Ring buffer and frame-by-frame audio streamer for streaming PCM frames.
5. `src/audio/pipeline.py`:
   - `AudioDenoisingPipeline` tying together framing, STFT, denoiser model, and iSTFT synthesis.
   - Includes hook points for profiler: `start_stage("pre_processing")`, `start_stage("tensor_compute")`, `start_stage("output_synthesis")`.
6. Unit Tests:
   - Create and run `tests/unit/test_stft.py` verifying perfect STFT/iSTFT reconstruction.
   - Create and run `tests/unit/test_denoiser.py` verifying >= 10 dB SNR improvement across white noise, pink noise, drone hum, and RF static.
   - Run tests and document passing test execution in your handoff report.

## 2026-09-06T18:02:27Z
You are Worker 1 (Audio Core & Denoising Pipeline Specialist).
Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1
Authoritative request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Project plan: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\PROJECT.md
Architecture & Math Survey: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model\report.md
Your dispatch instructions: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1\DISPATCH.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership:
You exclusively own:
- src/__init__.py
- src/audio/__init__.py
- src/audio/stft.py
- src/audio/stream.py
- src/audio/dataset.py
- src/audio/pipeline.py
- src/models/__init__.py
- src/models/denoiser.py
- tests/unit/test_stft.py
- tests/unit/test_denoiser.py

Implement Milestone 1 using pure NumPy and SciPy:
1. Streaming STFT/iSTFT with square-root Hann window, perfect reconstruction (<1e-14 error), 1-hop algorithmic delay.
2. Lightweight recurrent GRU-MaskNet and Decision-Directed Wiener spectral filter.
3. Synthetic speech generator (harmonic glottal pulses + 3 vowel formants + syllabic envelope) and 4 noise generators (White, Pink, Drone hum, RF static).
4. Broadband SNR metric calculation: 10 * log10(sum(clean^2) / sum((denoised - clean)^2)). Must achieve >= 10 dB SNR improvement on 0 dB mixtures!
5. Audio stream chunker / ring buffer and AudioDenoisingPipeline class with hook points for 3-stage latency profiling.
6. Create unit tests in tests/unit/test_stft.py and tests/unit/test_denoiser.py. Run tests with pytest or python.
7. Write your handoff report to C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1\handoff.md with all test commands and passing outputs, and send a message to parent orchestrator.


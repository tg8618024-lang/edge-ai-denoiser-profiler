# BRIEFING — 2026-09-06T17:48:00Z

## Mission
Survey and design the real-time neural audio denoising pipeline, lightweight recurrent mask estimator / DSP filter, synthetic benchmark dataset generator, and streaming I/O architecture.

## 🔒 My Identity
- Archetype: explorer
- Roles: Audio Denoising Pipeline & Model Specialist
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model
- Original parent: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Milestone: Audio Denoising Pipeline Architecture & Model Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production source code outside .agents
- Real-time frame budget constraint: <= 20ms per frame, target tensor compute < 5ms on CPU
- Quality constraint: >= 10 dB SNR improvement on noisy benchmark audio clips
- Zero distortion / click-free STFT/iSTFT overlap-add synthesis
- Support both live microphone streaming and pre-recorded benchmark files

## Current Parent
- Conversation ID: 2f5181a5-717d-442c-9f20-392a7c3a7fc1
- Updated: 2026-09-06T18:05:00Z

## Investigation State
- **Explored paths**: Python 3.13.7 environment inspection, STFT/iSTFT overlap-add COLA validation, lightweight GRU mask estimator and decision-directed Wiener filter design, synthetic speech & noise generation benchmarks, multi-precision quantization analysis.
- **Key findings**:
  1. STFT/iSTFT with periodic Hann window ($N=512, H=256$) achieves machine-precision reconstruction ($1.33 \times 10^{-15}$ error) with exact 1-hop ($16\text{ ms}$) algorithmic latency.
  2. Total per-frame CPU latency is $\sim 0.26\text{ ms}$ (Pre: 0.031 ms, Tensor: 0.197 ms, Synth: 0.032 ms) with $> 96\%$ real-time budget headroom on a 16 ms chunk.
  3. Denoising pipeline achieves $+10.5\text{ dB}$ to $+11.8\text{ dB}$ SNR improvement on challenging 0 dB synthetic benchmark mixtures (white noise, drone motor hum, RF static bursts) and $> 26\text{ dB}$ attenuation during speech pauses.
  4. INT8 quantization shrinks model footprint from 305 KB to 76.2 KB with high fidelity (39.05 dB SQNR).
  5. Pure NumPy/SciPy execution avoids heavy ~2GB PyTorch dependencies and operates natively on Windows Python 3.13.
- **Unexplored areas**: None for Explorer 1 scope. Next phase is implementation by Builder agents.

## Key Decisions Made
- Design a standalone, zero-heavy-dependency neural + DSP pipeline using NumPy and SciPy (with optional Numba acceleration).
- Fix sample rate to 16 kHz, window size $N=512$, hop size $H=256$ (16 ms) for the golden speech enhancement standard.
- Instrument the 3-stage latency profiler at the frame loop boundaries using `time.perf_counter()`.
- Use a 58k-parameter recurrent GRU-MaskNet architecture with FP32, FP16, and INT8 modes.

## Artifact Index
- report.md — comprehensive survey and architecture design report
- handoff.md — 5-component handoff report (Hard handoff)
- progress.md — liveness heartbeat


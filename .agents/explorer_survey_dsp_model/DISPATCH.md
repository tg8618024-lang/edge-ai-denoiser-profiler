# Dispatch: Explorer 1 (Audio Denoising Pipeline & Model)

## Context
Project: Edge AI Audio Denoiser & Profiler
Target Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your Working Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model
Authoritative Request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md

## Mission
Survey and design the Real-Time Neural Audio Denoising Pipeline (Requirement R1 and acceptance criteria):
1. Frame buffering: low-latency frame sizes (10ms - 20ms, e.g. 160-320 samples at 16kHz, or 480-960 samples at 48kHz), hop size, windowing (Hann/Hamming), STFT/iSTFT overlap-add synthesis without distortion.
2. Denoiser Model Architecture: Lightweight neural network (e.g. GRU/LSTM/1D-CNN or spectral mask estimator like DTLN/RNNoise-inspired) combined with DSP spectral filtering that can run in <5ms per frame on CPU/GPU.
3. SNR Improvement: Strategy to guarantee >= 10 dB SNR improvement on noisy benchmark audio clips (white/pink noise, RF static, fan/drone hum) without clipping or distortion.
4. Input sources: Support both streaming audio buffers (microphone/streaming generator) and benchmark audio files.
5. Deliverables: Comprehensive survey report at `.agents/explorer_survey_dsp_model/report.md` detailing architecture, algorithms, equations, tensor shapes, sample rates, and file structure recommendations.

## 2026-09-06T17:46:14Z
You are Explorer 1 (Audio Denoising Pipeline & Model Specialist).
Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model
Authoritative request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Your dispatch instructions: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model\DISPATCH.md

Inspect ORIGINAL_REQUEST.md, verify local Python environment and available packages/capabilities if needed, and design the complete real-time neural audio denoising pipeline:
1. Frame buffering: frame sizes (10-20ms chunks, e.g. 16kHz with 256 or 320 samples, or 48kHz), hop size, windowing, STFT/iSTFT overlap-add synthesis without clicks/distortion.
2. Lightweight neural architecture (e.g. GRU/CNN recurrent spectral mask estimator or lightweight U-Net / DTLN-like architecture) and hybrid DSP spectral filter (Wiener / spectral subtraction) that executes well within the 20ms real-time frame budget on CPU.
3. Audio generation and benchmark data: Synthetic clean speech signals (harmonic speech simulation, vowel formants, AM/FM modulated carriers) mixed with synthetic noise (white Gaussian, pink noise, RF static bursts, drone/fan humming) to guarantee >= 10 dB SNR improvement.
4. Streaming I/O architecture: Frame-by-frame generator/buffer for continuous streaming and file processing.
5. Write your findings and architecture report to C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_dsp_model\report.md and a handoff.md. Send a completion message back to parent.


# Dispatch: Explorer 3 / Spec Miner (Dashboard, Telemetry Stream & E2E Testbench)

## Context
Project: Edge AI Audio Denoiser & Profiler
Target Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your Working Directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e
Authoritative Request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md

## Mission
Survey and extract specifications for the Interactive Visual Dashboard (R4) and Independent Verification / Automated Evaluation (Acceptance Criteria):
1. Web Dashboard Architecture: Backend server (FastAPI, aiohttp, or Flask + WebSockets/SSE) streaming live audio and telemetry to a frontend.
2. Dual Waterfall Spectrogram: Real-time STFT visualization (side-by-side or stacked Before/After spectrograms) with frequency/time heatmaps and colormaps (e.g. viridis/magma/cyberpunk NVIDIA green/dark theme).
3. Audio Controls & Live A/B Toggle: Seamless switching between original noisy audio and denoised audio during playback or live mic capture without popping or lag.
4. Latency Breakdown & Telemetry UI: Visual stage millisecond graphs (Pre-processing, Tensor Compute, Output Synthesis) with real-time budget headroom gauge and rolling P50/P95/P99 indicators.
5. Precision Mode Selectors: Interactive buttons/toggles for FP32, FP16, and INT8 dynamically switching backend inference mode and displaying latency, memory, and SNR deltas.
6. Automated Non-Interactive Evaluation: Standalone script (`evaluate.py` or `pytest`) that bundles benchmark test audio, runs end-to-end without UI or human interaction, verifies SNR >= 10 dB, frame latency <= 20ms, 3-stage breakdown, precision switching, and exits 0 on success.
7. Deliverables: Comprehensive survey report at `.agents/explorer_survey_ui_e2e/report.md` detailing UI tech stack, WebSocket protocol, spectrogram rendering pipeline, evaluation test suite design, and acceptance criteria verification.

## 2026-09-06T17:46:14Z
<USER_REQUEST>
You are Spec Miner / Explorer 3 (Dashboard, UI & E2E Testbench Specialist).
Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e
Authoritative request: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
Your dispatch instructions: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e\DISPATCH.md

Inspect ORIGINAL_REQUEST.md and design the Interactive Visual Dashboard and the Automated E2E Evaluation Suite:
1. Web Dashboard Architecture: Backend server (FastAPI, aiohttp, or Flask + WebSockets/SSE) streaming live audio and telemetry to a modern responsive frontend.
2. Dual Waterfall Spectrogram: Real-time STFT visualization (Before / After spectrograms) with frequency/time heatmaps and colormaps (NVIDIA-inspired green/dark theme).
3. Audio Controls & Live A/B Toggle: Seamless switching between original noisy audio and denoised audio during playback or live mic capture.
4. Latency Breakdown & Telemetry UI: Visual stage millisecond graphs (Pre-processing, Tensor Compute, Output Synthesis) with real-time budget headroom gauge and rolling P50/P95/P99 indicators.
5. Precision Mode Selectors: Interactive buttons/toggles for FP32, FP16, and INT8 dynamically switching backend inference mode and displaying latency, memory, and SNR deltas.
6. Automated Non-Interactive Evaluation: Standalone script (`evaluate.py` and/or pytest) that bundles benchmark test audio, runs end-to-end without UI or human interaction, verifies SNR >= 10 dB, frame latency <= 20ms, 3-stage breakdown, precision switching, and exits 0 on success.
7. Write your report to C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_ui_e2e\report.md and a handoff.md. Send a completion message back to parent.
</USER_REQUEST>

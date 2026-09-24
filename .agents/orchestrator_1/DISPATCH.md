## 2026-09-06T17:44:00Z

You are the Project Orchestrator for the Edge AI Audio Denoiser & Profiler project.
Workspace directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler
Your agent working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\orchestrator_1
Authoritative user request file: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md

Please inspect ORIGINAL_REQUEST.md, decompose the requirements, dispatch specialists, and drive the project to completion.
Key deliverables:
1. Real-time audio denoising pipeline (frame buffers 10-20ms, streaming & benchmark clips, >= 10 dB SNR improvement).
2. NVIDIA-style latency profiler across 3 distinct stages (Pre-processing, Tensor Compute, Output Synthesis) with rolling median/P95/P99 stats.
3. Multi-precision comparison mode (FP32, FP16, INT8) measuring latency, memory, and SNR.
4. Interactive visual dashboard with dual waterfall spectrogram, live A/B audio toggle, latency breakdown, and precision selectors.
5. End-to-end automated non-interactive evaluation script (e.g. pytest or evaluate.py) that exits 0 with all assertions passing.

Maintain progress.md and BRIEFING.md in your working directory at C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\orchestrator_1.
When done, report completion and handoff back to Sentinel.

## 2026-09-23T11:44:35Z

You are Explorer R1 & R3 Specialist for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r1_r3
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r1_r3\context.md

Your Task:
Investigate and produce a detailed architectural analysis and remediation specification for:
1. R1: Authentic Neural DNSMOS & Psychoacoustic Evaluator Replacement:
   - Audit `src/telemetry/dnsmos.py` to identify all heuristic sigmoid curve-fits, analytical approximations, and fake metric functions.
   - Investigate how to integrate authentic neural DNSMOS (e.g., ONNX model graph for ITU-T P.835 SIG/BAK/OVRL) or authentic psychoacoustic evaluation (genuine STOI/PESQ implementations e.g. 1/3-octave band correlation).
   - Design the asynchronous background profiler thread architecture so that DNSMOS and psychoacoustic evaluations execute out-of-band without adding latency to the 16.0 ms real-time audio pipeline.
2. R3: End-to-End Complex Spectral Mapping (E2E CRM Phase Preservation):
   - Audit `src/models/denoiser.py` and `GRUMaskNet`.
   - Identify how magnitude gain masking and heuristic phase/boost scalars are currently used.
   - Design the refactoring of `GRUMaskNet` to genuine Complex Spectral Mapping: directly estimating real ($M_r$) and imaginary ($M_i$) ratio masks, and applying complex multiplication $S = Y \cdot M = (Y_r M_r - Y_i M_i) + j(Y_r M_i + Y_i M_r)$ to reconstruct clean real and imaginary STFT components without heuristic phase alteration or empirical boost scalars.
   - Specify weight matrices, parameter counts, tensor dimensions, and initialization to guarantee numerical stability and >=10 dB SNR improvement.

Deliverables:
- Write full findings to `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r1_r3\analysis.md`.
- Write `handoff.md` following standard format.
- Send a completion message back to parent.

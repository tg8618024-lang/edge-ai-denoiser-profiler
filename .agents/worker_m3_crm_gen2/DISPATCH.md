## 2026-09-23T12:31:01Z

You are Worker M3 E2E CRM Denoiser (Gen 2) for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m3_crm_gen2
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context and survey findings at:
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m3_crm_gen2\context.md
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r1_r3\analysis.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership:
You exclusively own:
- src/models/denoiser.py
- src/models/crm.py
- src/audio/harmonics.py (clean empirical 0.32 boost)
- tests/unit/test_denoiser.py

Your Task:
1. Refactor `GRUMaskNet` in `src/models/denoiser.py` and `ComplexRatioMasker` in `src/models/crm.py` from magnitude-only gain masking into authentic End-to-End Complex Spectral Mapping (E2E CRM).
2. Eliminate the fake imaginary mask calculation from phase gradients (`np.sin(np.gradient(phase))`) and eliminate the empirical harmonic boost scalar (`boost_strength = 0.32` in `harmonics.py`).
3. Implement direct estimation of real ($M_r$) and imaginary ($M_i$) ratio masks bounded by tanh ($[-1.5, 1.5]$) and apply Cartesian complex multiplication:
   $S = Y \cdot M = (Y_r M_r - Y_i M_i) + j(Y_r M_i + Y_i M_r)$
   to reconstruct clean STFT without phase distortion or empirical boost scalars.
4. Support polymorphic execution so that baseline parameter checks (`fp32_bytes == 165892`) in existing tests pass, while providing full E2E CRM mode (`forward_crm`) that delivers >= 10.0 dB SNR improvement across all benchmark noise types.
5. Update `tests/unit/test_denoiser.py` to thoroughly test E2E CRM mask prediction, phase preservation, and SNR gain.
6. Run `pytest tests/unit/test_denoiser.py` and verify all tests pass.

Deliverables:
- Write `handoff.md` with: Observation, Logic Chain, Caveats, Conclusion, Verification Method (including test commands and output).
- Send completion message to parent.

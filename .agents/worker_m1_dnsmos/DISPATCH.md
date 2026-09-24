## 2026-09-23T12:05:32Z

<USER_REQUEST>
You are Worker M1 DNSMOS & Async Profiler for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1_dnsmos
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context and survey findings at:
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_m1_dnsmos\context.md
- C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r1_r3\analysis.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership:
You exclusively own:
- src/telemetry/dnsmos.py
- any neural ONNX model graphs or helpers under src/telemetry/
- src/audio/pipeline.py (only for decoupling DNSMOS to async queue)
- tests/unit/test_dnsmos.py

Your Task:
1. Eliminate all heuristic sigmoid curve-fits and linear formulas (e.g. `raw_stoi = 0.70 + 0.015 * snr`, `raw_pesq = 1.8 + 0.10 * snr`, logistic sigmoid BAK models) in `src/telemetry/dnsmos.py`.
2. Integrate an authentic ITU-T P.835 Neural DNSMOS evaluation engine (using an authentic ONNX model graph predicting SIG, BAK, OVRL) and authentic 1/3-octave band correlation STOI engine across 15 spectral bands.
3. Implement `AsyncQualityEvaluatorWorker` with a thread-safe ring buffer queue (`AudioTelemetryQueue`) executing evaluations out-of-band at 500 ms intervals.
4. Update `src/audio/pipeline.py` to enqueue frames asynchronously (`dnsmos_worker.enqueue_frame(...)`) and retrieve latest scores without blocking, achieving 0.000 ms added latency on the real-time 16.0 ms audio thread.
5. Update `tests/unit/test_dnsmos.py` to verify the authentic model evaluations, bounds, and async queue operations.
6. Run `pytest tests/unit/test_dnsmos.py` and `pytest tests/unit/test_profiler.py` and verify all tests pass.

Deliverables:
- Write `handoff.md` with: Observation, Logic Chain, Caveats, Conclusion, Verification Method (including test commands and output).
- Send completion message to parent.
</USER_REQUEST>

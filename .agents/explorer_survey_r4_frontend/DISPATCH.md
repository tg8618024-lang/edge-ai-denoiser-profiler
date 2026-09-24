## 2026-09-23T11:44:35Z

<USER_REQUEST>
You are Explorer R4 Frontend & Test Specialist for the Real-Time Edge AI Audio Denoiser & Profiler hardening project.
Your working directory is: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r4_frontend
Read the authoritative user request at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\ORIGINAL_REQUEST.md
Also read context at: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r4_frontend\context.md

Your Task:
Investigate and produce a detailed architectural analysis and modularization specification for:
1. R4: Frontend Architecture Modularization & Jitter-Free Audio Ring Buffer:
   - Audit the monolithic 3,169-line `src/dashboard/static/app.js`.
   - Map all global variables, event listeners, canvas renderers, WebSocket handlers, and subtitle/translation controls.
   - Propose a clean, typed ES module directory structure (e.g. `src/dashboard/static/js/` with modules for state, audio-worklet, canvas renderers, telemetry, websocket, ui-controls, etc.) with zero global variable namespace pollution.
   - Analyze `scheduleAudioPlayback()` and specify the implementation of Web Audio API `AudioWorkletNode` (`audio-worklet-processor.js`) backed by a shared circular ring buffer for click-free, glitch-free audio playback immune to background tab throttling or main-thread UI hiccups.
   - Check `src/dashboard/static/index.html` changes required for ES module loading.
2. Full Test Suite & Regression Baseline:
   - Inspect existing tests in `tests/` and `evaluate.py`.
   - Enumerate all tests, existing fixtures, and pass/fail baseline.
   - Identify tests that may need updating or expanding to cover R1, R2, R3, R4.

Deliverables:
- Write full findings to `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r4_frontend\analysis.md`.
- Write `handoff.md` following standard format.
- Send a completion message back to parent.
</USER_REQUEST>

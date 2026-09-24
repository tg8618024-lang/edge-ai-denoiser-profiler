## 2026-09-22T19:58:11Z
You are Worker 3 implementing R4 (Complete Multilingual Translation) and R5 (Browser & Automated Regression Verification).
Your working directory is C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\worker_translation_browser.

MANDATORY FIRST STEP: Read the authoritative requirements at:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\ORIGINAL_REQUEST.md
and read the detailed explorer findings in:
C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_3_gen2\report.md
and C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_3_gen2\handoff.md.

EXCLUSIVE FILE OWNERSHIP:
You have exclusive write access to:
- src/models/translator.py
- tests/unit/test_speech_translation.py
- tests/integration/test_pipeline_stream.py
- tests/e2e/test_dashboard_browser_audit.py
DO NOT edit any other files.

Requirements:
1. R4: In src/models/translator.py:
   - Pre-normalize hyphenated compounds (e.g. 'real-time', 'sub-millisecond') in the tokenizer.
   - Expand VOCAB_MAP and phrase dictionaries to ensure complete, natural sentences across all 9 Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi) and 6 Global languages (Spanish, French, German, Japanese, Chinese, Italian) without leftover untranslated English fragments.
   - Eliminate token-dropping in Japanese and Chinese.
   - Replace raw English fallback with complete vocabulary coverage.
2. R5:
   - In tests/integration/test_pipeline_stream.py, adjust the p50 latency assertion threshold to <= 3.0 ms to eliminate intermittent test-runner load failures (frame budget is 20ms).
   - Implement an automated end-to-end browser / UI audit test in tests/e2e/test_dashboard_browser_audit.py verifying that all 5 workspaces (viewBroadcast, viewEqSuite, viewVectorscope, viewBatch, viewTelemetry), 4 canvases (Oscilloscope, Waterfall Spectrogram, Parametric EQ, Polar Vectorscope), and audio controls load and operate with zero JavaScript console errors or dropped Web Audio buffers.
3. Verification: Run tests using run_command:
   pytest tests/unit/test_speech_translation.py tests/integration/test_pipeline_stream.py tests/e2e/test_dashboard_browser_audit.py
   Ensure all tests pass.
4. Write detailed report to your directory and self-contained handoff.md, then send a message back.

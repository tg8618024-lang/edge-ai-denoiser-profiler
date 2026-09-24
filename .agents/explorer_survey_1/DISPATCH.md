## 2026-09-22T18:41:00Z
Explorer 1 investigating R1 (Robust Pitch Detection & Harmonic Comb Filtering) and R2 (Responsive Parametric EQ & Canvas Node Synchronization).
Scope:
1. R1: Locate HarmonicEnhancer and pitch detection / harmonic comb filtering code in src/. Analyze autocorrelation calculation, current window size (56-sample truncation), pitch lag range 40..200 samples, circular 512-sample history window maintaining >=312 samples continuous overlap. Exact lines, math, failure modes on transient noises.
2. R2: Locate Parametric EQ implementation in frontend (src/dashboard/static/app.js) and backend (src/audio/pipeline.py or audio processing modules). Check how EQ is enabled, dragging biquad filter nodes / studio presets updating audio stream, and changes required for active & reactive by default with instant live audio alteration.
Read-only investigation. Output report.md and handoff.md.

# Handoff Report — Explorer 1: R1 & R2 Investigation

**Investigator**: Explorer 1  
**Working Directory**: `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_1`  
**Reference Report**: `report.md`  

---

## 1. Observation

### R1: Pitch Detection & Harmonic Comb Filtering
1. **Source Location**:
   - `src/audio/harmonics.py:14-106`: `HarmonicEnhancer` class implements `estimate_f0` and `enhance_gain_mask`.
   - `src/audio/pipeline.py:69-72, 547-553`: `AudioDenoisingPipeline` initializes `HarmonicEnhancer` and calls `estimate_f0(frame_pcm)` inside `process_frame`.
   - `src/audio/harmonics.py:31-32`:
     ```python
     self.min_lag = int(self.sample_rate / self.f0_max)  # ~40 samples
     self.max_lag = int(self.sample_rate / self.f0_min)  # ~200 samples
     ```
   - `src/audio/harmonics.py:46-48`:
     ```python
     corr = np.correlate(frame, frame, mode="full")
     mid = len(corr) // 2
     half_corr = corr[mid:] / (energy + 1e-9)
     ```
   - `src/audio/pipeline.py:467, 548`: `frame_pcm` passed to `estimate_f0` has length equal to `self.hop_length = 256` samples.
   - For length $N = 256$ and lag $k \in [40, 200]$, the number of non-zero overlapping samples in `np.correlate(frame, frame)` is $N - k$. At $k = 200$, the overlap is $256 - 200 = 56$ samples ($3.5\text{ ms}$).
   - `src/audio/harmonics.py` has no history buffer (`self.history` does not exist); it computes correlation purely within the single 256-sample frame.
   - `src/audio/pipeline.py:549-553`:
     ```python
     if conf > 0.55 and f0_est > 80.0:
         gain_mask = self.harmonic_enhancer.enhance_gain_mask(
             gain_mask, f0_est, conf, boost_strength=0.08
         )
     ```
   - `src/audio/harmonics.py:87-104`: `enhance_gain_mask` applies Gaussian boosts across up to 24 harmonic multiples ($h \cdot f_0$) to `gain_mask`.

### R2: Responsive Parametric EQ & Canvas Node Synchronization
2. **Backend EQ Engine & Pipeline**:
   - `src/audio/parametric_eq.py:189`:
     ```python
     def __init__(self, sample_rate: int = 16000, enabled: bool = False) -> None:
     ```
   - `src/audio/pipeline.py:120`:
     ```python
     self.eq = ParametricEQ(sample_rate=self.sample_rate, enabled=False)
     ```
   - `src/audio/pipeline.py:570-573`:
     ```python
     # 5-Band Studio Parametric EQ Sculptor
     if hasattr(self, "eq") and self.eq.get_enabled():
         out_pcm = self.eq.process_frame(out_pcm)
     ```
   - `src/audio/parametric_eq.py:286-287`:
     ```python
     if not self.enabled:
         return pcm
     ```
3. **Backend Server (FastAPI & WebSocket)**:
   - `src/dashboard/app.py:1181`: `client_pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)` has `client_pipeline.eq.enabled = False`.
   - `src/dashboard/app.py:1600-1628`: WebSocket handler `set_eq_band` calls `client_pipeline.set_eq_band(...)` but never calls `set_eq_enabled(True)`.
   - `src/dashboard/app.py:1630-1640`: WebSocket handler `set_eq_preset` calls `client_pipeline.apply_eq_preset(...)` but never calls `set_eq_enabled(True)`.
   - `src/dashboard/app.py:394-405`: REST endpoint `POST /api/eq/preset` calls `state.pipeline.apply_eq_preset(...)` but never sets `enabled=True`.
   - `src/dashboard/app.py:372-392`: REST endpoint `POST /api/eq/configure` only sets `enabled` if `req.enabled is not None`. When only band parameters are submitted, `req.enabled` is `None`.
   - `src/dashboard/app.py:1227-1234`: In `client_dual_mode`, `out_frame = dual_res.audio_mix` is produced without calling `client_pipeline.eq.process_frame(out_frame)`.
4. **Frontend Controller (app.js) & UI**:
   - `src/dashboard/static/app.js:1788-1796`:
     Mousemove sends `set_eq_band` on every raw mouse event without throttling or setting `enabled: true`.
   - `src/dashboard/static/app.js:1818-1837`:
     Clicking `.btn-eq-preset` sends `set_eq_preset` and `POST /api/eq/preset` without setting `toggleEqMaster.checked = true` or sending `set_eq_enabled: true`.
   - `src/dashboard/static/app.js:1340-1342, 1691-1728`:
     Telemetry packet includes `"eq": client_pipeline.get_eq_curve()`. Every 16ms, `updateEqCurveUI(msg.eq)` sets `toggleEqMaster.checked = curve.enabled` (`false`) and overwrites `eqBands` without checking `if (eqIsDragging && i === eqActiveBand)`, causing visual jitter during user dragging.
   - `src/dashboard/static/app.js:525-567`:
     Audio is played through Web Audio directly from `audio.denoised` received from the server. There is no client-side Web Audio biquad filter node. All equalization must occur server-side in Python.

---

## 2. Logic Chain

1. **R1 Autocorrelation Truncation**:
   - In `src/audio/pipeline.py:548`, `time_frame` has length $N = 256$ samples.
   - At $f_0 = 80\text{ Hz}$, $T_0 = 200$ samples. Correlating a 256-sample frame with itself shifted by 200 samples produces an overlap of $256 - 200 = 56$ samples (Observation 1).
   - 56 samples is $3.5\text{ ms}$, or $0.28$ of a single pitch period.
   - In Gaussian noise, sample correlation across 56 samples has standard deviation $\sigma \approx 1/\sqrt{56} \approx 0.134$. A random peak easily reaches $> 0.50$.
   - On transient noises (desk bumps, clicks, taps), concentrated burst energy within the 56 overlapping samples causes the normalized cross-correlation peak to spike above $0.55$.
   - When confidence exceeds $0.55$, `enhance_gain_mask` boosts up to 24 harmonic multiples in the suppression mask. The wideband transient noise is thus multiplied by harmonic comb teeth, creating audible metallic ringing / chirping artifacts.
   - Implementing a circular 512-sample history window maintains at least $512 - 200 = 312$ samples of continuous overlap across all lags $k \in [40, 200]$ ($19.5\text{ ms}$ at 16 kHz), covering at least 1.56 pitch periods at 80 Hz and up to 12.8 periods at 400 Hz.
   - Across $\ge 312$ samples, Gaussian noise variance drops to $\sigma \le 1/\sqrt{312} \approx 0.0566$ (making a $0.55$ peak a $9.7\sigma$ statistical impossibility), and isolated transient bursts cannot correlate with non-transient audio, completely eliminating false pitch triggers.

2. **R2 Inactive & Unreactive EQ**:
   - `ParametricEQ` starts with `enabled=False` (Observation 2).
   - In `AudioDenoisingPipeline.process_frame`, `out_pcm = self.eq.process_frame(out_pcm)` is guarded by `self.eq.get_enabled()`, which is `False` (Observation 2).
   - Dragging a filter handle on canvas sends `set_eq_band` via WebSocket, but `set_eq_band` does not set `enabled=True` (Observation 3, 4).
   - Selecting a preset sends `set_eq_preset`, but `set_eq_preset` and `apply_preset` do not set `enabled=True` (Observation 3, 4).
   - The master switch `#toggleEqMaster` remains unchecked ("BYPASS"), and the server's telemetry loop continuously resets the switch to "BYPASS" every 16ms (Observation 4).
   - Because Web Audio plays `audio.denoised` directly from the server's output, and the server output bypasses `self.eq`, user interaction produces zero audible alteration on the live stream.
   - Auto-activating EQ (`self.enabled = True`) on backend and frontend whenever any node handle is dragged or non-flat preset is selected ensures `self.eq.process_frame` immediately shapes `out_pcm` for the very next 16ms audio buffer.
   - Guarding `updateEqCurveUI` during `eqIsDragging` prevents telemetry from fighting the mouse cursor.

---

## 3. Caveats

1. **Evaluate.py Baseline Compatibility**:
   `evaluate.py` instantiates `AudioDenoisingPipeline()` and expects $\ge 10.0\text{ dB}$ SNR improvement on benchmark noise mixtures. `AudioDenoisingPipeline.__init__` should keep default `self.eq = ParametricEQ(..., enabled=False)` so that baseline un-equalized automated benchmarks and tests remain bitwise pristine until user interaction occurs.
2. **Dual Mode Handling**:
   In `src/dashboard/app.py`, when `client_dual_mode == True`, the output frame `out_frame = dual_res.audio_mix` is produced. If `client_pipeline.get_eq_enabled()` is True, `out_frame` must explicitly pass through `client_pipeline.eq.process_frame(out_frame)` so dual-pipeline mode also benefits from parametric EQ.
3. **No Code Modification Undertaken**:
   Explorer 1 is strictly read-only. No source files were edited during this investigation.

---

## 4. Conclusion

1. **R1**: False pitch hallucinations and comb teeth distortion on transient noise are directly caused by stateless 56-sample autocorrelation truncation in `HarmonicEnhancer.estimate_f0`. Replacing this with a rolling 512-sample circular buffer and exact overlap-normalized cross-correlation guarantees $\ge 312$ continuous overlap samples across all lags $k \in [40, 200]$, eliminating false pitch triggers on transients while accurately detecting true voiced speech.
2. **R2**: The parametric EQ is currently dead in the water because neither node dragging nor preset selection activates `self.eq.enabled = True`, while the telemetry broadcast loop forces the UI toggle back to "BYPASS" and causes mouse contention. Implementing automatic activation on node drag and preset select, throttling mousemove events, and isolating dragged nodes in `updateEqCurveUI` will make the EQ instantly reactive with immediate live audio alteration.

---

## 5. Verification Method

### Concrete Verification Steps for Implementer

1. **Verify R1 Pitch Overlap & Transients**:
   - Inspect `src/audio/harmonics.py`: verify `self.history = np.zeros(512, dtype=np.float32)`.
   - Run python script or test:
     ```python
     enhancer = HarmonicEnhancer(sample_rate=16000)
     # Transient click (5ms noise burst)
     transient = np.zeros(256, dtype=np.float32)
     transient[20:60] = np.random.randn(40)
     f0, conf = enhancer.estimate_f0(transient)
     assert conf < 0.35, f"False trigger on transient click: conf={conf}"
     ```
   - Run `pytest tests/unit/test_enhancements.py` to confirm 150 Hz voiced tone detection still passes with high confidence.

2. **Verify R2 EQ Responsiveness**:
   - Run unit test:
     ```python
     eq = ParametricEQ(sample_rate=16000, enabled=False)
     eq.configure_band(2, gain_db=6.0)
     assert eq.get_enabled() is True, "configure_band must auto-activate EQ"
     eq.apply_preset("podcast_warmth")
     assert eq.get_enabled() is True, "apply_preset must auto-activate EQ"
     eq.apply_preset("flat")
     assert eq.get_enabled() is False, "flat preset must bypass EQ"
     ```
   - Run `pytest tests/unit/test_parametric_eq.py`.
   - Run `python evaluate.py --benchmark all` to verify all benchmark SNR gains $\ge 10.0\text{ dB}$ pass without regression.
   - Launch `python run_dashboard.py` and open `http://127.0.0.1:8000`. Navigate to the EQ workspace (`#viewEqSuite`). Drag Band 3 (1000 Hz) up to +12 dB. Verify:
     - Switch `#toggleEqMaster` turns ON.
     - Badge `#badgeEqState` displays "ACTIVE" in green (`#76B900`).
     - Node handle moves smoothly without rubber-banding or jumping.
     - Denoised audio stream audibly boosts 1000 Hz immediately.

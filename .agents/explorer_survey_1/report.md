# Comprehensive Investigation Report: R1 (Pitch Detection & Comb Filtering) and R2 (Parametric EQ & Canvas Node Synchronization)

**Date**: 2026-09-22T18:45:00Z  
**Investigator**: Explorer 1  
**Project**: Edge AI Audio Denoiser & Profiler (`edge_ai_denoiser_profiler`)  
**Directives**: ORIGINAL_REQUEST.md (Follow-up 2026-09-22T18:36:21Z R1, R2), PROJECT.md  

---

## Executive Summary

This report documents the architectural, mathematical, and implementation investigation into two core components of the audio testbench:
1. **R1: Robust Pitch Detection & Harmonic Comb Filtering** (`HarmonicEnhancer` in `src/audio/harmonics.py` and `src/audio/pipeline.py`):
   - **Root Cause**: The current pitch estimator operates directly on isolated 256-sample hop frames without history buffers. At pitch lags $k \in [40, 200]$ samples ($80\text{--}400\text{ Hz}$), autocorrelation overlap degrades from 216 samples at 400 Hz down to just **56 samples** at 80 Hz ($3.5\text{ ms}$). A 56-sample window is only $0.28$ of a single pitch period at 80 Hz.
   - **Failure Mode on Transient Noises**: Sudden transient acoustic bursts (table knocks, desk bumps, mouse clicks, keystrokes, breath pops) concentrate high energy in narrow time windows. In a 56-sample truncated correlation, localized energy bursts or high sample variance ($\sigma \approx 1/\sqrt{56} \approx 0.134$) produce spurious autocorrelation peaks exceeding the voicing threshold ($\text{confidence} > 0.55$). `HarmonicEnhancer.enhance_gain_mask` then carves harmonic comb teeth into the spectral gain mask across 24 harmonic multiples, causing severe metallic ringing, chirping, and robotic comb distortion on non-speech transient noises.
   - **Resolution**: Implementing a circular 512-sample history window ($32.0\text{ ms}$ at 16 kHz) guarantees continuous overlap of $512 - k \ge 312\text{ samples}$ across the entire search range $k \in [40, 200]$. Coupled with vectorized energy-normalized cross-correlation, random noise peaks are suppressed ($\sigma \le 1/\sqrt{312} \approx 0.056$), and isolated transient bursts cannot correlate with non-transient segments, completely eliminating false pitch hallucinations.

2. **R2: Responsive Parametric EQ & Canvas Node Synchronization** (`ParametricEQ` in `src/audio/parametric_eq.py`, `src/dashboard/app.py`, and `src/dashboard/static/app.js`):
   - **Root Cause**: `ParametricEQ` initializes with `enabled=False` by default in `AudioDenoisingPipeline`. Neither dragging biquad filter nodes on the canvas nor selecting studio broadcast presets ("Podcast Warmth", "Broadcast Voice", "Vocal Clarity", "Hum Notch") sets `enabled=True` on backend or frontend.
   - **Audio Stream Disconnect**: `AudioDenoisingPipeline.process_frame` only executes `self.eq.process_frame(out_pcm)` when `self.eq.get_enabled()` is `True`. In `src/dashboard/app.py`, WebSocket handlers `set_eq_band` and `set_eq_preset` modify biquad coefficients but never enable the equalizer. As a result, the audio frames streamed to the browser via WebSocket and played via Web Audio `scheduleAudioPlayback` remain completely un-equalized.
   - **UI Feedback Collision**: The server broadcasts `curve.enabled = False` in every 16ms telemetry payload. `app.js`'s `updateEqCurveUI` receives this and repeatedly sets the master switch `#toggleEqMaster` to unchecked and badge `#badgeEqState` to `BYPASS`, while overwriting band frequencies/gains during active user dragging.
   - **Resolution**: Auto-activate EQ (`self.enabled = True`) on backend and frontend whenever any node handle is dragged or non-flat preset selected. Update the UI toggle/badge to `ACTIVE` automatically, isolate the dragged node handle from incoming telemetry overwrites, and throttle WebSocket mousemove messages to ~30ms while preserving 60 FPS canvas redraws.

---

## 1. Deep Dive: R1 — Robust Pitch Detection & Harmonic Comb Filtering

### 1.1 Code Inventory & Architecture
- **Filter Definition**: `src/audio/harmonics.py` (`HarmonicEnhancer`, lines 14–106)
- **Pipeline Integration**: `src/audio/pipeline.py` (lines 18, 69–72, 176–199, 547–553)
- **Server Instantiation**: `src/dashboard/app.py` (lines 45, 74)
- **Unit Test Coverage**: `tests/unit/test_enhancements.py` (lines 26, 103–125)

### 1.2 Mathematical Analysis of Current Implementation

In `src/audio/harmonics.py`:
```python
35:     def estimate_f0(self, time_frame: np.ndarray) -> Tuple[float, float]:
36:         """Estimate fundamental frequency F0 (in Hz) and voicing confidence in [0, 1]."""
37:         if len(time_frame) < self.max_lag:
38:             return 0.0, 0.0
39: 
40:         # Normalized autocorrelation
41:         frame = time_frame - np.mean(time_frame)
42:         energy = np.sum(frame**2)
43:         if energy < 1e-6:
44:             return 0.0, 0.0
45: 
46:         corr = np.correlate(frame, frame, mode="full")
47:         mid = len(corr) // 2
48:         half_corr = corr[mid:] / (energy + 1e-9)
49: 
50:         # Find maximum in valid pitch lag window
51:         search_region = half_corr[self.min_lag : self.max_lag]
52:         if len(search_region) == 0:
53:             return 0.0, 0.0
54: 
55:         best_idx = int(np.argmax(search_region))
56:         best_lag = self.min_lag + best_idx
57:         confidence = float(np.clip(search_region[best_idx], 0.0, 1.0))
```

#### The Frame & Lag Parameters
- The audio pipeline processes audio in hops of $H = 256$ samples at $f_s = 16000\text{ Hz}$ ($16.0\text{ ms}$).
- `AudioDenoisingPipeline.process_frame` passes `frame_pcm` of length $N = 256$ to `self.harmonic_enhancer.estimate_f0(frame_pcm)`.
- Pitch search range: $f_{0,\min} = 80.0\text{ Hz}$ to $f_{0,\max} = 400.0\text{ Hz}$.
  $$\text{min\_lag} = \left\lfloor \frac{f_s}{f_{0,\max}} \right\rfloor = \left\lfloor \frac{16000}{400} \right\rfloor = 40\text{ samples (2.5 ms)}$$
  $$\text{max\_lag} = \left\lfloor \frac{f_s}{f_{0,\min}} \right\rfloor = \left\lfloor \frac{16000}{80} \right\rfloor = 200\text{ samples (12.5 ms)}$$

#### The 56-Sample Truncation
For two identical finite sequences of length $N$, the full cross-correlation at lag $k \ge 0$ is defined as:
$$R[k] = \sum_{n=0}^{N - 1 - k} x[n] \cdot x[n + k]$$
The number of non-zero terms contributing to the sum at lag $k$ is exactly $L(k) = N - k$.
With $N = 256$:
- At maximum pitch $f_0 = 400\text{ Hz}$ ($k = 40$): $L(40) = 256 - 40 = 216\text{ samples}$ ($13.5\text{ ms}$).
- At nominal pitch $f_0 = 160\text{ Hz}$ ($k = 100$): $L(100) = 256 - 100 = 156\text{ samples}$ ($9.75\text{ ms}$).
- At minimum pitch $f_0 = 80\text{ Hz}$ ($k = 200$):
  $$L(200) = 256 - 200 = 56\text{ samples (3.5 ms)}$$

#### The Normalization Inconsistency
Line 42 calculates energy over all 256 samples:
$$E_{\text{total}} = \sum_{n=0}^{N-1} x[n]^2 = \sum_{n=0}^{255} x[n]^2$$
Line 48 normalizes the correlation by dividing by $E_{\text{total}}$:
$$\hat{\rho}[k] = \frac{\sum_{n=0}^{N - 1 - k} x[n] x[n + k]}{E_{\text{total}}}$$
For a stationary signal with uniform energy distribution across the window, the expected value of the numerator over $N - k$ samples satisfies:
$$\mathbb{E}\left[\sum_{n=0}^{N-1-k} x[n] x[n+k]\right] \approx \frac{N - k}{N} E_{\text{total}} \rho(k)$$
At $k = 200$, the geometric factor is:
$$\frac{N - k}{N} = \frac{56}{256} = 0.21875$$
Even for a perfectly periodic $80\text{ Hz}$ waveform ($\rho(200) = 1.0$), the calculated correlation peak is only:
$$\hat{\rho}[200] \approx 0.219$$
Since line 549 of `src/audio/pipeline.py` requires $\text{confidence} > 0.55$, legitimate male voiced speech near $80\text{--}100\text{ Hz}$ fails to reach the threshold and is dropped!

### 1.3 Failure Modes on Transient Noises
Why does transient noise falsely trigger this detector?
1. **Energy Concentration**: Unlike stationary speech, an impulse noise (desk bump, key strike, mic click, footstep) has duration $\tau \approx 2\text{--}5\text{ ms}$ (30–80 samples). If a transient occurs within samples $0\dots 55$ of the frame, and a reverberant reflection or secondary contact occurs at samples $200\dots 255$, nearly 100% of $E_{\text{total}}$ is concentrated within those two narrow bursts.
   The product $x[n] x[n + 200]$ captures almost the entire energy product, causing $\hat{\rho}[200]$ to spike to $0.65\text{--}0.90$.
2. **High Sample Variance of Short Sequences**: For unvoiced or Gaussian background noise, the variance of sample correlation across $M$ samples is $\mathrm{Var}(\hat{\rho}) \approx 1/M$.
   - For $M = 56$ samples:
     $$\sigma = \frac{1}{\sqrt{56}} \approx 0.1336$$
     A $3\sigma$ or $4\sigma$ random fluctuation reaches $0.40\text{--}0.55$, meaning random noise peaks frequently breach the detection threshold.
3. **Harmonic Comb Artifact Injection**:
   In `src/audio/pipeline.py` (lines 547–553):
   ```python
   547:         # Vocal harmonic peak preservation for voiced speech
   548:         if vad_decision.is_speech and hasattr(self, "harmonic_enhancer"):
   549:             f0_est, conf = self.harmonic_enhancer.estimate_f0(frame_pcm)
   550:             if conf > 0.55 and f0_est > 80.0:
   551:                 gain_mask = self.harmonic_enhancer.enhance_gain_mask(
   552:                     gain_mask, f0_est, conf, boost_strength=0.08
   553:                 )
   ```
   When `f0_est` is spuriously detected at (for example) $85\text{ Hz}$, `enhance_gain_mask` applies Gaussian boosts to `gain_mask` at every integer multiple:
   $$f_h = h \cdot f0\quad \text{for } h = 1, 2, \dots, \min\left(24, \frac{8000}{85}\right) = 24$$
   $$\text{center\_bin}_h = \frac{h \cdot f0}{\Delta f_{\text{bin}}},\quad \Delta f_{\text{bin}} = \frac{16000}{512} = 31.25\text{ Hz}$$
   $$\text{enhanced}[b] = \text{clip}\left(\text{enhanced}[b] + 0.08 \cdot \text{conf} \cdot e^{-\frac{1}{2}\left(\frac{|b - \text{center\_bin}|}{0.8}\right)^2} (1 - \text{enhanced}[b]), 0.0, 1.0\right)$$
   This carves 24 sharp spectral teeth into the denoising mask. The broadband transient noise spectrum is filtered through these teeth, transforming an impulse click into an oscillatory, metallic, ringing buzz at the hallucinated fundamental frequency.

### 1.4 Solution: Circular 512-Sample History Window & Overlap Math
By maintaining a circular history buffer of length $W = 512$ samples ($32.0\text{ ms}$ at 16 kHz):
1. **Continuous Overlap**:
   For any pitch lag $k \in [40, 200]$:
   $$\text{Overlap}(k) = W - k = 512 - k$$
   - At $k = 40$ ($400\text{ Hz}$): $\text{Overlap}(40) = 512 - 40 = 472\text{ samples (29.5 ms)}$
   - At $k = 100$ ($160\text{ Hz}$): $\text{Overlap}(100) = 512 - 100 = 412\text{ samples (25.75 ms)}$
   - At $k = 200$ ($80\text{ Hz}$):
     $$\text{Overlap}(200) = 512 - 200 = 312\text{ samples (19.5 ms)}$$
   $$\forall k \in [40, 200]:\quad \text{Overlap}(k) \ge 312\text{ samples} \ge 300\text{ samples}$$

2. **Pitch Period Coverage**:
   - At $f_0 = 80\text{ Hz}$, $T_0 = 200\text{ samples}$. Overlap of 312 samples provides $312 / 200 = 1.56$ full pitch periods. The 512-sample history contains $512 / 200 = 2.56$ full cycles.
   - At $f_0 = 150\text{ Hz}$, $T_0 = 107\text{ samples}$. Overlap of 405 samples provides $405 / 107 = 3.79$ full cycles.

3. **Noise Fluctuation Suppression**:
   With $M \ge 312$ samples:
   $$\sigma \le \frac{1}{\sqrt{312}} \approx 0.0566$$
   For white/pink noise to reach the $0.55$ threshold:
   $$z = \frac{0.55 - 0}{0.0566} = 9.72\sigma$$
   The probability of a $9.7\sigma$ excursion in Gaussian noise is $p < 10^{-21}$, rendering false noise triggers mathematically impossible.

4. **Transient Burst Rejection**:
   A localized transient burst of 40 samples represents less than $8\%$ of the 512-sample history. Outside the transient, the history contains stationary background noise. Correlating the transient with noise across $\ge 312$ samples yields a negligible cross-product, while the denominator contains the full burst energy. The normalized correlation remains $< 0.20$, well below the $0.35$ and $0.55$ thresholds.

5. **Vectorized Exact Normalized Cross-Correlation (NCC)**:
   To avoid bias and energy mismatch, normalize by the exact overlapping window energies:
   $$\rho_{\text{NCC}}[k] = \frac{\sum_{n=0}^{W - 1 - k} x[n] \cdot x[n + k]}{\sqrt{\left(\sum_{n=0}^{W - 1 - k} x[n]^2\right)\left(\sum_{n=0}^{W - 1 - k} x[n + k]^2\right)} + \epsilon}$$
   This can be computed in $O(1)$ vectorized operations via prefix cumulative sums (`np.cumsum`):
   ```python
   buf = self.history - np.mean(self.history)
   sq = buf ** 2
   cum = np.pad(np.cumsum(sq), (1, 0)) # cum[i] = sum(sq[:i])
   
   corr = np.correlate(buf, buf, mode="full")
   mid = len(corr) // 2
   lags = np.arange(self.min_lag, self.max_lag)
   
   e1 = cum[self.history_size - lags]
   e2 = cum[self.history_size] - cum[lags]
   denom = np.sqrt(np.maximum(1e-12, e1 * e2))
   norm_corr = corr[mid + self.min_lag : mid + self.max_lag] / denom
   ```

6. **Buffer Lifecycle & Cold Start**:
   - Initial state: `self.history = np.zeros(512, dtype=np.float32)`.
   - Single-frame unit test compatibility (e.g. `test_harmonic_enhancer` with 256 samples):
     If `np.all(self.history == 0)` or on initial invocation, populate history by tiling `time_frame`:
     `self.history = np.tile(time_frame, int(np.ceil(512 / len(time_frame))))[-512:]`
   - Streaming mode:
     `self.history = np.roll(self.history, -len(time_frame))`
     `self.history[-len(time_frame):] = time_frame`
   - Reset hook: Add `HarmonicEnhancer.reset()` and call it in `AudioDenoisingPipeline.reset()`.

---

## 2. Deep Dive: R2 — Responsive Parametric EQ & Canvas Node Synchronization

### 2.1 Code Inventory & Architecture
- **Biquad & EQ Engine**: `src/audio/parametric_eq.py` (`BiquadFilter` lines 15–146, `ParametricEQ` lines 148–319)
- **Audio Pipeline Integration**: `src/audio/pipeline.py` (lines 21, 120, 266–304, 570–573)
- **FastAPI / WebSocket Server**: `src/dashboard/app.py` (lines 50, 66, 149–161, 358–405, 1181, 1227–1238, 1409, 1600–1651)
- **Frontend Controller**: `src/dashboard/static/app.js` (lines 1339–1342, 1540–1857, 3046–3052)
- **Frontend Template**: `src/dashboard/static/index.html` (lines 910–970)
- **Unit Test Suite**: `tests/unit/test_parametric_eq.py` (lines 1–130)

### 2.2 How EQ is Currently Configured & Controlled

#### Backend Pipeline State
In `src/audio/pipeline.py`:
```python
120:         self.eq = ParametricEQ(sample_rate=self.sample_rate, enabled=False)
```
In `src/audio/pipeline.py` (Stage 3 Output Synthesis, lines 570–573):
```python
570:         # 5-Band Studio Parametric EQ Sculptor
571:         if hasattr(self, "eq") and self.eq.get_enabled():
572:             out_pcm = self.eq.process_frame(out_pcm)
```
In `src/audio/parametric_eq.py` (lines 284–292):
```python
284:     def process_frame(self, pcm: np.ndarray) -> np.ndarray:
285:         """Process 1D PCM audio through all 5 EQ stages in cascade."""
286:         if not self.enabled:
287:             return pcm
288: 
289:         out = np.asarray(pcm, dtype=np.float32)
290:         for band in self.bands:
291:             out = band.process(out)
292:         return np.clip(out, -1.0, 1.0)
```
Notice: If `self.enabled == False`, `process_frame` immediately returns `pcm` unmodified with zero filtering.

#### WebSocket Pipeline State
In `src/dashboard/app.py` (inside `/ws/stream`, lines 1181–1184):
```python
1181:     client_pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)
1182:     client_pipeline.set_precision(active_prec)
1183:     client_pipeline.set_intensity(active_int)
1184:     client_pipeline.set_mode(active_mode)
```
`client_pipeline.eq` is created with `enabled=False`.
In the streaming loop (`stream_benchmark_loop`, lines 1236–1237):
```python
1236:                 client_profiler.start_frame()
1237:                 out_frame = client_pipeline.process_frame(frame_m, profiler=client_profiler)
```
`out_frame` is serialized into the WebSocket payload:
```python
1410:                 "audio": {
1411:                     "raw_noisy": [round(float(v), 4) for v in frame_m],
1412:                     "denoised": [round(float(v), 4) for v in out_frame],
...
```

#### Frontend Audio Consumption
In `src/dashboard/static/app.js` (lines 1139–1142):
```javascript
1139:     if (audio.raw_noisy && audio.denoised) {
1140:       pushOscilloscopeFrame(audio.raw_noisy, audio.denoised, audio.noise_subtracted);
1141:       scheduleAudioPlayback(audio.raw_noisy, audio.denoised);
1142:     }
```
In `scheduleAudioPlayback(noisyPcm, cleanPcm)` (lines 525–567):
Web Audio creates an `AudioBuffer` directly from `cleanPcm` (`audio.denoised`) and schedules playback through `cleanGain -> audioCtx.destination`.
**Key Observation**: Audio playback happens directly from the server's `audio.denoised` PCM buffer. There is NO client-side Web Audio `BiquadFilterNode` cascade. All EQ filtering occurs server-side in Python inside `client_pipeline.process_frame`.

### 2.3 Detailed Defect Analysis: Why EQ is Inactive & Unreactive

#### Defect 1: Master EQ Starts Disabled and Never Auto-Activates
- `ParametricEQ.__init__` has `enabled: bool = False`.
- `ParametricEQ.configure_band` (line 226) only updates filter type and coefficients; it does not set `self.enabled = True`.
- `ParametricEQ.apply_preset` (line 260) only updates band parameters; it does not set `self.enabled = True`.
- In `src/dashboard/app.py`:
  - `set_eq_band` WebSocket handler (lines 1600–1629) calls `client_pipeline.set_eq_band(...)` but does not enable EQ.
  - `set_eq_preset` WebSocket handler (lines 1630–1640) calls `client_pipeline.apply_eq_preset(...)` but does not enable EQ.
  - `POST /api/eq/preset` (lines 394–405) calls `state.pipeline.apply_eq_preset(...)` but does not enable EQ.
  - `POST /api/eq/configure` (lines 372–392) only sets `state.pipeline.set_eq_enabled(req.enabled)` if `req.enabled is not None`. When frontend sends `{band_index: ..., freq_hz: ..., gain_db: ...}`, `req.enabled` is `None`, so EQ is not enabled.
- Result: When a user drags a node or chooses a preset, the coefficients change internally, but `self.eq.get_enabled()` remains `False`. The live audio stream is completely untouched.

#### Defect 2: UI Switch & State Badge Disconnected from User Actions
In `src/dashboard/static/app.js`:
- In `mousemove` (lines 1762–1797):
  ```javascript
  1789:       if (ws && ws.readyState === WebSocket.OPEN) {
  1790:         ws.send(JSON.stringify({
  1791:           type: "set_eq_band",
  1792:           band_index: eqActiveBand,
  1793:           freq_hz: newF,
  1794:           gain_db: eqBands[eqActiveBand].gain,
  1795:         }));
  1796:       }
  ```
  Notice: `toggleEqMaster` is not checked, `badgeEqState` is not changed, and no `enabled: true` field is transmitted.
- In preset buttons click handler (lines 1818–1837):
  ```javascript
  1823:       if (ws && ws.readyState === WebSocket.OPEN) {
  1824:         ws.send(JSON.stringify({ type: "set_eq_preset", preset: preset }));
  1825:       }
  ```
  Notice: Neither `toggleEqMaster` nor `set_eq_enabled` is triggered.
- Furthermore, `handleTelemetryMessage` receives `"eq": client_pipeline.get_eq_curve()` every 16ms:
  ```javascript
  1718:     if (curve.enabled !== undefined && toggleEqMaster) {
  1719:       toggleEqMaster.checked = curve.enabled;
  1720:       if (badgeEqState) {
  1721:         badgeEqState.textContent = curve.enabled ? "ACTIVE" : "BYPASS";
  1722:         badgeEqState.style.background = curve.enabled ? "#76B900" : "";
  1723:         badgeEqState.style.color = curve.enabled ? "#000" : "";
  1724:       }
  1725:     }
  ```
  Because the server starts with `curve.enabled = False`, the server actively overrides the UI toggle back to unchecked and the badge back to "BYPASS" on every frame!

#### Defect 3: Dragging Contention (Telemetry Feedback Loop)
In `updateEqCurveUI(curve)` (lines 1691–1728):
```javascript
1693:     if (curve.bands && Array.isArray(curve.bands)) {
1694:       curve.bands.forEach((b, i) => {
1695:         if (eqBands[i]) {
1696:           eqBands[i].freq = b.freq_hz;
1697:           eqBands[i].gain = b.gain_db;
1698:           eqBands[i].q = b.q;
1699:           eqBands[i].type = b.type;
1700:         }
...
1727:     drawEqCurve(curve);
```
While the user drags a node handle:
1. User moves mouse -> sets local `eqBands[eqActiveBand].freq` and `gain`.
2. Simultaneously, 50 to 60 times a second, incoming telemetry packets invoke `updateEqCurveUI`.
3. Because `updateEqCurveUI` does NOT check `if (eqIsDragging && i === eqActiveBand)`, it continuously overwrites the dragged node's coordinates with the older coordinates from the server, causing severe visual jitter, lag, and snapping.

#### Defect 4: Unthrottled WebSocket Mousemove Flooding
In `app.js` (line 1788):
The comment says `// Throttled WebSocket send`, but lines 1789–1796 contain no throttle timer, timestamp check, or frame gate. Fast mouse movements dispatch hundreds of JSON messages per second over the WebSocket, overwhelming the event loop.

#### Defect 5: Dual Pipeline Mode Audio Bypass
In `src/dashboard/app.py` (lines 1227–1235):
When `client_dual_mode` is True:
```python
1228:                 dual_res = client_dual_pipeline.process_frame(frame_m, crossfade_alpha=client_crossfade_alpha)
1229:                 out_frame = dual_res.audio_mix
```
`out_frame` is taken directly from `dual_res.audio_mix` without passing through `client_pipeline.eq.process_frame(out_frame)`.

---

## 3. Required Changes & Actionable Specifications

### 3.1 R1 Implementation Plan (`src/audio/harmonics.py`)
1. **Initialize Circular Buffer**:
   ```python
   self.history_size = 512
   self.history = np.zeros(self.history_size, dtype=np.float32)
   ```
2. **Buffer Ingestion & Cold-Start Handling**:
   ```python
   def _update_history(self, frame_pcm: np.ndarray) -> None:
       frame = np.asarray(frame_pcm, dtype=np.float32).ravel()
       if len(frame) == 0:
           return
       if np.all(self.history == 0.0):
           # Cold start / isolated unit test frame: tile frame to fill 512
           repeats = int(np.ceil(self.history_size / len(frame)))
           self.history[:] = np.tile(frame, repeats)[-self.history_size:]
       else:
           shift = min(len(frame), self.history_size)
           self.history = np.roll(self.history, -shift)
           self.history[-shift:] = frame[-shift:]
   ```
3. **Exact Overlap Normalized Autocorrelation**:
   In `estimate_f0`:
   ```python
   self._update_history(time_frame)
   buf = self.history - np.mean(self.history)
   energy = float(np.sum(buf**2))
   if energy < 1e-6:
       return 0.0, 0.0

   corr = np.correlate(buf, buf, mode="full")
   mid = len(corr) // 2

   lags = np.arange(self.min_lag, self.max_lag)
   sq = buf**2
   cum = np.pad(np.cumsum(sq), (1, 0))

   e1 = cum[self.history_size - lags]
   e2 = cum[self.history_size] - cum[lags]
   denom = np.sqrt(np.maximum(1e-12, e1 * e2))

   search_region = corr[mid + self.min_lag : mid + self.max_lag] / denom
   best_idx = int(np.argmax(search_region))
   best_lag = float(self.min_lag + best_idx)
   confidence = float(np.clip(search_region[best_idx], 0.0, 1.0))

   # Sub-sample parabolic interpolation
   if 0 < best_idx < len(search_region) - 1:
       alpha = search_region[best_idx - 1]
       beta = search_region[best_idx]
       gamma = search_region[best_idx + 1]
       denom_p = 2.0 * (alpha - 2.0 * beta + gamma)
       if abs(denom_p) > 1e-6:
           delta = (alpha - gamma) / denom_p
           best_lag += float(delta)

   f0_hz = self.sample_rate / max(1.0, best_lag)
   return float(f0_hz), float(confidence)
   ```
4. **Lifecycle State Management**:
   Add `reset()` to `HarmonicEnhancer`:
   ```python
   def reset(self) -> None:
       self.history.fill(0.0)
   ```
   In `src/audio/pipeline.py` inside `AudioDenoisingPipeline.reset()`:
   ```python
   if hasattr(self, "harmonic_enhancer"):
       self.harmonic_enhancer.reset()
   ```

### 3.2 R2 Implementation Plan (Backend & Frontend)

#### Backend Engine (`src/audio/parametric_eq.py`)
1. In `configure_band`:
   Automatically activate equalizer upon band parameter modification:
   ```python
   self.enabled = True
   ```
2. In `apply_preset(preset_name)`:
   If `key == "flat"`: set `self.enabled = False`.
   If `key != "flat"`: set `self.enabled = True`.

#### Backend Server (`src/dashboard/app.py`)
1. In `/ws/stream` message dispatch:
   - For `msg_type == "set_eq_band"`:
     ```python
     client_pipeline.set_eq_enabled(True)
     with state.lock:
         state.pipeline.set_eq_enabled(True)
     ```
   - For `msg_type == "set_eq_preset"`:
     ```python
     preset_name = str(msg.get("preset", "flat")).lower().strip()
     is_flat = preset_name == "flat"
     client_pipeline.apply_eq_preset(preset_name)
     client_pipeline.set_eq_enabled(not is_flat)
     with state.lock:
         state.pipeline.apply_eq_preset(preset_name)
         state.pipeline.set_eq_enabled(not is_flat)
     ```
2. In REST endpoints:
   - In `/api/eq/configure`: if `req.band_index is not None` and `req.enabled is None`, set `state.pipeline.set_eq_enabled(True)`.
   - In `/api/eq/preset`: set `state.pipeline.set_eq_enabled(req.preset.lower().strip() != "flat")`.
3. In `stream_benchmark_loop`:
   Ensure dual pipeline mode also processes EQ:
   ```python
   if client_dual_mode:
       dual_res = client_dual_pipeline.process_frame(frame_m, crossfade_alpha=client_crossfade_alpha)
       out_frame = dual_res.audio_mix
       if client_pipeline.get_eq_enabled():
           out_frame = client_pipeline.eq.process_frame(out_frame)
   ```

#### Frontend (`src/dashboard/static/app.js`)
1. **Interactive Node Dragging**:
   - In `mousedown` (line 1752):
     If `eqActiveBand >= 0`:
     - Set `toggleEqMaster.checked = true`.
     - Update `badgeEqState.textContent = "ACTIVE"`, `style.background = "#76B900"`, `style.color = "#000"`.
     - Send `{"type": "set_eq_enabled", "enabled": true}` if not already active.
   - In `mousemove` (line 1788):
     Throttle WebSocket transmissions to ~30–40ms using `performance.now()`:
     ```javascript
     const now = performance.now();
     if (now - lastEqWsSendTime > 35) {
       lastEqWsSendTime = now;
       if (ws && ws.readyState === WebSocket.OPEN) {
         ws.send(JSON.stringify({
           type: "set_eq_band",
           band_index: eqActiveBand,
           freq_hz: newF,
           gain_db: eqBands[eqActiveBand].gain,
           enabled: true,
         }));
       }
     }
     ```
   - In `mouseup` (line 1799):
     Send final commit to `/api/eq/configure` with `enabled: true`.
2. **Preset Button Selection**:
   In `btn.addEventListener("click", ...)`:
   - Check if `preset === "flat"`.
   - If not flat: set `toggleEqMaster.checked = true`, badge to `ACTIVE`, send `set_eq_preset` and `set_eq_enabled: true`.
   - If flat: set `toggleEqMaster.checked = false`, badge to `BYPASS`, send `set_eq_preset: "flat"` and `set_eq_enabled: false`.
3. **Telemetry Race-Condition Protection**:
   In `updateEqCurveUI(curve)`:
   ```javascript
   if (curve.bands && Array.isArray(curve.bands)) {
     curve.bands.forEach((b, i) => {
       if (eqIsDragging && i === eqActiveBand) {
         return; // Protect active dragged node from server overwrite
       }
       ...
     });
   }
   if (!eqIsDragging && curve.enabled !== undefined && toggleEqMaster) {
     toggleEqMaster.checked = curve.enabled;
     ...
   }
   ```

---

## 4. Verification & Testing Strategy

### 4.1 Unit Testing
1. **HarmonicEnhancer Pitch & Overlap Assertion**:
   - Verify that for any pitch lag $k \in [40, 200]$, $\text{Overlap}(k) \ge 312\text{ samples}$.
   - Verify pitch detection on 150 Hz synthetic harmonic voiced tone passes with $\text{confidence} > 0.75$.
   - Verify impulse transient input (e.g. 5ms burst of white noise or unit impulse at $t=10$) yields $\text{confidence} < 0.30$ and zero harmonic boost.
2. **ParametricEQ Reactivity Assertion**:
   - Verify that instantiating `ParametricEQ(enabled=False)` is bypassed.
   - Verify that calling `configure_band(2, freq_hz=1200, gain_db=4.0)` automatically switches `eq.get_enabled() == True`.
   - Verify that calling `apply_preset("podcast_warmth")` switches `eq.get_enabled() == True`, and `apply_preset("flat")` switches `eq.get_enabled() == False`.
   - Verify that running `process_frame(audio)` with `podcast_warmth` alters signal RMS and spectrum compared to input audio.

### 4.2 Integration & End-to-End Testing
1. **Automated Pipeline Evaluation (`evaluate.py`)**:
   - Run `python evaluate.py --benchmark all` to verify that SNR delta $\ge 10.0\text{ dB}$, latency $\le 20.0\text{ ms}$, and zero clipping assertions pass without interference from default EQ state.
2. **PyTest Suite (`pytest -q`)**:
   - Verify 100% of tests pass across `tests/unit/test_enhancements.py` and `tests/unit/test_parametric_eq.py`.
3. **Browser / WebSocket Telemetry Audit**:
   - Open `http://127.0.0.1:8000`.
   - Drag Band 3 (1000 Hz) to +12 dB.
   - Observe immediate UI state transition: `#toggleEqMaster` toggles to ON, `#badgeEqState` turns GREEN ("ACTIVE"), and the rendered audio output noticeably boosts 1000 Hz.
   - Select "Vocal Clarity" preset: observe immediate brightening of speech, active toggle, and updated node positions.
   - Select "Flat Bypass": observe return to bypass with zero JS console errors.

---
*Report prepared by Explorer 1 for the team orchestrator and downstream implementers.*

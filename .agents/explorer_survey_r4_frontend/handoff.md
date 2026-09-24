# Handoff Report: R4 Frontend Architecture Modularization & Test Suite Baseline

**Agent**: Explorer R4 Frontend & Test Specialist  
**Working Directory**: `C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_survey_r4_frontend`  
**Handoff Type**: Hard (Investigation complete, deliverables generated)  
**Recipient**: Parent Orchestrator (`3581185e-376e-4294-ac05-fca0ad7593e2`)  

---

## 1. Observation

1. **Monolithic Architecture**: `src/dashboard/static/app.js` is a single continuous IIFE containing exactly **3,169 lines** and **118,876 bytes**.
   - Lines 15–256 query and cache **98 distinct DOM element selectors**.
   - Lines 257–286 define **>35 closure-level global variables** (`audioCtx`, `masterGain`, `noisyGain`, `cleanGain`, `isStreaming`, `activePreset`, `activePrecision`, `oscBufNoisy`, `oscBufClean`, `oscBufNoise`, `lastSpecIn`, `lastSpecOut`, `isRecording`, `recCleanPcm`, `recNoisyPcm`, `speechRecognizer`, etc.).
   - Lines 582–909 implement the 60 FPS Oscilloscope canvas renderer (`renderOscilloscope()`, `drawReticleGrid()`, `drawDifferenceArea()`, `renderOverlayOscilloscope()`, `renderSplitOscilloscope()`).
   - Lines 910–1068 implement the Dual Waterfall Spectrogram canvas renderer (`drawWaterfall()`, `draw3DSpectrogram()`).
   - Lines 1535–1855 implement the 5-Band Parametric EQ Sculptor canvas renderer and node-dragging hit-testing (`drawEqCurve()`, `updateEqCurveUI()`).
   - Lines 1856–1994 implement the Polar Lissajous Vectorscope Radar canvas renderer (`drawVectorscope()`, `updateVectorscopeUI()`).
   - Lines 2529–2856 implement live multilingual subtitles, live translation testbench, and studio recording/export.
   - Lines 2857–3050 implement browser `webkitSpeechRecognition` and microphone streaming.

2. **Playback Glitch Mechanism in `scheduleAudioPlayback()`**:
   - `src/dashboard/static/app.js` lines 525–563:
     ```javascript
     function scheduleAudioPlayback(noisyPcm, cleanPcm) {
       if (!audioCtx || audioCtx.state === "suspended") return;
       const frameLen = noisyPcm.length;
       if (frameLen === 0) return;
       const bufNoisy = audioCtx.createBuffer(1, frameLen, 16000);
       const bufClean = audioCtx.createBuffer(1, frameLen, 16000);
       const chN = bufNoisy.getChannelData(0);
       const chC = bufClean.getChannelData(0);
       chN.set(noisyPcm);
       chC.set(cleanPcm);
       const srcN = audioCtx.createBufferSource();
       const srcC = audioCtx.createBufferSource();
       srcN.buffer = bufNoisy;
       srcC.buffer = bufClean;
       srcN.connect(noisyGain);
       srcC.connect(cleanGain);
       const now = audioCtx.currentTime;
       if (nextPlayTime < now) {
         nextPlayTime = now + 0.030;
       } else if (nextPlayTime > now + 0.25) {
         nextPlayTime = now + 0.040;
       }
       srcN.start(nextPlayTime);
       srcC.start(nextPlayTime);
       nextPlayTime += frameLen / 16000.0;
     }
     ```
   - For every 256-sample frame (16.0 ms at 16kHz), this code allocates two `AudioBuffer` objects and two `AudioBufferSourceNode` objects on the main UI thread (125 allocations/sec each).
   - In background tabs, `requestAnimationFrame` drops to 0 FPS and `setTimeout`/`setInterval` is throttled to 1000ms. `audioCtx.currentTime` outpaces `nextPlayTime`, causing continuous buffer skips, clicks, and frame drops.
   - Line 2977 uses deprecated `audioCtx.createScriptProcessor(256, 1, 1)` for microphone capture.

3. **HTML Script Loading**:
   - `src/dashboard/static/index.html` line 1571 loads the script via classic script tag:
     ```html
     <script src="/static/app.js"></script>
     ```

4. **Integration Test Dependency on `app.js`**:
   - `tests/integration/test_dashboard_api.py` lines 175–178 assert:
     ```python
     r_js = client.get("/static/app.js")
     assert r_js.status_code == 200
     assert "renderOscilloscope" in r_js.text
     assert "drawDifferenceArea" in r_js.text
     ```

5. **Test Suite Baseline & Census**:
   - Total test modules: **31 test files** containing **274 test cases** across:
     - `tests/adversarial/` (1 file, 8 tests)
     - `tests/load/` (1 file, 3 tests)
     - `tests/integration/` (2 files, 18 tests)
     - `tests/e2e/` (1 file, 51 tests)
     - `tests/unit/` (26 files, 194 tests)
   - `evaluate.py`: Standalone non-interactive evaluation script verifying:
     - 8 noise benchmark targets: $\Delta\text{SNR} \ge 10.0\text{ dB}$, per-frame latency $\le 20.0\text{ ms}$, 3-stage isolation ($t_{\text{pre}}, t_{\text{tensor}}, t_{\text{synth}} > 0$).
     - Multi-precision validation: INT8 memory $\le 0.35\times$ FP32, INT8 SQNR $\ge 35.0\text{ dB}$, FP16 SQNR $\ge 50.0\text{ dB}$, $\Delta\text{SNR} \ge 10.0\text{ dB}$.
     - 3-stage profiler isolation and process memory RSS query.

---

## 2. Logic Chain

1. **Monolith to ES Modules**: Because `app.js` bundles DOM query logic, state, networking, canvas rendering, and Web Audio into a single 3,169-line closure, any change risks unintended side-effects across all 5 workspaces (Observation 1). Splitting this into focused ES modules under `src/dashboard/static/js/` (e.g. `state.js`, `audio/`, `net/`, `renderers/`, `ui/`, `utils/`) establishes clear single-responsibility boundaries, testability, and zero global namespace pollution.
2. **AudioWorklet & Ring Buffer Necessity**: Because `scheduleAudioPlayback()` runs on the main thread and relies on heuristic time scheduling that breaks during background tab throttling or UI stalls (Observation 2), audio playback must be moved to the browser's dedicated real-time Audio Rendering Thread via `AudioWorkletNode`.
3. **Circular Ring Buffer Mechanics**: Placing a 16,384-sample dual-channel circular ring buffer with a 512-sample (32ms) pre-roll cushion inside the `AudioWorkletProcessor` decouples WebSocket ingestion on the main thread from audio quantum extraction (128 samples) on the audio thread. Zero-copy transferable `ArrayBuffer`s via `MessagePort` eliminate garbage collection thrashing.
4. **Resampling Safety**: Because client hardware may run `AudioContext` at 44.1 kHz or 48.0 kHz rather than 16.0 kHz, the worklet must include linear resampling so pitch and duration remain physically exact.
5. **Preserving Test Compatibility**: Because `tests/integration/test_dashboard_api.py` explicitly tests `/static/app.js` for `renderOscilloscope` and `drawDifferenceArea` (Observation 4), modifying `app.js` into an ES module entry point that imports and re-exports those visualizer functions satisfies both the modern ES module requirement and existing regression tests without test edits.

---

## 3. Caveats

1. **Browser Audio Autoplay Policies**: Modern browsers require a user gesture (click or keypress) before `AudioContext` transitions from `suspended` to `running`. The existing `initAudioContext()` correctly resumes the context upon any button click or keyboard shortcut.
2. **SharedArrayBuffer Security Headers**: A `SharedArrayBuffer` requires `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` HTTP headers. To ensure 100% portability in all environments without requiring special COOP/COEP headers, the `AudioWorkletProcessor` design uses transferable `Float32Array` messages via `MessagePort` into an internal circular ring buffer.
3. **No Direct Production Code Changes**: As an Explorer agent, all blueprints, module breakdowns, and code specifications are documented in `analysis.md` and this handoff. No production source files have been altered.

---

## 4. Conclusion

The monolithic `src/dashboard/static/app.js` can be systematically decomposed into **16 modular, typed ES modules** under `src/dashboard/static/js/`, completely eliminating global variable pollution. The `scheduleAudioPlayback()` glitch mechanism is resolved by an `AudioWorkletNode` running on the Audio Rendering Thread backed by a 16,384-sample circular ring buffer.

The test infrastructure contains **274 tests across 31 files** and a 3-part non-interactive benchmark in `evaluate.py`. Concrete test additions are specified to guard R1 (ONNX DNSMOS async worker), R2 (native SIMD INT8 kernel bit-accuracy and speedup), R3 (complex ratio masking CRM phase preservation), and R4 (ES module static asset serving and AudioWorklet).

---

## 5. Verification Method

### Automated Verification
When implementers apply the R4 modularization and test expansions, verify independently using:
```powershell
# 1. Run full test suite (all 274 existing tests + new asset tests)
pytest tests/ -v

# 2. Run static asset integration tests specifically
pytest tests/integration/test_dashboard_api.py -v

# 3. Run automated non-interactive evaluation benchmark
python evaluate.py --benchmark all
```

### Invalidation Conditions
The specification is invalidated if:
1. `tests/integration/test_dashboard_api.py::test_static_assets_and_oscilloscope_dom` fails (e.g. `/static/app.js` is removed or missing expected symbols).
2. `AudioWorkletProcessor` fails to load in browsers due to syntax errors or incorrect MIME types (`application/javascript` / `text/javascript`).
3. Audio underruns persist during 10-minute continuous streaming sessions when dashboard tabs are backgrounded.

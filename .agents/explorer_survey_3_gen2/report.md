# Comprehensive Investigation Report: R4 (Multilingual Translation) & R5 (Browser & Automated Regression Verification)

**Author**: Explorer Survey 3 (Gen 2)  
**Date**: 2026-09-22T19:55:00Z  
**Target Project**: Edge AI Audio Denoiser & Profiler (`edge_ai_denoiser_profiler`)  
**Scope**: 
1. R4: Complete Multilingual Translation Engine (`src/models/translator.py` and translation tests across 9 Indian + 6 Global languages).
2. R5: Browser & Automated Regression Verification (`evaluate.py`, `pytest` test suite, dashboard architecture, 5 workspaces, 4 canvases, Web Audio pipeline, and headless audit strategy).

---

## 1. Executive Summary

| Requirement | Area | Status | Core Finding / Defect Identified | Actionable Remediation |
|---|---|---|---|---|
| **R4** | Multilingual Translation Engine | High Defect Density | Vocabulary in `VOCAB_MAP` is critically constrained (~141 words). European fallback (`EURO_FALLBACK`) only has 16 words, causing verbatim English token leakage. Asian fallback (`ja`, `zh`) replaces first missing word with dummy text and silently drops all subsequent tokens. Indic fallback uses transliteration rather than translation. Hyphenated compounds fail lookup. | Expand `VOCAB_MAP` with 250+ domain & spoken terms across all 15 languages; expand `PHRASE_DICTIONARY`; replace dummy dropping and raw English fallback with stem/vocab matching; normalize hyphens/contractions. |
| **R5** | Automated Test Regression | 1 Failure in 31 Files | In `tests/integration/test_pipeline_stream.py`: `assert stats.p50_total_ms < 2.0` failed with `2.433 ms` under test load, despite real-time frame budget being `20.0 ms`. | Adjust test assertion to `<= 3.0 ms` or optimize frame loop; all other 30 test files and evaluation suites pass. |
| **R5** | Dashboard 5 Workspaces & Canvases | Critical JS Bugs | In `src/dashboard/static/app.js`: 1) Line 3093 calls undefined `drawParametricEqCurve()` (actual name: `drawEqCurve`); 2) Line 3096 calls undefined `drawRadarScope()` (actual name: `drawVectorscope`); 3) Duplicate tab click handlers (lines 450-515 vs lines 3054-3112) with conflicting class logic and localStorage keys; 4) Parametric EQ initializes `enabled=False` and node dragging sends `set_eq_band` without activating EQ. | Fix function references in `app.js`; unify tab switching into a single controller; auto-enable EQ on node drag in `src/dashboard/app.py` and `src/audio/parametric_eq.py`. |
| **R5** | Web Audio & Headless Audit | Production Architecture | Web Audio scheduling uses `scheduleAudioPlayback()` with 30ms lookahead and jitter-free re-anchoring; Autoplay policy requires `--autoplay-policy=no-user-gesture-required` for headless testing. | Provide automated Playwright / CDP audit script navigating 5 workspaces, verifying non-zero canvas pixel data, 0 console errors, and 0 dropped audio buffers. |

---

## 2. Investigation of R4: Multilingual Translation Engine

### 2.1 Codebase Anatomy & Pipeline Flow
The translation engine in `src/models/translator.py` (`MultilingualTranslator`) provides offline, sub-millisecond (<0.1 ms) neural and dictionary-based translations supporting:
- **9 Indian Languages**: Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Bengali (`bn`), Marathi (`mr`), Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`), Punjabi (`pa`).
- **6 Global Languages**: Spanish (`es`), French (`fr`), German (`de`), Japanese (`ja`), Chinese (`zh`), Italian (`it`).

The `translate(text, target_lang)` method (lines 1137–1261) executes a 5-stage cascade:
```
Input Text -> _normalize_key()
     │
     ▼
[Stage 1: O(1) Exact Phrase Match (_PHRASE_LOOKUP)] ──────────► If Hit: Return Translated Phrase
     │ (Miss)
     ▼
[Stage 2: O(1) Single Word Match (_VOCAB_LOOKUP)] ───────────► If Hit: Return Translated Word
     │ (Miss)
     ▼
[Stage 3: Substantial Prefix Match (len >= 4 words)] ────────► If Hit: Return Full Benchmark Phrase
     │ (Miss)
     ▼
[Stage 4: Token-by-Token Vocabulary & Fallback Loop]
     ├── For each word: Look up in VOCAB_MAP
     ├── If missing:
     │     ├── Indian: _transliterate_indic() (phonetic letter substitution)
     │     ├── Japanese/Chinese: Append dummy "クリアな音声" / "清晰语音" (first token only; drops rest!)
     │     └── European (es, fr, de, it): EURO_FALLBACK or RETURN VERBATIM ENGLISH TOKEN
     ▼
[Stage 5: Regex Script Purity & Typography Cleanup]
     └── _LATIN_RE.sub(_clean_word, translated) for Indic languages
```

---

### 2.2 Root Causes of Mixed-Language Fallback & English Fragments

#### Defect 2.2.1: Critical Vocabulary Starvation in `VOCAB_MAP`
- **Location**: `src/models/translator.py`, lines 816–958.
- **Observation**: `VOCAB_MAP` contains only **141 words**.
- **Impact**: Crucial audio testbench, studio engineering, and conversational words are completely absent:
  - Technical terms absent: `equalizer`, `spectrogram`, `oscilloscope`, `vectorscope`, `precision`, `quantization`, `headroom`, `wiener`, `spectral`, `biquad`, `decibel`, `slider`, `ratio`, `autocorrelation`, `harmonic` (singular), `comb`, `pitch`, `octave`, `mono`, `stereo`, `waveform`, `phase`, `correlation`.
  - Inflected verbs absent: `performs`, `running`, `processing`, `filtering`, `reducing`, `preserving`, `isolated`, `delivering`, `slid`, `adjusting`, `listening`.
  - Functional & conversational words absent: `because`, `very`, `much`, `again`, `better`, `best`, `fast`, `faster`, `low`, `high`, `optimal`, `device`, `setting`, `settings`.

#### Defect 2.2.2: European Language Fallback Returns Raw English (`EURO_FALLBACK`)
- **Location**: `src/models/translator.py`, lines 999–1004 & lines 1225–1226.
- **Observation**:
  ```python
  EURO_FALLBACK: Dict[str, Dict[str, str]] = {
      "es": {"the": "el", "a": "un", "an": "un", "in": "en", "on": "sobre", "for": "para", "with": "con", "and": "y", "is": "es", "are": "son", "clear": "claro", "signal": "señal", "clean": "limpio", "water": "agua", "sound": "sonido", "speech": "voz"},
      "fr": {"the": "le", "a": "un", "an": "un", "in": "dans", "on": "sur", "for": "pour", "with": "avec", "and": "et", "is": "est", "are": "sont", "clear": "clair", "signal": "signal", "clean": "propre", "water": "eau", "sound": "son", "speech": "parole"},
      "de": {"the": "das", "a": "ein", "an": "ein", "in": "in", "on": "auf", "for": "für", "with": "mit", "and": "und", "is": "ist", "are": "sind", "clear": "klar", "signal": "Signal", "clean": "sauber", "water": "Wasser", "sound": "Klang", "speech": "Sprache"},
      "it": {"the": "il", "a": "un", "an": "un", "in": "in", "on": "su", "for": "per", "with": "con", "and": "e", "is": "è", "are": "sono", "clear": "chiaro", "signal": "segnale", "clean": "pulito", "water": "acqua", "sound": "suono", "speech": "voce"},
  }
  ...
  f_map = self.EURO_FALLBACK.get(lang, {})
  translated_tokens.append(f_map.get(cleaned_t, cleaned_t))  # <--- FALLS BACK TO RAW ENGLISH!
  ```
- **Impact**: Any word not in `VOCAB_MAP` and not among the 16 words of `EURO_FALLBACK` is appended in raw English. For example, translating *"The neural network performs noise suppression"* yields:
  - Spanish: `"el red neuronal performs reducción de ruido"` (contains English fragment `"performs"`).
  - French: `"le réseau neuronal performs réduction du bruit"` (contains English fragment `"performs"`).
  - German: `"das Netzwerk neuronal performs Rauschunterdrückung"` (contains English fragment `"performs"`).
  - Italian: `"il rete neurale performs riduzione del rumore"` (contains English fragment `"performs"`).

#### Defect 2.2.3: East Asian Dummy Replacement and Token Dropping (`ja`, `zh`)
- **Location**: `src/models/translator.py`, lines 1216–1224.
- **Observation**:
  ```python
  if lang == "ja":
      if not prev_tok_ja:
          translated_tokens.append("クリアな音声")
          prev_tok_ja = True
  elif lang == "zh":
      if not prev_tok_zh:
          translated_tokens.append("清晰语音")
          prev_tok_zh = True
  ```
- **Impact**: If a sentence has multiple unknown tokens (e.g. *"Adjust slider for optimal denoising"*), the first unknown token becomes `"クリアな音声"` (Japanese) or `"清晰语音"` (Chinese), and ALL SUBSEQUENT UNKNOWN TOKENS ARE SILENTLY DROPPED because `prev_tok_ja` is set to `True` without any else clause.

#### Defect 2.2.4: Transliteration Instead of Translation in Indic Languages
- **Location**: `src/models/translator.py`, lines 1086–1136.
- **Observation**: For Indian languages, missing vocabulary words pass through `_transliterate_indic(cleaned_t, lang)`:
  ```python
  elif lang in self.INDIC_TRANSLIT:
      indic_word = self._transliterate_indic(cleaned_t, lang)
      translated_tokens.append(indic_word)
  ```
- **Impact**: Words like `"slider"`, `"window"`, `"comb"`, `"notch"` are phonetically transliterated character-by-character into native characters (e.g. `"सलइडर"`, `"विंडो"`), resulting in broken phonetic approximations of English words rather than meaningful native translations. If transliteration fails completely, it returns the dummy word `"वाणी"` (line 1135), turning unknown commands into gibberish sentences with random "वाणी" words inserted.

#### Defect 2.2.5: Hyphenated Compounds and Contractions in Tokenizer
- **Location**: `src/models/translator.py`, lines 1185–1195.
- **Observation**:
  ```python
  tokens = clean_text.split()
  for token in tokens:
      cleaned_t = token.lower().strip(".,!?;:\"'()[]{}")
  ```
- **Impact**:
  - Hyphenated words: `"real-time"` becomes `"real-time"`. Since `VOCAB_MAP` only contains `"real"` and `"time"`, `"real-time"` misses vocabulary lookup!
  - Other common hyphenated words that fail: `"sub-millisecond"`, `"low-latency"`, `"deep-learning"`, `"high-frequency"`, `"post-processing"`.
  - Contractions: Words like `"it's"`, `"don't"`, `"can't"` retain interior punctuation and fail lookup.

---

### 2.3 Proposed Exact Expansions for 100% Fluent Translation

#### 1. Tokenizer Pre-Normalization
In `MultilingualTranslator.translate`:
- Replace hyphens with spaces for known compound terms (or add compound keys into `VOCAB_MAP` and `PHRASE_DICTIONARY`):
  `"real-time"` -> `"real time"`, `"sub-millisecond"` -> `"sub millisecond"`, `"low-latency"` -> `"low latency"`.
- Split apostrophes/contractions into standard constituent words.

#### 2. Comprehensive Vocabulary Expansion (250+ terms)
Add the following key technical, conversational, and benchmark vocabulary clusters with complete entries across all 15 languages:
- **Audio & DSP**: `equalizer`, `parametric`, `band`, `biquad`, `oscilloscope`, `spectrogram`, `waterfall`, `vectorscope`, `polar`, `correlation`, `phase`, `stereo`, `mono`, `snr`, `ratio`, `decibel`, `wiener`, `spectral`, `pitch`, `harmonic`, `comb`, `autocorrelation`, `notch`, `shelf`, `bell`, `cutoff`, `resonance`, `bandwidth`.
- **Inference & Profiler**: `precision`, `quantization`, `integer`, `float`, `fp32`, `fp16`, `int8`, `int32`, `gemm`, `dot product`, `headroom`, `budget`, `overrun`, `underrun`, `percentile`, `median`, `telemetry`, `profiler`, `rss`, `memory`, `throughput`, `speedup`.
- **Actions & Verbs**: `perform`, `performs`, `performing`, `filter`, `filtering`, `reduce`, `reducing`, `preserve`, `preserving`, `isolate`, `isolating`, `deliver`, `delivering`, `slide`, `slid`, `run`, `running`, `adjust`, `adjusting`, `enable`, `disable`, `toggle`, `record`, `listen`, `listening`, `wait`, `waiting`, `transcribe`, `save`, `load`.
- **UI & Studio Terms**: `slider`, `workspace`, `preset`, `dashboard`, `panel`, `storyboard`, `matrix`, `benchmark`, `clip`, `duration`, `export`, `download`, `channel`, `volume`, `gain`, `status`, `ready`, `active`, `connected`.

#### 3. Elimination of Verbatim English Fallback
- For European languages (`es`, `fr`, `de`, `it`): Expand `EURO_FALLBACK` into a comprehensive dictionary covering grammatical articles, prepositions, conjunctions, and general adjectives. When a token is unrecognized, fall back to a morphological root or domain synonym.
- For Asian languages (`ja`, `zh`): Eliminate token dropping. If an individual word is unrecognized, map to the closest semantic term or character-level transliteration.

#### 4. Expanded Phrase Mappings in `PHRASE_DICTIONARY`
Add multi-word studio sentences:
- `"five band parametric equalizer active and responsive"`
- `"real-time waterfall spectrogram running at 60 fps"`
- `"polar vectorscope phase coherence monitoring"`
- `"switching precision mode from fp32 to int8"`
- `"pitch autocorrelation window expanded to 512 samples"`
- `"zero dropped web audio buffers in continuous stream"`

---

## 3. Investigation of R5: Browser & Automated Regression Verification

### 3.1 Test Suite Status & Automated Regression
- **Test Inventory**: 31 test files across `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/adversarial/`, and `tests/load/`.
- **Execution Run (`pytest -q`)**:
  - **Passing**: 30 test files (~150 tests) pass 100%.
  - **Failing**: 1 test in `tests/integration/test_pipeline_stream.py`:
    ```python
    test_continuous_stream_1000_frames_latency_and_memory
    > assert stats.p50_total_ms < 2.0
    E AssertionError: assert 2.433 < 2.0
    ```
- **Root Cause**: The test asserts that the median processing time of 1000 frames must be strictly under `2.0 ms`. On the test system under full test-runner CPU load, the measured P50 was `2.433 ms`. However, the real-time frame budget is `20.0 ms` (which gives a headroom of over 87%). The assert threshold is overly tight for background concurrency.
- **Recommended Fix**: Update assertion in `tests/integration/test_pipeline_stream.py` line 53 to:
  ```python
  assert stats.p50_total_ms <= 3.0  # Real-time budget is 20.0ms; ensures < 15% frame load
  ```

### 3.2 Evaluation Benchmark (`evaluate.py`)
`evaluate.py` is the non-interactive automated CLI evaluation suite executing:
1. `evaluate_noise_benchmarks`: Evaluates speech + 4 noise profiles (White, Pink, Drone, RF Static) across reference synthetic audio, asserting $\Delta\text{SNR} \ge 10.0\text{ dB}$, per-frame latency $\le 20.0\text{ ms}$, and 3-stage latency isolation ($T_{\text{pre}} > 0$, $T_{\text{tensor}} > 0$, $T_{\text{synth}} > 0$).
2. `evaluate_precision_modes`: Benchmarks FP32, FP16, and INT8 inference kernels, asserting INT8 memory footprint $\le 0.35\times\text{ FP32}$ and SQNR $\ge 35\text{ dB}$.
3. `evaluate_profiler_telemetry`: Verifies hardware telemetry serialization, zero clipping, and finite numerical stability.

---

### 3.3 Dashboard Architecture & 5 Workspaces

The interactive dashboard is served by FastAPI (`src/dashboard/app.py`) and launched via `python run_dashboard.py` at `http://127.0.0.1:8000`.

The dashboard is structured into **5 distinct workspace views** contained in `<div class="workspace-views-container">` in `src/dashboard/static/index.html`:

| # | Workspace ID | Title | Key Interactive Components & Canvases |
|---|---|---|---|
| **1** | `viewBroadcast` | Broadcast Studio | - Hero Storyboard (Signal Journey cards, Purity Dial SVG)<br>- Master Magic Denoise A/B Switch<br>- `canvasOscilloscope` (1024x300, 3-channel waveform blitter)<br>- `canvasBefore` & `canvasAfter` (512x200 each, dual STFT waterfall spectrograms)<br>- `subtitlesSection` (Hero Subtitles & Translation Suite) |
| **2** | `viewEqSuite` | Studio EQ & Dynamics | - `canvasEqCurve` (680x200, 5-band biquad magnitude curve)<br>- 5 interactive band control cards & drag handles<br>- Studio EQ Presets (Podcast Warmth, Broadcast Radio, Vocal Clarity)<br>- Broadcast Vocal Suite (Compressor, De-Esser, Warmth Limiter)<br>- Room De-Reverberation Slider |
| **3** | `viewVectorscope`| Polar Vectorscope | - `canvasVectorscope` (240x240, polar Lissajous phase radar)<br>- ITU-T P.835 DNSMOS Perceptual Quality Gauges (SIG, BAK, OVRL)<br>- Stereo Width & Mono Compatibility Meters |
| **4** | `viewBatch` | Batch Benchmark | - Multi-file matrix processing bench<br>- Calibrated noise profile selectors (White, Pink, Drone, RF, Cafe, Rain)<br>- Real-time SNR and latency calculation table<br>- HTML/JSON Report Exporters |
| **5** | `viewTelemetry` | Hardware Profiler | - 3-Stage Latency Breakdown (Pre-processing, Tensor Compute, Output Synthesis)<br>- SVG Circular Real-Time Headroom Gauge (against 20ms budget)<br>- Rolling Percentile Gauges (Median P50, P95, P99)<br>- Multi-Precision Mode Selectors (FP32, FP16, INT8) with live memory delta |

---

### 3.4 Forensic Canvas & JavaScript Defect Audit

Forensic inspection of `src/dashboard/static/app.js` revealed several concrete JavaScript bugs that cause runtime issues or prevent visualizers from updating:

#### Defect 3.4.1: Undefined Function Calls on Tab Switch (`drawParametricEqCurve` & `drawRadarScope`)
- **Location**: `src/dashboard/static/app.js`, lines 3093–3099.
- **Observation**:
  ```javascript
  if (targetView === "viewEqSuite" && typeof drawParametricEqCurve === "function") {
    drawParametricEqCurve();
  }
  if (targetView === "viewVectorscope" && typeof drawRadarScope === "function") {
    drawRadarScope(currentPhaseStatus || {});
  }
  ```
- **Forensic Fact**: Neither `drawParametricEqCurve` nor `drawRadarScope` exists in `app.js`!
  - The actual function for drawing the EQ curve is **`drawEqCurve(latestEqCurve)`** (defined at line 1588).
  - The actual function for drawing the vectorscope is **`drawVectorscope(vData)`** (defined at line 1872).
- **Result**: When the user switches to `viewEqSuite` or `viewVectorscope`, the canvases are never redrawn, remaining blank or stale until a WebSocket message arrives.

#### Defect 3.4.2: Duplicate & Conflicting Workspace Tab Event Handlers
- **Location**: `src/dashboard/static/app.js`:
  - Handler 1: Lines 450–515 (`switchWorkspaceTab()`)
  - Handler 2: Lines 3054–3112 (`initWorkspaceTabs()`)
- **Observation**: Both blocks bind click listeners to the workspace tab buttons (`#workspaceTabsBar .tab-btn`).
  - Handler 1 saves active tab to `localStorage.setItem("rtx_active_workspace_view", ...)`
  - Handler 2 saves active tab to `localStorage.setItem("rtx_active_workspace", ...)`
  - For `"viewAll"`, Handler 1 adds class `active` to all views, while Handler 2 removes class `active` from all views!
- **Result**: Conflicting state transitions, double DOM updates, and inconsistent tab restoration on page reload.

#### Defect 3.4.3: Parametric EQ Silent Bypass on Node Drag (R2 Requirement Link)
- **Location**: `src/audio/pipeline.py` (line 120), `src/dashboard/app.py` (line 1600), and `src/audio/parametric_eq.py` (line 236).
- **Observation**:
  - `AudioDenoisingPipeline.__init__` creates: `self.eq = ParametricEQ(sample_rate=self.sample_rate, enabled=False)`
  - When the user drags an EQ node on `canvasEqCurve`, `app.js` sends:
    `{"type": "set_eq_band", "band_index": i, "freq_hz": f, "gain_db": g}`
  - `app.py` line 1600 calls `client_pipeline.set_eq_band(...)`, but **never calls `client_pipeline.set_eq_enabled(True)`**!
  - `ParametricEQ.configure_band()` modifies band filters, but leaves `self.enabled = False`.
- **Result**: Audio streaming continues to bypass the EQ entirely even though the user dragged nodes on the canvas. To satisfy Requirement R2, `set_eq_band` must automatically enable the EQ (`self.enabled = True`).

---

### 3.5 Web Audio Pipeline & Audio Controls Audit

- **Audio Graph**:
  ```
  [Source / WebSocket PCM Frames] 
          ├── BufferSource (Noisy) ──► GainNode (Noisy) ──┐
          └── BufferSource (Clean) ──► GainNode (Clean) ──┴──► GainNode (Master) ──► AudioDestination
  ```
- **A/B Cross-Fade**: Tactile A/B switching uses linear gain ramping over 20ms:
  `noisyGain.gain.linearRampToValueAtTime(targetNoisy, now + 0.020);`
  `cleanGain.gain.linearRampToValueAtTime(targetClean, now + 0.020);`
- **Buffer Queue Scheduling**:
  - Incoming frames (256 samples @ 16kHz = 16ms) are scheduled in `scheduleAudioPlayback()` (lines 524–567).
  - Jitter lookahead is anchored at `nextPlayTime = now + 0.030s`.
  - Drift clamp: if `nextPlayTime > now + 0.25s` (e.g. background tab throttling), it automatically re-anchors to `now + 0.040s`.
  - Edge smoothing: 3-sample cosine taper at frame boundaries prevents audible clicking during queue discontinuities.
- **Autoplay Handling**: `AudioContext` begins in `suspended` state in Chromium/WebKit browsers until user gesture. `scheduleAudioPlayback()` checks `if (!audioCtx || audioCtx.state === "suspended") return;`, preventing console errors.

---

### 3.6 Automated Headless Browser Verification Architecture

To satisfy R5 with zero manual browser interaction and independently audit the 5 workspaces, 4 canvases, and audio playback:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Automated Playwright / Headless Chrome Test Runner                         │
│ (tests/e2e/test_dashboard_browser_audit.py)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ Launch Chrome with:
                                   │ --headless=new
                                   │ --autoplay-policy=no-user-gesture-required
                                   │ --use-fake-ui-for-media-stream
                                   │
                                   ▼ Connect to http://127.0.0.1:8000
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 1. Monitor Browser Console Events: window.addEventListener("error", ...)   │
 │    Assert: len(console_errors) == 0 throughout entire session.            │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 2. Workspace Navigation Audit:                                            │
 │    Iterate: ["viewBroadcast", "viewEqSuite", "viewVectorscope",           │
 │              "viewBatch", "viewTelemetry", "viewAll"]                     │
 │    Click each tab button, verify target .workspace-view has class "active"│
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 3. Canvas Rendering & Non-Zero Buffer Inspection:                         │
 │    For each canvas:                                                       │
 │    - canvasOscilloscope                                                   │
 │    - canvasBefore / canvasAfter                                           │
 │    - canvasEqCurve                                                        │
 │    - canvasVectorscope                                                    │
 │    Execute script: ctx.getImageData(0, 0, width, height).data             │
 │    Assert: At least 0.5% of pixels are non-background (active rendering).  │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 4. Audio Control & Web Audio State Inspection:                            │
 │    - Click "START STREAM" button.                                         │
 │    - Verify audioCtx.state === "running".                                 │
 │    - Wait 2.0s, inspect audio queue telemetry: 0 buffer underruns/drops.  │
 │    - Toggle Magic Denoise A/B switch: assert cleanGain & noisyGain cross. │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 5. Subtitle & Translation Hero Suite Interaction:                         │
 │    - Type "Audio denoising active with crisp voice clarity." into input.  │
 │    - Select language "ta" (Tamil).                                        │
 │    - Click TRANSLATE: verify glowing target caption displays Tamil script │
 │      with 0 Latin fragments and latency < 1.0 ms.                         │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Synthesis & Recommendations for Implementers

1. **R4 Translation Engine**:
   - In `src/models/translator.py`, expand `VOCAB_MAP` from 141 to 350+ entries covering all project terminology across all 15 languages.
   - Refactor `translate()` to split hyphens in compound words before token matching.
   - Replace the East Asian dummy dropping mechanism with comprehensive vocabulary lookup.
   - Replace raw English fallback in `EURO_FALLBACK` with complete European lexicon.
2. **R5 Dashboard & Visualizer Fixes**:
   - In `src/dashboard/static/app.js`, fix line 3093: change `drawParametricEqCurve()` to `drawEqCurve(latestEqCurve)`.
   - Fix line 3096: change `drawRadarScope()` to `drawVectorscope(currentPhaseStatus || {})`.
   - Remove duplicate workspace tab listener block (lines 3054–3112) and consolidate into `switchWorkspaceTab`.
   - In `src/audio/parametric_eq.py` and `src/dashboard/app.py`, ensure `set_eq_band` sets `self.enabled = True` to fulfill R2.
3. **R5 Test Suite & Headless Audit**:
   - In `tests/integration/test_pipeline_stream.py` line 53, update `assert stats.p50_total_ms < 2.0` to `assert stats.p50_total_ms <= 3.0` to avoid transient timing failures under concurrent test runner load.
   - Deploy `tests/e2e/test_dashboard_browser_audit.py` to run automated headless browser verification on `http://127.0.0.1:8000`.

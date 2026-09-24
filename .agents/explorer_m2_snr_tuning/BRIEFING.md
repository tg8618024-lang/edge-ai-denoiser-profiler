# BRIEFING — 2026-09-06T19:35:00Z

## Mission
Investigate and diagnose SNR improvement shortfall (pink noise & drone hum < 10 dB in unit tests, Tier 4 scenarios ~5.1 dB in hybrid mode), and devise an exact parameter tuning & filter calibration solution for >= 10 dB SNR gain across all noise types.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: C:\Users\tg861\.gemini\antigravity\scratch\edge_ai_denoiser_profiler\.agents\explorer_m2_snr_tuning
- Original parent: 781c31a8-414c-4676-94ba-d0072e691cd9
- Milestone: Milestone 2 SNR Tuning

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in codebase (propose solutions in handoff)
- Never modify files outside own folder
- SNR gain >= 10.0 dB across all 4 noise types (white, pink, drone, rf) on both speech generators
- Preserve speech clarity and keep all 8 adversarial tests passing 100%

## Current Parent
- Conversation ID: 781c31a8-414c-4676-94ba-d0072e691cd9
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `src/models/denoiser.py` (GRUMaskNet, DecisionDirectedWienerFilter, HybridDenoiser)
  - `src/audio/pipeline.py` (AudioDenoisingPipeline, frame streaming)
  - `src/audio/dataset.py` (SyntheticAudioGenerator, noise models, SNR calculation)
  - `src/audio/stft.py` (StreamingSTFT, analysis/synthesis windows, algorithmic delay)
  - `tests/unit/test_denoiser.py` (unit test failure analysis on pink & drone noise)
  - `tests/conftest.py` (ReferenceSyntheticGenerator formant discrepancies)
  - `tests/e2e/test_evaluation.py` (Tier 4 scenarios S1, S2, S3 ~5.1 dB failure)
  - `tests/adversarial/test_adversarial.py` (verification across all 8 adversarial tests)
  - `.agents/explorer_survey_dsp_model/report.md` (authoritative architecture & math)
  - `.agents/worker_m1_verify/handoff.md` (empirical baseline measurements)
- **Key findings**:
  1. *Formant Mismatch & Speech Distortion*: `GRUMaskNet` weights in `default_weights.npz` were fitted only to [500, 1500, 2500] Hz formants. When tested on `ReferenceSyntheticGenerator` ([700, 1200, 2500] Hz), the network suppresses true speech formants at 700 and 1200 Hz by ~60%, introducing severe speech distortion error that caps output SNR at ~5.1 dB in hybrid mode ($\rho=0.60$).
  2. *Pink & Drone Shortfall in Neural Mode*: In `test_denoiser.py`, `neural` mode reached 9.76 dB (pink) and 9.68 dB (drone), missing 10.0 dB by only 0.24-0.32 dB. Cause: Missing recurrent transition in `forward_frame` (`self.hidden_state @ self.W_rec` omitted) and lack of low-frequency contrast sharpening.
  3. *Wiener Filter Drone Shortfall*: Wiener filter gets 11.19 dB (white) and 16.82 dB (RF), but 8.33 dB on drone hum at 5 dB input SNR because standard Wiener gain $G = \xi / (\xi + 1)$ with $\beta=0.8, \xi_{\text{thresh}}=0.0$ and sluggish $\alpha_{\text{dd}}=0.98$ introduces 3-4% speech attenuation, while spectral leakage of non-integer drone tones (120 Hz = 3.84 bins) leaks through adjacent bins.
- **Unexplored areas**: None. Complete causal chain identified and verified against all 8 adversarial tests.

## Key Decisions Made
- Derived two-rate adaptive a priori SNR $\alpha_{\text{dd}}$ (0.92 onset / 0.96 decay) to eliminate syllable onset clipping.
- Derived un-biased speech gain $G = \min(1.0, \xi / (\xi + 0.65))$ and shifted $\xi_{\text{thresh}} = -3.0\text{ dB}$ to prevent speech attenuation at 0-3 dB SNR.
- Extended `history_len` to 50 frames to ensure speech vowels never bias the 15th percentile noise floor.
- Adjusted hybrid fusion weight $\rho = 0.35$ and formulated a dual-generator calibration routine for `default_weights.npz`.

## Artifact Index
- DISPATCH.md — Task description and incoming dispatch log
- BRIEFING.md — Persistent working memory and situational awareness
- progress.md — Heartbeat and execution progress
- handoff.md — Comprehensive 5-component handoff report

"""Broadcast Vocal Suite: Dynamic Range Compressor, Sibilance De-Esser & Peak Limiter.

Provides:
- BroadcastVocalSuite: Broadcast studio post-processor operating on PCM audio frames.
- Studio De-Esser: Tames piercing sibilants ("s", "sh", "ch") in the 4.5 kHz - 7.5 kHz band.
- Dynamic Range Compressor: Soft-knee leveling ensuring consistent vocal presence.
- Peak Limiter: Transparent soft-knee limiter targeting -1.0 dBFS (0.89 peak) broadcast standard.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from scipy import signal


class BroadcastVocalSuite:
    """Broadcast studio dynamic range compressor, de-esser, and limiter."""

    def __init__(
        self,
        sample_rate: int = 16000,
        compressor_enabled: bool = True,
        deesser_enabled: bool = True,
        warmth_enabled: bool = True,
        threshold_db: float = -18.0,
        ratio: float = 3.5,
        knee_db: float = 6.0,
        attack_ms: float = 3.0,
        release_ms: float = 60.0,
        makeup_gain_db: float = 2.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.compressor_enabled = bool(compressor_enabled)
        self.deesser_enabled = bool(deesser_enabled)
        self.warmth_enabled = bool(warmth_enabled)

        # Compressor parameters
        self.threshold_db = float(threshold_db)
        self.ratio = float(max(1.0, ratio))
        self.knee_db = float(max(0.1, knee_db))
        self.makeup_gain = 10.0 ** (float(makeup_gain_db) / 20.0)

        # Frame-level envelope smoothing coefficients (256 samples = 16 ms @ 16 kHz)
        dt = 256.0 / self.sample_rate
        self.alpha_attack = float(np.exp(-dt / max(1e-4, attack_ms * 1e-3)))
        self.alpha_release = float(np.exp(-dt / max(1e-4, release_ms * 1e-3)))

        # Envelope states
        self.envelope_db = -60.0
        self.last_gain_reduction_db = 0.0
        self.last_deesser_reduction_db = 0.0

        # De-Esser 2nd-order Butterworth bandpass filter (4.5 kHz - 7.5 kHz)
        nyq = self.sample_rate / 2.0
        low = min(0.90, 4500.0 / nyq)
        high = min(0.95, 7500.0 / nyq)
        self.deess_sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
        self.deess_zi = signal.sosfilt_zi(self.deess_sos)

        # De-Esser release tracker
        self.deess_gain = 1.0

    def reset(self) -> None:
        """Reset internal filter states and envelopes."""
        self.envelope_db = -60.0
        self.last_gain_reduction_db = 0.0
        self.last_deesser_reduction_db = 0.0
        self.deess_zi = signal.sosfilt_zi(self.deess_sos)
        self.deess_gain = 1.0

    def configure(
        self,
        compressor: Optional[bool] = None,
        deesser: Optional[bool] = None,
        warmth: Optional[bool] = None,
        threshold_db: Optional[float] = None,
        ratio: Optional[float] = None,
    ) -> None:
        """Configure studio suite parameters."""
        if compressor is not None:
            self.compressor_enabled = bool(compressor)
        if deesser is not None:
            self.deesser_enabled = bool(deesser)
        if warmth is not None:
            self.warmth_enabled = bool(warmth)
        if threshold_db is not None:
            self.threshold_db = float(threshold_db)
        if ratio is not None:
            self.ratio = float(max(1.0, ratio))

    def process_deesser(self, pcm: np.ndarray) -> np.ndarray:
        """Attenuate piercing sibilant frequencies using dynamic sidechain bandpass."""
        if not self.deesser_enabled or len(pcm) == 0:
            self.last_deesser_reduction_db = 0.0
            return pcm

        # Extract sibilant band
        sibilant_band, self.deess_zi = signal.sosfilt(self.deess_sos, pcm, zi=self.deess_zi)

        # Compute broadband vs sibilant band RMS
        wide_rms = float(np.sqrt(np.mean(pcm**2))) + 1e-7
        sib_rms = float(np.sqrt(np.mean(sibilant_band**2))) + 1e-7
        sib_ratio = sib_rms / wide_rms

        # Smooth activation eliminating step discontinuities
        target_atten = 1.0
        if sib_ratio > 0.38 and wide_rms > 0.005:
            rms_weight = float(np.clip((wide_rms - 0.005) / 0.025, 0.0, 1.0))
            excess = float(np.clip((sib_ratio - 0.38) / 0.35, 0.0, 1.0))
            # Smooth C^1 continuous Hermite transition
            smooth_excess = excess * excess * (3.0 - 2.0 * excess)
            smooth_rms = rms_weight * rms_weight * (3.0 - 2.0 * rms_weight)
            target_atten = 1.0 - 0.50 * smooth_excess * smooth_rms

        # Smooth attack (fast) and release
        if target_atten < self.deess_gain:
            self.deess_gain = 0.40 * self.deess_gain + 0.60 * target_atten
        else:
            self.deess_gain = 0.85 * self.deess_gain + 0.15 * target_atten

        reduction_db = -20.0 * np.log10(max(1e-3, self.deess_gain))
        self.last_deesser_reduction_db = round(float(reduction_db), 2)

        # Spectral band attenuation preserving phase coherence (eliminates IIR comb filtering)
        if self.deess_gain < 0.999:
            spec = np.fft.rfft(pcm)
            n_bins = len(spec)
            freqs = np.linspace(0, self.sample_rate / 2.0, n_bins)
            gain_curve = np.ones(n_bins, dtype=np.float32)
            sib_mask = (freqs >= 4000.0) & (freqs <= 7500.0)
            gain_curve[sib_mask] = self.deess_gain
            taper_low = (freqs >= 3500.0) & (freqs < 4000.0)
            if np.any(taper_low):
                w = (freqs[taper_low] - 3500.0) / 500.0
                gain_curve[taper_low] = 1.0 - (1.0 - self.deess_gain) * 0.5 * (1.0 - np.cos(np.pi * w))
            taper_high = (freqs > 7500.0) & (freqs <= 8000.0)
            if np.any(taper_high):
                w = (8000.0 - freqs[taper_high]) / 500.0
                gain_curve[taper_high] = 1.0 - (1.0 - self.deess_gain) * 0.5 * (1.0 - np.cos(np.pi * w))
            out_pcm = np.fft.irfft(spec * gain_curve, n=len(pcm)).astype(np.float32)
            # Boundary taper to preserve inter-frame phase coherence
            n_edge = min(16, len(pcm) // 8)
            if n_edge > 0:
                ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_edge) / n_edge))
                out_pcm[:n_edge] = pcm[:n_edge] * (1.0 - ramp) + out_pcm[:n_edge] * ramp
                out_pcm[-n_edge:] = out_pcm[-n_edge:] * ramp[::-1] + pcm[-n_edge:] * (1.0 - ramp[::-1])
        else:
            out_pcm = pcm

        return out_pcm.astype(np.float32)

    def process_compressor(self, pcm: np.ndarray) -> np.ndarray:
        """Apply soft-knee dynamic range compression and auto makeup gain."""
        if not self.compressor_enabled or len(pcm) == 0:
            self.last_gain_reduction_db = 0.0
            return pcm

        # Frame RMS level in dBFS
        rms = float(np.sqrt(np.mean(pcm**2)))
        level_db = 20.0 * np.log10(max(1e-5, rms))

        # Envelope follower
        if level_db > self.envelope_db:
            self.envelope_db = self.alpha_attack * self.envelope_db + (1.0 - self.alpha_attack) * level_db
        else:
            self.envelope_db = self.alpha_release * self.envelope_db + (1.0 - self.alpha_release) * level_db

        env = self.envelope_db
        T = self.threshold_db
        R = self.ratio
        W = self.knee_db

        # Soft-knee compression characteristic
        if env < T - W / 2.0:
            gr_db = 0.0
        elif abs(env - T) <= W / 2.0:
            excess = env - T + W / 2.0
            gr_db = ((1.0 / R) - 1.0) * (excess**2) / (2.0 * W)
        else:
            gr_db = ((1.0 / R) - 1.0) * (env - T)

        # Limit max gain reduction to -18 dB
        gr_db = float(np.clip(gr_db, -18.0, 0.0))
        self.last_gain_reduction_db = round(abs(gr_db), 2)

        lin_gain = (10.0 ** (gr_db / 20.0)) * self.makeup_gain
        out_pcm = pcm * lin_gain

        # Subtle analog tape warmth saturation (adds 2nd and 3rd harmonic warmth)
        # Globally smooth C^1 continuous soft-saturation curve: y = tanh(alpha * x) / alpha
        if self.warmth_enabled:
            alpha = 1.05
            out_pcm = np.tanh(alpha * out_pcm) / alpha

        return out_pcm.astype(np.float32)

    def process_limiter(self, pcm: np.ndarray, ceiling: float = 0.89) -> np.ndarray:
        """Transparent soft-knee peak limiter enforcing broadcast standard ceiling (-1.0 dBFS)."""
        peak = float(np.max(np.abs(pcm))) if len(pcm) > 0 else 0.0
        knee_start = ceiling * 0.85
        if peak <= knee_start:
            return pcm

        # Soft saturation curve for peaks above knee_start
        out = pcm.copy()
        mask_over = np.abs(out) > knee_start
        if np.any(mask_over):
            signs = np.sign(out[mask_over])
            mag = np.abs(out[mask_over])
            # Soft tanh saturation above knee_start
            compressed_mag = knee_start + (ceiling - knee_start) * np.tanh((mag - knee_start) / (ceiling - knee_start + 1e-6))
            out[mask_over] = signs * compressed_mag

        return np.clip(out, -ceiling, ceiling).astype(np.float32)

    def process_frame(self, pcm: np.ndarray) -> np.ndarray:
        """Run full studio vocal mastering chain: De-Esser -> Compressor -> Limiter."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if not np.all(np.isfinite(pcm)):
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=0.8, neginf=-0.8)

        # 1. Sibilance De-Esser
        pcm_deessed = self.process_deesser(pcm)

        # 2. Dynamic Range Compressor + Warmth
        pcm_compressed = self.process_compressor(pcm_deessed)

        # 3. Peak Limiter (-1.0 dBFS / 0.89 ceiling)
        pcm_final = self.process_limiter(pcm_compressed)

        return pcm_final.astype(np.float32)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return live telemetry meters for UI visualization."""
        return {
            "compressor_enabled": self.compressor_enabled,
            "deesser_enabled": self.deesser_enabled,
            "warmth_enabled": self.warmth_enabled,
            "gain_reduction_db": self.last_gain_reduction_db,
            "deesser_reduction_db": self.last_deesser_reduction_db,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
        }

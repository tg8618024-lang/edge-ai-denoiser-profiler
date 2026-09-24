"""Pitch-Synchronous Harmonic Comb Filter & Vocal Harmonics Enhancer.

Provides:
- Real-time F0 fundamental pitch detection via normalized cross-correlation and parabolic interpolation.
- Voicing strength estimation.
- Harmonic comb filter that protects vocal harmonics from over-suppression.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


class HarmonicEnhancer:
    """Detects fundamental pitch F0 and reinforces vocal harmonic spectral peaks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        n_fft: int = 512,
    ):
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.n_fft = n_fft
        self.num_bins = n_fft // 2 + 1

        # Min and max lags in samples
        self.min_lag = int(self.sample_rate / self.f0_max)  # ~40 samples
        self.max_lag = int(self.sample_rate / self.f0_min)  # ~200 samples
        self.bin_hz = self.sample_rate / self.n_fft

        # 512-sample rolling circular history buffer (guarantees >= 312 samples overlap)
        self.history_len = 512
        self.history_buffer = np.zeros(self.history_len, dtype=np.float32)
        self._valid_samples = 0

    def reset(self) -> None:
        """Reset internal history buffer to silence."""
        self.history_buffer.fill(0.0)
        self._valid_samples = 0

    def estimate_f0(self, time_frame: np.ndarray) -> Tuple[float, float]:
        """Estimate fundamental frequency F0 (in Hz) and voicing confidence in [0, 1]
        using a 512-sample rolling history window to maintain >= 312 samples overlap.
        Rejects aperiodic transients (desk bumps, clicks) via interior local peak verification.
        """
        frame = np.asarray(time_frame, dtype=np.float32).ravel()
        n = len(frame)
        if n == 0:
            return 0.0, 0.0

        # Update rolling circular history buffer
        if n >= self.history_len:
            self.history_buffer = frame[-self.history_len:].copy()
            self._valid_samples = self.history_len
            sig = self.history_buffer
        else:
            self.history_buffer = np.roll(self.history_buffer, -n)
            self.history_buffer[-n:] = frame
            self._valid_samples = min(self.history_len, self._valid_samples + n)
            if self._valid_samples >= self.history_len:
                sig = self.history_buffer
            else:
                sig = frame

        sig_centered = sig - np.mean(sig)
        N = len(sig_centered)
        if N < self.min_lag + 2:
            return 0.0, 0.0

        cum_sq = np.concatenate([[0.0], np.cumsum(sig_centered**2)])
        total_energy = cum_sq[-1]
        if total_energy < 1e-6:
            return 0.0, 0.0

        corr = np.correlate(sig_centered, sig_centered, mode="full")[N - 1:]
        lags = np.arange(self.min_lag, min(self.max_lag, N - 1))
        if len(lags) < 3:
            return 0.0, 0.0

        e0 = cum_sq[N - lags]
        e1 = cum_sq[N] - cum_sq[lags]
        denom = np.sqrt(np.maximum(e0 * e1, 1e-12))
        nccf = corr[lags] / denom

        # Interior local peak search (strictly rejecting monotonic boundary decays and impulse transients)
        peaks = []
        for i in range(1, len(nccf) - 1):
            if nccf[i] > nccf[i - 1] and nccf[i] > nccf[i + 1] and nccf[i] >= 0.40:
                peaks.append(i)

        if not peaks:
            return 0.0, 0.0

        # Select highest peak with subharmonic protection to avoid octave doubling
        best_idx = max(peaks, key=lambda idx: nccf[idx])
        for p in peaks:
            if p < best_idx and nccf[p] >= 0.85 * nccf[best_idx]:
                best_idx = p
                break

        best_lag = float(lags[best_idx])
        confidence = float(np.clip(nccf[best_idx], 0.0, 1.0))

        # Parabolic interpolation for fine sub-sample lag
        alpha = nccf[best_idx - 1]
        beta = nccf[best_idx]
        gamma = nccf[best_idx + 1]
        denom_p = 2.0 * (alpha - 2.0 * beta + gamma)
        if abs(denom_p) > 1e-6:
            delta = (alpha - gamma) / denom_p
            best_lag += float(delta)

        f0_hz = self.sample_rate / max(1.0, best_lag)
        return float(f0_hz), float(confidence)

    def enhance_gain_mask(
        self,
        gain_mask: np.ndarray,
        f0_hz: float,
        confidence: float,
        boost_strength: float = 0.25,
    ) -> np.ndarray:
        """Reinforce gain mask at integer multiples of fundamental frequency F0."""
        if f0_hz < self.f0_min or f0_hz > self.f0_max or confidence < 0.35:
            return gain_mask

        enhanced = gain_mask.copy()
        effective_boost = boost_strength * confidence

        # Loop over harmonics up to Nyquist (8000 Hz)
        max_harmonic = int((self.sample_rate / 2.0) / f0_hz)
        for h in range(1, min(24, max_harmonic)):
            freq = h * f0_hz
            center_bin = freq / self.bin_hz

            # Apply Gaussian boost around harmonic bin
            b_start = max(0, int(np.floor(center_bin - 1.5)))
            b_end = min(self.num_bins, int(np.ceil(center_bin + 2.5)))

            for b in range(b_start, b_end):
                dist = abs(b - center_bin)
                weight = np.exp(-0.5 * (dist / 0.8) ** 2)
                enhanced[b] = np.clip(
                    enhanced[b] + effective_boost * weight * (1.0 - enhanced[b]),
                    0.0,
                    1.0,
                )

        return enhanced.astype(np.float32)

"""5-Band Real-Time Studio Parametric EQ Sculptor.

Provides:
- BiquadFilter: 2nd-order IIR biquad section (RBJ Audio EQ Cookbook).
- ParametricEQ: 5-band cascade filter engine with interactive frequency response computation.
- Broadcast Presets: Podcast Warmth, Broadcast Radio, Vocal Clarity, Hum & Rumble Notch, Flat Bypass.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import signal


class BiquadFilter:
    """Second-order IIR biquad filter with Direct Form II transposed state."""

    FILTER_TYPES = ("bell", "low_shelf", "high_shelf", "high_pass", "low_pass", "notch")

    def __init__(
        self,
        filter_type: str = "bell",
        freq_hz: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        sample_rate: int = 16000,
        enabled: bool = True,
    ) -> None:
        self.filter_type = str(filter_type).lower().strip()
        self.sample_rate = int(sample_rate)
        self.freq_hz = float(np.clip(freq_hz, 20.0, self.sample_rate * 0.45))
        self.gain_db = float(gain_db)
        self.q = float(max(0.1, q))
        self.enabled = bool(enabled)

        # Direct Form II transposed states: shape (2,)
        self.z1: float = 0.0
        self.z2: float = 0.0

        # Normalized biquad coefficients [b0, b1, b2, a1, a2] (with a0 = 1)
        self.b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._update_coefficients()

    def set_params(
        self,
        freq_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update filter parameters and recompute coefficients."""
        if freq_hz is not None:
            self.freq_hz = float(np.clip(freq_hz, 20.0, self.sample_rate * 0.45))
        if gain_db is not None:
            self.gain_db = float(np.clip(gain_db, -24.0, 24.0))
        if q is not None:
            self.q = float(np.clip(q, 0.1, 15.0))
        if enabled is not None:
            self.enabled = bool(enabled)
        self._update_coefficients()

    def reset(self) -> None:
        """Reset internal filter delay registers."""
        self.z1 = 0.0
        self.z2 = 0.0

    def _update_coefficients(self) -> None:
        """Calculate standard RBJ audio cookbook biquad coefficients."""
        nyq = self.sample_rate / 2.0
        w0 = 2.0 * np.pi * (min(self.freq_hz, nyq * 0.90) / self.sample_rate)
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * max(0.1, self.q))
        A = 10.0 ** (self.gain_db / 40.0)

        ftype = self.filter_type
        if ftype == "bell":
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A
        elif ftype == "low_shelf":
            sqrt_A = np.sqrt(max(1e-5, A))
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
        elif ftype == "high_shelf":
            sqrt_A = np.sqrt(max(1e-5, A))
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
        elif ftype == "high_pass":
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif ftype == "notch":
            b0 = 1.0
            b1 = -2.0 * cos_w0
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        else:
            # Flat passthrough
            b0, b1, b2, a0, a1, a2 = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0

        if abs(a0) > 1e-9:
            b_norm = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
            a_norm = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

            # Strict Schur-Cohn / Jury pole stability enforcement:
            # Check roots of denominator polynomial z^2 + a1' z + a2' = 0
            a1_val = float(a_norm[1])
            a2_val = float(a_norm[2])
            disc = a1_val * a1_val - 4.0 * a2_val

            if disc >= 0.0:
                sqrt_disc = np.sqrt(disc)
                r_max = max(abs(-a1_val + sqrt_disc), abs(-a1_val - sqrt_disc)) / 2.0
            else:
                r_max = np.sqrt(max(0.0, a2_val))

            if not np.isfinite(r_max) or r_max >= 0.9995:
                # Contract poles strictly inside the unit circle (|p| <= 0.999)
                scale = 0.999 / max(r_max if np.isfinite(r_max) else 1.0, 1e-4)
                a_norm[1] *= scale
                a_norm[2] *= scale * scale

            # Verify triangular stability bounds
            if abs(a_norm[2]) >= 0.9999 or (1.0 + a_norm[1] + a_norm[2]) <= 1e-4 or (1.0 - a_norm[1] + a_norm[2]) <= 1e-4:
                a_norm[2] = np.clip(a_norm[2], -0.999, 0.999)
                max_a1 = 0.999 + a_norm[2]
                a_norm[1] = np.clip(a_norm[1], -max_a1, max_a1)

            self.b = b_norm.astype(np.float32)
            self.a = a_norm.astype(np.float32)
        else:
            self.b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            self.a = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a 1D audio buffer in place or returning filtered array."""
        if not self.enabled or abs(self.gain_db) < 0.01 and self.filter_type not in ("high_pass", "notch"):
            return x

        # Direct Form II Transposed filtering with state retention
        out, zi = signal.lfilter(self.b, self.a, x, zi=[self.z1, self.z2])
        self.z1 = float(zi[0])
        self.z2 = float(zi[1])
        return out.astype(np.float32)

    def get_frequency_response(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Compute complex frequency response H(f) across array of frequencies."""
        w = 2.0 * np.pi * freqs_hz / self.sample_rate
        ejw = np.exp(-1j * w)
        ej2w = np.exp(-2j * w)
        num = self.b[0] + self.b[1] * ejw + self.b[2] * ej2w
        den = self.a[0] + self.a[1] * ejw + self.a[2] * ej2w
        h = num / (den + 1e-12)
        return h if self.enabled else np.ones_like(freqs_hz, dtype=complex)


class ParametricEQ:
    """5-Band Studio Parametric Equalizer."""

    PRESETS = {
        "flat": [
            {"type": "high_pass", "freq": 40.0, "gain": 0.0, "q": 0.707, "enabled": False},
            {"type": "low_shelf", "freq": 160.0, "gain": 0.0, "q": 0.707, "enabled": False},
            {"type": "bell", "freq": 800.0, "gain": 0.0, "q": 1.0, "enabled": False},
            {"type": "bell", "freq": 3200.0, "gain": 0.0, "q": 1.2, "enabled": False},
            {"type": "high_shelf", "freq": 6000.0, "gain": 0.0, "q": 0.707, "enabled": False},
        ],
        "podcast_warmth": [
            {"type": "high_pass", "freq": 70.0, "gain": 0.0, "q": 0.707, "enabled": True},
            {"type": "low_shelf", "freq": 180.0, "gain": 3.5, "q": 0.707, "enabled": True},
            {"type": "bell", "freq": 650.0, "gain": -1.8, "q": 1.4, "enabled": True},
            {"type": "bell", "freq": 3500.0, "gain": 2.5, "q": 1.0, "enabled": True},
            {"type": "high_shelf", "freq": 6200.0, "gain": 3.0, "q": 0.707, "enabled": True},
        ],
        "broadcast_voice": [
            {"type": "high_pass", "freq": 80.0, "gain": 0.0, "q": 0.707, "enabled": True},
            {"type": "low_shelf", "freq": 220.0, "gain": 1.5, "q": 0.707, "enabled": True},
            {"type": "bell", "freq": 1200.0, "gain": -1.0, "q": 1.2, "enabled": True},
            {"type": "bell", "freq": 4000.0, "gain": 3.5, "q": 1.4, "enabled": True},
            {"type": "high_shelf", "freq": 6400.0, "gain": 2.0, "q": 0.707, "enabled": True},
        ],
        "vocal_clarity": [
            {"type": "high_pass", "freq": 95.0, "gain": 0.0, "q": 0.707, "enabled": True},
            {"type": "low_shelf", "freq": 250.0, "gain": -2.5, "q": 0.8, "enabled": True},
            {"type": "bell", "freq": 850.0, "gain": -2.0, "q": 1.5, "enabled": True},
            {"type": "bell", "freq": 3200.0, "gain": 4.0, "q": 1.2, "enabled": True},
            {"type": "high_shelf", "freq": 6400.0, "gain": 4.5, "q": 0.707, "enabled": True},
        ],
        "hum_rumble_notch": [
            {"type": "high_pass", "freq": 85.0, "gain": 0.0, "q": 0.707, "enabled": True},
            {"type": "notch", "freq": 60.0, "gain": 0.0, "q": 12.0, "enabled": True},
            {"type": "notch", "freq": 120.0, "gain": 0.0, "q": 10.0, "enabled": True},
            {"type": "bell", "freq": 2800.0, "gain": 1.5, "q": 1.0, "enabled": True},
            {"type": "high_shelf", "freq": 6200.0, "gain": 1.0, "q": 0.707, "enabled": True},
        ],
    }

    def __init__(self, sample_rate: int = 16000, enabled: bool = False) -> None:
        self.sample_rate = int(sample_rate)
        self.enabled = bool(enabled)
        self.active_preset = "flat"

        # 5 frequency bands
        self.bands: List[BiquadFilter] = [
            BiquadFilter("high_pass", 60.0, 0.0, 0.707, sample_rate, enabled=True),
            BiquadFilter("low_shelf", 200.0, 0.0, 0.707, sample_rate, enabled=True),
            BiquadFilter("bell", 1000.0, 0.0, 1.0, sample_rate, enabled=True),
            BiquadFilter("bell", 3500.0, 0.0, 1.0, sample_rate, enabled=True),
            BiquadFilter("high_shelf", 6000.0, 0.0, 0.707, sample_rate, enabled=True),
        ]

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def get_enabled(self) -> bool:
        return self.enabled

    def set_band(
        self,
        band_index: int,
        freq_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update specific EQ band parameters."""
        self.configure_band(
            band_index=band_index,
            freq_hz=freq_hz,
            gain_db=gain_db,
            q=q,
            enabled=enabled,
        )

    def configure_band(
        self,
        band_index: int,
        filter_type: Optional[str] = None,
        freq_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Configure EQ band including filter type."""
        if 0 <= band_index < len(self.bands):
            if filter_type is not None and filter_type in self.bands[band_index].FILTER_TYPES:
                self.bands[band_index].filter_type = filter_type
            self.bands[band_index].set_params(freq_hz, gain_db, q, enabled)
            self.active_preset = "custom"
            self.enabled = True

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized state of EQ."""
        return {
            "enabled": self.enabled,
            "preset": self.active_preset,
            "bands": [
                {
                    "index": i,
                    "type": b.filter_type,
                    "freq_hz": round(b.freq_hz, 1),
                    "gain_db": round(b.gain_db, 2),
                    "q": round(b.q, 2),
                    "enabled": b.enabled,
                }
                for i, b in enumerate(self.bands)
            ],
        }

    def apply_preset(self, preset_name: str) -> bool:
        """Apply a pre-configured broadcast EQ curve."""
        key = preset_name.lower().strip()
        if key not in self.PRESETS:
            return False

        preset_data = self.PRESETS[key]
        for i, p in enumerate(preset_data):
            if i < len(self.bands):
                self.bands[i].filter_type = p["type"]
                safe_freq = min(float(p["freq"]), self.sample_rate * 0.45)
                self.bands[i].set_params(
                    freq_hz=safe_freq,
                    gain_db=p["gain"],
                    q=p["q"],
                    enabled=p["enabled"],
                )
        self.active_preset = key
        self.enabled = (key != "flat")
        return True

    def reset(self) -> None:
        """Reset internal filter states across all bands."""
        for b in self.bands:
            b.reset()

    def process_frame(self, pcm: np.ndarray) -> np.ndarray:
        """Process 1D PCM audio through all 5 EQ stages in cascade."""
        if not self.enabled:
            return pcm

        out = np.asarray(pcm, dtype=np.float32)
        for band in self.bands:
            out = band.process(out)
        return np.clip(out, -1.0, 1.0)

    def get_curve(self, num_points: int = 128) -> Dict[str, Any]:
        """Compute composite frequency response curve in dB across log-spaced frequencies."""
        freqs = np.geomspace(20.0, self.sample_rate * 0.48, num_points)
        total_resp = np.ones(num_points, dtype=complex)
        for band in self.bands:
            total_resp *= band.get_frequency_response(freqs)

        curve_db = 20.0 * np.log10(np.maximum(1e-4, np.abs(total_resp)))
        return {
            "freqs_hz": [round(float(f), 1) for f in freqs],
            "gain_db": [round(float(g), 2) for g in curve_db],
            "bands": [
                {
                    "index": i,
                    "type": b.filter_type,
                    "freq_hz": round(b.freq_hz, 1),
                    "gain_db": round(b.gain_db, 2),
                    "q": round(b.q, 2),
                    "enabled": b.enabled,
                }
                for i, b in enumerate(self.bands)
            ],
            "enabled": self.enabled,
            "preset": self.active_preset,
        }

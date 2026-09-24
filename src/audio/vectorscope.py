"""Polar Lissajous Vectorscope & Stereo Phase Coherence Radar.

Provides:
- PhaseCorrelationAnalyzer: Computes real-time mono compatibility, phase correlation index, and Lissajous orbit coordinates.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class PhaseCorrelationAnalyzer:
    """Analyzes stereo phase coherence, mono compatibility, and generates Lissajous coordinates."""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_orbit_points: int = 32,
        orbit_decimation: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_orbit_points = int(num_orbit_points)
        self.orbit_decimation = orbit_decimation
        self.smoothed_correlation: Optional[float] = None

    def reset(self) -> None:
        """Reset internal smoothed phase correlation tracker."""
        self.smoothed_correlation = None

    def analyze(
        self,
        audio_l: np.ndarray,
        audio_r: Optional[np.ndarray] = None,
        right_pcm: Optional[np.ndarray] = None,
        smooth: bool = True,
    ) -> Dict[str, Any]:
        """Compute phase correlation, mono compatibility, and Lissajous orbit points.

        If audio_r is None (mono input), the left channel is duplicated to both stereo
        channels (L = R) in accordance with broadcast monitoring standards (EBU R128),
        accurately yielding true +1.0 correlation, 100% mono compatibility, and a vertical
        mid-axis Lissajous line without artificial delay distortion.
        """
        if audio_r is None and right_pcm is not None:
            audio_r = right_pcm

        l_arr = np.asarray(audio_l, dtype=np.float32).ravel()
        n = len(l_arr)
        if audio_r is None:
            # Monophonic broadcast alignment: L = R
            r_arr = l_arr.copy()
        else:
            r_arr = np.asarray(audio_r, dtype=np.float32).ravel()

        if n == 0:
            return {
                "phase_correlation": 1.0,
                "mono_compatibility_pct": 100.0,
                "stereo_width": 0.0,
                "status": "Silence / Standby",
                "orbit_x": [],
                "orbit_y": [],
            }

        sum_l2 = float(np.sum(l_arr**2))
        sum_r2 = float(np.sum(r_arr**2))

        # Handle complete digital silence safely
        if sum_l2 < 1e-9 and sum_r2 < 1e-9:
            return {
                "phase_correlation": 1.0,
                "mono_compatibility_pct": 100.0,
                "stereo_width": 0.0,
                "status": "Silence / Standby",
                "orbit_x": [0.0] * self.num_orbit_points,
                "orbit_y": [0.0] * self.num_orbit_points,
            }

        # 1. Pearson Cross-Correlation Coefficient in [-1.0, +1.0]
        dot_lr = float(np.sum(l_arr * r_arr))
        denom = np.sqrt(max(1e-12, sum_l2 * sum_r2))
        raw_corr = float(np.clip(dot_lr / denom, -1.0, 1.0))

        # Temporal smoothing: initialize on first frame or instant update if not smooth
        if not smooth or self.smoothed_correlation is None:
            self.smoothed_correlation = raw_corr
        else:
            self.smoothed_correlation = 0.80 * self.smoothed_correlation + 0.20 * raw_corr
        corr = self.smoothed_correlation

        # 2. Mono Compatibility Rating: 100% at +1.0, 50% at 0.0, 0% at -1.0
        mono_pct = float(np.clip((corr + 1.0) * 50.0, 0.0, 100.0))

        # 3. Stereo Width (Side vs Mid energy ratio via energy identities)
        # sum_mid2 = 0.5 * (sum_l2 + sum_r2 + 2 * dot_lr)
        # sum_side2 = 0.5 * (sum_l2 + sum_r2 - 2 * dot_lr)
        sum_mid2 = max(1e-12, 0.5 * (sum_l2 + sum_r2 + 2.0 * dot_lr))
        sum_side2 = max(1e-12, 0.5 * (sum_l2 + sum_r2 - 2.0 * dot_lr))
        width = float(np.clip(np.sqrt(sum_side2 / sum_mid2), 0.0, 2.0))

        if corr > 0.65:
            status = "Rock Solid Phase (+1.0)"
        elif corr > 0.20:
            status = "Acceptable Phase"
        elif corr > -0.50:
            status = "Stereo Wide (0.0)"
        else:
            status = "Phase Inverted / Cancellation Hazard (-1.0)"

        # 4. 45-degree Rotated Lissajous Orbit Points: (Side, Mid)
        step = max(1, n // self.num_orbit_points)
        sub_l = l_arr[::step][:self.num_orbit_points]
        sub_r = r_arr[::step][:self.num_orbit_points]

        # Vectorized 45-degree rotation: X = (L - R)/sqrt(2), Y = (L + R)/sqrt(2)
        ox = np.clip((sub_l - sub_r) * 0.7071, -1.0, 1.0)
        oy = np.clip((sub_l + sub_r) * 0.7071, -1.0, 1.0)
        orbit_x = np.round(ox, 3).tolist()
        orbit_y = np.round(oy, 3).tolist()

        return {
            "phase_correlation": round(corr, 2),
            "mono_compatibility_pct": round(mono_pct, 1),
            "stereo_width": round(width, 2),
            "status": status,
            "orbit_x": orbit_x,
            "orbit_y": orbit_y,
        }

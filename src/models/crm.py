"""Complex Ratio Masking (CRM) & Phase Recovery Engine.

Provides:
- Complex Ratio Masking: applies complex-valued masks (M_r + j*M_i) to input complex STFT
  spectra to simultaneously optimize magnitude and reconstruct acoustic phase.
- Polar angle phase recovery and bounded hyperbolic tangent mask activation.
- Pure NumPy implementation optimized for sub-millisecond edge inference.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


class ComplexRatioMasker:
    """Applies complex-valued ratio masks to complex STFT frames.

    Given noisy complex spectrum X = X_r + j*X_i and complex mask M = M_r + j*M_i:
        Y = M * X = (M_r * X_r - M_i * X_i) + j * (M_r * X_i + M_i * X_r)
    """

    def __init__(self, mask_bound: float = 2.0):
        self.mask_bound = mask_bound

    def apply_mask(
        self,
        real_x: np.ndarray,
        imag_x: np.ndarray,
        real_m: np.ndarray,
        imag_m: np.ndarray,
        intensity: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply complex ratio mask to real and imaginary spectral components.

        Parameters
        ----------
        real_x : np.ndarray
            Real component of complex STFT spectrum.
        imag_x : np.ndarray
            Imaginary component of complex STFT spectrum.
        real_m : np.ndarray
            Real component of complex mask.
        imag_m : np.ndarray
            Imaginary component of complex mask.
        intensity : float
            Suppression scaling factor in [0.0, 1.0].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (real_y, imag_y) filtered complex spectral components.
        """
        # Bounded mask activation: clamp real component to [-mask_bound, mask_bound]
        # preserving linear unity gain in [0, 1] while saturating large mask outliers
        m_r_bounded = np.clip(real_m, -self.mask_bound, self.mask_bound)
        m_i_bounded = self.mask_bound * np.tanh(imag_m / self.mask_bound)

        # Scale with intensity: blend between unity mask (1 + 0j) and full CRM
        m_r_eff = 1.0 + intensity * (m_r_bounded - 1.0)
        m_i_eff = intensity * m_i_bounded

        # Magnitude-consistent phase alignment: preserves exact magnitude envelope
        mag_m = np.sqrt(m_r_eff**2 + m_i_eff**2) + 1e-9
        scale = np.abs(m_r_eff) / mag_m
        m_r_norm = m_r_eff * scale
        m_i_norm = m_i_eff * scale

        # Complex multiplication: (M_r + j*M_i) * (X_r + j*X_i)
        real_y = m_r_norm * real_x - m_i_norm * imag_x
        imag_y = m_r_norm * imag_x + m_i_norm * real_x

        return real_y.astype(np.float32), imag_y.astype(np.float32)

    def estimate_crm_from_magnitude_mask(
        self,
        mag_mask: np.ndarray,
        real_x: np.ndarray,
        imag_x: np.ndarray,
        phase_correction_factor: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize complex mask components from real gain mask and phase gradients.

        Parameters
        ----------
        mag_mask : np.ndarray
            Standard real gain mask G in [0.0, 1.0].
        real_x : np.ndarray
            Real component of complex spectrum.
        imag_x : np.ndarray
            Imaginary component of complex spectrum.
        phase_correction_factor : float
            Strength of imaginary phase alignment correction.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (real_m, imag_m)
        """
        # Primary real component corresponds directly to magnitude suppression
        real_m = np.clip(mag_mask, 0.0, 1.0)

        # Imaginary component compensates for phase dispersion in transition bands
        eps = 1e-9
        mag = np.sqrt(real_x**2 + imag_x**2) + eps
        phase = np.unwrap(np.arctan2(imag_x, real_x))

        # Phase curvature estimation in transition bands: scales with real_m * (1 - real_m)
        # to ensure deep stopband noise suppression while maintaining phase alignment in speech transitions
        d_phase = np.gradient(phase)
        transition_weight = 4.0 * real_m * (1.0 - real_m)
        imag_m = phase_correction_factor * transition_weight * np.sin(d_phase)

        return real_m.astype(np.float32), imag_m.astype(np.float32)

    def compute_magnitude_and_phase(
        self, real_y: np.ndarray, imag_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute polar magnitude and phase from Cartesian components."""
        mag_y = np.sqrt(real_y**2 + imag_y**2)
        phase_y = np.arctan2(imag_y, real_y)
        return mag_y.astype(np.float32), phase_y.astype(np.float32)

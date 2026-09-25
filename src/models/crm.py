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

    Grounded in Williamson, Wang, & Wang (2016) and Hu et al. DCCRN (2020).
    Given noisy complex spectrum Y = Y_r + j*Y_i and complex mask M = M_r + j*M_i:
        S = M * Y = (M_r * Y_r - M_i * Y_i) + j * (M_r * Y_i + M_i * Y_r)
    """

    def __init__(self, mask_bound: float = 2.0, c_param: float = 1.0):
        self.mask_bound = float(mask_bound)
        self.c_param = float(c_param)

    def compute_cirm(
        self,
        clean_spec: np.ndarray,
        noisy_spec: np.ndarray,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Ideal Complex Ratio Mask (cIRM) from clean and noisy complex spectra.

        Parameters
        ----------
        clean_spec : np.ndarray
            Clean speech complex spectrum S = S_r + j*S_i.
        noisy_spec : np.ndarray
            Noisy speech complex spectrum Y = Y_r + j*Y_i.
        eps : float
            Regularization constant to prevent division-by-zero.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (real_m, imag_m) uncompressed ideal complex ratio mask.
        """
        s_r = np.real(clean_spec)
        s_i = np.imag(clean_spec)
        y_r = np.real(noisy_spec)
        y_i = np.imag(noisy_spec)

        denom = y_r**2 + y_i**2 + eps
        real_m = (y_r * s_r + y_i * s_i) / denom
        imag_m = (y_r * s_i - y_i * s_r) / denom

        # Strict Hermitian conjugate symmetry invariant at DC and Nyquist bins:
        # For real-valued time-domain signals, imaginary component must be strictly zero.
        if imag_m.ndim == 1 and len(imag_m) > 1:
            imag_m[0] = 0.0
            imag_m[-1] = 0.0
        elif imag_m.ndim == 2 and imag_m.shape[0] > 1:
            imag_m[0, :] = 0.0
            imag_m[-1, :] = 0.0

        return real_m.astype(np.float32), imag_m.astype(np.float32)

    def compress_mask(
        self, real_m: np.ndarray, imag_m: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress unconstrained mask components into [-K, K] using Williamson et al. (2016) tanh scaling.

        M_x' = K * tanh(0.5 * C * M_x)
        """
        k = self.mask_bound
        c = self.c_param
        eps = 1e-6
        comp_r = np.clip(k * np.tanh(0.5 * c * real_m), -k + eps, k - eps)
        comp_i = np.clip(k * np.tanh(0.5 * c * imag_m), -k + eps, k - eps)

        if comp_i.ndim == 1 and len(comp_i) > 1:
            comp_i[0] = 0.0
            comp_i[-1] = 0.0
        elif comp_i.ndim == 2 and comp_i.shape[0] > 1:
            comp_i[0, :] = 0.0
            comp_i[-1, :] = 0.0

        return comp_r.astype(np.float32), comp_i.astype(np.float32)

    def decompress_mask(
        self, comp_r: np.ndarray, comp_i: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress tanh-compressed mask back to unconstrained domain.

        M_x = (2 / C) * arctanh(M_x' / K)
        """
        k = self.mask_bound
        c = self.c_param
        eps = 1e-6
        ratio_r = np.clip(comp_r / k, -1.0 + eps, 1.0 - eps)
        ratio_i = np.clip(comp_i / k, -1.0 + eps, 1.0 - eps)

        decomp_r = (2.0 / c) * np.arctanh(ratio_r)
        decomp_i = (2.0 / c) * np.arctanh(ratio_i)

        if decomp_i.ndim == 1 and len(decomp_i) > 1:
            decomp_i[0] = 0.0
            decomp_i[-1] = 0.0
        elif decomp_i.ndim == 2 and decomp_i.shape[0] > 1:
            decomp_i[0, :] = 0.0
            decomp_i[-1, :] = 0.0

        return decomp_r.astype(np.float32), decomp_i.astype(np.float32)

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
        # Bounded mask activation: clamp components to [-mask_bound, mask_bound]
        # preserving exact linear response within bounds while saturating large mask outliers
        m_r_bounded = np.clip(real_m, -self.mask_bound, self.mask_bound)
        m_i_bounded = np.clip(imag_m, -self.mask_bound, self.mask_bound)

        # Scale with intensity: linear blend between unity passthrough (1 + 0j) and full CRM
        m_r_eff = 1.0 + intensity * (m_r_bounded - 1.0)
        m_i_eff = intensity * m_i_bounded

        # Strict Hermitian symmetry constraint at DC (bin 0) and Nyquist (bin -1):
        # Imaginary mask components must be identically zero to prevent non-real DC drift
        if m_i_eff.ndim == 1 and len(m_i_eff) > 1:
            m_i_eff[0] = 0.0
            m_i_eff[-1] = 0.0
        elif m_i_eff.ndim == 2 and m_i_eff.shape[0] > 1:
            m_i_eff[0, :] = 0.0
            m_i_eff[-1, :] = 0.0

        # Exact Cartesian complex multiplication: (M_r + j*M_i) * (X_r + j*X_i)
        # S_r = M_r * X_r - M_i * X_i
        # S_i = M_r * X_i + M_i * X_r
        real_y = m_r_eff * real_x - m_i_eff * imag_x
        imag_y = m_r_eff * imag_x + m_i_eff * real_x

        # Enforce zero imaginary component at DC and Nyquist on synthesized spectrum
        if imag_y.ndim == 1 and len(imag_y) > 1:
            imag_y[0] = 0.0
            imag_y[-1] = 0.0
        elif imag_y.ndim == 2 and imag_y.shape[0] > 1:
            imag_y[0, :] = 0.0
            imag_y[-1, :] = 0.0

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

        # Compute polar angle and local phase gradient
        eps = 1e-9
        mag = np.sqrt(real_x**2 + imag_x**2) + eps
        phase = np.arctan2(imag_x, real_x)

        # Central difference phase gradient wrapped to [-pi, pi]
        d_phase = np.gradient(phase)
        d_phase_wrapped = (d_phase + np.pi) % (2.0 * np.pi) - np.pi

        # Magnitude-weighting: suppress phase adjustments in deep noise notches
        # where phase is purely stochastic and prone to unwrapping jumps
        mag_norm = mag / (np.max(mag) + eps)

        # Transition-band weighting: 4 * G * (1 - G) focuses phase correction in speech transitions
        transition_weight = 4.0 * real_m * (1.0 - real_m)
        imag_m = phase_correction_factor * transition_weight * mag_norm * np.sin(d_phase_wrapped)

        # Strict Hermitian symmetry: DC and Nyquist imaginary components must be zero
        if imag_m.ndim == 1 and len(imag_m) > 1:
            imag_m[0] = 0.0
            imag_m[-1] = 0.0
        elif imag_m.ndim == 2 and imag_m.shape[0] > 1:
            imag_m[0, :] = 0.0
            imag_m[-1, :] = 0.0

        return real_m.astype(np.float32), imag_m.astype(np.float32)

    def compute_magnitude_and_phase(
        self, real_y: np.ndarray, imag_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute polar magnitude and phase from Cartesian components."""
        mag_y = np.sqrt(real_y**2 + imag_y**2)
        phase_y = np.arctan2(imag_y, real_y)
        return mag_y.astype(np.float32), phase_y.astype(np.float32)

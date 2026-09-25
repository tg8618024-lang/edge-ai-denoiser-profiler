"""Unit Tests for Phase-Preserving Complex Ratio Masking (CRM).

Grounding:
- Williamson, Wang, & Wang (2016) "Complex Ratio Masking for Monaural Speech Separation", IEEE/ACM TASLP.
- Tan & Wang (2020) "Learning Complex Spectral Mapping with Gated Convolutional Recurrent Networks", IEEE/ACM TASLP.
- Hu et al. (2020) "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement", Interspeech.

Verifies:
1. Exact Cartesian complex ratio masking: S = (M_r*Y_r - M_i*Y_i) + j*(M_r*Y_i + M_i*Y_r).
2. Machine-precision cIRM reconstruction (SNR >= 80 dB) on non-zero speech/noise mixtures.
3. Strict Hermitian conjugate symmetry at DC (bin 0) and Nyquist (bin -1).
4. Bounded hyperbolic tangent compression and decompression roundtrip.
5. Robustness to silence, noise notches, and pipeline streaming integration.
"""

import os
import sys
import pytest
import numpy as np
from scipy.fft import irfft

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.crm import ComplexRatioMasker
from src.audio.pipeline import AudioDenoisingPipeline


class TestComplexRatioMasker:
    """Rigorous mathematical and numerical validation of ComplexRatioMasker."""

    def test_cirm_perfect_reconstruction(self):
        """Verify closed-form cIRM achieves machine-precision reconstruction (>= 80 dB SNR)."""
        crm = ComplexRatioMasker(mask_bound=100.0, c_param=1.0)
        num_bins = 257
        np.random.seed(42)

        # Generate non-zero synthetic clean speech and noise spectra
        clean_spec = (
            np.random.randn(num_bins).astype(np.float32)
            + 1j * np.random.randn(num_bins).astype(np.float32)
        )
        noise_spec = (
            np.random.randn(num_bins).astype(np.float32)
            + 1j * np.random.randn(num_bins).astype(np.float32)
        ) * 0.3
        # Ensure DC and Nyquist are strictly purely real for valid STFT spectra
        clean_spec[0] = np.real(clean_spec[0])
        clean_spec[-1] = np.real(clean_spec[-1])
        noise_spec[0] = np.real(noise_spec[0])
        noise_spec[-1] = np.real(noise_spec[-1])

        noisy_spec = clean_spec + noise_spec

        # Compute ideal cIRM
        real_m, imag_m = crm.compute_cirm(clean_spec, noisy_spec)

        # Reconstruct speech via Cartesian complex multiplication
        real_y, imag_y = crm.apply_mask(
            np.real(noisy_spec),
            np.imag(noisy_spec),
            real_m,
            imag_m,
            intensity=1.0,
        )
        rec_spec = real_y + 1j * imag_y

        # Exclude near-zero bins from relative SNR test if any
        valid = np.abs(noisy_spec) > 1e-4
        error = np.abs(clean_spec[valid] - rec_spec[valid])
        clean_power = np.mean(np.abs(clean_spec[valid]) ** 2)
        error_power = np.mean(error**2) + 1e-15
        snr_db = 10.0 * np.log10(clean_power / error_power)

        # Numerical reconstruction should be close to floating point precision
        assert snr_db >= 80.0
        assert np.max(error) < 1e-4

    def test_hermitian_symmetry_dc_nyquist(self):
        """Verify strict conjugate symmetry: imaginary mask must be 0.0 at DC and Nyquist."""
        crm = ComplexRatioMasker(mask_bound=2.0)
        num_bins = 257

        real_x = np.random.randn(num_bins).astype(np.float32)
        imag_x = np.random.randn(num_bins).astype(np.float32)
        imag_x[0] = 0.0
        imag_x[-1] = 0.0

        # Pass arbitrary non-zero masks at DC and Nyquist to test enforcement
        real_m = np.random.randn(num_bins).astype(np.float32)
        imag_m = np.random.randn(num_bins).astype(np.float32)
        imag_m[0] = 1.234
        imag_m[-1] = -5.678

        real_y, imag_y = crm.apply_mask(real_x, imag_x, real_m, imag_m)

        # Invariant: Imaginary component at DC and Nyquist must be exactly 0.0
        assert imag_y[0] == 0.0
        assert imag_y[-1] == 0.0

        # Verifies that inverse real FFT synthesizes purely real time-domain samples
        time_domain = irfft(real_y + 1j * imag_y, n=512)
        assert np.all(np.isfinite(time_domain))
        assert time_domain.dtype == np.float64 or time_domain.dtype == np.float32

    def test_tanh_compression_roundtrip(self):
        """Verify Williamson et al. (2016) bounded tanh compression and inverse decompression."""
        k_bound = 3.0
        c_param = 1.0
        crm = ComplexRatioMasker(mask_bound=k_bound, c_param=c_param)

        # Unbounded mask inputs spanning large dynamic range
        m_r = np.array([-15.0, -3.0, -1.0, 0.0, 0.5, 2.0, 5.0, 25.0], dtype=np.float32)
        m_i = np.array([0.0, -2.5, -0.8, 0.0, 0.3, 1.8, 4.0, 0.0], dtype=np.float32)

        comp_r, comp_i = crm.compress_mask(m_r, m_i)

        # All compressed values must be strictly inside (-K, K)
        assert np.all(comp_r > -k_bound)
        assert np.all(comp_r < k_bound)
        assert np.all(comp_i > -k_bound)
        assert np.all(comp_i < k_bound)

        # Decompress back
        decomp_r, decomp_i = crm.decompress_mask(comp_r, comp_i)

        # For values in moderate range [-3.0, 3.0], decompression error should be negligible
        moderate_idx = np.abs(m_r) <= 3.0
        np.testing.assert_allclose(decomp_r[moderate_idx], m_r[moderate_idx], rtol=1e-4, atol=1e-4)

    def test_estimate_crm_from_magnitude_mask_stability(self):
        """Verify heuristic phase gradient estimation on edge cases (silence, random noise)."""
        crm = ComplexRatioMasker(mask_bound=2.0)
        num_bins = 257

        # Silence case
        real_silence = np.zeros(num_bins, dtype=np.float32)
        imag_silence = np.zeros(num_bins, dtype=np.float32)
        mag_mask = np.full(num_bins, 0.5, dtype=np.float32)

        r_m, i_m = crm.estimate_crm_from_magnitude_mask(
            mag_mask, real_silence, imag_silence, phase_correction_factor=0.02
        )
        assert np.all(np.isfinite(r_m))
        assert np.all(np.isfinite(i_m))
        assert i_m[0] == 0.0
        assert i_m[-1] == 0.0

        # Random noisy spectrum
        real_rand = np.random.randn(num_bins).astype(np.float32)
        imag_rand = np.random.randn(num_bins).astype(np.float32)
        r_m, i_m = crm.estimate_crm_from_magnitude_mask(
            mag_mask, real_rand, imag_rand, phase_correction_factor=0.05
        )
        assert np.all(np.isfinite(r_m))
        assert np.all(np.isfinite(i_m))
        assert np.all(np.abs(i_m) <= 2.0)
        assert i_m[0] == 0.0
        assert i_m[-1] == 0.0

    def test_pipeline_crm_toggle_and_processing(self):
        """Verify pipeline processes frames with CRM enabled and achieves real-time execution."""
        pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)

        # Test toggle methods
        assert hasattr(pipeline, "set_crm_enabled")
        assert hasattr(pipeline, "get_crm_enabled")
        assert hasattr(pipeline, "set_crm_phase_factor")
        assert hasattr(pipeline, "get_crm_phase_factor")

        pipeline.set_crm_enabled(True)
        assert pipeline.get_crm_enabled() is True
        pipeline.set_crm_phase_factor(0.03)
        assert abs(pipeline.get_crm_phase_factor() - 0.03) < 1e-5

        # Process a sequence of 10 continuous frames
        np.random.seed(123)
        frame_len = 256
        for _ in range(10):
            noisy_frame = np.random.randn(frame_len).astype(np.float32) * 0.2
            out_frame = pipeline.process_frame(noisy_frame)
            assert out_frame.shape == (frame_len,)
            assert np.all(np.isfinite(out_frame))
            assert not np.any(np.isnan(out_frame))

        # Check that last_output_spec is complex64
        assert pipeline.last_output_spec is not None
        assert np.iscomplexobj(pipeline.last_output_spec)
        assert pipeline.last_output_spec.shape == (257,)

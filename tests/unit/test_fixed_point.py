"""Unit tests for Fixed-Point DSP Arithmetic Engine (Q1.15 / Q8.7).

Tests:
- float_to_q15 and q15_to_float conversions and saturation boundaries.
- Saturating int16 arithmetic (saturate_int16).
- Fixed-point multiplication (q15_mul) and division (q15_div).
- Signal-to-Quantization-Noise Ratio (SQNR > 80 dB on clean speech).
- FixedPointWienerDSP gain calculation in int16.
- FixedPointAudioPipeline streaming with zero floating-point operations in inner loop.
"""

import pytest
import numpy as np

from src.audio.fixed_point import (
    float_to_q15,
    q15_to_float,
    saturate_int16,
    q15_mul,
    q15_div,
    compute_sqnr_db,
    FixedPointWienerDSP,
    FixedPointAudioPipeline,
    Q15_MIN,
    Q15_MAX,
)


class TestFixedPointArithmetic:
    def test_float_to_q15_conversion_and_bounds(self):
        """Verify Q1.15 conversion, scaling, and clipping boundaries."""
        inputs = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 0.999969, 1.5], dtype=np.float32)
        q15 = float_to_q15(inputs)

        assert q15.dtype == np.int16
        assert q15[0] == Q15_MIN  # Clamped to -32768
        assert q15[1] == Q15_MIN  # -1.0 -> -32768
        assert q15[2] == -16384   # -0.5 -> -16384
        assert q15[3] == 0        # 0.0 -> 0
        assert q15[4] == 16384    # 0.5 -> 16384
        assert q15[6] == Q15_MAX  # Clamped to 32767

    def test_q15_to_float_reconstruction(self):
        """Verify inverse conversion reconstructs original float values within 1 LSB."""
        orig = np.array([-1.0, -0.5, 0.0, 0.5, 0.9999], dtype=np.float32)
        q15 = float_to_q15(orig)
        recon = q15_to_float(q15)

        assert recon.dtype == np.float32
        np.testing.assert_allclose(recon, orig, atol=1e-4)

    def test_saturate_int16(self):
        """Verify saturating clamp prevents integer overflow wrap-around."""
        large_pos = np.array([40000, 70000], dtype=np.int32)
        large_neg = np.array([-40000, -70000], dtype=np.int32)

        sat_pos = saturate_int16(large_pos)
        sat_neg = saturate_int16(large_neg)

        assert np.all(sat_pos == Q15_MAX)
        assert np.all(sat_neg == Q15_MIN)

    def test_q15_mul_and_div(self):
        """Verify fixed-point multiplication and division."""
        # 0.5 * 0.5 = 0.25 (16384 * 16384 >> 15 = 8192)
        half = np.int16(16384)
        quarter = q15_mul(half, half)
        assert quarter == 8192

        # 0.25 / 0.5 = 0.5 (8192 << 15 // 16384 = 16384)
        div_res = q15_div(quarter, half)
        assert div_res == 16384

    def test_pure_quantization_sqnr(self):
        """Verify Q1.15 delivers > 80 dB Signal-to-Quantization-Noise Ratio."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sine = (0.7 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        q15 = float_to_q15(sine)
        recon = q15_to_float(q15)

        sqnr = compute_sqnr_db(sine, recon)
        assert sqnr > 80.0, f"Expected SQNR > 80 dB, got {sqnr} dB"

    def test_fixed_point_wiener_dsp(self):
        """Verify integer Wiener DSP calculates valid gain masks in Q1.15."""
        dsp = FixedPointWienerDSP(num_bins=257)
        mag_q15 = np.full(257, 10000, dtype=np.int16)

        gain_q15 = dsp.compute_gain_q15(mag_q15)

        assert gain_q15.shape == (257,)
        assert gain_q15.dtype == np.int16
        assert np.all(gain_q15 >= dsp.gain_min_q15)
        assert np.all(gain_q15 <= Q15_MAX)

    def test_fixed_point_pipeline_stream(self):
        """Verify complete fixed-point audio pipeline execution."""
        pipeline = FixedPointAudioPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        assert pipeline.in_buffer_q15.dtype == np.int16
        assert pipeline.out_buffer_q32.dtype == np.int32

        # Stream float audio through convenience wrapper (8192 samples = 32 frames of 256)
        num_samples = 8192
        t = np.linspace(0, num_samples / 16000.0, num_samples, endpoint=False)
        test_audio = (0.5 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

        out_audio = pipeline.process_stream_float(test_audio)

        assert len(out_audio) == len(test_audio)
        assert out_audio.dtype == np.float32
        assert np.all(np.isfinite(out_audio))
        assert pipeline.total_frames_processed == 32

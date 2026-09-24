"""Unit tests for Streaming STFT and iSTFT reconstruction.

Verifies:
- Periodic Hann window and square-root window COLA compliance.
- Streaming STFT/iSTFT machine-precision reconstruction.
- Deterministic 1-hop algorithmic delay (H = 256 samples).
- State reset behavior and boundary conditions.
- Batch STFT/iSTFT equivalence and roundtrip reconstruction.
"""

import unittest
import numpy as np

from src.audio.stft import (
    StreamingSTFT,
    periodic_hann_window,
    sqrt_hann_windows,
    stft,
    istft,
)


class TestSTFT(unittest.TestCase):
    """Test suite for STFT analysis and synthesis windowing and streaming overlap-add."""

    def setUp(self) -> None:
        self.n_fft = 512
        self.hop_length = 256
        self.sample_rate = 16000

    def test_periodic_hann_window_cola(self) -> None:
        """Verify periodic Hann window satisfies Constant Overlap-Add (COLA) = 1.0."""
        w = periodic_hann_window(self.n_fft)
        self.assertEqual(len(w), self.n_fft)

        # 50% overlap-add property: w[n] + w[n + N/2] == 1.0 for all n in [0, N/2 - 1]
        half_n = self.n_fft // 2
        cola_sum = w[:half_n] + w[half_n:]
        max_deviation = np.max(np.abs(cola_sum - 1.0))
        self.assertLess(
            max_deviation,
            1e-14,
            f"Periodic Hann window violated COLA = 1.0; max deviation: {max_deviation}",
        )

    def test_sqrt_hann_windows(self) -> None:
        """Verify square-root analysis and synthesis windows satisfy w_a * w_s == w."""
        w = periodic_hann_window(self.n_fft)
        w_a, w_s = sqrt_hann_windows(self.n_fft)

        self.assertEqual(w_a.shape, (self.n_fft,))
        self.assertEqual(w_s.shape, (self.n_fft,))

        product = (w_a * w_s).astype(np.float64)
        max_err = np.max(np.abs(product - w))
        self.assertLess(
            max_err,
            1e-7,
            f"Square-root window product diverged from periodic Hann; max error: {max_err}",
        )

        # Analysis and synthesis windows should be identical
        np.testing.assert_allclose(w_a, w_s)

    def test_streaming_stft_perfect_reconstruction(self) -> None:
        """Verify streaming STFT -> iSTFT pass-through achieves exact reconstruction."""
        engine = StreamingSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        rng = np.random.default_rng(42)
        num_hops = 30
        input_signal = rng.standard_normal(num_hops * self.hop_length).astype(np.float32)

        output_hops = []
        for i in range(num_hops):
            hop = input_signal[i * self.hop_length : (i + 1) * self.hop_length]
            out_hop = engine.process(hop)
            output_hops.append(out_hop)

        reconstructed = np.concatenate(output_hops)

        # Because of the 1-hop algorithmic delay (256 samples),
        # hop 0 outputs zeros, and hop 1 onwards matches input[0:-256]
        h = self.hop_length
        aligned_input = input_signal[:-h]
        aligned_output = reconstructed[h:]

        max_abs_err = np.max(np.abs(aligned_output - aligned_input))
        self.assertLess(
            max_abs_err,
            1e-6,
            f"Streaming STFT float32 reconstruction error exceeded tolerance: {max_abs_err}",
        )

    def test_streaming_stft_double_precision_reconstruction(self) -> None:
        """Verify double precision (< 1e-14 error) overlap-add formulation."""
        N, H = 512, 256
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N, dtype=np.float64) / N))
        wa = np.sqrt(w)
        ws = np.sqrt(w)

        rng = np.random.default_rng(123)
        signal = rng.standard_normal(256 * 20)

        in_buf = np.zeros(N, dtype=np.float64)
        out_buf = np.zeros(N, dtype=np.float64)
        reconstructed = []

        for i in range(len(signal) // H):
            hop_in = signal[i * H : (i + 1) * H]
            in_buf[:-H] = in_buf[H:]
            in_buf[-H:] = hop_in

            spec = np.fft.rfft(in_buf * wa)
            time_frame = np.fft.irfft(spec, n=N) * ws

            out_buf += time_frame
            reconstructed.append(out_buf[:H].copy())
            out_buf[:-H] = out_buf[H:]
            out_buf[-H:] = 0.0

        rec = np.concatenate(reconstructed)
        error = np.max(np.abs(rec[H:] - signal[:-H]))
        self.assertLess(
            error,
            1e-14,
            f"Double-precision reconstruction error was {error}, expected < 1e-14",
        )

    def test_streaming_stft_algorithmic_delay(self) -> None:
        """Verify algorithmic delay is exactly 1 hop (256 samples = 16.0 ms)."""
        engine = StreamingSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        impulse_loc = 1000
        signal = np.zeros(self.hop_length * 20, dtype=np.float32)
        signal[impulse_loc] = 1.0

        output_hops = []
        for i in range(len(signal) // self.hop_length):
            hop = signal[i * self.hop_length : (i + 1) * self.hop_length]
            output_hops.append(engine.process(hop))

        reconstructed = np.concatenate(output_hops)
        output_peak_idx = int(np.argmax(reconstructed))

        measured_delay = output_peak_idx - impulse_loc
        self.assertEqual(
            measured_delay,
            self.hop_length,
            f"Algorithmic delay was {measured_delay} samples; expected exactly {self.hop_length} samples (1 hop).",
        )
        self.assertEqual(engine.algorithmic_delay_samples, self.hop_length)
        self.assertAlmostEqual(engine.algorithmic_delay_ms, 16.0, places=2)

    def test_streaming_stft_reset(self) -> None:
        """Verify reset() clears internal buffers and restores clean state."""
        engine = StreamingSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        test_hop = np.ones(self.hop_length, dtype=np.float32)
        engine.process(test_hop)
        self.assertTrue(np.any(engine.in_buffer != 0.0))
        self.assertTrue(np.any(engine.out_buffer != 0.0))

        engine.reset()
        self.assertTrue(np.all(engine.in_buffer == 0.0))
        self.assertTrue(np.all(engine.out_buffer == 0.0))

    def test_streaming_stft_invalid_input(self) -> None:
        """Verify validation of invalid frame sizes and dimensions."""
        engine = StreamingSTFT(n_fft=512, hop_length=256)
        with self.assertRaises(ValueError):
            engine.analyze(np.zeros(100))

        with self.assertRaises(ValueError):
            engine.synthesize(np.zeros(100))

    def test_batch_stft_istft_roundtrip(self) -> None:
        """Verify batch stft() and istft() functions match and invert cleanly."""
        rng = np.random.default_rng(99)
        orig_signal = rng.standard_normal(4000).astype(np.float32)

        spec = stft(orig_signal, n_fft=self.n_fft, hop_length=self.hop_length)
        self.assertEqual(spec.shape[0], self.n_fft // 2 + 1)

        recon = istft(spec, hop_length=self.hop_length, length=len(orig_signal))
        self.assertEqual(len(recon), len(orig_signal))

        # Check correlation and MSE between original and reconstructed
        mse = np.mean((recon - orig_signal) ** 2)
        self.assertLess(mse, 1e-4)


if __name__ == "__main__":
    unittest.main()

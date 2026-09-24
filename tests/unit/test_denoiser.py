"""Unit tests for Denoising Models, Dataset Generation, Ring Buffering, and Pipeline.

Verifies:
- Synthetic speech physics (glottal excitation, formants, prosody).
- 4 calibrated noise generators (White, Pink, Drone hum, RF static).
- Mixture synthesis and target broadband SNR calibration.
- GRUMaskNet forward pass, mask bounds [0, 1], parameter counts, and save/load.
- Decision-Directed Wiener spectral filtering and noise tracking.
- AudioRingBuffer circular buffering and AudioChunker framing.
- 3-stage latency profiler instrumentation hooks.
- Broadband SNR improvement >= 10.0 dB across benchmark noise types on 0 dB mixtures.
"""

import os
import tempfile
import unittest
import numpy as np

from src.audio.dataset import (
    generate_synthetic_speech,
    generate_noise,
    create_mixture,
    calculate_snr,
    compute_snr_gain,
    SyntheticAudioGenerator,
)
from src.audio.stream import AudioRingBuffer, AudioChunker, AudioStreamer
from src.models.denoiser import (
    GRUMaskNet,
    DecisionDirectedWienerFilter,
    HybridDenoiser,
)
from src.audio.pipeline import AudioDenoisingPipeline


class MockProfiler:
    """Mock profiler tracking stage calls for latency instrumentation verification."""

    def __init__(self) -> None:
        self.stage_starts: list[str] = []
        self.stage_ends: list[str] = []

    def start_stage(self, stage_name: str) -> None:
        self.stage_starts.append(stage_name)

    def end_stage(self, stage_name: str) -> None:
        self.stage_ends.append(stage_name)


class TestDenoiser(unittest.TestCase):
    """Comprehensive test suite for Milestone 1 Audio Core & Denoising Pipeline."""

    def setUp(self) -> None:
        self.sample_rate = 16000
        self.duration_sec = 3.0
        self.n_fft = 512
        self.hop_length = 256
        self.num_bins = self.n_fft // 2 + 1

    def test_synthetic_speech_generation(self) -> None:
        """Verify synthetic speech waveform properties, bounds, and absence of NaNs."""
        speech = generate_synthetic_speech(
            duration_sec=self.duration_sec,
            sample_rate=self.sample_rate,
            seed=42,
        )
        expected_samples = int(self.duration_sec * self.sample_rate)
        self.assertEqual(len(speech), expected_samples)
        self.assertFalse(np.any(np.isnan(speech)))
        self.assertFalse(np.any(np.isinf(speech)))

        # Speech should be bounded within [-1.0, 1.0] without clipping
        peak = np.max(np.abs(speech))
        self.assertLessEqual(peak, 0.90)
        self.assertGreater(peak, 0.50)

        # Check that energy is non-zero
        rms = np.sqrt(np.mean(speech**2))
        self.assertGreater(rms, 0.02)

    def test_noise_generators(self) -> None:
        """Verify 4 calibrated noise generators produce zero-mean, unit-variance signals."""
        num_samples = 16000
        noise_types = ["white", "pink", "drone", "rf_static"]

        for ntype in noise_types:
            noise = generate_noise(ntype, num_samples, sample_rate=self.sample_rate, seed=123)
            self.assertEqual(len(noise), num_samples)
            self.assertFalse(np.any(np.isnan(noise)))
            self.assertFalse(np.any(np.isinf(noise)))

            # Mean should be near 0, standard deviation near 1.0
            mean = float(np.mean(noise))
            std = float(np.std(noise))
            self.assertAlmostEqual(mean, 0.0, delta=0.1)
            self.assertAlmostEqual(std, 1.0, delta=0.1)

        # Invalid noise type should raise ValueError
        with self.assertRaises(ValueError):
            generate_noise("invalid_noise", num_samples)

    def test_create_mixture_and_snr_calibration(self) -> None:
        """Verify mixture generator produces exact target input SNR."""
        speech = generate_synthetic_speech(duration_sec=2.0, sample_rate=self.sample_rate, seed=10)
        noise = generate_noise("white", len(speech), sample_rate=self.sample_rate, seed=20)

        for target_snr in [0.0, 5.0, 10.0]:
            mix, s_clean, n_scaled = create_mixture(speech, noise, target_snr_db=target_snr)
            self.assertEqual(len(mix), len(speech))

            measured_snr = calculate_snr(s_clean, mix, delay_samples=0)
            self.assertAlmostEqual(
                measured_snr,
                target_snr,
                places=1,
                msg=f"Mixture target SNR was {target_snr} dB, but measured {measured_snr:.2f} dB",
            )

    def test_broadband_snr_formula(self) -> None:
        """Verify broadband SNR metric against known theoretical values."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(10000)

        # SNR with identical signal is infinite (capped at 100 dB)
        snr_clean = calculate_snr(signal, signal)
        self.assertGreaterEqual(snr_clean, 99.0)

        # Add noise with known power: SNR = 10 dB
        noise = rng.standard_normal(10000)
        scale = np.sqrt(np.mean(signal**2) / (np.mean(noise**2) * 10.0))
        noisy_10db = signal + noise * scale
        measured_snr = calculate_snr(signal, noisy_10db)
        self.assertAlmostEqual(measured_snr, 10.0, places=1)

        # Delay compensation check: 256 samples
        delayed_signal = np.pad(signal, (256, 0))[: len(signal)]
        snr_delayed = calculate_snr(signal, delayed_signal, delay_samples=256)
        self.assertGreaterEqual(snr_delayed, 99.0)

    def test_gru_masknet_inference_and_bounds(self) -> None:
        """Verify GRUMaskNet output dimensions, mask values in [0, 1], and parameters."""
        model = GRUMaskNet(input_dim=self.num_bins, hidden_dim=64, output_dim=self.num_bins)

        # Verify parameter count is ~58k and memory footprint is positive
        self.assertGreater(model.param_count, 40000)
        self.assertGreater(model.memory_footprint_bytes, 100000)

        # Single frame forward pass
        log_mag = np.zeros(self.num_bins, dtype=np.float32)
        mask = model.forward_frame(log_mag)

        self.assertEqual(mask.shape, (self.num_bins,))
        self.assertTrue(np.all(mask >= 0.0))
        self.assertTrue(np.all(mask <= 1.0))
        self.assertFalse(np.any(np.isnan(mask)))

        # Reset state clears recurrent hidden state
        model.reset_state()
        self.assertTrue(np.all(model.hidden_state == 0.0))

    def test_decision_directed_wiener_filter(self) -> None:
        """Verify DecisionDirectedWienerFilter gain calculation and floor clamping."""
        wf = DecisionDirectedWienerFilter(num_bins=self.num_bins, gain_min=0.005)

        mag = np.ones(self.num_bins, dtype=np.float32) * 2.0
        gain = wf.compute_gain(mag)

        self.assertEqual(gain.shape, (self.num_bins,))
        self.assertTrue(np.all(gain >= 0.005))
        self.assertTrue(np.all(gain <= 1.0))

        wf.reset()
        self.assertEqual(len(wf.mag_history), 0)
        self.assertTrue(np.all(wf.prev_clean_mag == 0.0))

    def test_hybrid_denoiser_modes(self) -> None:
        """Verify HybridDenoiser supports 'neural', 'wiener', and 'hybrid' modes."""
        denoiser = HybridDenoiser(num_bins=self.num_bins)

        spec = np.ones(self.num_bins, dtype=np.complex64)

        for mode in ["neural", "wiener", "hybrid"]:
            denoiser.set_mode(mode)
            gain = denoiser.compute_gain(spec)
            self.assertEqual(gain.shape, (self.num_bins,))
            self.assertTrue(np.all(gain >= 0.0))
            self.assertTrue(np.all(gain <= 1.0))

        with self.assertRaises(ValueError):
            denoiser.set_mode("unknown_mode")

    def test_denoiser_weights_save_load(self) -> None:
        """Verify parameter saving and loading in NumPy .npz format."""
        model_a = GRUMaskNet(input_dim=self.num_bins, hidden_dim=64, output_dim=self.num_bins)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_weights.npz")
            model_a.save_weights(save_path)
            self.assertTrue(os.path.isfile(save_path))

            model_b = GRUMaskNet(input_dim=self.num_bins, hidden_dim=64, output_dim=self.num_bins)
            model_b.load_weights(save_path)

            # Assert all weights match exactly
            weights_a = model_a.get_weights()
            weights_b = model_b.get_weights()
            for key in weights_a:
                np.testing.assert_allclose(weights_a[key], weights_b[key])

    def test_audio_ring_buffer(self) -> None:
        """Verify AudioRingBuffer circular write, read, peek, and overwrite behavior."""
        rb = AudioRingBuffer(capacity=100)
        self.assertEqual(len(rb), 0)
        self.assertEqual(rb.available_write, 100)

        # Write 40 samples
        data1 = np.arange(40, dtype=np.float32)
        written = rb.write(data1)
        self.assertEqual(written, 40)
        self.assertEqual(len(rb), 40)
        self.assertEqual(rb.available_write, 60)

        # Peek without advancing
        peeked = rb.peek(20)
        self.assertEqual(len(peeked), 20)
        np.testing.assert_allclose(peeked, data1[:20])
        self.assertEqual(len(rb), 40)

        # Read 30 samples
        read1 = rb.read(30)
        self.assertEqual(len(read1), 30)
        np.testing.assert_allclose(read1, data1[:30])
        self.assertEqual(len(rb), 10)

        # Overflow write with overwrite
        data2 = np.ones(120, dtype=np.float32) * 5.0
        rb.write(data2, allow_overwrite=True)
        self.assertEqual(len(rb), 100)

        # Clear
        rb.clear()
        self.assertEqual(len(rb), 0)

    def test_audio_chunker(self) -> None:
        """Verify AudioChunker partitions variable block inputs into fixed chunks."""
        chunker = AudioChunker(chunk_size=256)

        # Push 600 samples -> should yield two 256-sample chunks (512 total), 88 remainder
        data = np.ones(600, dtype=np.float32)
        chunks = chunker.push(data)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 256)
        self.assertEqual(len(chunks[1]), 256)
        self.assertEqual(len(chunker.remainder), 88)

        # Flush remainder
        tail = chunker.flush()
        self.assertIsNotNone(tail)
        self.assertEqual(len(tail), 256)
        self.assertEqual(len(chunker.remainder), 0)

    def test_pipeline_profiler_hooks(self) -> None:
        """Verify AudioDenoisingPipeline invokes profiler hooks for all 3 latency stages."""
        pipeline = AudioDenoisingPipeline(n_fft=self.n_fft, hop_length=self.hop_length)
        profiler = MockProfiler()

        frame = np.random.randn(self.hop_length).astype(np.float32)
        pipeline.process_frame(frame, profiler=profiler)

        # Stages must match interface contract: pre_processing, tensor_compute, output_synthesis
        expected_stages = ["pre_processing", "tensor_compute", "output_synthesis"]
        self.assertEqual(profiler.stage_starts, expected_stages)
        self.assertEqual(profiler.stage_ends, expected_stages)
        self.assertEqual(pipeline.total_frames_processed, 1)

    def test_snr_improvement_ge_10db_benchmark(self) -> None:
        """Verify denoising pipeline achieves >= 10.0 dB SNR improvement on 0 dB mixtures."""
        generator = SyntheticAudioGenerator(sample_rate=self.sample_rate)
        benchmark_suite = generator.generate_benchmark_suite(
            duration_sec=3.0,
            target_snr_db=0.0,
            seed=42,
        )

        pipeline = AudioDenoisingPipeline(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            denoiser_mode="neural",
        )

        for noise_type, data in benchmark_suite.items():
            clean = data["clean"]
            mixture = data["mixture"]

            denoised = pipeline.process_stream(mixture, reset_before=True)

            snr_in, snr_out, snr_gain = compute_snr_gain(
                clean,
                mixture,
                denoised,
                delay_samples=pipeline.algorithmic_delay_samples,
            )

            # Assert input was 0 dB (+/- 0.1 dB)
            self.assertAlmostEqual(snr_in, 0.0, delta=0.1)
            print(f"[{noise_type}] In: {snr_in:.2f} dB, Out: {snr_out:.2f} dB, Gain: {snr_gain:.2f} dB")

            # Assert >= 10 dB SNR improvement
            self.assertGreaterEqual(
                snr_gain,
                10.0,
                f"Benchmark noise '{noise_type}' failed acceptance criteria: "
                f"expected >= 10.0 dB SNR gain, got {snr_gain:.2f} dB (In: {snr_in:.2f} dB, Out: {snr_out:.2f} dB)",
            )

            # Verify no clipping occurred in output
            max_peak = np.max(np.abs(denoised))
            self.assertLessEqual(max_peak, 1.0, f"Denoised output clipped at {max_peak}")


if __name__ == "__main__":
    unittest.main()

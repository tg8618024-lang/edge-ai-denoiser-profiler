"""Unit tests for Dual-Model Concurrent A/B Engine and Crossfader.

Tests:
- DualModelPipeline initialization and component integrity.
- Concurrent frame processing of Model A (GRUMaskNet) and Model B (Wiener DSP).
- Linear audio crossfading verification across alpha in [0.0, 1.0].
- Independent 3-stage latency isolation for both models.
- Budget headroom compliance (< 20.0 ms per frame).
- State reset and precision delegation.
"""

import pytest
import numpy as np

from src.audio.dual_pipeline import DualModelPipeline, DualPipelineResult, ModelMetrics


class TestDualModelPipeline:
    def test_initialization(self):
        """Verify initialization of analysis STFT, synthesis STFTs, and both models."""
        pipeline = DualModelPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        assert pipeline.n_fft == 512
        assert pipeline.hop_length == 256
        assert pipeline.sample_rate == 16000
        assert pipeline.crossfade_alpha == 0.5
        assert pipeline.frame_budget_ms == 16.0
        assert pipeline.model_a is not None
        assert pipeline.model_b is not None

    def test_process_frame_shape_and_types(self):
        """Verify process_frame outputs valid audio shapes and data structures."""
        pipeline = DualModelPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        in_pcm = (np.random.randn(256) * 0.3).astype(np.float32)

        res = pipeline.process_frame(in_pcm, crossfade_alpha=0.5)

        assert isinstance(res, DualPipelineResult)
        assert res.audio_a.shape == (256,)
        assert res.audio_b.shape == (256,)
        assert res.audio_mix.shape == (256,)
        assert res.audio_mix.dtype == np.float32
        assert np.all(np.isfinite(res.audio_mix))
        assert np.all(res.audio_mix >= -1.0) and np.all(res.audio_mix <= 1.0)
        assert res.crossfade_alpha == 0.5

    def test_crossfade_extremes(self):
        """Verify crossfade at alpha=0.0 matches Model A and alpha=1.0 matches Model B."""
        pipeline = DualModelPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        in_pcm = (np.random.randn(256) * 0.2).astype(np.float32)

        # Alpha = 0.0 -> 100% Model A
        res_a = pipeline.process_frame(in_pcm, crossfade_alpha=0.0)
        np.testing.assert_allclose(res_a.audio_mix, res_a.audio_a, atol=1e-5)

        # Alpha = 1.0 -> 100% Model B
        res_b = pipeline.process_frame(in_pcm, crossfade_alpha=1.0)
        np.testing.assert_allclose(res_b.audio_mix, res_b.audio_b, atol=1e-5)

        # Alpha = 0.5 -> Exactly midpoint blend
        res_mid = pipeline.process_frame(in_pcm, crossfade_alpha=0.5)
        expected_blend = 0.5 * res_mid.audio_a + 0.5 * res_mid.audio_b
        np.testing.assert_allclose(res_mid.audio_mix, expected_blend, atol=1e-5)

    def test_independent_latency_telemetry(self):
        """Verify isolated stage latency tracking for both models."""
        pipeline = DualModelPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        in_pcm = (np.random.randn(256) * 0.2).astype(np.float32)

        res = pipeline.process_frame(in_pcm)

        assert isinstance(res.telemetry_a, ModelMetrics)
        assert isinstance(res.telemetry_b, ModelMetrics)

        # Model A should include neural weights footprint (> 0)
        assert res.telemetry_a.model_size_bytes > 0
        assert res.telemetry_a.total_latency_ms > 0.0

        # Model B is analytical DSP (size = 0)
        assert res.telemetry_b.model_size_bytes == 0
        assert res.telemetry_b.total_latency_ms > 0.0

        # Concurrent total execution must remain well within real-time frame budget (16.0 ms)
        assert res.concurrent_latency_ms < 16.0
        assert res.telemetry_a.headroom_pct > 80.0
        assert res.telemetry_b.headroom_pct > 80.0

    def test_precision_switching_and_reset(self):
        """Verify precision propagation to Model A and state reset."""
        pipeline = DualModelPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        pipeline.set_precision("INT8")
        assert pipeline.model_a.get_precision() == "INT8"

        # Process frames
        for _ in range(5):
            in_pcm = (np.random.randn(256) * 0.1).astype(np.float32)
            pipeline.process_frame(in_pcm)

        assert pipeline.total_frames_processed == 5
        pipeline.reset()
        assert pipeline.total_frames_processed == 0

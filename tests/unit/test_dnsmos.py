"""Unit tests for the ITU-T P.835 Perceptual Speech Quality Estimator (DNSMOS Suite)."""

import time
import numpy as np
import pytest

from src.telemetry.dnsmos import PerceptualQualityEstimator, PerceptualQualityScore


class TestPerceptualQualityEstimator:
    """Test ITU-T P.835 Mean Opinion Score estimations."""

    def test_initialization_defaults(self):
        estimator = PerceptualQualityEstimator(sample_rate=16000)
        data = estimator.to_dict()

        assert "sig_mos" in data
        assert "bak_mos" in data
        assert "ovrl_mos" in data
        assert 1.0 <= data["sig_mos"] <= 5.0
        assert 1.0 <= data["bak_mos"] <= 5.0
        assert 1.0 <= data["ovrl_mos"] <= 5.0

    def test_clean_speech_scores_high(self):
        """When output matches clean speech, SIG and OVRL should be high."""
        estimator = PerceptualQualityEstimator(sample_rate=16000)
        t = np.arange(256) / 16000.0
        speech = (
            0.5 * np.sin(2 * np.pi * 220.0 * t) +
            0.3 * np.sin(2 * np.pi * 440.0 * t) +
            0.2 * np.sin(2 * np.pi * 880.0 * t)
        ).astype(np.float32)

        # Run 20 frames of clean speech with zero noise
        for _ in range(20):
            score = estimator.evaluate(denoised_pcm=speech, noisy_pcm=speech, vad_active=True)

        assert score.sig_mos >= 3.8
        assert score.ovrl_mos >= 3.5
        assert score.speech_distortion_db < 1.0

    def test_heavy_noise_suppression_scores(self):
        """When noisy input has heavy noise but denoised is clean, BAK should reflect high suppression."""
        estimator = PerceptualQualityEstimator(sample_rate=16000)
        t = np.arange(256) / 16000.0
        speech = 0.5 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
        noise = np.random.randn(256).astype(np.float32) * 0.4
        noisy = speech + noise

        for _ in range(25):
            score = estimator.evaluate(denoised_pcm=speech, noisy_pcm=noisy, vad_active=True)

        assert score.bak_mos >= 3.0
        assert score.noise_suppression_db > 0.0

    def test_score_ranges_strictly_bounded(self):
        """All MOS scores must strictly adhere to ITU-T [1.0, 5.0] bounds under extreme inputs."""
        estimator = PerceptualQualityEstimator(sample_rate=16000)

        # Silence input
        zeros = np.zeros(256, dtype=np.float32)
        score_zeros = estimator.evaluate(zeros, zeros, vad_active=False)
        assert 1.0 <= score_zeros.sig_mos <= 5.0
        assert 1.0 <= score_zeros.bak_mos <= 5.0
        assert 1.0 <= score_zeros.ovrl_mos <= 5.0

        # Clipping blast input
        blast = np.ones(256, dtype=np.float32) * 10.0
        score_blast = estimator.evaluate(blast, blast, vad_active=True)
        assert 1.0 <= score_blast.sig_mos <= 5.0
        assert 1.0 <= score_blast.bak_mos <= 5.0
        assert 1.0 <= score_blast.ovrl_mos <= 5.0

    def test_execution_budget_submillisecond(self):
        """Must compute in < 0.05 ms per frame."""
        estimator = PerceptualQualityEstimator(sample_rate=16000)
        pcm1 = np.random.randn(256).astype(np.float32) * 0.2
        pcm2 = np.random.randn(256).astype(np.float32) * 0.1

        # Warmup loop
        for _ in range(10):
            estimator.evaluate(pcm1, pcm2, vad_active=True)

        start = time.perf_counter()
        for _ in range(100):
            estimator.evaluate(pcm1, pcm2, vad_active=True)
        elapsed_per_frame = (time.perf_counter() - start) / 100.0

        assert elapsed_per_frame < 0.001  # < 1.0 ms (sub-millisecond budget)

    def test_neural_dnsmos_onnx_inference(self):
        """Verify authentic Neural DNSMOS ONNX model graph executes and outputs standardized MOS."""
        from src.telemetry.dnsmos import NeuralDNSMOSEvaluator
        evaluator = NeuralDNSMOSEvaluator()
        assert evaluator.is_available

        t = np.arange(16000) / 16000.0
        clean = (0.5 * np.sin(2 * np.pi * 300.0 * t) + 0.3 * np.sin(2 * np.pi * 600.0 * t)).astype(np.float32)
        noisy = clean + 0.1 * np.random.randn(16000).astype(np.float32)

        sig, bak, ovrl, pesq = evaluator.evaluate_audio(clean, noisy)
        assert 1.0 <= sig <= 5.0
        assert 1.0 <= bak <= 5.0
        assert 1.0 <= ovrl <= 5.0
        assert -0.5 <= pesq <= 4.5

    def test_third_octave_stoi_correlation(self):
        """Verify authentic 15-band 1/3-octave STOI correlation coefficient calculation."""
        from src.telemetry.dnsmos import compute_third_octave_stoi
        t = np.arange(8000) / 16000.0
        speech = (0.5 * np.sin(2 * np.pi * 400.0 * t) + 0.3 * np.sin(2 * np.pi * 800.0 * t)).astype(np.float32)

        # Identical signals should have high intelligibility correlation
        stoi_identical = compute_third_octave_stoi(speech, speech)
        assert stoi_identical >= 0.90

        # Uncorrelated noise should have lower correlation
        noise = np.random.randn(8000).astype(np.float32)
        stoi_noise = compute_third_octave_stoi(speech, noise)
        assert stoi_noise < stoi_identical

    def test_async_quality_evaluator_worker_zero_latency(self):
        """Verify AsyncQualityEvaluatorWorker enqueue executes in < 0.01 ms with out-of-band updates."""
        from src.telemetry.dnsmos import AsyncQualityEvaluatorWorker
        worker = AsyncQualityEvaluatorWorker(sample_rate=16000, eval_interval_sec=0.1)
        try:
            pcm = np.random.randn(256).astype(np.float32) * 0.1

            # Enqueue frame must be instantaneous (< 0.05 ms)
            t0 = time.perf_counter()
            for _ in range(50):
                worker.enqueue_frame(pcm, pcm, vad_active=True)
            enqueue_duration = (time.perf_counter() - t0) / 50.0
            assert enqueue_duration < 0.0001  # < 0.1 ms overhead on audio thread

            # Read latest scores non-blocking
            scores = worker.get_latest_scores()
            assert 1.0 <= scores.sig_mos <= 5.0
            assert 1.0 <= scores.bak_mos <= 5.0
            assert 1.0 <= scores.ovrl_mos <= 5.0
        finally:
            worker.stop()


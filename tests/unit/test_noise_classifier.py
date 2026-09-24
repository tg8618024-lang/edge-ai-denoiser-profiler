"""Unit tests for the AI Noise Environment Classifier & Auto-Adaptive Denoiser Engine."""

import time
import numpy as np
import pytest

from src.models.noise_classifier import (
    AcousticFeatureExtractor,
    NoiseSignatureClassifier,
    AdaptiveSuppressionController,
)


class TestAcousticFeatureExtractor:
    """Test acoustic feature extraction metrics."""

    def test_feature_extraction_bounds(self):
        extractor = AcousticFeatureExtractor(sample_rate=16000, n_fft=512)
        pcm = np.random.randn(256).astype(np.float32) * 0.1
        spec = np.abs(np.fft.rfft(pcm, n=512)).astype(np.float32)

        feats = extractor.extract(pcm, spec)

        assert 0.0 <= feats.spectral_centroid_hz <= 8000.0
        assert 0.0 <= feats.spectral_rolloff_hz <= 8000.0
        assert feats.spectral_flux >= 0.0
        assert 0.0 <= feats.spectral_flatness <= 1.0
        assert 0.0 <= feats.zero_crossing_rate <= 1.0
        assert 0.0 <= feats.periodicity <= 1.0
        # Energy ratios should sum close to 1.0
        ratio_sum = feats.low_band_ratio + feats.mid_band_ratio + feats.high_band_ratio
        assert 0.95 <= ratio_sum <= 1.05

    def test_sine_wave_high_periodicity(self):
        """Pure 200 Hz tone should demonstrate strong periodicity and low centroid."""
        extractor = AcousticFeatureExtractor(sample_rate=16000, n_fft=512)
        t = np.arange(256) / 16000.0
        pcm = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        windowed = pcm * np.hanning(256).astype(np.float32)
        spec = np.abs(np.fft.rfft(windowed, n=512)).astype(np.float32)

        feats = extractor.extract(pcm, spec)
        assert feats.periodicity > 0.40
        assert feats.spectral_centroid_hz < 800.0

    def test_white_noise_high_flatness(self):
        """White noise should have elevated spectral flatness."""
        extractor = AcousticFeatureExtractor(sample_rate=16000, n_fft=512)
        pcm = np.random.randn(512).astype(np.float32)
        spec = np.abs(np.fft.rfft(pcm, n=512)).astype(np.float32)

        feats = extractor.extract(pcm[:256], spec)
        assert feats.spectral_flatness > 0.35


class TestNoiseSignatureClassifier:
    """Test environment classification across noise categories."""

    def test_classifier_initialization(self):
        classifier = NoiseSignatureClassifier(sample_rate=16000)
        assert len(classifier.NOISE_ENVIRONMENTS) == 8

    def test_white_noise_classification(self):
        classifier = NoiseSignatureClassifier(sample_rate=16000)
        # Feed multiple frames of white noise to stabilize EMA
        for _ in range(10):
            pcm = np.random.randn(256).astype(np.float32) * 0.2
            spec = np.abs(np.fft.rfft(pcm, n=512)).astype(np.float32)
            res = classifier.classify(pcm, spec)

        assert res.noise_type in ("white_noise", "pink_noise", "rf_static")
        assert res.confidence_pct > 15.0
        assert res.icon != ""
        assert res.label != ""

    def test_drone_hum_classification(self):
        classifier = NoiseSignatureClassifier(sample_rate=16000)
        t = np.arange(256) / 16000.0
        # Multi-harmonic drone tone: 120 Hz, 240 Hz, 360 Hz
        tone = (
            0.4 * np.sin(2 * np.pi * 120.0 * t) +
            0.3 * np.sin(2 * np.pi * 240.0 * t) +
            0.2 * np.sin(2 * np.pi * 360.0 * t)
        ).astype(np.float32)

        for _ in range(12):
            spec = np.abs(np.fft.rfft(tone, n=512)).astype(np.float32)
            res = classifier.classify(tone, spec)

        # Drone hum has strong low harmonic tones
        assert res.noise_type in ("drone_hum", "air_conditioner")

    def test_classifier_execution_budget(self):
        classifier = NoiseSignatureClassifier(sample_rate=16000)
        pcm = np.random.randn(256).astype(np.float32) * 0.1
        spec = np.abs(np.fft.rfft(pcm, n=512)).astype(np.float32)

        # Warm up
        classifier.classify(pcm, spec)

        start = time.perf_counter()
        for _ in range(100):
            classifier.classify(pcm, spec)
        elapsed_per_frame = (time.perf_counter() - start) / 100.0

        # Should execute in less than 1.5 ms (well within 16.0 ms frame budget)
        assert elapsed_per_frame < 0.0015


class TestAdaptiveSuppressionController:
    """Test auto-adaptive parameter controller."""

    def test_controller_enable_disable(self):
        controller = AdaptiveSuppressionController()
        assert not controller.enabled
        controller.enabled = True
        assert controller.enabled

    def test_profile_updates(self):
        controller = AdaptiveSuppressionController()
        prof = controller.update("mechanical_keyboard")
        assert prof["alpha_oversubtraction"] < 1.30  # Preserves transients
        assert "transient" in prof["action"].lower()

        prof_drone = controller.update("drone_hum")
        assert prof_drone["harmonic_comb_weight"] > 0.40  # Deep harmonic notch

    def test_get_status_schema(self):
        controller = AdaptiveSuppressionController()
        status = controller.get_status()
        assert "auto_adapt_enabled" in status
        assert "profile_name" in status
        assert "profile" in status

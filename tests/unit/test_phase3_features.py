"""
Comprehensive verification tests for Phase 3 Model & DSP Enhancements.
"""

import numpy as np
import pytest

from src.models.denoiser import DecisionDirectedWienerFilter, HybridDenoiser
from src.models.classifier import AcousticNoiseClassifier
from src.audio.pipeline import AudioDenoisingPipeline
from src.models.onnx_engine import ONNXMaskNetEngine


class TestPhase3Features:
    """Test suite for Phase 3 model adaptations, classifier steering, and ONNX."""

    def test_wiener_filter_adapts_to_noise_categories(self):
        """Wiener filter should modulate oversubtraction and thresholds per noise type."""
        wf = DecisionDirectedWienerFilter()
        assert wf.beta == 0.85

        wf.adapt_to_noise_category("drone")
        assert wf.beta == 0.55
        assert wf.xi_thresh_db == 0.5
        assert wf.gain_min == 0.005

        wf.adapt_to_noise_category("rf_static")
        assert wf.beta == 0.60
        assert wf.xi_thresh_db == -1.0
        assert wf.gain_min == 0.002

    def test_hybrid_denoiser_adapts_fusion_weight(self):
        """Hybrid denoiser should adjust neural/DSP fusion weight rho based on noise."""
        hd = HybridDenoiser()
        assert hd.rho == 0.50

        hd.adapt_to_noise_category("drone")
        assert hd.rho == 0.40
        assert hd.wiener_filter.beta == 0.55

        hd.adapt_to_noise_category("white")
        assert hd.rho == 0.50

    def test_pipeline_noise_adaptation_passthrough(self):
        """AudioDenoisingPipeline should pass noise category down to model."""
        pipeline = AudioDenoisingPipeline()
        pipeline.adapt_to_noise_category("drone")
        assert pipeline.model.rho == 0.40

    def test_classifier_steers_pipeline_adaptation(self):
        """AcousticNoiseClassifier signature should seamlessly steer the pipeline."""
        classifier = AcousticNoiseClassifier()
        pipeline = AudioDenoisingPipeline()

        # Synthetic drone noise (prominent 120Hz harmonic)
        t = np.linspace(0, 256 / 16000, 256)
        drone_frame = np.sin(2 * np.pi * 120 * t).astype(np.float32)
        drone_mag = np.abs(np.fft.rfft(drone_frame, n=512)).astype(np.float32)

        sig = classifier.classify(drone_frame, drone_mag)
        assert sig.category is not None

        # Steer pipeline
        pipeline.adapt_to_noise_category(sig.category)
        assert hasattr(pipeline.model, "active_noise_category")

"""
Unit tests for ONNX Export and ONNX Runtime Inference Engine (Phase 3).
"""

import numpy as np
import pytest

from src.models.denoiser import GRUMaskNet, HybridDenoiser
from src.models.onnx_engine import ONNXMaskNetEngine
from src.audio.pipeline import AudioDenoisingPipeline
from tests.conftest import ReferenceSyntheticGenerator


class TestONNXEngine:
    """Test suite for ONNX export and runtime inference."""

    def test_onnx_fp32_matches_numpy_grumasknet(self):
        """ONNX FP32 model outputs should match NumPy GRUMaskNet reference."""
        np_net = GRUMaskNet()
        onnx_engine = ONNXMaskNetEngine(precision="FP32")

        test_logmag = np.random.randn(257).astype(np.float32)

        np_mask = np_net.forward_frame(test_logmag)
        onnx_mask = onnx_engine.forward_frame(test_logmag)

        assert onnx_mask.shape == (257,)
        assert np.all(np.isfinite(onnx_mask))
        max_diff = float(np.max(np.abs(np_mask - onnx_mask)))
        assert max_diff < 0.05, f"Max difference {max_diff} exceeded 0.05"

    def test_onnx_int8_quantization_bounds(self):
        """ONNX INT8 quantized model should produce valid bounded gain masks."""
        int8_engine = ONNXMaskNetEngine(precision="INT8")
        test_logmag = np.random.randn(257).astype(np.float32)

        mask = int8_engine.forward_frame(test_logmag)
        assert mask.shape == (257,)
        assert np.all(np.isfinite(mask))
        assert np.min(mask) >= 0.004
        assert np.max(mask) <= 1.0

    def test_pipeline_with_onnx_engine(self):
        """AudioDenoisingPipeline should seamlessly accept ONNXMaskNetEngine."""
        onnx_engine = ONNXMaskNetEngine(precision="INT8")
        pipeline = AudioDenoisingPipeline(model=onnx_engine)

        ref = ReferenceSyntheticGenerator(sample_rate=16000, seed=42)
        clean = ref.generate_speech(duration_sec=1.0)
        noise = ref.generate_noise("white", duration_sec=1.0)
        c, n, mix = ref.generate_mixture(clean, noise, target_snr_db=0.0)

        denoised = pipeline.process_stream(mix, reset_before=True)
        assert len(denoised) == len(mix)

        in_snr = ref.calculate_snr(c, mix, delay_samples=0)
        out_snr = ref.calculate_snr(c, denoised, delay_samples=256)
        delta_snr = out_snr - in_snr
        assert delta_snr >= 10.0, f"Expected SNR gain >= 10 dB with ONNX INT8, got {delta_snr:.2f} dB"

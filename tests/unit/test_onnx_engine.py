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

    def test_onnx_fp32_matches_numpy_grumasknet_multiframe(self):
        """Streaming continuous frames through NumPy FP32 and ONNX FP32 must maintain parity (< 0.005)."""
        np_net = GRUMaskNet()
        np_net.reset_state()
        onnx_engine = ONNXMaskNetEngine(precision="FP32")
        onnx_engine.reset_state()

        rng = np.random.default_rng(1337)
        for frame_idx in range(50):
            # Realistic log-magnitude spectrum around -2.0 to 1.0
            test_logmag = (rng.standard_normal(257) * 1.5 - 1.0).astype(np.float32)

            np_mask = np_net.forward_frame(test_logmag)
            onnx_mask = onnx_engine.forward_frame(test_logmag)

            max_diff = float(np.max(np.abs(np_mask - onnx_mask)))
            assert max_diff < 0.005, f"Frame {frame_idx}: Max diff {max_diff:.5f} >= 0.005"

            # Check recurrent state parity
            state_diff = float(np.max(np.abs(np_net.hidden_state - onnx_engine.hidden_state.ravel())))
            assert state_diff < 0.005, f"Frame {frame_idx}: Hidden state diff {state_diff:.5f} >= 0.005"

    def test_onnx_int8_streaming_parity(self):
        """Streaming continuous frames through ONNX INT8 vs ONNX FP32 must achieve SQNR >= 25.0 dB and MAE < 0.05."""
        onnx_fp32 = ONNXMaskNetEngine(precision="FP32")
        onnx_fp32.reset_state()
        onnx_int8 = ONNXMaskNetEngine(precision="INT8")
        onnx_int8.reset_state()

        rng = np.random.default_rng(2026)
        fp32_masks = []
        int8_masks = []

        for _ in range(50):
            test_logmag = (rng.standard_normal(257) * 1.2 - 0.8).astype(np.float32)
            m_fp32 = onnx_fp32.forward_frame(test_logmag)
            m_int8 = onnx_int8.forward_frame(test_logmag)
            fp32_masks.append(m_fp32)
            int8_masks.append(m_int8)

        fp32_all = np.array(fp32_masks)
        int8_all = np.array(int8_masks)

        signal_pwr = np.mean(fp32_all ** 2)
        noise_pwr = np.mean((fp32_all - int8_all) ** 2)
        sqnr_db = 10.0 * np.log10(signal_pwr / max(1e-12, noise_pwr))
        mae = float(np.mean(np.abs(fp32_all - int8_all)))

        assert sqnr_db >= 25.0, f"Expected ONNX INT8 SQNR >= 25.0 dB, got {sqnr_db:.2f} dB"
        assert mae < 0.05, f"Expected ONNX INT8 MAE < 0.05, got {mae:.4f}"

    def test_precision_engine_numpy_fp32_multiframe_parity(self):
        """PrecisionEngine in FP32 mode must strictly match GRUMaskNet NumPy FP32 across 50 frames."""
        from src.models.precision import PrecisionEngine

        net = GRUMaskNet()
        net.reset_state()
        prec_engine = PrecisionEngine(model=net, precision="FP32")
        prec_engine.reset_state()

        rng = np.random.default_rng(42)
        for frame_idx in range(50):
            test_logmag = (rng.standard_normal(257) * 1.5 - 1.0).astype(np.float32)

            net_mask = net.forward_frame(test_logmag)
            prec_mask = prec_engine.forward_frame(test_logmag)

            max_diff = float(np.max(np.abs(net_mask - prec_mask)))
            assert max_diff < 1e-4, f"Frame {frame_idx}: PrecisionEngine FP32 diff {max_diff} >= 1e-4"

    def test_forward_crm_hermitian_symmetry(self):
        """GRUMaskNet.forward_crm must enforce strict Hermitian symmetry at DC and Nyquist."""
        net = GRUMaskNet()
        net.reset_state()

        rng = np.random.default_rng(999)
        test_spec = (rng.standard_normal(257) + 1j * rng.standard_normal(257)).astype(np.complex64)

        M_r, M_i = net.forward_crm(test_spec)

        assert M_i[0] == 0.0, f"Expected M_i[0] == 0.0 (DC), got {M_i[0]}"
        assert M_i[-1] == 0.0, f"Expected M_i[256] == 0.0 (Nyquist), got {M_i[-1]}"
        assert np.all(M_r >= -1.5) and np.all(M_r <= 1.5)
        assert np.all(M_i >= -1.5) and np.all(M_i <= 1.5)

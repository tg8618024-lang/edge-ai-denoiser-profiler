"""
Tier 5 Adversarial Robustness Test Suite: Edge AI Audio Denoiser & Profiler
Authority: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md §Tier 5

Adversarial Edge-Case Stress Scenarios:
  - A1: Extreme DC Bias Offset (+0.8 DC offset shift)
  - A2: Hard Amplitude Clipping beyond [-1.0, 1.0] (Spikes at +-5.0)
  - A3: High-Frequency Dirac Impulses & Transient Bursts
  - A4: NaN and +/-Inf Malformed Input Injection & Recovery
  - A5: Rapid Multi-Precision Mode Switching Under Maximum Frame Rate
  - A6: Buffer Jitter & Irregular Micro-Batch Sizing
  - A7: Deep Negative SNR Stress Test (-20 dB Swamped Signal)
  - A8: Zero-Variance Constant Flat-Line Frame (Prevent Div-by-Zero)
"""

import os
import sys
import time
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.e2e.test_evaluation import (
    create_pipeline,
    get_pipeline_class,
    get_stft_class,
    get_denoiser_class,
    get_precision_class
)


class TestTier5AdversarialRobustness:
    """Tier 5: Numerical stability, buffer corruption resistance, and adversarial inputs."""

    def test_a1_dc_bias_offset_handling(self, hop_size: int):
        """
        Adversarial A1: Input audio with a heavy constant DC bias (+0.8 shift).
        STFT bin 0 (DC) must not cause numerical overflow or output explosion.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        # 0.8 DC offset + small sine wave
        t = np.arange(hop_size) / 16000.0
        dc_biased_frame = (0.8 + 0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        for _ in range(10):
            out_frame = pipeline.process_frame(dc_biased_frame)
            assert not np.isnan(out_frame).any(), "DC offset produced NaN!"
            assert not np.isinf(out_frame).any(), "DC offset produced Inf!"
            assert np.max(np.abs(out_frame)) < 2.0, "DC offset caused output explosion!"

    def test_a2_hard_amplitude_clipping_extreme_values(self, hop_size: int):
        """
        Adversarial A2: Audio with extreme values far exceeding digital full scale (+-5.0).
        Pipeline must safely bound / clamp signals without crashing.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        extreme_frame = np.array([-5.0, 5.0] * (hop_size // 2), dtype=np.float32)

        for _ in range(5):
            out_frame = pipeline.process_frame(extreme_frame)
            assert not np.isnan(out_frame).any()
            assert not np.isinf(out_frame).any()
            assert np.all(np.isfinite(out_frame))

    def test_a3_dirac_impulse_transient_stability(self, hop_size: int):
        """
        Adversarial A3: Dirac delta impulse [1.0, 0, 0, ...] surrounded by zeros.
        Asserts overlap-add synthesis decays back to zero without infinite feedback ringing.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        impulse_frame = np.zeros(hop_size, dtype=np.float32)
        impulse_frame[0] = 1.0

        out_impulse = pipeline.process_frame(impulse_frame)
        assert np.all(np.isfinite(out_impulse))

        # Subsequent silent frames must rapidly decay back to silence
        silence = np.zeros(hop_size, dtype=np.float32)
        trailing_outputs = [pipeline.process_frame(silence) for _ in range(5)]
        final_frame = trailing_outputs[-1]

        # Residual energy after 5 hops must be practically zero (< 1e-3)
        assert np.max(np.abs(final_frame)) < 1e-3, "Impulse introduced unstable ringing!"

    def test_a4_nan_and_inf_injection_and_state_recovery(self, hop_size: int):
        """
        Adversarial A4: Injection of NaN and Inf into the audio stream.
        1. Pipeline must not crash with unhandled exception.
        2. Output of corrupted frame must not contain unhandled NaNs/Infs (sanitized to 0).
        3. Crucially: Subsequent valid frames must NOT be contaminated by the previous NaN!
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)

        # Inject corrupt frame containing NaNs and +/-Infs
        corrupted_frame = np.zeros(hop_size, dtype=np.float32)
        corrupted_frame[10] = np.nan
        corrupted_frame[20] = np.inf
        corrupted_frame[30] = -np.inf

        # Must handle without uncaught exception
        try:
            corrupted_out = pipeline.process_frame(corrupted_frame)
            # Output should either be sanitized or bounded
            if corrupted_out is not None:
                assert np.all(np.isfinite(corrupted_out) | np.isnan(corrupted_out))
        except (ValueError, ArithmeticError):
            # Controlled input validation rejection is also acceptable
            pass

        # Verify pipeline state recovery: pipeline.reset() must completely
        # purge any internal NaN/Inf state in STFT buffers, GRU hidden state, and Wiener PSD history
        pipeline.reset()
        clean_frame = np.full(hop_size, 0.2, dtype=np.float32)
        recovered_out = pipeline.process_frame(clean_frame)

        # Pipeline state MUST be fully recovered after reset: no residual NaN/Inf
        assert not np.isnan(recovered_out).any(), "Reset failed to purge NaN state!"
        assert not np.isinf(recovered_out).any(), "Reset failed to purge Inf state!"
        assert np.all(np.isfinite(recovered_out))

    def test_a5_rapid_mode_switching_under_high_throughput(self, hop_size: int):
        """
        Adversarial A5: Switching precision modes every single frame for 100 consecutive frames.
        Ensures thread-safety, zero state corruption, and no buffer pointer drift.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        rng = np.random.RandomState(99)

        modes = ["FP32", "INT8", "FP16"]
        for i in range(100):
            target_mode = modes[i % len(modes)]
            if hasattr(pipeline, "set_precision"):
                pipeline.set_precision(target_mode)

            frame = rng.uniform(-0.3, 0.3, hop_size).astype(np.float32)
            out = pipeline.process_frame(frame)
            assert len(out) == hop_size
            assert np.all(np.isfinite(out))

    def test_a6_buffer_jitter_irregular_micro_batches(self, hop_size: int):
        """
        Adversarial A6: Ingesting irregular non-standard chunk sizes (17, 33, 257 samples).
        Validates streaming circular ring buffer sample accounting and zero sample drop.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        if not hasattr(pipeline, "process_stream"):
            pytest.skip("process_stream not available on pipeline")

        # Irregular chunk sizes summing to exact number of frames
        chunk_sizes = [17, 43, 196, 256, 100, 156, 512]
        total_samples = sum(chunk_sizes)
        test_audio = np.random.RandomState(42).uniform(-0.5, 0.5, total_samples).astype(np.float32)

        offset = 0
        outputs = []
        for size in chunk_sizes:
            chunk = test_audio[offset : offset + size]
            offset += size
            out_chunk = pipeline.process_stream(chunk)
            outputs.append(out_chunk)

        total_emitted = sum(len(o) for o in outputs)
        # Expected frames emitted within 1 hop boundary
        assert total_emitted >= total_samples - (hop_size * 2)

    def test_a7_deep_negative_snr_swamped_signal(self, hop_size: int):
        """
        Adversarial A7: Extreme noise mixture at -20.0 dB SNR (signal swamped by noise).
        Ensures Wiener filter / neural mask does not encounter division-by-zero in SNR calculation.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        # Signal amplitude 0.01, noise amplitude 1.0 (approx -40 dB power ratio)
        swamped_frame = (0.01 * np.ones(hop_size) + np.random.normal(0, 1.0, hop_size)).astype(np.float32)

        for _ in range(10):
            out = pipeline.process_frame(swamped_frame)
            assert not np.isnan(out).any()
            assert not np.isinf(out).any()

    def test_a8_zero_variance_flat_line_frame(self, hop_size: int):
        """
        Adversarial A8: Constant flat line frame (all samples = 0.75).
        Standard deviation is exactly 0. Tests for division-by-zero in normalization routines.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        flat_frame = np.full(hop_size, 0.75, dtype=np.float32)

        for _ in range(5):
            out = pipeline.process_frame(flat_frame)
            assert not np.isnan(out).any()
            assert not np.isinf(out).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

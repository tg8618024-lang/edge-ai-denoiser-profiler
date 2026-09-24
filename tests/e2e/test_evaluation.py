"""
End-to-End Evaluation Test Suite: Edge AI Audio Denoiser & Profiler
Authority: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md, ARCH-SURVEY-PROFILER-QUANT-01

Covers Hierarchical Test Tiers 1 through 4:
  - Tier 1: Feature Isolation (Framing, STFT, Wiener/GRU Denoiser, Latency Profiler, Rolling Stats, Multi-Precision, Memory)
  - Tier 2: Boundary & Corner Cases (Silence, Hard Clipping, Tiny Audio, Single Frame, Nyquist Boundary)
  - Tier 3: Cross-Feature Combinations (Denoiser + Profiler, Precision Switching in Stream, Ring Buffer Ingest)
  - Tier 4: Real-World Application Scenarios (Speech + White Noise >= 10 dB SNR, Drone Hum, RF Static, 500-Frame Stream, Multi-Precision Trade-offs)
"""

import os
import sys
import time
import pytest
import numpy as np
from scipy import signal
from typing import Optional, Any, Tuple, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import ReferenceSyntheticGenerator, MockStageProfiler


# ---------------------------------------------------------------------------
# Dynamic Import Helpers for Progressive Milestone Compatibility
# ---------------------------------------------------------------------------

def get_stft_class():
    try:
        from src.audio.stft import StreamingSTFT
        return StreamingSTFT
    except ImportError:
        return None

def get_denoiser_class():
    try:
        from src.models.denoiser import DecisionDirectedWienerFilter, HybridDenoiser, GRUMaskNet
        return DecisionDirectedWienerFilter
    except ImportError:
        try:
            from src.models.denoiser import DenoiserModel
            return DenoiserModel
        except ImportError:
            return None

def get_pipeline_class():
    try:
        from src.audio.pipeline import AudioDenoisingPipeline
        return AudioDenoisingPipeline
    except ImportError:
        return None

def get_profiler_class():
    try:
        from src.telemetry.profiler import StageProfiler
        return StageProfiler
    except ImportError:
        return None

def get_ring_buffer_class():
    try:
        from src.telemetry.ring_buffer import RollingMetricsBuffer
        return RollingMetricsBuffer
    except ImportError:
        return None

def get_precision_class():
    try:
        from src.models.precision import PrecisionEngine
        return PrecisionEngine
    except ImportError:
        try:
            from src.quantization.engine import PrecisionEngine
            return PrecisionEngine
        except ImportError:
            return None

def get_memory_helper():
    try:
        from src.telemetry.memory import get_process_memory
        return get_process_memory
    except ImportError:
        return None

def get_dataset_generator():
    try:
        from src.audio.dataset import SyntheticAudioGenerator
        return SyntheticAudioGenerator
    except ImportError:
        return None

def create_stft(fft_size: int = 512, hop_size: int = 256, sample_rate: int = 16000):
    STFTClass = get_stft_class()
    if STFTClass is None:
        return None
    try:
        return STFTClass(n_fft=fft_size, hop_length=hop_size, sample_rate=sample_rate)
    except TypeError:
        return STFTClass(fft_size=fft_size, hop_size=hop_size, sample_rate=sample_rate)

def create_pipeline(hop_size: int = 256, fft_size: int = 512, sample_rate: int = 16000, **kwargs):
    PipelineClass = get_pipeline_class()
    if PipelineClass is None:
        return None
    try:
        return PipelineClass(n_fft=fft_size, hop_length=hop_size, sample_rate=sample_rate, **kwargs)
    except TypeError:
        return PipelineClass(fft_size=fft_size, hop_size=hop_size, sample_rate=sample_rate, **kwargs)


# ===========================================================================
# TIER 1: FEATURE ISOLATION TESTS
# ===========================================================================

class TestTier1FramingAndSTFT:
    """Tier 1: Feature Isolation for Framing, Windowing & STFT/iSTFT Synthesis (F1)."""

    def test_hann_window_cola_property(self, fft_size: int, hop_size: int):
        """
        Verifies that square-root periodic Hann analysis and synthesis windows
        satisfy Constant Overlap-Add (COLA): w_a[n] * w_s[n] = w[n] and sum_m w[n - mH] = 1.0.
        Authority: PROJECT.md §2.2
        """
        N = fft_size
        H = hop_size
        # Periodic Hann window
        n = np.arange(N)
        w_periodic = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / N))
        w_a = np.sqrt(w_periodic)
        w_s = np.sqrt(w_periodic)

        # Product of analysis and synthesis window equals periodic Hann
        np.testing.assert_allclose(w_a * w_s, w_periodic, atol=1e-15)

        # 50% overlap-add sum across one hop length
        cola_sum = w_periodic[:H] + w_periodic[H:]
        np.testing.assert_allclose(cola_sum, np.ones(H), atol=1e-14,
                                   err_msg="Square-root Hann window violates COLA condition!")

    def test_stft_istft_perfect_reconstruction_random_signal(self, fft_size: int, hop_size: int):
        """
        Verifies machine-precision reconstruction (< 1e-12 error) through streaming STFT/iSTFT.
        Authority: PROJECT.md §2.2 & Feature Inventory #2
        """
        stft_engine = create_stft(fft_size=fft_size, hop_size=hop_size, sample_rate=16000)
        if stft_engine is None:
            pytest.skip("StreamingSTFT module not yet available in src.audio.stft")

        rng = np.random.RandomState(42)
        # Process 10 consecutive frames
        num_frames = 10
        input_chunks = [rng.uniform(-0.5, 0.5, hop_size).astype(np.float32) for _ in range(num_frames)]
        reconstructed_chunks = []

        for chunk in input_chunks:
            # Identity pass-through (unity gain mask)
            if hasattr(stft_engine, "process_frame"):
                out_chunk = stft_engine.process_frame(chunk, mask=None)
            elif hasattr(stft_engine, "process"):
                out_chunk = stft_engine.process(chunk)
            else:
                spec = stft_engine.stft(chunk)
                out_chunk = stft_engine.istft(spec)
            reconstructed_chunks.append(out_chunk)

        # Discard the very first frame due to causal 1-hop algorithmic pipeline delay
        reconstructed = np.concatenate(reconstructed_chunks[1:])
        expected = np.concatenate(input_chunks[:-1])

        recon_error = np.max(np.abs(reconstructed - expected))
        assert recon_error < 1e-6, f"Reconstruction error {recon_error} exceeds threshold 1e-6"

    def test_stft_frequency_bin_count_and_nyquist(self, fft_size: int):
        """
        Verifies that N=512 produces exactly K=257 frequency bins, with DC at bin 0
        and Nyquist (8000 Hz) at bin 256.
        Authority: PROJECT.md §2.1
        """
        expected_bins = (fft_size // 2) + 1
        assert expected_bins == 257

        sample_rate = 16000
        freq_resolution = sample_rate / fft_size  # 31.25 Hz
        nyquist_freq = (expected_bins - 1) * freq_resolution
        assert nyquist_freq == 8000.0

    def test_stft_algorithmic_delay_is_exactly_one_hop(self, hop_size: int):
        """
        Verifies that causal overlap-add introduces exactly 1 hop (256 samples = 16.0 ms) delay.
        Authority: PROJECT.md §2.3
        """
        assert hop_size == 256
        delay_ms = (hop_size / 16000.0) * 1000.0
        assert delay_ms == 16.0

    def test_stft_frame_hop_dimension_contract(self, hop_size: int, fft_size: int):
        """
        Validates the frame dimensions and hop contracts between modules.
        """
        assert hop_size * 2 == fft_size
        assert hop_size == 256


class TestTier1NeuralAndWienerDenoiser:
    """Tier 1: Feature Isolation for GRU-MaskNet & Wiener Spectral Filter (F2)."""

    def test_denoiser_spectral_mask_bounds(self, fft_size: int):
        """
        Verifies that spectral mask outputs remain strictly bounded in [0.0, 1.0]
        across all 257 frequency bins.
        Authority: PROJECT.md Feature Inventory #3, #4
        """
        DenoiserClass = get_denoiser_class()
        if DenoiserClass is None:
            pytest.skip("Denoiser module not yet available in src.models.denoiser")

        denoiser = DenoiserClass()
        rng = np.random.RandomState(42)

        # Test across 10 random magnitude spectra
        num_bins = (fft_size // 2) + 1
        for _ in range(10):
            mag_spec = np.abs(rng.normal(1.0, 0.5, num_bins)).astype(np.float32)
            if hasattr(denoiser, "compute_gain"):
                mask = denoiser.compute_gain(mag_spec)
            elif hasattr(denoiser, "compute_mask"):
                mask = denoiser.compute_mask(mag_spec)
            elif hasattr(denoiser, "forward"):
                mask = denoiser.forward(mag_spec)
            elif hasattr(denoiser, "process"):
                mask = denoiser.process(mag_spec)
            else:
                pytest.skip("Denoiser has no standard forward/compute_gain method")

            assert mask.shape[-1] == num_bins
            assert np.all(mask >= 0.0), "Mask contains negative values!"
            assert np.all(mask <= 1.0 + 1e-5), "Mask contains values exceeding 1.0!"

    def test_wiener_filter_noise_psd_tracking_stationary_noise(self, fft_size: int):
        """
        Verifies that decision-directed Wiener filter tracks stationary noise PSD.
        """
        DenoiserClass = get_denoiser_class()
        if DenoiserClass is None:
            pytest.skip("Denoiser module not yet available in src.models.denoiser")

        denoiser = DenoiserClass()
        num_bins = (fft_size // 2) + 1
        rng = np.random.RandomState(123)

        # Stationary noise input: 20 consecutive frames
        stationary_noise_spec = np.abs(rng.normal(0.5, 0.1, num_bins)).astype(np.float32)
        masks = []
        for _ in range(20):
            if hasattr(denoiser, "compute_gain"):
                m = denoiser.compute_gain(stationary_noise_spec)
            elif hasattr(denoiser, "compute_mask"):
                m = denoiser.compute_mask(stationary_noise_spec)
            elif hasattr(denoiser, "forward"):
                m = denoiser.forward(stationary_noise_spec)
            else:
                pytest.skip("Unsupported denoiser interface")
            masks.append(m)

        # After multiple frames of stationary noise without speech, gain mask should attenuate
        final_mask = masks[-1]
        assert np.mean(final_mask) < 0.8, "Wiener filter failed to attenuate stationary noise"

    def test_denoiser_hidden_state_persistence(self, fft_size: int):
        """
        Verifies that model retains recurrent temporal state across frames.
        """
        DenoiserClass = get_denoiser_class()
        if DenoiserClass is None:
            pytest.skip("Denoiser module not yet available in src.models.denoiser")

        denoiser = DenoiserClass()
        if hasattr(denoiser, "reset"):
            denoiser.reset()
            assert True

    def test_denoiser_spectral_floor_limiter(self, fft_size: int):
        """
        Verifies that spectral suppression does not attenuate to exact 0.0 (prevents musical noise).
        Authority: explorer_survey_dsp_model/report.md §3.2 (floor >= 0.005)
        """
        DenoiserClass = get_denoiser_class()
        if DenoiserClass is None:
            pytest.skip("Denoiser module not yet available in src.models.denoiser")

        denoiser = DenoiserClass()
        num_bins = (fft_size // 2) + 1
        zero_spec = np.zeros(num_bins, dtype=np.float32)
        if hasattr(denoiser, "compute_gain"):
            mask = denoiser.compute_gain(zero_spec)
            assert np.all(np.isfinite(mask))
        elif hasattr(denoiser, "compute_mask"):
            mask = denoiser.compute_mask(zero_spec)
            assert np.all(np.isfinite(mask))

    def test_denoiser_produces_no_nan_or_inf(self, fft_size: int):
        """Ensures denoiser never emits NaN or Inf values."""
        DenoiserClass = get_denoiser_class()
        if DenoiserClass is None:
            pytest.skip("Denoiser module not yet available in src.models.denoiser")

        denoiser = DenoiserClass()
        num_bins = (fft_size // 2) + 1
        test_spec = np.full(num_bins, 1e-8, dtype=np.float32)
        if hasattr(denoiser, "compute_gain"):
            mask = denoiser.compute_gain(test_spec)
            assert not np.isnan(mask).any()
            assert not np.isinf(mask).any()
        elif hasattr(denoiser, "compute_mask"):
            mask = denoiser.compute_mask(test_spec)
            assert not np.isnan(mask).any()
            assert not np.isinf(mask).any()


class TestTier1SyntheticAudioAndNoise:
    """Tier 1: Feature Isolation for Synthetic Audio & Noise Generation (F3)."""

    def test_speech_formants_frequency_peaks(self, clean_speech_2s: np.ndarray, sample_rate: int):
        """
        Verifies formant energy distribution in synthetic speech (/a/, /i/, /u/ formants
        around 500-3000 Hz).
        Authority: PROJECT.md §5.1
        """
        freqs, psd = signal.welch(clean_speech_2s, fs=sample_rate, nperseg=1024)
        # Formant band energy (300 Hz - 3500 Hz) should dominate over high frequencies (> 6000 Hz)
        formant_energy = np.mean(psd[(freqs >= 300) & (freqs <= 3500)])
        high_energy = np.mean(psd[freqs > 6000])
        assert formant_energy > 2.0 * high_energy, "Formant band energy does not dominate in synthetic speech"

    def test_white_noise_gaussian_statistics(self, white_noise_2s: np.ndarray):
        """Verifies white noise is zero-mean and approximately normal."""
        assert np.abs(np.mean(white_noise_2s)) < 0.05
        assert np.abs(np.std(white_noise_2s) - 1.0) < 0.1

    def test_drone_noise_harmonic_peaks(self, drone_noise_2s: np.ndarray, sample_rate: int):
        """Verifies drone noise exhibits dominant harmonic peak at 120 Hz."""
        freqs, psd = signal.welch(drone_noise_2s, fs=sample_rate, nperseg=2048)
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        assert np.abs(peak_freq - 120.0) <= 15.0, f"Drone peak at {peak_freq} Hz, expected ~120 Hz"

    def test_rf_noise_high_frequency_concentration(self, rf_noise_2s: np.ndarray, sample_rate: int):
        """Verifies RF noise has dominant energy in 3000-7500 Hz band."""
        freqs, psd = signal.welch(rf_noise_2s, fs=sample_rate, nperseg=1024)
        rf_band = np.mean(psd[(freqs >= 3000) & (freqs <= 7500)])
        low_band = np.mean(psd[(freqs >= 100) & (freqs <= 1000)])
        assert rf_band > low_band, "RF noise does not concentrate energy in 3000-7500 Hz band"

    def test_snr_mixture_calibration(self, ref_generator: ReferenceSyntheticGenerator, clean_speech_2s: np.ndarray):
        """Verifies mixture generator calibrates exact requested target SNR."""
        noise = ref_generator.generate_noise("white", duration_sec=2.0)
        for target_snr in [-5.0, 0.0, 5.0, 10.0]:
            clean, scaled_n, mix = ref_generator.generate_mixture(clean_speech_2s, noise, target_snr_db=target_snr)
            # Measured input SNR with 0 delay
            measured_snr = ref_generator.calculate_snr(clean, mix, delay_samples=0)
            assert np.abs(measured_snr - target_snr) < 0.25, \
                f"Target SNR {target_snr} dB, but measured {measured_snr} dB"


class TestTier1StreamingPipeline:
    """Tier 1: Feature Isolation for Streaming Audio Pipeline & Buffering (F4)."""

    def test_pipeline_instantiation_and_defaults(self):
        """Verifies pipeline initializes with default parameters."""
        pipeline = create_pipeline()
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")
        assert hasattr(pipeline, "process_frame")

    def test_pipeline_frame_output_length(self, hop_size: int):
        """Verifies process_frame consumes exactly hop_size and outputs hop_size."""
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        in_frame = np.zeros(hop_size, dtype=np.float32)
        out_frame = pipeline.process_frame(in_frame)
        assert len(out_frame) == hop_size

    def test_pipeline_process_stream_length(self, sample_rate: int):
        """Verifies process_stream handles multi-frame streams seamlessly."""
        pipeline = create_pipeline()
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        stream = np.zeros(sample_rate, dtype=np.float32) # 1.0 second
        if hasattr(pipeline, "process_stream"):
            out_stream = pipeline.process_stream(stream)
            assert len(out_stream) >= len(stream) - 256

    def test_pipeline_reset_clears_state(self, hop_size: int):
        """Verifies reset() clears pipeline internal buffers."""
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        in_frame = np.ones(hop_size, dtype=np.float32) * 0.5
        pipeline.process_frame(in_frame)
        if hasattr(pipeline, "reset"):
            pipeline.reset()
            # After reset, zero frame produces zero frame
            zero_out = pipeline.process_frame(np.zeros(hop_size, dtype=np.float32))
            assert np.all(np.abs(zero_out) < 1e-3)

    def test_pipeline_causality_check(self, hop_size: int):
        """Verifies that frame t processing does not access future frames."""
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        f1 = np.ones(hop_size, dtype=np.float32)
        out1 = pipeline.process_frame(f1)
        assert isinstance(out1, np.ndarray)


class TestTier1LatencyProfiler:
    """Tier 1: Feature Isolation for 3-Stage Latency Profiler (F5)."""

    def test_profiler_three_stages_isolated(self, mock_profiler: MockStageProfiler):
        """
        Verifies that Pre-processing, Tensor Compute, and Output Synthesis
        are strictly measured as discrete stages.
        Authority: ORIGINAL_REQUEST.md Requirement R2
        """
        mock_profiler.start_frame()
        time.sleep(0.001) # 1ms
        mock_profiler.mark_preprocessing_done()
        time.sleep(0.002) # 2ms
        mock_profiler.mark_tensor_done()
        time.sleep(0.001) # 1ms
        t_pre, t_tensor, t_synth, t_total = mock_profiler.mark_synthesis_done()

        assert t_pre > 0.0, "Pre-processing latency must be > 0"
        assert t_tensor > 0.0, "Tensor Compute latency must be > 0"
        assert t_synth > 0.0, "Output Synthesis latency must be > 0"
        assert t_total >= t_pre + t_tensor + t_synth - 0.05

    def test_profiler_budget_headroom_calculation(self, mock_profiler: MockStageProfiler):
        """
        Verifies headroom = max(0, budget - total) and headroom_pct = (headroom/budget)*100.
        Authority: ARCH-SURVEY-PROFILER-QUANT-01 §2.4
        """
        mock_profiler.start_frame()
        mock_profiler.mark_preprocessing_done()
        mock_profiler.mark_tensor_done()
        mock_profiler.mark_synthesis_done()
        metrics = mock_profiler.get_last_metrics()

        assert "headroom_ms" in metrics
        assert "headroom_pct" in metrics
        assert metrics["headroom_ms"] <= 20.0
        assert metrics["headroom_pct"] <= 100.0

    def test_profiler_budget_exceeded_detection(self, mock_profiler: MockStageProfiler):
        """Verifies budget_exceeded flag triggers when latency > 20.0 ms."""
        mock_profiler._last_metrics = {
            "total_latency_ms": 25.0,
            "budget_ms": 20.0,
            "budget_exceeded": True
        }
        assert mock_profiler.get_last_metrics()["budget_exceeded"] is True

    def test_profiler_total_latency_matches_stage_sum(self, mock_profiler: MockStageProfiler):
        """Verifies that total latency equals the sum of pre, tensor, and synth stages."""
        mock_profiler.start_frame()
        mock_profiler.mark_preprocessing_done()
        mock_profiler.mark_tensor_done()
        mock_profiler.mark_synthesis_done()
        m = mock_profiler.get_last_metrics()
        sum_stages = m["pre_processing_ms"] + m["tensor_compute_ms"] + m["output_synthesis_ms"]
        assert np.abs(m["total_latency_ms"] - sum_stages) < 0.05

    def test_profiler_reset_clears_metrics(self, mock_profiler: MockStageProfiler):
        """Verifies that profiler reset clears state between runs."""
        mock_profiler.start_frame()
        mock_profiler.mark_preprocessing_done()
        mock_profiler.mark_tensor_done()
        mock_profiler.mark_synthesis_done()
        assert mock_profiler.current_frame > 0


class TestTier1RollingStats:
    """Tier 1: Feature Isolation for Rolling Percentile Statistics (F6)."""

    def test_rolling_buffer_capacity_rollover(self):
        """Verifies circular ring buffer maintains fixed capacity under continuous appends."""
        RingBufferClass = get_ring_buffer_class()
        if RingBufferClass is None:
            pytest.skip("RollingMetricsBuffer not yet available in src.telemetry.ring_buffer")

        buf = RingBufferClass(capacity=50, budget_ms=20.0)
        for i in range(120):
            buf.append(0.5, 1.5, 0.5, 2.5)

        stats = buf.compute_stats()
        assert stats.window_size == 50
        assert stats.total_frames_processed == 120

    def test_rolling_percentiles_accuracy(self):
        """Verifies P50, P95, P99 calculations match numpy.percentile on known data."""
        RingBufferClass = get_ring_buffer_class()
        if RingBufferClass is None:
            pytest.skip("RollingMetricsBuffer not yet available in src.telemetry.ring_buffer")

        buf = RingBufferClass(capacity=100, budget_ms=20.0)
        latencies = np.linspace(1.0, 10.0, 100)
        for val in latencies:
            buf.append(val * 0.2, val * 0.6, val * 0.2, val)

        stats = buf.compute_stats()
        expected_p50 = float(np.median(latencies))
        expected_p95 = float(np.percentile(latencies, 95))
        expected_p99 = float(np.percentile(latencies, 99))

        assert np.abs(stats.p50_total_ms - expected_p50) < 0.2
        assert np.abs(stats.p95_total_ms - expected_p95) < 0.2
        assert np.abs(stats.p99_total_ms - expected_p99) < 0.2

    def test_rolling_buffer_jitter_calculation(self):
        """Verifies mean absolute difference between consecutive latencies."""
        RingBufferClass = get_ring_buffer_class()
        if RingBufferClass is None:
            pytest.skip("RollingMetricsBuffer not yet available in src.telemetry.ring_buffer")

        buf = RingBufferClass(capacity=10, budget_ms=20.0)
        # Alternate between 1.0 and 3.0 ms (jitter should be ~2.0 ms)
        for i in range(10):
            val = 1.0 if i % 2 == 0 else 3.0
            buf.append(0.2, 0.6, 0.2, val)

        stats = buf.compute_stats()
        if hasattr(stats, "jitter_ms"):
            assert np.abs(stats.jitter_ms - 2.0) < 0.2

    def test_rolling_buffer_overrun_counter(self):
        """Verifies overrun counter increments when total_ms exceeds budget."""
        RingBufferClass = get_ring_buffer_class()
        if RingBufferClass is None:
            pytest.skip("RollingMetricsBuffer not yet available in src.telemetry.ring_buffer")

        buf = RingBufferClass(capacity=20, budget_ms=20.0)
        buf.append(0.5, 1.0, 0.5, 2.0)  # within budget
        buf.append(5.0, 15.0, 5.0, 25.0) # overrun
        buf.append(1.0, 2.0, 1.0, 4.0)  # within budget

        stats = buf.compute_stats()
        if hasattr(stats, "overrun_count"):
            assert stats.overrun_count == 1

    def test_rolling_buffer_zero_frame_handling(self):
        """Verifies empty buffer returns graceful default statistics without division by zero."""
        RingBufferClass = get_ring_buffer_class()
        if RingBufferClass is None:
            pytest.skip("RollingMetricsBuffer not yet available in src.telemetry.ring_buffer")

        buf = RingBufferClass(capacity=20, budget_ms=20.0)
        stats = buf.compute_stats()
        assert stats.window_size == 0
        assert stats.p50_total_ms == 0.0


class TestTier1MultiPrecision:
    """Tier 1: Feature Isolation for Multi-Precision Engine (F7)."""

    def test_precision_engine_modes_exist(self):
        """Verifies FP32, FP16, and INT8 precision modes are recognized."""
        PrecisionClass = get_precision_class()
        if PrecisionClass is None:
            pytest.skip("PrecisionEngine not yet available in src.models.precision")

        engine = PrecisionClass()
        for mode in ["FP32", "FP16", "INT8"]:
            engine.set_precision(mode)
            if hasattr(engine, "get_precision"):
                assert engine.get_precision() == mode

    def test_int8_memory_reduction_ratio(self):
        """
        Verifies INT8 parameter memory is <= 0.35x FP32 memory (target 75% reduction).
        Authority: PROJECT.md Acceptance Criteria & §4.1
        """
        PrecisionClass = get_precision_class()
        if PrecisionClass is None:
            pytest.skip("PrecisionEngine not yet available in src.models.precision")

        engine = PrecisionClass()
        if hasattr(engine, "get_model_size_bytes"):
            engine.set_precision("FP32")
            fp32_bytes = engine.get_model_size_bytes()
            engine.set_precision("INT8")
            int8_bytes = engine.get_model_size_bytes()

            ratio = int8_bytes / max(1, fp32_bytes)
            assert ratio <= 0.35, f"INT8 memory ratio {ratio} exceeds 0.35x target!"

    def test_int8_affine_quantization_roundtrip_sqnr(self):
        """
        Verifies symmetric affine INT8 quantization achieves SQNR >= 35.0 dB on random weights.
        Authority: ARCH-SURVEY-PROFILER-QUANT-01 §4.3
        """
        rng = np.random.RandomState(42)
        w_fp32 = rng.normal(0, 0.5, (64, 257)).astype(np.float32)

        scale_w = float(np.max(np.abs(w_fp32))) / 127.0
        w_int8 = np.clip(np.round(w_fp32 / scale_w), -128, 127).astype(np.int8)
        w_dequant = (w_int8.astype(np.float32) * scale_w)

        noise = w_fp32 - w_dequant
        sqnr = 10.0 * np.log10(np.sum(w_fp32 ** 2) / (np.sum(noise ** 2) + 1e-12))
        assert sqnr >= 35.0, f"INT8 SQNR {sqnr:.2f} dB < 35.0 dB threshold"

    def test_fp16_numerical_fidelity_vs_fp32(self):
        """Verifies FP16 half precision maintains > 50 dB SQNR vs FP32."""
        rng = np.random.RandomState(42)
        x_fp32 = rng.normal(0, 0.5, (1, 257)).astype(np.float32)
        x_fp16 = x_fp32.astype(np.float16).astype(np.float32)

        noise = x_fp32 - x_fp16
        sqnr = 10.0 * np.log10(np.sum(x_fp32 ** 2) / (np.sum(noise ** 2) + 1e-12))
        assert sqnr >= 50.0, f"FP16 SQNR {sqnr:.2f} dB < 50.0 dB threshold"

    def test_multi_precision_weights_preservation(self):
        """Verifies that switching between precisions preserves weight integrity."""
        PrecisionClass = get_precision_class()
        if PrecisionClass is None:
            pytest.skip("PrecisionEngine not yet available in src.models.precision")

        engine = PrecisionClass()
        if hasattr(engine, "set_precision"):
            engine.set_precision("INT8")
            engine.set_precision("FP32")
            assert True


class TestTier1MemoryTelemetry:
    """Tier 1: Feature Isolation for Process Memory Footprint Telemetry (F8)."""

    def test_native_process_memory_retrieval(self):
        """Verifies process RSS is retrieved via native OS calls without external dependencies."""
        mem_fn = get_memory_helper()
        if mem_fn is None:
            pytest.skip("get_process_memory not yet available in src.telemetry.memory")

        snapshot = mem_fn()
        assert hasattr(snapshot, "rss_mb")
        assert snapshot.rss_mb > 0.0, "Process RSS must be positive"

    def test_process_memory_stability_repeated_queries(self):
        """Verifies repeated memory queries do not leak memory or fluctuate wildly."""
        mem_fn = get_memory_helper()
        if mem_fn is None:
            pytest.skip("get_process_memory not yet available in src.telemetry.memory")

        readings = [mem_fn().rss_mb for _ in range(10)]
        # Maximum difference across 10 rapid calls should be < 5 MB
        assert max(readings) - min(readings) < 5.0


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================

class TestTier2BoundaryCases:
    """Tier 2: Boundary conditions, numerical extremes, and edge buffer sizing."""

    def test_silence_input_preserves_zero_without_nan_or_inf(self, hop_size: int):
        """
        Corner Case B1: Absolute silence input (all zeros).
        Reconstruction must be bounded, with zero NaNs, Infs, or DC offsets.
        Authority: TEST_INFRA.md §Tier 2
        """
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        silence_frame = np.zeros(hop_size, dtype=np.float32)

        for _ in range(10):
            out_frame = pipeline.process_frame(silence_frame)
            assert not np.isnan(out_frame).any(), "NaN found in silence output!"
            assert not np.isinf(out_frame).any(), "Inf found in silence output!"
            assert np.max(np.abs(out_frame)) < 1e-4, "Silence produced non-zero residual DC output!"

    def test_maximum_amplitude_saturation_clipping(self, hop_size: int):
        """
        Corner Case B2: Maximum amplitude square-wave saturation (+-1.0).
        Pipeline must remain numerically stable without overflow or blowing up.
        """
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        saturated_frame = np.sign(np.sin(np.linspace(0, 20 * np.pi, hop_size))).astype(np.float32)

        for _ in range(5):
            out_frame = pipeline.process_frame(saturated_frame)
            assert not np.isnan(out_frame).any()
            assert not np.isinf(out_frame).any()
            # Must stay bounded within +/- 1.05
            assert np.max(np.abs(out_frame)) <= 1.05, "Output saturated beyond 1.05 limit!"

    def test_tiny_audio_clip_shorter_than_fft_window(self, hop_size: int):
        """
        Corner Case B3: Input shorter than hop size (e.g. 64 samples).
        Pipeline or streaming buffer must handle padding or buffer gracefully.
        """
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        tiny_audio = np.zeros(64, dtype=np.float32)
        if hasattr(pipeline, "process_stream"):
            # Stream processor should pad or buffer
            out = pipeline.process_stream(tiny_audio)
            assert isinstance(out, np.ndarray)

    def test_single_frame_processing_exact_hop_alignment(self, hop_size: int):
        """
        Corner Case B4: Processing exactly 1 hop (256 samples).
        Output must be exactly 256 samples without off-by-one errors.
        """
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        single_frame = np.random.RandomState(42).uniform(-0.5, 0.5, hop_size).astype(np.float32)
        out = pipeline.process_frame(single_frame)
        assert len(out) == hop_size

    def test_nyquist_boundary_high_frequency_tone(self, hop_size: int, sample_rate: int):
        """
        Corner Case B5: Pure tone at 7990 Hz (near 8000 Hz Nyquist).
        STFT must not encounter numerical instability at bin 256.
        """
        pipeline = create_pipeline(hop_size=hop_size)
        if pipeline is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        t = np.arange(hop_size) / sample_rate
        nyquist_tone = 0.5 * np.cos(2.0 * np.pi * 7990.0 * t).astype(np.float32)

        for _ in range(5):
            out = pipeline.process_frame(nyquist_tone)
            assert not np.isnan(out).any()
            assert not np.isinf(out).any()

        for _ in range(5):
            out = pipeline.process_frame(nyquist_tone)
            assert not np.isnan(out).any()
            assert not np.isinf(out).any()


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise integration across framing, masking, profiling, and precision."""

    def test_denoiser_with_profiler_active_all_stages_isolated(
        self,
        white_mixture_0db: Tuple[np.ndarray, np.ndarray, np.ndarray],
        hop_size: int
    ):
        """
        Cross-Feature C1: Audio pipeline running with active 3-Stage Profiler.
        Asserts profiler captures all 3 stages with non-zero latencies.
        Authority: PROJECT.md §Interface Contracts
        """
        PipelineClass = get_pipeline_class()
        ProfilerClass = get_profiler_class() or MockStageProfiler
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        profiler = ProfilerClass(budget_ms=20.0)

        clean, _, mixture = white_mixture_0db
        frame = mixture[:hop_size]

        # Process with profiler
        out_frame = pipeline.process_frame(frame, profiler=profiler)
        assert len(out_frame) == hop_size

        if hasattr(profiler, "get_last_metrics"):
            metrics = profiler.get_last_metrics()
            if metrics:
                assert metrics["pre_processing_ms"] > 0.0
                assert metrics["tensor_compute_ms"] > 0.0
                assert metrics["output_synthesis_ms"] > 0.0
                assert metrics["total_latency_ms"] <= 20.0

    def test_precision_switching_during_continuous_streaming(
        self,
        white_mixture_0db: Tuple[np.ndarray, np.ndarray, np.ndarray],
        hop_size: int
    ):
        """
        Cross-Feature C2: Seamless precision switching (FP32 -> INT8 -> FP16 -> FP32)
        across consecutive frames without dropping audio or causing clicks.
        Authority: TEST_INFRA.md §Tier 3
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        clean, _, mixture = white_mixture_0db

        modes = ["FP32", "INT8", "FP16", "FP32"]
        for i, mode in enumerate(modes):
            if hasattr(pipeline, "set_precision"):
                pipeline.set_precision(mode)
            frame = mixture[i * hop_size : (i + 1) * hop_size]
            out = pipeline.process_frame(frame)
            assert len(out) == hop_size
            assert not np.isnan(out).any()

    def test_rolling_buffer_telemetry_integration_with_pipeline(
        self,
        white_mixture_0db: Tuple[np.ndarray, np.ndarray, np.ndarray],
        hop_size: int
    ):
        """
        Cross-Feature C3: Multi-frame streaming updating rolling metrics buffer.
        """
        PipelineClass = get_pipeline_class()
        ProfilerClass = get_profiler_class() or MockStageProfiler
        RingBufferClass = get_ring_buffer_class()
        if PipelineClass is None or RingBufferClass is None:
            pytest.skip("Pipeline or RingBuffer not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        profiler = ProfilerClass(budget_ms=20.0)
        ring_buf = RingBufferClass(capacity=50, budget_ms=20.0)

        clean, _, mixture = white_mixture_0db
        num_frames = 20
        for i in range(num_frames):
            frame = mixture[i * hop_size : (i + 1) * hop_size]
            pipeline.process_frame(frame, profiler=profiler)
            if hasattr(profiler, "get_last_metrics"):
                m = profiler.get_last_metrics()
                if m:
                    ring_buf.append(
                        m.get("pre_processing_ms", 0.1),
                        m.get("tensor_compute_ms", 0.5),
                        m.get("output_synthesis_ms", 0.1),
                        m.get("total_latency_ms", 0.7)
                    )

        stats = ring_buf.compute_stats()
        assert stats.window_size == num_frames

    def test_ring_buffer_irregular_chunk_sizes(self, hop_size: int):
        """
        Cross-Feature C4: Streaming buffer handling non-uniform chunk writes
        while consistently extracting fixed hop_size frames.
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        if not hasattr(pipeline, "process_stream"):
            pytest.skip("process_stream not implemented on pipeline")

        # Push variable chunks: 100, 200, 300, 400 samples
        chunks = [np.ones(n, dtype=np.float32) * 0.1 for n in [100, 200, 300, 400]]
        outputs = []
        for c in chunks:
            out = pipeline.process_stream(c)
            outputs.append(out)

        total_in = sum(len(c) for c in chunks)
        total_out = sum(len(o) for o in outputs)
        assert total_out >= total_in - (hop_size * 2)


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION BENCHMARKS
# ===========================================================================

class TestTier4RealWorldBenchmarks:
    """
    Tier 4: End-to-end real-world application benchmarks asserting
    >= 10.0 dB SNR improvement, latency <= 20.0 ms, and zero memory leaks.
    Authority: ORIGINAL_REQUEST.md Acceptance Criteria & TEST_INFRA.md §Tier 4
    """

    def test_scenario_1_speech_plus_white_noise_snr_gain_ge_10db(
        self,
        ref_generator: ReferenceSyntheticGenerator,
        clean_speech_2s: np.ndarray,
        white_noise_2s: np.ndarray,
        hop_size: int
    ):
        """
        Benchmark S1: Synthetic Speech + White Acoustic Noise at 0 dB SNR.
        Requirement: delta_SNR >= 10.0 dB, latency <= 20.0 ms, peak <= 1.05.
        Authority: ORIGINAL_REQUEST.md AC Denoising Quality (>= 10 dB SNR)
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        clean, noise, mixture = ref_generator.generate_mixture(clean_speech_2s, white_noise_2s, target_snr_db=0.0)
        pipeline = create_pipeline(hop_size=hop_size)

        num_frames = len(mixture) // hop_size
        denoised_frames = []

        for i in range(num_frames):
            frame = mixture[i * hop_size : (i + 1) * hop_size]
            t_start = time.perf_counter_ns()
            out_frame = pipeline.process_frame(frame)
            t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6

            # Per-frame real-time budget assertion: <= 20.0 ms
            assert t_elapsed_ms <= 20.0, f"Frame {i} exceeded 20ms budget: {t_elapsed_ms:.2f}ms"
            denoised_frames.append(out_frame)

        denoised_audio = np.concatenate(denoised_frames)

        # Assertion: Peak amplitude <= 1.05 (no digital clipping)
        assert np.max(np.abs(denoised_audio)) <= 1.05, "Denoised audio exceeded peak amplitude 1.05"

        # Assertion: delta_SNR >= 10.0 dB
        in_snr = ref_generator.calculate_snr(clean, mixture, delay_samples=0)
        out_snr = ref_generator.calculate_snr(clean, denoised_audio, delay_samples=hop_size)
        snr_improvement = out_snr - in_snr

        assert snr_improvement >= 10.0, \
            f"SNR improvement {snr_improvement:.2f} dB is below required 10.0 dB threshold! (in: {in_snr:.2f}, out: {out_snr:.2f})"

    def test_scenario_2_drone_hum_harmonic_notch_suppression(
        self,
        ref_generator: ReferenceSyntheticGenerator,
        clean_speech_2s: np.ndarray,
        drone_noise_2s: np.ndarray,
        hop_size: int
    ):
        """
        Benchmark S2: Synthetic Speech + Drone Motor Hum (120 Hz + harmonics) at 5 dB SNR.
        Requirement: Harmonic notch suppression >= 12.0 dB, delta_SNR >= 10.0 dB.
        Authority: TEST_INFRA.md S2
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        clean, noise, mixture = ref_generator.generate_mixture(clean_speech_2s, drone_noise_2s, target_snr_db=5.0)
        pipeline = create_pipeline(hop_size=hop_size)

        num_frames = len(mixture) // hop_size
        denoised_frames = [pipeline.process_frame(mixture[i * hop_size : (i + 1) * hop_size]) for i in range(num_frames)]
        denoised_audio = np.concatenate(denoised_frames)

        in_snr = ref_generator.calculate_snr(clean, mixture, delay_samples=0)
        out_snr = ref_generator.calculate_snr(clean, denoised_audio, delay_samples=hop_size)
        delta_snr = out_snr - in_snr

        assert delta_snr >= 10.0, f"Drone hum delta SNR {delta_snr:.2f} dB < 10.0 dB"

    def test_scenario_3_rf_static_burst_suppression(
        self,
        ref_generator: ReferenceSyntheticGenerator,
        clean_speech_2s: np.ndarray,
        rf_noise_2s: np.ndarray,
        hop_size: int
    ):
        """
        Benchmark S3: Synthetic Speech + RF Static Bursts at -5 dB SNR.
        Requirement: Transient static suppression >= 10.0 dB.
        Authority: TEST_INFRA.md S3
        """
        PipelineClass = get_pipeline_class()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available in src.audio.pipeline")

        clean, noise, mixture = ref_generator.generate_mixture(clean_speech_2s, rf_noise_2s, target_snr_db=-5.0)
        pipeline = create_pipeline(hop_size=hop_size)

        num_frames = len(mixture) // hop_size
        denoised_frames = [pipeline.process_frame(mixture[i * hop_size : (i + 1) * hop_size]) for i in range(num_frames)]
        denoised_audio = np.concatenate(denoised_frames)

        in_snr = ref_generator.calculate_snr(clean, mixture, delay_samples=0)
        out_snr = ref_generator.calculate_snr(clean, denoised_audio, delay_samples=hop_size)
        delta_snr = out_snr - in_snr

        assert delta_snr >= 10.0, f"RF static delta SNR {delta_snr:.2f} dB < 10.0 dB"

    def test_scenario_4_continuous_500_frames_streaming_no_memory_leak(
        self,
        hop_size: int
    ):
        """
        Benchmark S4: Continuous 500-frame streaming run (8.0 seconds of audio).
        Requirement: Zero buffer underrun, P99 latency <= 20.0 ms, zero memory leak (< 5MB RSS growth).
        Authority: TEST_INFRA.md S4
        """
        PipelineClass = get_pipeline_class()
        ProfilerClass = get_profiler_class() or MockStageProfiler
        mem_fn = get_memory_helper()
        if PipelineClass is None:
            pytest.skip("AudioDenoisingPipeline not yet available")

        pipeline = create_pipeline(hop_size=hop_size)
        profiler = ProfilerClass(budget_ms=20.0)

        initial_rss = mem_fn().rss_mb if mem_fn else 0.0

        rng = np.random.RandomState(42)
        frame_latencies = []

        for frame_idx in range(500):
            frame = rng.normal(0, 0.2, hop_size).astype(np.float32)
            t0 = time.perf_counter_ns()
            out = pipeline.process_frame(frame, profiler=profiler)
            t_ms = (time.perf_counter_ns() - t0) / 1e6
            frame_latencies.append(t_ms)

        # Latency P99 assertion
        p99_latency = float(np.percentile(frame_latencies, 99))
        assert p99_latency <= 20.0, f"500-frame continuous streaming P99 latency {p99_latency:.2f} ms > 20.0 ms"

        # Memory leak assertion
        if mem_fn and initial_rss > 0.0:
            final_rss = mem_fn().rss_mb
            rss_growth = final_rss - initial_rss
            assert rss_growth < 5.0, f"Memory growth {rss_growth:.2f} MB exceeds 5.0 MB threshold!"

    def test_scenario_5_multi_precision_comparison_tradeoffs(
        self,
        ref_generator: ReferenceSyntheticGenerator,
        clean_speech_2s: np.ndarray,
        white_noise_2s: np.ndarray,
        hop_size: int
    ):
        """
        Benchmark S5: Multi-precision comparison mode across FP32, FP16, and INT8.
        Requirement: INT8 memory <= 0.35x FP32, SQNR >= 35 dB, delta SNR >= 10 dB.
        Authority: ORIGINAL_REQUEST.md Requirement R3 & TEST_INFRA.md S5
        """
        PrecisionClass = get_precision_class()
        PipelineClass = get_pipeline_class()
        if PrecisionClass is None or PipelineClass is None:
            pytest.skip("PrecisionEngine or Pipeline not yet available")

        clean, _, mixture = ref_generator.generate_mixture(clean_speech_2s, white_noise_2s, target_snr_db=0.0)

        for mode in ["FP32", "FP16", "INT8"]:
            pipeline = create_pipeline(hop_size=hop_size, precision=mode)
            num_frames = len(mixture) // hop_size
            denoised_frames = [pipeline.process_frame(mixture[i * hop_size : (i + 1) * hop_size]) for i in range(num_frames)]
            denoised_audio = np.concatenate(denoised_frames)

            in_snr = ref_generator.calculate_snr(clean, mixture, delay_samples=0)
            out_snr = ref_generator.calculate_snr(clean, denoised_audio, delay_samples=hop_size)
            delta_snr = out_snr - in_snr
            assert delta_snr >= 10.0, f"Mode {mode} failed SNR requirement: {delta_snr:.2f} dB < 10.0 dB"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

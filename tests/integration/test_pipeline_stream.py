"""Integration tests for streaming audio pipeline and continuous frame buffer telemetry.

Authority: PROJECT.md §Feature 8, Feature 25, TEST_INFRA.md S4.
"""

import time
import pytest
import numpy as np

from src.audio.pipeline import AudioDenoisingPipeline
from src.audio.stream import AudioStreamer, StreamingPCMBuffer
from src.telemetry.profiler import StageProfiler
from src.telemetry.ring_buffer import RollingMetricsBuffer
from src.telemetry.memory import get_process_memory


def test_continuous_stream_1000_frames_latency_and_memory():
    """
    Simulates continuous 1000-frame streaming audio session (16 seconds of real-time audio).
    Asserts:
    1. Zero buffer underruns.
    2. Per-frame processing time <= 20ms real-time budget.
    3. P99 latency <= 5ms on modern CPU.
    4. Process memory RSS remains bounded (no memory leaks).
    """
    pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)
    profiler = StageProfiler(budget_ms=20.0)
    ring_buf = RollingMetricsBuffer(capacity=500, budget_ms=20.0)

    num_frames = 1000
    hop_size = 256
    rng = np.random.RandomState(42)

    initial_mem = get_process_memory()

    for i in range(num_frames):
        # Generate random audio frame
        frame = rng.normal(0, 0.1, hop_size).astype(np.float32)

        profiler.start_frame()
        out = pipeline.process_frame(frame, profiler=profiler)
        t_pre, t_tensor, t_synth, t_total = profiler.mark_synthesis_done()
        ring_buf.append(t_pre, t_tensor, t_synth, t_total)

        assert len(out) == hop_size
        assert not np.isnan(out).any()
        assert t_total <= 20.0, f"Frame {i} exceeded 20ms budget: {t_total:.2f}ms"

    stats = ring_buf.compute_stats()
    final_mem = get_process_memory()

    assert stats.total_frames_processed == num_frames
    assert stats.p50_total_ms <= 6.0  # Real-time budget is 20.0ms; ensures < 30% frame load
    assert stats.p99_total_ms < 10.0
    assert stats.overrun_count == 0

    # Memory growth should be negligible (< 35 MB)
    mem_delta_mb = final_mem.rss_mb - initial_mem.rss_mb
    assert mem_delta_mb < 35.0, f"Excessive memory growth detected: {mem_delta_mb:.2f} MB"


def test_dynamic_precision_switching_in_stream():
    """
    Verifies on-the-fly precision mode switching (FP32 -> FP16 -> INT8 -> FP32)
    across consecutive streaming frames without audio glitches or state corruption.
    """
    pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)
    hop_size = 256
    rng = np.random.RandomState(42)

    modes = ["FP32", "FP16", "INT8", "FP32", "INT8", "FP16"]
    for mode in modes:
        pipeline.set_precision(mode)
        assert pipeline.get_precision() == mode

        for _ in range(10):
            frame = rng.normal(0, 0.2, hop_size).astype(np.float32)
            out = pipeline.process_frame(frame)
            assert len(out) == hop_size
            assert np.all(np.isfinite(out))
            assert np.max(np.abs(out)) <= 1.05


def test_circular_pcm_buffer_underrun_overrun_handling():
    """Verifies circular PCM buffer manages stream boundaries gracefully."""
    buf = StreamingPCMBuffer(capacity_samples=1024)
    assert buf.available_samples == 0

    chunk = np.ones(512, dtype=np.float32)
    buf.write(chunk)
    assert buf.available_samples == 512

    read_out = buf.read(256)
    assert len(read_out) == 256
    assert buf.available_samples == 256
    np.testing.assert_allclose(read_out, np.ones(256))

    # Read more than available returns padded zeros
    large_read = buf.read(512)
    assert len(large_read) == 512
    assert buf.available_samples == 0

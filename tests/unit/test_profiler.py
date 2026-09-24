"""Comprehensive Unit Tests for 3-Stage Profiler & Telemetry Subsystem.

Authority: ORIGINAL_REQUEST.md Requirement R2, PROJECT.md Milestone 2, TEST_INFRA.md F5/F6/F8.
Covers Tier 1 (Feature Isolation) and Tier 2 (Boundary & Corner Cases).
"""

import time
import pytest
import numpy as np

from src.telemetry.profiler import StageProfiler, FrameMetrics
from src.telemetry.ring_buffer import RollingMetricsBuffer, AggregatedStats
from src.telemetry.memory import get_process_memory, MemorySnapshot
from src.telemetry.schema import TelemetryPayload, PrecisionModeStats


# =============================================================================
# TIER 1: FEATURE ISOLATION TESTS
# =============================================================================

class TestTier1StageProfiler:
    """Tier 1: Feature Isolation for StageProfiler (F5)."""

    def test_profiler_three_stages_isolated_start_end(self):
        """Verifies start_stage and end_stage isolate pre, tensor, and synth stages."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_stage("pre_processing")
        time.sleep(0.001)
        t_pre = profiler.end_stage("pre_processing")

        profiler.start_stage("tensor_compute")
        time.sleep(0.002)
        t_tensor = profiler.end_stage("tensor_compute")

        profiler.start_stage("output_synthesis")
        time.sleep(0.001)
        t_synth = profiler.end_stage("output_synthesis")

        assert t_pre > 0.0
        assert t_tensor > 0.0
        assert t_synth > 0.0

        metrics = profiler.get_last_metrics()
        assert metrics["pre_processing_ms"] == t_pre
        assert metrics["tensor_compute_ms"] == t_tensor
        assert metrics["output_synthesis_ms"] == t_synth
        assert metrics["total_latency_ms"] >= t_pre + t_tensor + t_synth - 0.05

    def test_profiler_mark_api(self):
        """Verifies start_frame and mark_* methods return positive latencies."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_frame()
        time.sleep(0.001)
        profiler.mark_preprocessing_done()
        time.sleep(0.002)
        profiler.mark_tensor_done()
        time.sleep(0.001)
        t_pre, t_tensor, t_synth, t_total = profiler.mark_synthesis_done()

        assert t_pre > 0.0
        assert t_tensor > 0.0
        assert t_synth > 0.0
        assert t_total > 0.0
        assert np.abs(t_total - (t_pre + t_tensor + t_synth)) < 0.1

    def test_profiler_headroom_within_budget(self):
        """Verifies headroom calculation when latency is within real-time budget."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_frame()
        profiler.mark_preprocessing_done()
        profiler.mark_tensor_done()
        profiler.mark_synthesis_done()

        m = profiler.get_last_metrics()
        assert "headroom_ms" in m
        assert "headroom_pct" in m
        assert m["headroom_ms"] <= 20.0
        assert m["headroom_pct"] <= 100.0
        assert m["budget_exceeded"] is False

    def test_profiler_budget_exceeded_flag(self):
        """Verifies budget_exceeded triggers when budget is exceeded."""
        profiler = StageProfiler(budget_ms=0.0001)  # tiny budget to trigger overrun
        profiler.start_frame()
        time.sleep(0.002)
        profiler.mark_preprocessing_done()
        profiler.mark_tensor_done()
        profiler.mark_synthesis_done()

        m = profiler.get_last_metrics()
        assert m["budget_exceeded"] is True
        assert m["headroom_ms"] == 0.0
        assert m["headroom_pct"] == 0.0

    def test_profiler_dual_access_frame_metrics(self):
        """Verifies FrameMetrics supports both dictionary and property access."""
        m = FrameMetrics(
            seq=1,
            pre_processing_ms=0.5,
            tensor_compute_ms=1.5,
            output_synthesis_ms=0.5,
            total_latency_ms=2.5,
            budget_ms=20.0,
            headroom_ms=17.5,
            headroom_pct=87.5,
            budget_exceeded=False,
        )
        # Dict access
        assert m["seq"] == 1
        assert m["pre_processing_ms"] == 0.5
        assert m.get("tensor_compute_ms") == 1.5
        assert "headroom_ms" in m

        # Property access
        assert m.total_latency_ms == 2.5
        assert m.headroom_ms == 17.5
        assert m.headroom_pct == 87.5
        assert m.budget_exceeded is False

    def test_profiler_reset(self):
        """Verifies profiler reset clears state completely."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_frame()
        profiler.mark_preprocessing_done()
        profiler.mark_tensor_done()
        profiler.mark_synthesis_done()

        assert profiler.current_frame == 1
        assert profiler.get_last_metrics() != {}

        profiler.reset()
        assert profiler.current_frame == 0
        assert profiler.get_last_metrics() == {}

    def test_profiler_mixed_api_no_double_increment(self):
        """Verifies mixed API (start_frame + start/end_stage + mark_synthesis_done) counts exactly 1 frame."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_frame()
        profiler.start_stage("pre_processing")
        profiler.end_stage("pre_processing")
        profiler.start_stage("tensor_compute")
        profiler.end_stage("tensor_compute")
        profiler.start_stage("output_synthesis")
        profiler.end_stage("output_synthesis")
        profiler.mark_synthesis_done()

        assert profiler.current_frame == 1
        m = profiler.get_last_metrics()
        assert m["seq"] == 1
        assert m["total_latency_ms"] > 0.0


class TestTier1RollingBuffer:
    """Tier 1: Feature Isolation for RollingMetricsBuffer (F6)."""

    def test_ring_buffer_append_and_stats(self):
        """Verifies ring buffer tracks percentiles and window size accurately."""
        buf = RollingMetricsBuffer(capacity=100, budget_ms=20.0)
        for i in range(1, 101):
            buf.append(0.2, 0.6, 0.2, float(i))

        stats = buf.compute_stats()
        assert stats.window_size == 100
        assert stats.total_frames_processed == 100
        assert np.abs(stats.p50_total_ms - 50.5) < 0.5
        assert np.abs(stats.p95_total_ms - 95.05) < 0.5
        assert np.abs(stats.p99_total_ms - 99.01) < 0.5

    def test_ring_buffer_circular_rollover(self):
        """Verifies ring buffer rolls over correctly when frames exceed capacity."""
        buf = RollingMetricsBuffer(capacity=20, budget_ms=20.0)
        for i in range(50):
            buf.append(0.1, 0.2, 0.1, 1.0)

        stats = buf.compute_stats()
        assert stats.window_size == 20
        assert stats.total_frames_processed == 50
        assert stats.p50_total_ms == 1.0

    def test_ring_buffer_jitter(self):
        """Verifies jitter calculation computes mean absolute consecutive delta."""
        buf = RollingMetricsBuffer(capacity=10, budget_ms=20.0)
        for i in range(10):
            val = 2.0 if i % 2 == 0 else 4.0
            buf.append(0.2, 0.4, 0.2, val)

        stats = buf.compute_stats()
        assert np.abs(stats.jitter_ms - 2.0) < 0.1

    def test_ring_buffer_overrun_tracking(self):
        """Verifies overrun counter increments when total_ms > budget_ms."""
        buf = RollingMetricsBuffer(capacity=10, budget_ms=20.0)
        buf.append(0.1, 0.2, 0.1, 15.0)  # ok
        buf.append(0.1, 0.2, 0.1, 25.0)  # overrun
        buf.append(0.1, 0.2, 0.1, 10.0)  # ok
        buf.append(0.1, 0.2, 0.1, 30.0)  # overrun

        stats = buf.compute_stats()
        assert stats.overrun_count == 2
        assert stats.overrun_pct == 50.0

    def test_ring_buffer_zero_frames(self):
        """Verifies empty buffer returns safe defaults without ZeroDivisionError."""
        buf = RollingMetricsBuffer(capacity=50, budget_ms=20.0)
        stats = buf.compute_stats()

        assert stats.window_size == 0
        assert stats.total_frames_processed == 0
        assert stats.p50_total_ms == 0.0
        assert stats.headroom_median_ms == 20.0
        assert stats.headroom_pct == 100.0

    def test_ring_buffer_clear_and_reset(self):
        """Verifies clear() and reset() reset state completely."""
        buf = RollingMetricsBuffer(capacity=50, budget_ms=20.0)
        buf.append(0.1, 0.2, 0.1, 0.4)
        assert buf.count == 1
        buf.clear()
        assert buf.count == 0
        assert buf.compute_stats().window_size == 0


class TestTier1MemoryTelemetry:
    """Tier 1: Feature Isolation for Process Memory Telemetry (F8)."""

    def test_memory_snapshot_rss_positive(self):
        """Verifies get_process_memory returns positive RSS in MB."""
        snapshot = get_process_memory()
        assert isinstance(snapshot, MemorySnapshot)
        assert hasattr(snapshot, "rss_mb")
        assert snapshot.rss_mb > 0.0

    def test_memory_snapshot_stability(self):
        """Verifies rapid memory queries do not leak memory or fluctuate wildly."""
        readings = [get_process_memory().rss_mb for _ in range(10)]
        assert max(readings) - min(readings) < 5.0


class TestTier1SchemaSerialization:
    """Tier 1: Feature Isolation for TelemetryPayload & PrecisionModeStats."""

    def test_telemetry_payload_flat_and_nested_dict(self):
        """Verifies serialization format matching WebSocket and PROJECT.md schemas."""
        payload = TelemetryPayload(
            stage_pre_ms=0.1,
            stage_tensor_ms=0.5,
            stage_synth_ms=0.2,
            total_latency_ms=0.8,
            budget_ms=20.0,
            headroom_ms=19.2,
            headroom_pct=96.0,
            rolling_p50_ms=0.79,
            rolling_p95_ms=0.85,
            rolling_p99_ms=0.92,
            speedup_vs_realtime=25.0,
            snr_in_db=0.0,
            snr_out_db=12.5,
            snr_delta_db=12.5,
            model_memory_kb=152.8,
            model_memory_reduction_pct=75.0,
            process_rss_mb=28.4,
            precision_mode="INT8",
        )

        flat = payload.to_dict()
        assert flat["stage_pre_ms"] == 0.1
        assert flat["precision_mode"] == "INT8"

        nested = payload.to_nested_dict()
        assert nested["type"] == "telemetry"
        assert nested["stages"]["pre_processing_ms"] == 0.1
        assert nested["budget"]["budget_ms"] == 20.0
        assert nested["rolling_stats"]["median_p50_ms"] == 0.79
        assert nested["metrics"]["snr_delta_db"] == 12.5

    def test_precision_mode_stats(self):
        """Verifies PrecisionModeStats dataclass serialization."""
        pms = PrecisionModeStats(
            precision="INT8",
            latency_p50_ms=1.95,
            latency_p95_ms=2.28,
            latency_p99_ms=2.65,
            speedup_factor=1.46,
            model_size_kb=152.8,
            memory_reduction_pct=75.0,
            process_rss_mb=34.8,
            snr_improvement_db=13.38,
            snr_delta_vs_fp32_db=-0.14,
        )
        d = pms.to_dict()
        assert d["precision"] == "INT8"
        assert d["memory_reduction_pct"] == 75.0


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2BoundaryCases:
    """Tier 2: Boundary conditions, extremes, and robustness."""

    def test_profiler_minimum_time_clamp(self):
        """Verifies sub-microsecond stages are clamped to positive non-zero."""
        profiler = StageProfiler(budget_ms=20.0)
        profiler.start_stage("pre_processing")
        t_pre = profiler.end_stage("pre_processing")
        assert t_pre >= 0.001, "Stage latency must be clamped to >= 0.001 ms"

    def test_ring_buffer_capacity_one(self):
        """Verifies buffer with capacity=1 behaves correctly without error."""
        buf = RollingMetricsBuffer(capacity=1, budget_ms=20.0)
        buf.append(0.1, 0.2, 0.1, 1.0)
        buf.append(0.2, 0.4, 0.2, 2.0)

        stats = buf.compute_stats()
        assert stats.window_size == 1
        assert stats.total_frames_processed == 2
        assert stats.p50_total_ms == 2.0
        assert stats.jitter_ms == 0.0

    def test_ring_buffer_identical_latencies_zero_jitter(self):
        """Verifies identical latencies result in 0.0 jitter and equal percentiles."""
        buf = RollingMetricsBuffer(capacity=10, budget_ms=20.0)
        for _ in range(10):
            buf.append(0.1, 0.3, 0.1, 1.5)

        stats = buf.compute_stats()
        assert stats.jitter_ms == 0.0
        assert stats.p50_total_ms == 1.5
        assert stats.p95_total_ms == 1.5
        assert stats.p99_total_ms == 1.5

    def test_ring_buffer_reset_clears_all(self):
        """Verifies buffer reset returns to zero state."""
        buf = RollingMetricsBuffer(capacity=10, budget_ms=20.0)
        buf.append(0.1, 0.2, 0.1, 25.0)
        assert buf.compute_stats().overrun_count == 1

        buf.reset()
        stats = buf.compute_stats()
        assert stats.window_size == 0
        assert stats.total_frames_processed == 0
        assert stats.overrun_count == 0

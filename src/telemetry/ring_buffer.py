"""Thread-safe circular ring buffer for rolling latency percentile telemetry.

Preallocates fixed memory to record stage latencies without garbage collection
pressure, computing P50, P95, P99 percentiles, jitter, and overrun stats.
"""

from __future__ import annotations
import threading
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class AggregatedStats:
    """Aggregated latency and headroom statistics over the rolling window."""

    window_size: int
    total_frames_processed: int
    p50_total_ms: float
    p95_total_ms: float
    p99_total_ms: float
    mean_total_ms: float
    min_total_ms: float
    max_total_ms: float
    jitter_ms: float
    p50_pre_ms: float
    p50_tensor_ms: float
    p50_synth_ms: float
    headroom_median_ms: float
    headroom_p95_ms: float
    headroom_pct: float
    realtime_factor: float
    throughput_fps: float
    overrun_count: int
    overrun_pct: float


class RollingMetricsBuffer:
    """Thread-safe circular ring buffer for rolling latency metrics.

    Parameters
    ----------
    capacity : int
        Maximum number of frame records stored in the circular window (default: 100).
    budget_ms : float
        Real-time per-frame budget constraint in ms (default: 20.0 ms).
    """

    def __init__(self, capacity: int = 100, budget_ms: float = 20.0) -> None:
        self.capacity = max(1, int(capacity))
        self.budget_ms = float(budget_ms)
        # Preallocated storage: columns = [pre_ms, tensor_ms, synth_ms, total_ms]
        self.data = np.zeros((self.capacity, 4), dtype=np.float64)
        self.head = 0
        self.count = 0
        self.total_frames = 0
        self.overrun_count = 0
        self._lock = threading.Lock()

    def append(
        self,
        pre_ms: float,
        tensor_ms: float,
        synth_ms: float,
        total_ms: float,
    ) -> None:
        """Append a single frame's stage and total latencies into the ring buffer."""
        with self._lock:
            self.data[self.head, 0] = pre_ms
            self.data[self.head, 1] = tensor_ms
            self.data[self.head, 2] = synth_ms
            self.data[self.head, 3] = total_ms

            self.head = (self.head + 1) % self.capacity
            if self.count < self.capacity:
                self.count += 1
            self.total_frames += 1

            if total_ms > self.budget_ms:
                self.overrun_count += 1

    def compute_stats(self) -> AggregatedStats:
        """Compute rolling percentiles, jitter, headroom, and overrun statistics."""
        with self._lock:
            k = self.count
            if k == 0:
                return AggregatedStats(
                    window_size=0,
                    total_frames_processed=self.total_frames,
                    p50_total_ms=0.0,
                    p95_total_ms=0.0,
                    p99_total_ms=0.0,
                    mean_total_ms=0.0,
                    min_total_ms=0.0,
                    max_total_ms=0.0,
                    jitter_ms=0.0,
                    p50_pre_ms=0.0,
                    p50_tensor_ms=0.0,
                    p50_synth_ms=0.0,
                    headroom_median_ms=self.budget_ms,
                    headroom_p95_ms=self.budget_ms,
                    headroom_pct=100.0,
                    realtime_factor=0.0,
                    throughput_fps=0.0,
                    overrun_count=self.overrun_count,
                    overrun_pct=0.0,
                )

            # Extract chronological active window
            if self.total_frames < self.capacity:
                active_data = self.data[:k].copy()
            else:
                # Ring has wrapped around; unroll chronologically
                active_data = np.empty((self.capacity, 4), dtype=np.float64)
                active_data[: self.capacity - self.head] = self.data[self.head :]
                active_data[self.capacity - self.head :] = self.data[: self.head]

            total_frames = self.total_frames
            overruns = self.overrun_count

        totals = active_data[:, 3]
        p50 = float(np.median(totals))
        p95 = float(np.percentile(totals, 95))
        p99 = float(np.percentile(totals, 99))
        mean_val = float(np.mean(totals))
        min_val = float(np.min(totals))
        max_val = float(np.max(totals))

        # Jitter: mean absolute delta between consecutive frame total latencies
        jitter = float(np.mean(np.abs(np.diff(totals)))) if k > 1 else 0.0

        p50_pre = float(np.median(active_data[:, 0]))
        p50_tensor = float(np.median(active_data[:, 1]))
        p50_synth = float(np.median(active_data[:, 2]))

        headroom_p50 = max(0.0, self.budget_ms - p50)
        headroom_p95 = max(0.0, self.budget_ms - p95)
        headroom_pct = (headroom_p50 / self.budget_ms) * 100.0 if self.budget_ms > 0 else 0.0

        rtf = (p50 / self.budget_ms) if self.budget_ms > 0 else 0.0
        throughput = (1000.0 / p50) if p50 > 0 else 0.0
        overrun_pct = (overruns / max(1, total_frames)) * 100.0

        return AggregatedStats(
            window_size=k,
            total_frames_processed=total_frames,
            p50_total_ms=round(p50, 3),
            p95_total_ms=round(p95, 3),
            p99_total_ms=round(p99, 3),
            mean_total_ms=round(mean_val, 3),
            min_total_ms=round(min_val, 3),
            max_total_ms=round(max_val, 3),
            jitter_ms=round(jitter, 3),
            p50_pre_ms=round(p50_pre, 3),
            p50_tensor_ms=round(p50_tensor, 3),
            p50_synth_ms=round(p50_synth, 3),
            headroom_median_ms=round(headroom_p50, 3),
            headroom_p95_ms=round(headroom_p95, 3),
            headroom_pct=round(headroom_pct, 2),
            realtime_factor=round(rtf, 4),
            throughput_fps=round(throughput, 1),
            overrun_count=overruns,
            overrun_pct=round(overrun_pct, 2),
        )

    def reset(self) -> None:
        """Reset the buffer state and clear all recorded metrics."""
        with self._lock:
            self.data.fill(0.0)
            self.head = 0
            self.count = 0
            self.total_frames = 0
            self.overrun_count = 0

    def clear(self) -> None:
        """Alias for reset() to match circular buffer naming conventions."""
        self.reset()

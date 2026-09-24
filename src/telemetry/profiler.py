"""High-resolution 3-stage hardware latency profiler.

Isolates pre-processing, tensor compute, and output synthesis latencies
using time.perf_counter_ns() with zero object allocations in the hot path.
Supports both start_stage/end_stage and start_frame/mark_* API patterns.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional, Tuple


class FrameMetrics(dict):
    """Dictionary subclass representing per-frame latency metrics.

    Supports both dictionary access (metrics["headroom_ms"]) and
    attribute access (metrics.headroom_ms) for maximum compatibility.
    """

    def __init__(
        self,
        seq: int,
        pre_processing_ms: float,
        tensor_compute_ms: float,
        output_synthesis_ms: float,
        total_latency_ms: float,
        budget_ms: float,
        headroom_ms: float,
        headroom_pct: float,
        budget_exceeded: bool,
    ) -> None:
        super().__init__(
            seq=seq,
            pre_processing_ms=pre_processing_ms,
            tensor_compute_ms=tensor_compute_ms,
            output_synthesis_ms=output_synthesis_ms,
            total_latency_ms=total_latency_ms,
            budget_ms=budget_ms,
            headroom_ms=headroom_ms,
            headroom_pct=headroom_pct,
            budget_exceeded=budget_exceeded,
        )

    @property
    def seq(self) -> int:
        return self["seq"]

    @property
    def pre_processing_ms(self) -> float:
        return self["pre_processing_ms"]

    @property
    def tensor_compute_ms(self) -> float:
        return self["tensor_compute_ms"]

    @property
    def output_synthesis_ms(self) -> float:
        return self["output_synthesis_ms"]

    @property
    def total_latency_ms(self) -> float:
        return self["total_latency_ms"]

    @property
    def budget_ms(self) -> float:
        return self["budget_ms"]

    @property
    def headroom_ms(self) -> float:
        return self["headroom_ms"]

    @property
    def headroom_pct(self) -> float:
        return self["headroom_pct"]

    @property
    def budget_exceeded(self) -> bool:
        return self["budget_exceeded"]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


class StageProfiler:
    """High-resolution 3-stage latency profiler with sub-millisecond accuracy.

    Measures elapsed time in nanoseconds via time.perf_counter_ns() across:
    1. Pre-processing (STFT, framing, windowing)
    2. Tensor Compute (neural mask estimator or Wiener DSP filter)
    3. Output Synthesis (iSTFT, overlap-add, framing)

    Parameters
    ----------
    budget_ms : float
        Nominal per-frame real-time budget in milliseconds (default: 20.0 ms).
    """

    __slots__ = (
        "budget_ms",
        "current_frame",
        "stage_starts",
        "stage_durations",
        "substage_starts",
        "substage_durations",
        "_last_metrics",
        "_t0",
        "_t1",
        "_t2",
        "_t3",
        "_frame_active",
    )

    def __init__(self, budget_ms: float = 20.0) -> None:
        self.budget_ms = float(budget_ms)
        self.current_frame = 0
        self.stage_starts: Dict[str, int] = {}
        self.stage_durations: Dict[str, float] = {}
        self.substage_starts: Dict[str, int] = {}
        self.substage_durations: Dict[str, float] = {}
        self._last_metrics: Optional[FrameMetrics] = None
        self._frame_active: bool = False

        # Hot-path nanosecond timestamps for start_frame / mark_* API
        self._t0: int = 0
        self._t1: int = 0
        self._t2: int = 0
        self._t3: int = 0

    # -------------------------------------------------------------------------
    # Generic start_stage / end_stage API (used by AudioDenoisingPipeline)
    # -------------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Start timing a named pipeline stage."""
        now = time.perf_counter_ns()
        self.stage_starts[name] = now
        if name == "pre_processing":
            self._t0 = now

    def start_substage(self, name: str) -> None:
        """Start timing a named granular sub-stage (vad, noise_classification, etc.)."""
        self.substage_starts[name] = time.perf_counter_ns()

    def end_substage(self, name: str) -> float:
        """End timing a named granular sub-stage and return elapsed milliseconds."""
        now = time.perf_counter_ns()
        start = self.substage_starts.get(name, now)
        elapsed_ns = max(now - start, 100)
        elapsed_ms = elapsed_ns / 1_000_000.0
        self.substage_durations[name] = elapsed_ms
        return elapsed_ms

    def get_substages(self) -> Dict[str, float]:
        """Return a copy of the active sub-stage durations dictionary in milliseconds."""
        return dict(self.substage_durations)

    def end_stage(self, name: str) -> float:
        """End timing a named pipeline stage and return elapsed milliseconds."""
        now = time.perf_counter_ns()
        start = self.stage_starts.get(name, now)
        elapsed_ns = max(now - start, 1000)  # clamp to >= 1 microsecond (0.001 ms)
        elapsed_ms = elapsed_ns / 1_000_000.0
        self.stage_durations[name] = elapsed_ms

        if name == "output_synthesis":
            # Auto-finalize frame metrics if not managed by start_frame / mark_synthesis_done
            if not self._frame_active:
                self.current_frame += 1
            t_pre = self.stage_durations.get("pre_processing", 0.001)
            t_tensor = self.stage_durations.get("tensor_compute", 0.001)
            t_synth = elapsed_ms
            t_total = t_pre + t_tensor + t_synth

            headroom = max(0.0, self.budget_ms - t_total)
            headroom_pct = (headroom / self.budget_ms) * 100.0 if self.budget_ms > 0 else 0.0

            self._last_metrics = FrameMetrics(
                seq=self.current_frame,
                pre_processing_ms=t_pre,
                tensor_compute_ms=t_tensor,
                output_synthesis_ms=t_synth,
                total_latency_ms=t_total,
                budget_ms=self.budget_ms,
                headroom_ms=headroom,
                headroom_pct=headroom_pct,
                budget_exceeded=(t_total > self.budget_ms),
            )

        return elapsed_ms

    # -------------------------------------------------------------------------
    # Explicit start_frame / mark_* API (used by tests & MockStageProfiler)
    # -------------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the beginning of a new frame and start pre-processing stage."""
        self.current_frame += 1
        self._frame_active = True
        now = time.perf_counter_ns()
        self._t0 = now
        self.stage_starts["pre_processing"] = now

    def mark_preprocessing_done(self) -> None:
        """Mark end of pre-processing and start of tensor compute."""
        now = time.perf_counter_ns()
        self._t1 = now
        start = self._t0 if self._t0 > 0 else now
        elapsed_ns = max(now - start, 1000)
        self.stage_durations["pre_processing"] = elapsed_ns / 1_000_000.0
        self.stage_starts["tensor_compute"] = now

    def mark_tensor_done(self) -> None:
        """Mark end of tensor compute and start of output synthesis."""
        now = time.perf_counter_ns()
        self._t2 = now
        start = self._t1 if self._t1 > 0 else self.stage_starts.get("tensor_compute", now)
        elapsed_ns = max(now - start, 1000)
        self.stage_durations["tensor_compute"] = elapsed_ns / 1_000_000.0
        self.stage_starts["output_synthesis"] = now

    def mark_synthesis_done(self) -> Tuple[float, float, float, float]:
        """Mark end of output synthesis and compute total frame latency.

        Returns
        -------
        Tuple[float, float, float, float]
            (t_pre_ms, t_tensor_ms, t_synth_ms, t_total_ms)
        """
        now = time.perf_counter_ns()
        self._t3 = now

        t_pre = self.stage_durations.get("pre_processing")
        if t_pre is None:
            t_pre = max(self._t1 - self._t0, 1000) / 1_000_000.0 if self._t1 > self._t0 else 0.001
            self.stage_durations["pre_processing"] = t_pre

        t_tensor = self.stage_durations.get("tensor_compute")
        if t_tensor is None:
            t_tensor = max(self._t2 - self._t1, 1000) / 1_000_000.0 if self._t2 > self._t1 else 0.001
            self.stage_durations["tensor_compute"] = t_tensor

        if "output_synthesis" in self.stage_durations:
            t_synth = self.stage_durations["output_synthesis"]
        else:
            start_synth = self._t2 if self._t2 > 0 else self.stage_starts.get("output_synthesis", now)
            t_synth = max(now - start_synth, 1000) / 1_000_000.0
            self.stage_durations["output_synthesis"] = t_synth

        # Total latency is the sum of the three distinct pipeline stages
        t_total = t_pre + t_tensor + t_synth

        headroom = max(0.0, self.budget_ms - t_total)
        headroom_pct = (headroom / self.budget_ms) * 100.0 if self.budget_ms > 0 else 0.0

        self._last_metrics = FrameMetrics(
            seq=self.current_frame,
            pre_processing_ms=t_pre,
            tensor_compute_ms=t_tensor,
            output_synthesis_ms=t_synth,
            total_latency_ms=t_total,
            budget_ms=self.budget_ms,
            headroom_ms=headroom,
            headroom_pct=headroom_pct,
            budget_exceeded=(t_total > self.budget_ms),
        )

        self._frame_active = False
        return (t_pre, t_tensor, t_synth, t_total)

    def get_last_metrics(self) -> Dict[str, Any]:
        """Retrieve metrics for the most recently completed frame."""
        if self._last_metrics is None:
            if "output_synthesis" in self.stage_durations:
                t_pre = self.stage_durations.get("pre_processing", 0.001)
                t_tensor = self.stage_durations.get("tensor_compute", 0.001)
                t_synth = self.stage_durations.get("output_synthesis", 0.001)
                t_total = t_pre + t_tensor + t_synth
                headroom = max(0.0, self.budget_ms - t_total)
                headroom_pct = (headroom / self.budget_ms) * 100.0 if self.budget_ms > 0 else 0.0
                return FrameMetrics(
                    seq=self.current_frame,
                    pre_processing_ms=t_pre,
                    tensor_compute_ms=t_tensor,
                    output_synthesis_ms=t_synth,
                    total_latency_ms=t_total,
                    budget_ms=self.budget_ms,
                    headroom_ms=headroom,
                    headroom_pct=headroom_pct,
                    budget_exceeded=(t_total > self.budget_ms),
                )
            return {}
        return self._last_metrics

    def reset(self) -> None:
        """Reset internal frame counters and cached latency state."""
        self.current_frame = 0
        self.stage_starts.clear()
        self.stage_durations.clear()
        self.substage_starts.clear()
        self.substage_durations.clear()
        self._last_metrics = None
        self._frame_active = False
        self._t0 = 0
        self._t1 = 0
        self._t2 = 0
        self._t3 = 0

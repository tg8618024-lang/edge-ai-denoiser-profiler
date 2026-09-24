"""Data models and serialization schemas for telemetry and precision metrics."""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass(slots=True)
class PrecisionModeStats:
    """Benchmark metrics comparing FP32, FP16, and INT8 precision modes."""

    precision: str  # "FP32", "FP16", "INT8"
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    speedup_factor: float
    model_size_kb: float
    memory_reduction_pct: float
    process_rss_mb: float
    snr_improvement_db: float
    snr_delta_vs_fp32_db: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TelemetryPayload:
    """Complete per-frame telemetry payload matching M2 <-> M3 WebSocket contracts."""

    stage_pre_ms: float = 0.0
    stage_tensor_ms: float = 0.0
    stage_synth_ms: float = 0.0
    total_latency_ms: float = 0.0
    vad_ms: float = 0.0
    noise_classification_ms: float = 0.0
    adaptive_controller_ms: float = 0.0
    dereverberation_ms: float = 0.0
    budget_ms: float = 20.0
    headroom_ms: float = 20.0
    headroom_pct: float = 100.0
    rolling_p50_ms: float = 0.0
    rolling_p95_ms: float = 0.0
    rolling_p99_ms: float = 0.0
    speedup_vs_realtime: float = 0.0
    snr_in_db: float = 0.0
    snr_out_db: float = 0.0
    snr_delta_db: float = 0.0
    model_memory_kb: float = 0.0
    model_memory_reduction_pct: float = 0.0
    process_rss_mb: float = 0.0
    precision_mode: str = "FP32"
    timestamp_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize as flat dictionary for WebSocket streaming."""
        d = asdict(self)
        if self.timestamp_ns == 0:
            d["timestamp_ns"] = time.time_ns()
        return d

    def to_nested_dict(self) -> Dict[str, Any]:
        """Serialize matching PROJECT.md §M2 <-> M3 Contract exact nested schema."""
        ts = self.timestamp_ns if self.timestamp_ns > 0 else time.time_ns()
        return {
            "type": "telemetry",
            "timestamp_ns": ts,
            "precision_mode": self.precision_mode,
            "stages": {
                "pre_processing_ms": round(self.stage_pre_ms, 3),
                "tensor_compute_ms": round(self.stage_tensor_ms, 3),
                "output_synthesis_ms": round(self.stage_synth_ms, 3),
                "total_latency_ms": round(self.total_latency_ms, 3),
            },
            "substages": {
                "vad_ms": round(self.vad_ms, 3),
                "noise_classification_ms": round(self.noise_classification_ms, 3),
                "adaptive_controller_ms": round(self.adaptive_controller_ms, 3),
                "dereverberation_ms": round(self.dereverberation_ms, 3),
            },
            "budget": {
                "budget_ms": round(self.budget_ms, 3),
                "headroom_ms": round(self.headroom_ms, 3),
                "headroom_pct": round(self.headroom_pct, 2),
            },
            "rolling_stats": {
                "median_p50_ms": round(self.rolling_p50_ms, 3),
                "p95_ms": round(self.rolling_p95_ms, 3),
                "p99_ms": round(self.rolling_p99_ms, 3),
            },
            "metrics": {
                "snr_input_db": round(self.snr_in_db, 2),
                "snr_output_db": round(self.snr_out_db, 2),
                "snr_delta_db": round(self.snr_delta_db, 2),
                "model_memory_kb": round(self.model_memory_kb, 1),
                "process_rss_mb": round(self.process_rss_mb, 2),
            },
        }

    @classmethod
    def from_profiler_and_stats(
        cls,
        profiler_metrics: Dict[str, Any],
        stats: Optional[Any] = None,
        snr_in: float = 0.0,
        snr_out: float = 0.0,
        model_memory_kb: float = 0.0,
        reduction_pct: float = 0.0,
        rss_mb: float = 0.0,
        precision: str = "FP32",
    ) -> TelemetryPayload:
        """Construct payload from active profiler metrics and rolling stats."""
        p50 = getattr(stats, "p50_total_ms", 0.0) if stats else 0.0
        p95 = getattr(stats, "p95_total_ms", 0.0) if stats else 0.0
        p99 = getattr(stats, "p99_total_ms", 0.0) if stats else 0.0
        rtf = getattr(stats, "realtime_factor", 0.0) if stats else 0.0
        speedup = (1.0 / rtf) if rtf > 0 else 0.0

        t_pre = profiler_metrics.get("pre_processing_ms", 0.0)
        t_tensor = profiler_metrics.get("tensor_compute_ms", 0.0)
        t_synth = profiler_metrics.get("output_synthesis_ms", 0.0)
        t_total = profiler_metrics.get("total_latency_ms", 0.0)
        budget = profiler_metrics.get("budget_ms", 20.0)
        headroom = profiler_metrics.get("headroom_ms", max(0.0, budget - t_total))
        headroom_pct = profiler_metrics.get(
            "headroom_pct", (headroom / budget) * 100.0 if budget > 0 else 0.0
        )
        t_vad = profiler_metrics.get("vad_ms", 0.0)
        t_nc = profiler_metrics.get("noise_classification_ms", 0.0)
        t_ac = profiler_metrics.get("adaptive_controller_ms", 0.0)
        t_dr = profiler_metrics.get("dereverberation_ms", 0.0)

        return cls(
            stage_pre_ms=t_pre,
            stage_tensor_ms=t_tensor,
            stage_synth_ms=t_synth,
            total_latency_ms=t_total,
            vad_ms=t_vad,
            noise_classification_ms=t_nc,
            adaptive_controller_ms=t_ac,
            dereverberation_ms=t_dr,
            budget_ms=budget,
            headroom_ms=headroom,
            headroom_pct=headroom_pct,
            rolling_p50_ms=p50,
            rolling_p95_ms=p95,
            rolling_p99_ms=p99,
            speedup_vs_realtime=round(speedup, 2),
            snr_in_db=snr_in,
            snr_out_db=snr_out,
            snr_delta_db=snr_out - snr_in,
            model_memory_kb=model_memory_kb,
            model_memory_reduction_pct=reduction_pct,
            process_rss_mb=rss_mb,
            precision_mode=precision,
            timestamp_ns=time.time_ns(),
        )

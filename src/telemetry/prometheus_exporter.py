"""Prometheus Time-Series Metrics Exporter for Edge AI Audio Pipeline.

Authority: Phase 2 Roadmap Requirement (Observability Upgrade).
Provides zero-overhead Prometheus instrumentation for real-time stage latencies,
audio enhancement metrics, memory footprint, active WebSocket clients, and VAD savings.
"""

from __future__ import annotations
import threading
from typing import Optional
import prometheus_client
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Real-time latency histogram buckets (in seconds): 0.1ms up to 50ms
LATENCY_BUCKETS = (
    0.0001,   # 0.10 ms
    0.00025,  # 0.25 ms
    0.0005,   # 0.50 ms
    0.001,    # 1.00 ms
    0.002,    # 2.00 ms
    0.004,    # 4.00 ms
    0.008,    # 8.00 ms
    0.012,    # 12.00 ms
    0.016,    # 16.00 ms (nominal 16ms budget)
    0.020,    # 20.00 ms (20ms budget limit)
    0.025,    # 25.00 ms
    0.050,    # 50.00 ms
)


class PrometheusMetricsExporter:
    """Zero-overhead Prometheus time-series metrics exporter for the audio testbench.

    Parameters
    ----------
    registry : Optional[CollectorRegistry]
        Custom Prometheus collector registry. Defaults to prometheus_client.REGISTRY.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or prometheus_client.REGISTRY
        self._lock = threading.Lock()

        # Latency breakdown histogram partitioned across stages and total
        self.frame_latency = Histogram(
            "audio_pipeline_frame_latency_seconds",
            "Latency of audio frame processing partitioned by pipeline stage and total.",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        # Processed frames counter partitioned by precision mode
        self.processed_frames = Counter(
            "audio_pipeline_processed_frames_total",
            "Total count of audio frames processed by the pipeline.",
            ["precision"],
            registry=self.registry,
        )

        # Real-time budget overruns counter
        self.budget_overruns = Counter(
            "audio_pipeline_budget_overruns_total",
            "Total count of frames whose processing latency exceeded the real-time budget.",
            registry=self.registry,
        )

        # Active concurrent WebSocket streaming sessions
        self.active_clients = Gauge(
            "audio_pipeline_active_clients",
            "Number of active concurrent WebSocket streaming audio clients.",
            registry=self.registry,
        )

        # Audio quality telemetry gauges
        self.snr_gain_db = Gauge(
            "audio_pipeline_snr_gain_db",
            "Instantaneous or rolling Signal-to-Noise Ratio (SNR) improvement in dB.",
            registry=self.registry,
        )
        self.snr_input_db = Gauge(
            "audio_pipeline_snr_input_db",
            "Estimated input Signal-to-Noise Ratio in dB.",
            registry=self.registry,
        )
        self.snr_output_db = Gauge(
            "audio_pipeline_snr_output_db",
            "Estimated output Signal-to-Noise Ratio in dB.",
            registry=self.registry,
        )
        self.sound_purity_pct = Gauge(
            "audio_pipeline_sound_purity_pct",
            "Estimated speech sound purity score (0.0 - 100.0%).",
            registry=self.registry,
        )

        # Voice Activity Detection (VAD) neural compute savings
        self.compute_savings_pct = Gauge(
            "audio_pipeline_compute_savings_pct",
            "Percentage of neural compute frames bypassed via VAD gating (0.0 - 100.0%).",
            registry=self.registry,
        )

        # Process Resident Set Size (RSS) memory in bytes
        self.process_rss_bytes = Gauge(
            "audio_pipeline_process_memory_rss_bytes",
            "Resident Set Size (RSS) physical memory used by the server process in bytes.",
            registry=self.registry,
        )

        # Physical Edge Hardware Telemetry (Thermal, Power, Throttling)
        self.hardware_temp_c = Gauge(
            "audio_pipeline_hardware_temperature_celsius",
            "Physical or simulated edge SoC temperature in degrees Celsius.",
            registry=self.registry,
        )
        self.hardware_power_w = Gauge(
            "audio_pipeline_hardware_power_watts",
            "Estimated or measured edge hardware power draw in Watts.",
            registry=self.registry,
        )
        self.hardware_throttled = Gauge(
            "audio_pipeline_hardware_throttled",
            "Hardware thermal/voltage throttling state (1.0 = throttled, 0.0 = nominal).",
            registry=self.registry,
        )

    def record_frame(
        self,
        pre_ms: float,
        tensor_ms: float,
        synth_ms: float,
        total_ms: float,
        precision: str = "FP32",
        budget_exceeded: bool = False,
    ) -> None:
        """Record fine-grained frame execution latencies and counters with sub-microsecond overhead.

        Parameters
        ----------
        pre_ms : float
            Pre-processing stage latency in milliseconds.
        tensor_ms : float
            Tensor compute stage latency in milliseconds.
        synth_ms : float
            Output synthesis stage latency in milliseconds.
        total_ms : float
            Total frame latency in milliseconds.
        precision : str
            Active precision format ('FP32', 'FP16', 'INT8').
        budget_exceeded : bool
            True if total_ms exceeded nominal real-time frame budget.
        """
        # Convert ms -> seconds for standard Prometheus units
        self.frame_latency.labels(stage="pre_processing").observe(max(0.0, pre_ms) / 1000.0)
        self.frame_latency.labels(stage="tensor_compute").observe(max(0.0, tensor_ms) / 1000.0)
        self.frame_latency.labels(stage="output_synthesis").observe(max(0.0, synth_ms) / 1000.0)
        self.frame_latency.labels(stage="total").observe(max(0.0, total_ms) / 1000.0)

        prec = precision.upper() if precision else "FP32"
        self.processed_frames.labels(precision=prec).inc()

        if budget_exceeded:
            self.budget_overruns.inc()

    def update_audio_metrics(
        self,
        snr_gain_db: float,
        snr_in_db: float = 0.0,
        snr_out_db: float = 0.0,
        purity_pct: float = 100.0,
    ) -> None:
        """Update instantaneous audio signal quality metrics."""
        self.snr_gain_db.set(float(snr_gain_db))
        self.snr_input_db.set(float(snr_in_db))
        self.snr_output_db.set(float(snr_out_db))
        self.sound_purity_pct.set(max(0.0, min(100.0, float(purity_pct))))

    def update_vad_savings(self, savings_pct: float) -> None:
        """Update Voice Activity Detection compute savings percentage."""
        self.compute_savings_pct.set(max(0.0, min(100.0, float(savings_pct))))

    def update_memory(self, rss_bytes: int) -> None:
        """Update Resident Set Size memory metric in bytes."""
        self.process_rss_bytes.set(max(0, int(rss_bytes)))

    def update_hardware_telemetry(self, temp_c: float, power_w: float, throttled: bool = False) -> None:
        """Update physical hardware SoC temperature, power draw, and throttling state."""
        self.hardware_temp_c.set(float(temp_c))
        self.hardware_power_w.set(float(power_w))
        self.hardware_throttled.set(1.0 if throttled else 0.0)

    def set_active_clients(self, count: int) -> None:
        """Set current count of active concurrent streaming clients."""
        self.active_clients.set(max(0, int(count)))

    def inc_active_clients(self) -> None:
        """Increment active WebSocket client count."""
        self.active_clients.inc()

    def dec_active_clients(self) -> None:
        """Decrement active WebSocket client count."""
        self.active_clients.dec()

    def generate_metrics(self) -> bytes:
        """Generate Prometheus exposition text representation."""
        return generate_latest(self.registry)


# Global default exporter instance hooked to the default Prometheus registry
metrics_exporter = PrometheusMetricsExporter()

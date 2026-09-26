"""BigQuery Streaming Telemetry Exporter and Dataform Schema Validator.

Provides:
- BigQueryTelemetryExporter: Thread-safe buffer and serializer formatting real-time edge
  audio telemetry into Google Cloud BigQuery records (NDJSON & Streaming Insert formats).
- Dataform schema validation and assertion pre-flight checks matching SQLX definitions.
"""

from __future__ import annotations
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BigQueryTelemetryExporter:
    """Serializes edge audio profiler telemetry into BigQuery lakehouse tables.

    Conforms to the Dataform schema defined in dataform/definitions/sources/audio_telemetry_raw.sqlx.
    """

    BIGQUERY_SCHEMA = [
        {"name": "session_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique client streaming session ID"},
        {"name": "timestamp_ns", "type": "INT64", "mode": "REQUIRED", "description": "Nanosecond UTC timestamp of frame"},
        {"name": "precision_mode", "type": "STRING", "mode": "NULLABLE", "description": "Active precision (FP32, FP16, INT8)"},
        {"name": "hardware_tier", "type": "STRING", "mode": "NULLABLE", "description": "Active SIMD tier (TIER_1_NATIVE_C, TIER_2_NUMBA, TIER_3_NUMPY)"},
        {"name": "stage_pre_ms", "type": "FLOAT64", "mode": "NULLABLE", "description": "Pre-processing latency in ms"},
        {"name": "stage_tensor_ms", "type": "FLOAT64", "mode": "NULLABLE", "description": "Neural tensor inference latency in ms"},
        {"name": "stage_synth_ms", "type": "FLOAT64", "mode": "NULLABLE", "description": "Output synthesis latency in ms"},
        {"name": "total_latency_ms", "type": "FLOAT64", "mode": "REQUIRED", "description": "Total end-to-end frame processing latency in ms"},
        {"name": "budget_exceeded", "type": "BOOL", "mode": "NULLABLE", "description": "Whether total latency exceeded 20.0 ms budget"},
        {"name": "snr_gain_db", "type": "FLOAT64", "mode": "NULLABLE", "description": "Acoustic SNR improvement in dB"},
        {"name": "sig_mos", "type": "FLOAT64", "mode": "NULLABLE", "description": "ITU-T P.835 Speech Quality MOS [1.0, 5.0]"},
        {"name": "bak_mos", "type": "FLOAT64", "mode": "NULLABLE", "description": "ITU-T P.835 Background Noise MOS [1.0, 5.0]"},
        {"name": "ovrl_mos", "type": "FLOAT64", "mode": "NULLABLE", "description": "ITU-T P.835 Overall Listening Quality MOS [1.0, 5.0]"},
        {"name": "stoi", "type": "FLOAT64", "mode": "NULLABLE", "description": "Short-Time Objective Intelligibility [0.0, 1.0]"},
        {"name": "pesq", "type": "FLOAT64", "mode": "NULLABLE", "description": "Perceptual Evaluation of Speech Quality [-0.5, 4.5]"},
        {"name": "noise_category", "type": "STRING", "mode": "NULLABLE", "description": "Detected acoustic noise category"},
        {"name": "vad_compute_saved_pct", "type": "FLOAT64", "mode": "NULLABLE", "description": "VAD compute reduction percentage"},
    ]

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = max(100, int(capacity))
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def format_record(
        self,
        session_id: str,
        total_latency_ms: float,
        timestamp_ns: Optional[int] = None,
        precision_mode: str = "FP32",
        hardware_tier: str = "TIER_3_NUMPY",
        stage_pre_ms: float = 0.0,
        stage_tensor_ms: float = 0.0,
        stage_synth_ms: float = 0.0,
        budget_exceeded: Optional[bool] = None,
        snr_gain_db: float = 0.0,
        sig_mos: float = 4.2,
        bak_mos: float = 4.0,
        ovrl_mos: float = 4.1,
        stoi: float = 0.92,
        pesq: float = 3.8,
        noise_category: str = "white",
        vad_compute_saved_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """Format a single frame's telemetry into a strict BigQuery schema record."""
        t_ns = int(timestamp_ns) if timestamp_ns is not None else time.time_ns()
        exceeded = budget_exceeded if budget_exceeded is not None else (total_latency_ms > 20.0)

        record = {
            "session_id": str(session_id),
            "timestamp_ns": t_ns,
            "precision_mode": str(precision_mode).upper(),
            "hardware_tier": str(hardware_tier).upper(),
            "stage_pre_ms": round(float(stage_pre_ms), 3),
            "stage_tensor_ms": round(float(stage_tensor_ms), 3),
            "stage_synth_ms": round(float(stage_synth_ms), 3),
            "total_latency_ms": round(float(total_latency_ms), 3),
            "budget_exceeded": bool(exceeded),
            "snr_gain_db": round(float(snr_gain_db), 2),
            "sig_mos": round(float(np.clip(sig_mos, 1.0, 5.0)), 2),
            "bak_mos": round(float(np.clip(bak_mos, 1.0, 5.0)), 2),
            "ovrl_mos": round(float(np.clip(ovrl_mos, 1.0, 5.0)), 2),
            "stoi": round(float(np.clip(stoi, 0.0, 1.0)), 3),
            "pesq": round(float(np.clip(pesq, -0.5, 4.5)), 2),
            "noise_category": str(noise_category).lower(),
            "vad_compute_saved_pct": round(float(np.clip(vad_compute_saved_pct, 0.0, 100.0)), 1),
        }
        return record

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a telemetry record against Dataform assertions."""
        errors: List[str] = []

        # Non-null assertions
        for req_field in ["session_id", "timestamp_ns", "total_latency_ms"]:
            if req_field not in record or record[req_field] is None:
                errors.append(f"Missing required field: {req_field}")

        # Row conditions
        if "total_latency_ms" in record:
            lat = record["total_latency_ms"]
            if lat < 0.0 or lat > 1000.0:
                errors.append(f"total_latency_ms out of bounds [0.0, 1000.0]: {lat}")

        for mos_field in ["sig_mos", "bak_mos", "ovrl_mos"]:
            if mos_field in record:
                val = record[mos_field]
                if val < 1.0 or val > 5.0:
                    errors.append(f"{mos_field} out of bounds [1.0, 5.0]: {val}")

        return (len(errors) == 0, errors)

    def buffer_record(self, record: Dict[str, Any]) -> None:
        """Buffer a validated telemetry record in memory with FIFO eviction."""
        with self._lock:
            if len(self._buffer) >= self.capacity:
                self._buffer.pop(0)
            self._buffer.append(record)

    def get_buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer.clear()

    def export_ndjson(self, limit: Optional[int] = None) -> str:
        """Export buffered records as Newline-Delimited JSON (NDJSON) for BigQuery."""
        with self._lock:
            records = list(self._buffer)
        if limit is not None and limit > 0:
            records = records[-limit:]

        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        return "\n".join(lines)

    @classmethod
    def get_schema(cls) -> List[Dict[str, str]]:
        """Return the BigQuery table schema definition."""
        return list(cls.BIGQUERY_SCHEMA)

    @classmethod
    def get_dataform_manifest(cls) -> Dict[str, Any]:
        """Return the Dataform DAG and model compilation manifest."""
        return {
            "project": "edge-ai-audio-profiler",
            "dataset": "edge_ai_telemetry",
            "location": "us-central1",
            "nodes": [
                {
                    "name": "audio_telemetry_raw",
                    "type": "declaration",
                    "path": "definitions/sources/audio_telemetry_raw.sqlx",
                },
                {
                    "name": "stg_audio_telemetry",
                    "type": "view",
                    "path": "definitions/staging/stg_audio_telemetry.sqlx",
                    "dependencies": ["audio_telemetry_raw"],
                    "assertions": ["nonNull", "rowConditions"],
                },
                {
                    "name": "int_hardware_latency_percentiles",
                    "type": "table",
                    "path": "definitions/intermediate/int_hardware_latency_percentiles.sqlx",
                    "dependencies": ["stg_audio_telemetry"],
                    "partition_by": "date",
                    "cluster_by": ["hardware_tier", "precision_mode"],
                },
                {
                    "name": "fct_psychoacoustic_quality_summary",
                    "type": "table",
                    "path": "definitions/marts/fct_psychoacoustic_quality_summary.sqlx",
                    "dependencies": ["stg_audio_telemetry"],
                    "partition_by": "date",
                    "cluster_by": ["noise_category", "precision_mode"],
                },
            ],
        }


# Global singleton instance
bigquery_exporter = BigQueryTelemetryExporter(capacity=2000)

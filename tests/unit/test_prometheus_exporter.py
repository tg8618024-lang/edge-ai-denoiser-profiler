"""Unit tests for Prometheus Metrics Exporter.

Authority: Phase 2 Roadmap Requirement (Observability Upgrade).
"""

import pytest
from prometheus_client import CollectorRegistry
from src.telemetry.prometheus_exporter import PrometheusMetricsExporter, LATENCY_BUCKETS


def test_prometheus_exporter_initialization():
    """Verifies PrometheusMetricsExporter initializes clean histograms, gauges, and counters."""
    registry = CollectorRegistry()
    exporter = PrometheusMetricsExporter(registry=registry)

    assert exporter.frame_latency is not None
    assert exporter.processed_frames is not None
    assert exporter.budget_overruns is not None
    assert exporter.active_clients is not None
    assert exporter.snr_gain_db is not None
    assert exporter.sound_purity_pct is not None
    assert exporter.compute_savings_pct is not None
    assert exporter.process_rss_bytes is not None


def test_prometheus_exporter_record_frame_latency():
    """Verifies record_frame records latencies across stages and increments counters."""
    registry = CollectorRegistry()
    exporter = PrometheusMetricsExporter(registry=registry)

    # Record 3 frames
    exporter.record_frame(pre_ms=0.25, tensor_ms=0.60, synth_ms=0.15, total_ms=1.0, precision="FP32", budget_exceeded=False)
    exporter.record_frame(pre_ms=0.20, tensor_ms=0.45, synth_ms=0.10, total_ms=0.75, precision="INT8", budget_exceeded=False)
    exporter.record_frame(pre_ms=5.0, tensor_ms=12.0, synth_ms=4.0, total_ms=21.0, precision="FP16", budget_exceeded=True)

    metrics_text = exporter.generate_metrics().decode("utf-8")

    # Assert metric families are exported
    assert "audio_pipeline_frame_latency_seconds_bucket" in metrics_text
    assert 'stage="pre_processing"' in metrics_text
    assert 'stage="tensor_compute"' in metrics_text
    assert 'stage="output_synthesis"' in metrics_text
    assert 'stage="total"' in metrics_text
    assert 'audio_pipeline_processed_frames_total{precision="FP32"} 1.0' in metrics_text
    assert 'audio_pipeline_processed_frames_total{precision="INT8"} 1.0' in metrics_text
    assert 'audio_pipeline_processed_frames_total{precision="FP16"} 1.0' in metrics_text
    assert "audio_pipeline_budget_overruns_total 1.0" in metrics_text


def test_prometheus_exporter_audio_and_system_gauges():
    """Verifies audio quality, VAD savings, client counts, and memory gauges update accurately."""
    registry = CollectorRegistry()
    exporter = PrometheusMetricsExporter(registry=registry)

    exporter.update_audio_metrics(snr_gain_db=12.5, snr_in_db=-2.0, snr_out_db=10.5, purity_pct=98.5)
    exporter.update_vad_savings(45.2)
    exporter.update_memory(48 * 1024 * 1024)
    exporter.set_active_clients(5)

    exporter.inc_active_clients()
    exporter.dec_active_clients()

    exporter.update_simd_tier(2)

    metrics_text = exporter.generate_metrics().decode("utf-8")

    assert "audio_pipeline_snr_gain_db 12.5" in metrics_text
    assert "audio_pipeline_snr_input_db -2.0" in metrics_text
    assert "audio_pipeline_snr_output_db 10.5" in metrics_text
    assert "audio_pipeline_sound_purity_pct 98.5" in metrics_text
    assert "audio_pipeline_compute_savings_pct 45.2" in metrics_text
    assert "audio_pipeline_active_clients 5.0" in metrics_text
    assert "audio_pipeline_simd_active_tier 2.0" in metrics_text
    assert "audio_pipeline_process_memory_rss_bytes" in metrics_text
    assert ("5.0331648e+07" in metrics_text or "50331648" in metrics_text)

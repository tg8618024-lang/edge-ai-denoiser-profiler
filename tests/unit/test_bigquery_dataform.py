"""Unit tests for BigQuery Telemetry Exporter and Dataform ELT Pipeline.

Authority: Phase 3 Telemetry Warehousing, Task 8 Specification.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from src.telemetry.bigquery_exporter import BigQueryTelemetryExporter, bigquery_exporter
from src.dashboard.app import app


@pytest.fixture
def clean_exporter():
    exporter = BigQueryTelemetryExporter(capacity=100)
    exporter.clear_buffer()
    return exporter


def test_bigquery_schema_structure():
    """Verifies BigQuery table schema matches enterprise lakehouse specifications."""
    schema = BigQueryTelemetryExporter.get_schema()
    assert isinstance(schema, list)
    assert len(schema) >= 15

    names = {col["name"]: col for col in schema}
    assert "session_id" in names
    assert names["session_id"]["type"] == "STRING"
    assert names["session_id"]["mode"] == "REQUIRED"

    assert "timestamp_ns" in names
    assert names["timestamp_ns"]["type"] == "INT64"
    assert names["timestamp_ns"]["mode"] == "REQUIRED"

    assert "total_latency_ms" in names
    assert names["total_latency_ms"]["type"] == "FLOAT64"
    assert names["total_latency_ms"]["mode"] == "REQUIRED"

    for opt_field in ["stage_pre_ms", "stage_tensor_ms", "stage_synth_ms", "snr_gain_db", "sig_mos", "bak_mos", "ovrl_mos"]:
        assert opt_field in names
        assert names[opt_field]["type"] == "FLOAT64"
        assert names[opt_field]["mode"] == "NULLABLE"


def test_format_record_clipping_and_defaults(clean_exporter):
    """Verifies format_record enforces mathematical ranges on psychoacoustic and latency fields."""
    rec = clean_exporter.format_record(
        session_id="test_sess_01",
        total_latency_ms=15.4567,
        stage_pre_ms=2.1234,
        stage_tensor_ms=10.5678,
        stage_synth_ms=2.7654,
        sig_mos=5.8,  # Out of range, should clip to 5.0
        bak_mos=0.5,  # Out of range, should clip to 1.0
        ovrl_mos=4.25,
        stoi=1.2,     # Out of range, should clip to 1.0
        pesq=5.0,     # Out of range, should clip to 4.5
        vad_compute_saved_pct=150.0,  # Clip to 100.0
    )

    assert rec["session_id"] == "test_sess_01"
    assert rec["total_latency_ms"] == 15.457
    assert rec["stage_pre_ms"] == 2.123
    assert rec["stage_tensor_ms"] == 10.568
    assert rec["stage_synth_ms"] == 2.765
    assert rec["sig_mos"] == 5.0
    assert rec["bak_mos"] == 1.0
    assert rec["ovrl_mos"] == 4.25
    assert rec["stoi"] == 1.0
    assert rec["pesq"] == 4.5
    assert rec["vad_compute_saved_pct"] == 100.0
    assert rec["budget_exceeded"] is False


def test_format_record_budget_exceeded(clean_exporter):
    """Verifies budget_exceeded flag triggers automatically when total latency > 20.0 ms."""
    rec = clean_exporter.format_record(
        session_id="test_sess_02",
        total_latency_ms=21.4,
    )
    assert rec["budget_exceeded"] is True


def test_validate_record_assertions(clean_exporter):
    """Verifies Dataform assertion checks catch malformed records before ingestion."""
    valid_rec = clean_exporter.format_record(
        session_id="sess_valid",
        total_latency_ms=12.0,
    )
    is_valid, errors = clean_exporter.validate_record(valid_rec)
    assert is_valid is True
    assert len(errors) == 0

    # Missing required field
    invalid_rec = dict(valid_rec)
    del invalid_rec["session_id"]
    is_valid, errors = clean_exporter.validate_record(invalid_rec)
    assert is_valid is False
    assert any("session_id" in e for e in errors)

    # Latency out of bounds
    invalid_rec2 = dict(valid_rec)
    invalid_rec2["total_latency_ms"] = -5.0
    is_valid2, errors2 = clean_exporter.validate_record(invalid_rec2)
    assert is_valid2 is False
    assert any("total_latency_ms out of bounds" in e for e in errors2)

    # MOS out of bounds
    invalid_rec3 = dict(valid_rec)
    invalid_rec3["sig_mos"] = 0.8
    is_valid3, errors3 = clean_exporter.validate_record(invalid_rec3)
    assert is_valid3 is False
    assert any("sig_mos out of bounds" in e for e in errors3)


def test_buffer_and_ndjson_export(clean_exporter):
    """Verifies buffer capacity, FIFO eviction, and NDJSON serialization."""
    # Buffer 10 records
    for i in range(10):
        rec = clean_exporter.format_record(
            session_id=f"sess_{i:02d}",
            total_latency_ms=10.0 + i,
        )
        clean_exporter.buffer_record(rec)

    assert clean_exporter.get_buffered_count() == 10

    # Export all NDJSON
    ndjson_data = clean_exporter.export_ndjson()
    lines = ndjson_data.strip().split("\n")
    assert len(lines) == 10

    first_obj = json.loads(lines[0])
    assert first_obj["session_id"] == "sess_00"
    last_obj = json.loads(lines[-1])
    assert last_obj["session_id"] == "sess_09"

    # Export limited subset (last 3)
    limited_ndjson = clean_exporter.export_ndjson(limit=3)
    lim_lines = limited_ndjson.strip().split("\n")
    assert len(lim_lines) == 3
    assert json.loads(lim_lines[0])["session_id"] == "sess_07"
    assert json.loads(lim_lines[-1])["session_id"] == "sess_09"

    # Test FIFO capacity eviction
    small_exporter = BigQueryTelemetryExporter(capacity=100)
    for i in range(120):
        rec = small_exporter.format_record(session_id=f"evict_{i}", total_latency_ms=10.0)
        small_exporter.buffer_record(rec)
    assert small_exporter.get_buffered_count() == 100


def test_dataform_manifest_and_sqlx_files():
    """Verifies that all Dataform pipeline SQLX definitions and manifests exist and are valid."""
    manifest = BigQueryTelemetryExporter.get_dataform_manifest()
    assert manifest["project"] == "edge-ai-audio-profiler"
    assert manifest["dataset"] == "edge_ai_telemetry"
    assert manifest["location"] == "us-central1"

    nodes = {n["name"]: n for n in manifest["nodes"]}
    assert "audio_telemetry_raw" in nodes
    assert nodes["audio_telemetry_raw"]["type"] == "declaration"

    assert "stg_audio_telemetry" in nodes
    assert nodes["stg_audio_telemetry"]["type"] == "view"
    assert "audio_telemetry_raw" in nodes["stg_audio_telemetry"]["dependencies"]

    assert "int_hardware_latency_percentiles" in nodes
    assert nodes["int_hardware_latency_percentiles"]["type"] == "table"
    assert "stg_audio_telemetry" in nodes["int_hardware_latency_percentiles"]["dependencies"]

    assert "fct_psychoacoustic_quality_summary" in nodes
    assert nodes["fct_psychoacoustic_quality_summary"]["type"] == "table"
    assert "stg_audio_telemetry" in nodes["fct_psychoacoustic_quality_summary"]["dependencies"]

    # Verify physical files on disk
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataform_dir = os.path.join(project_root, "dataform")

    assert os.path.isfile(os.path.join(dataform_dir, "workflow_settings.yaml"))
    assert os.path.isfile(os.path.join(dataform_dir, "package.json"))
    assert os.path.isfile(os.path.join(dataform_dir, "dataform.json"))

    for node in manifest["nodes"]:
        file_path = os.path.join(dataform_dir, node["path"])
        assert os.path.isfile(file_path), f"Missing SQLX definition: {file_path}"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "config {" in content


def test_api_bigquery_and_dataform_endpoints():
    """Verifies FastAPI REST endpoints for BigQuery export and Dataform specifications."""
    client = TestClient(app)

    # 1. Schema endpoint
    resp_schema = client.get("/api/telemetry/bigquery/schema")
    assert resp_schema.status_code == 200
    data_schema = resp_schema.json()
    assert "schema" in data_schema
    assert len(data_schema["schema"]) >= 15

    # 2. Dataform spec endpoint
    resp_spec = client.get("/api/telemetry/dataform/spec")
    assert resp_spec.status_code == 200
    data_spec = resp_spec.json()
    assert data_spec["project"] == "edge-ai-audio-profiler"
    assert "nodes" in data_spec
    assert len(data_spec["nodes"]) == 4

    # 3. BigQuery NDJSON export endpoint
    # Seed singleton with a record
    rec = bigquery_exporter.format_record(
        session_id="api_test_session",
        total_latency_ms=11.2,
    )
    bigquery_exporter.buffer_record(rec)

    resp_ndjson = client.get("/api/telemetry/bigquery/export?format=ndjson")
    assert resp_ndjson.status_code == 200
    assert "application/x-ndjson" in resp_ndjson.headers["content-type"]
    assert "api_test_session" in resp_ndjson.text

    # 4. BigQuery JSON export endpoint
    resp_json = client.get("/api/telemetry/bigquery/export?format=json")
    assert resp_json.status_code == 200
    data_json = resp_json.json()
    assert "buffered_count" in data_json
    assert "records" in data_json
    assert any(r["session_id"] == "api_test_session" for r in data_json["records"])

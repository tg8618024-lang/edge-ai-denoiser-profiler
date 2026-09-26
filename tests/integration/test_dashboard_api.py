"""Integration tests for FastAPI endpoints & WebSocket telemetry server.

Authority: ORIGINAL_REQUEST.md Requirement R4, PROJECT.md Milestone 3, TEST_INFRA.md F9/F10.
"""

import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.dashboard.app import app, state


@pytest.fixture
def client():
    return TestClient(app)


def test_get_index_html(client: TestClient):
    """Verifies that root URL serves the interactive dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "EDGE AI AUDIO DENOISER" in response.text


def test_api_status_endpoint(client: TestClient):
    """Verifies /api/status reports active precision, memory footprint, and frame budget."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["precision"] in ("FP32", "FP16", "INT8")
    assert data["model_size_bytes"] > 0
    assert data["frame_budget_ms"] == 16.0 or data["frame_budget_ms"] == 20.0
    assert data["process_rss_mb"] > 0
    assert "white" in data["presets"]


def test_api_precision_switching(client: TestClient):
    """Verifies /api/precision switches precision and returns accurate compression metrics."""
    # Switch to FP16
    r16 = client.post("/api/precision", json={"precision": "FP16"})
    assert r16.status_code == 200
    d16 = r16.json()
    assert d16["success"] is True
    assert d16["precision"] == "FP16"
    assert d16["compression_pct"] == 50.0

    # Switch to INT8
    r8 = client.post("/api/precision", json={"precision": "INT8"})
    assert r8.status_code == 200
    d8 = r8.json()
    assert d8["success"] is True
    assert d8["precision"] == "INT8"
    assert d8["compression_pct"] >= 65.0

    # Switch back to FP32
    r32 = client.post("/api/precision", json={"precision": "FP32"})
    assert r32.status_code == 200
    assert r32.json()["precision"] == "FP32"


def test_api_precision_invalid_mode_rejected(client: TestClient):
    """Verifies /api/precision rejects unsupported precision strings with 422."""
    response = client.post("/api/precision", json={"precision": "BFLOAT16"})
    assert response.status_code == 422


def test_api_benchmarks_list(client: TestClient):
    """Verifies /api/benchmarks returns metadata for all 4 benchmark profiles."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    presets = response.json()["presets"]
    assert len(presets) == 4
    preset_ids = {p["id"] for p in presets}
    assert "white" in preset_ids
    assert "pink" in preset_ids
    assert "drone" in preset_ids
    assert "rf_static" in preset_ids


def test_api_benchmark_generate_audio(client: TestClient):
    """Verifies /api/benchmark/generate synthesizes valid mixtures."""
    response = client.post(
        "/api/benchmark/generate",
        json={"preset": "white", "duration_sec": 2.0, "target_snr_db": 0.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["preset"] == "white"
    assert data["sample_rate"] == 16000
    assert data["num_samples"] == 32000
    assert len(data["mixture_preview"]) == 1024


def test_websocket_stream_control_and_audio(client: TestClient):
    """Verifies WebSocket /ws/stream processes live audio frames and control messages."""
    with client.websocket_connect("/ws/stream") as ws:
        # Test precision control
        ws.send_text(json.dumps({"type": "set_precision", "precision": "INT8"}))
        resp_text = ws.receive_text()
        resp = json.loads(resp_text)
        assert resp["type"] == "precision_updated"
        assert resp["precision"] == "INT8"

        # Test live audio frame ingestion
        dummy_pcm = (np.sin(2.0 * np.pi * 440.0 * np.arange(256) / 16000.0) * 0.5).tolist()
        ws.send_text(json.dumps({"type": "audio_frame", "pcm": dummy_pcm}))

        frame_resp_text = ws.receive_text()
        frame_resp = json.loads(frame_resp_text)
        assert frame_resp["type"] == "live_telemetry"
        assert "timestamp_ns" in frame_resp
        assert "precision_mode" in frame_resp
        assert "stages" in frame_resp
        assert frame_resp["stages"]["total_latency_ms"] <= 20.0
        assert frame_resp["budget"]["headroom_ms"] > 0
        assert len(frame_resp["audio"]["denoised"]) == 256
        assert len(frame_resp["audio"]["noise_subtracted"]) == 256
        assert len(frame_resp["audio"]["spec_out"]) == 64
        assert len(frame_resp["audio"]["gain_mask"]) == 64
        assert "snr_delta_db" in frame_resp["metrics"]
        assert "noise_erased_pct" in frame_resp["metrics"]
        assert "purity_pct" in frame_resp["metrics"]


def test_websocket_benchmark_streaming(client: TestClient):
    """Verifies WebSocket /ws/stream starts and stops synthetic benchmark stream."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({
            "type": "start_benchmark",
            "preset": "white",
            "target_snr": 0.0
        }))
        resp_text = ws.receive_text()
        resp = json.loads(resp_text)
        assert resp["type"] == "telemetry"
        assert "stages" in resp
        assert "budget" in resp
        assert "rolling_stats" in resp
        assert "metrics" in resp
        assert "audio" in resp
        assert len(resp["audio"]["raw_noisy"]) == 256
        assert len(resp["audio"]["denoised"]) == 256
        assert len(resp["audio"]["noise_subtracted"]) == 256
        assert len(resp["audio"]["spec_in"]) == 64
        assert len(resp["audio"]["spec_out"]) == 64
        assert "noise_erased_pct" in resp["metrics"]
        assert "purity_pct" in resp["metrics"]
        assert "rms_noise" in resp["metrics"]

        ws.send_text(json.dumps({"type": "stop_benchmark"}))


def test_static_assets_and_oscilloscope_dom(client: TestClient):
    """Verifies static assets (CSS, JS) and oscilloscope DOM elements are served properly."""
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "canvasOscilloscope" in r_index.text
    assert "purityGaugeBar" in r_index.text
    assert "magicDenoiseBtn" in r_index.text
    assert "btnHeroStream" in r_index.text
    assert "btnOscDiff" in r_index.text
    assert "MUDDY / NOISY AUDIO IN" in r_index.text
    assert "CRYSTAL CLEAN VOICE OUT" in r_index.text
    assert "GARBAGE NOISE ERASED" in r_index.text
    assert "SUBTRACTED NOISE" in r_index.text

    r_css = client.get("/static/styles.css")
    assert r_css.status_code == 200
    assert "canvasOscilloscope" in r_css.text
    assert "btn-hero-stream" in r_css.text

    r_js = client.get("/static/app.js")
    assert r_js.status_code == 200
    assert "renderOscilloscope" in r_js.text
    assert "drawDifferenceArea" in r_js.text


def test_pipeline_state_reset():
    """Verifies that PipelineState.reset() executes cleanly without AttributeError."""
    state.reset()
    assert state.pipeline.total_frames_processed == 0
    assert state.profiler.current_frame == 0
    assert state.ring_buffer.count == 0


def test_api_settings_update(client: TestClient):
    """Verifies /api/settings updates suppression intensity, precision, and model mode."""
    r = client.post(
        "/api/settings",
        json={"intensity": 0.85, "precision": "FP16", "mode": "hybrid"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["intensity"] == 0.85
    assert data["precision"] == "FP16"

    # Reset back to 1.0 and FP32
    r_reset = client.post(
        "/api/settings",
        json={"intensity": 1.0, "precision": "FP32", "mode": "hybrid"},
    )
    assert r_reset.status_code == 200
    assert r_reset.json()["intensity"] == 1.0
    assert r_reset.json()["precision"] == "FP32"


def test_api_presets_all_8(client: TestClient):
    """Verifies /api/presets returns all 8 calibrated benchmark profiles."""
    r = client.get("/api/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) == 8
    preset_ids = {p["id"] for p in presets}
    expected = {
        "white",
        "pink",
        "drone",
        "rf_static",
        "cafe",
        "rain",
        "keyboard",
        "air_conditioner",
    }
    assert expected.issubset(preset_ids)


def test_api_audio_upload_and_download(client: TestClient):
    """Verifies /api/audio/upload processes custom WAV and /api/audio/download exports channels."""
    import io
    import scipy.io.wavfile as wavfile

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    test_signal = np.sin(2.0 * np.pi * 440.0 * t) * 0.3 + np.random.randn(sr) * 0.1
    test_int16 = np.int16(np.clip(test_signal, -1.0, 1.0) * 32767)

    buf = io.BytesIO()
    wavfile.write(buf, sr, test_int16)
    wav_bytes = buf.getvalue()

    upload_resp = client.post(
        "/api/audio/upload?intensity=1.0&filename=test_synthetic.wav",
        content=wav_bytes,
        headers={"Content-Type": "audio/wav"},
    )
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    assert up_data["success"] is True
    audio_id = up_data["audio_id"]
    assert up_data["filename"] == "test_synthetic.wav"
    assert up_data["duration_sec"] == 1.0
    assert "preview" in up_data
    assert len(up_data["preview"]["noisy"]) == 256
    assert len(up_data["preview"]["denoised"]) == 256
    assert up_data["metrics"]["purity_pct"] > 0

    # Test clean WAV download
    r_clean = client.get(f"/api/audio/download/{audio_id}?channel=denoised")
    assert r_clean.status_code == 200
    assert r_clean.headers["content-type"] == "audio/wav"
    assert "attachment; filename=" in r_clean.headers["content-disposition"]
    assert len(r_clean.content) > 44

    # Test raw noisy WAV download
    r_noisy = client.get(f"/api/audio/download/{audio_id}?channel=noisy")
    assert r_noisy.status_code == 200
    assert r_noisy.headers["content-type"] == "audio/wav"

    # Test subtracted noise delta WAV download
    r_sub = client.get(f"/api/audio/download/{audio_id}?channel=noise")
    assert r_sub.status_code == 200
    assert r_sub.headers["content-type"] == "audio/wav"


def test_api_report_export_json_and_csv(client: TestClient):
    """Verifies /api/report/export generates valid JSON and CSV benchmark telemetry reports."""
    # JSON export
    r_json = client.get("/api/report/export?format=json")
    assert r_json.status_code == 200
    json_data = r_json.json()
    assert "engine" in json_data
    assert "precision_mode" in json_data
    assert "latency_percentiles" in json_data
    assert "stages_ms" in json_data
    assert "audio_signal_metrics" in json_data

    # CSV export
    r_csv = client.get("/api/report/export?format=csv")
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers.get("content-type", "")
    assert "Metric,Value,Unit" in r_csv.text
    assert "Stage 1: Pre-processing" in r_csv.text


def test_prometheus_metrics_endpoint(client: TestClient):
    """Verifies /metrics endpoint returns valid Prometheus time-series exposition."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    text = response.text
    assert "audio_pipeline_frame_latency_seconds" in text
    assert "audio_pipeline_processed_frames_total" in text
    assert "audio_pipeline_active_clients" in text
    assert "audio_pipeline_process_memory_rss_bytes" in text
    assert "audio_pipeline_hardware_temperature_celsius" in text
    assert "audio_pipeline_hardware_power_watts" in text
    assert "audio_pipeline_hardware_throttled" in text
    assert "audio_pipeline_simd_active_tier" in text


def test_api_studio_integrations_endpoint(client: TestClient):
    """Verifies /api/studio/integrations reports OBS, VST3, WebRTC, WebGPU, and Extension metadata."""
    response = client.get("/api/studio/integrations")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "integrations" in data
    integ = data["integrations"]

    assert "obs_studio" in integ
    assert integ["obs_studio"]["port"] == 18890
    assert "protocol" in integ["obs_studio"]

    assert "vst3_clap" in integ
    assert integ["vst3_clap"]["latency_samples_pdc"] == 256
    assert "stereo" in integ["vst3_clap"]["channels"]
    assert 128 in integ["vst3_clap"]["supported_block_sizes"]

    assert "webrtc" in integ
    assert integ["webrtc"]["endpoints"]["offer"] == "/api/webrtc/offer"

    assert "webgpu" in integ
    assert integ["webgpu"]["endpoint"] == "/webgpu"

    assert "chrome_extension" in integ
    assert integ["chrome_extension"]["manifest_version"] == 3


def test_api_obs_status_endpoint(client: TestClient):
    """Verifies /api/obs/status reports OBS server status and metrics."""
    response = client.get("/api/obs/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "host" in data
    assert "port" in data
    assert "active_clients" in data
    assert "total_frames_processed" in data




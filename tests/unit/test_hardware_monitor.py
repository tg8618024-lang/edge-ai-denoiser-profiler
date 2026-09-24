"""Unit tests for Edge Hardware Telemetry Monitor.

Authority: Phase 4 Roadmap Requirement (Real Edge Deployment).
"""

import pytest
from prometheus_client import CollectorRegistry
from src.telemetry.hardware_monitor import EdgeHardwareMonitor, HardwareSnapshot
from src.telemetry.prometheus_exporter import PrometheusMetricsExporter


def test_parse_tegrastats():
    """Verifies parsing of NVIDIA Jetson tegrastats log line."""
    sample_log = (
        "RAM 2145/3964MB (lfb 118x4MB) SWAP 0/1982MB (cached 0MB) "
        "CPU [25%@1479,35%@1479,15%@1479,25%@1479] EMC_FREQ 10%@1600 "
        "GR3D_FREQ 0%@921 NVENC 115 NVDEC 115 AO@45C GPU@43C PMIC@48C CPU@46.5C thermal@46.2C "
        "POM_5V_IN 4250/4250 POM_5V_CPU 1520/1520 POM_5V_GPU 480/480"
    )
    monitor = EdgeHardwareMonitor(device_profile="jetson_nano")
    data = monitor.parse_tegrastats(sample_log)

    # Average CPU usage = (25 + 35 + 15 + 25) / 4 = 25.0%
    assert data["cpu_usage_pct"] == 25.0
    assert data["temps"]["CPU"] == 46.5
    assert data["temps"]["GPU"] == 43.0
    assert data["temps"]["AO"] == 45.0
    assert data["powers_mw"]["POM_5V_IN"] == 4250.0
    assert data["powers_mw"]["POM_5V_CPU"] == 1520.0


def test_parse_vcgencmd_nominal():
    """Verifies Raspberry Pi vcgencmd parsing in normal nominal state (0x0)."""
    temp_output = "temp=48.2'C\n"
    throttle_output = "throttled=0x0\n"

    monitor = EdgeHardwareMonitor(device_profile="raspberry_pi_4")
    data = monitor.parse_vcgencmd(temp_output, throttle_output)

    assert data["temp_c"] == 48.2
    assert data["throttled"] is False
    assert len(data["reasons"]) == 0
    assert data["hex"] == "0x0"


def test_parse_vcgencmd_throttled_bits():
    """Verifies Raspberry Pi vcgencmd parsing with under-voltage and frequency capping."""
    temp_output = "temp=82.5'C\n"
    # 0x50005: Under-voltage detected (0x1) + Currently throttled (0x4) + Under-voltage has occurred (0x10000) + Throttling has occurred (0x40000)
    throttle_output = "throttled=0x50005\n"

    monitor = EdgeHardwareMonitor(device_profile="raspberry_pi_4")
    data = monitor.parse_vcgencmd(temp_output, throttle_output)

    assert data["temp_c"] == 82.5
    assert data["throttled"] is True
    assert "Under-voltage detected" in data["reasons"]
    assert "Currently throttled" in data["reasons"]
    assert "Under-voltage has occurred" in data["reasons"]
    assert "Throttling has occurred" in data["reasons"]


def test_calibrated_edge_profiles():
    """Verifies calibrated hardware models for Pi Zero, Pi 4, Jetson Nano, and Jetson Orin."""
    for profile in ["raspberry_pi_zero_2", "raspberry_pi_4", "jetson_nano", "jetson_orin_nano"]:
        monitor = EdgeHardwareMonitor(device_profile=profile)
        snap = monitor.poll(frame_latency_ms=0.5, precision="INT8")

        assert isinstance(snap, HardwareSnapshot)
        assert snap.cpu_temp_c > 30.0
        assert snap.power_watts > 1.0
        assert snap.energy_per_frame_uj > 0.0
        assert snap.device_profile == profile
        assert snap.throttled is False


def test_thermal_throttling_transition():
    """Verifies thermal accumulation triggers throttling when reaching the thermal limit."""
    monitor = EdgeHardwareMonitor(device_profile="raspberry_pi_4")
    # Artificially set temperature right at the thermal limit (80.0 C)
    monitor._current_temp = 80.5

    snap = monitor.poll(frame_latency_ms=15.0, precision="FP32")
    assert snap.throttled is True
    assert "Thermal throttling active" in snap.throttled_reasons[0]


def test_prometheus_hardware_metrics_export():
    """Verifies hardware temperature, power, and throttling state are exposed in Prometheus format."""
    registry = CollectorRegistry()
    exporter = PrometheusMetricsExporter(registry=registry)

    exporter.update_hardware_telemetry(temp_c=52.4, power_w=5.12, throttled=False)
    metrics_text = exporter.generate_metrics().decode("utf-8")

    assert "audio_pipeline_hardware_temperature_celsius 52.4" in metrics_text
    assert "audio_pipeline_hardware_power_watts 5.12" in metrics_text
    assert "audio_pipeline_hardware_throttled 0.0" in metrics_text

    # Throttled state
    exporter.update_hardware_telemetry(temp_c=85.0, power_w=10.0, throttled=True)
    metrics_throttled = exporter.generate_metrics().decode("utf-8")
    assert "audio_pipeline_hardware_throttled 1.0" in metrics_throttled

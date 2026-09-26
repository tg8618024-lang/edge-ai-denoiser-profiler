"""Unit Tests for 1st-Order Silicon Thermal Differential ODE & Prometheus Exporter.

Authority: Phase 12 Architectural Blueprint & SLA Guarantees.
Validates:
1. Analytical exponential solution: T(t) = T_eq + (T(0) - T_eq) * exp(-t / tau)
2. Precision-dependent power dissipation (FP32 vs FP16 vs INT8)
3. Duty-cycle scaling under variable processing latency
4. Thermal throttling threshold trigger at >= 85.0 C
5. Prometheus metrics exposition for temperature, power, and throttling state
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry.energy import HardwareEnergyProfiler, EnergyMetrics
from src.telemetry.prometheus_exporter import PrometheusMetricsExporter, CollectorRegistry


class TestThermalDifferentialODE:
    """Verifies the physical thermal differential ODE and analytical solution."""

    def test_analytical_step_response_fp32(self):
        """Verify numerical step integration matches closed-form analytical solution within 0.05 C."""
        profiler = HardwareEnergyProfiler()
        profiler.reset(initial_temp_c=42.0)

        # FP32 at 100% duty cycle (16.0 ms latency)
        # Power = 8.5 W, Duty Cycle = 1.0 -> T_eq = 42.0 + 1.8 * 8.5 * 1.0 = 57.3 C
        t_eq = profiler.get_equilibrium_temp(precision="FP32", duty_cycle=1.0)
        assert abs(t_eq - 57.3) < 1e-4

        t0 = 42.0
        tau = profiler.tau_sec  # 10.0 seconds

        # Simulate 10 seconds in steps of 0.5s (20 steps)
        dt = 0.5
        total_time = 0.0
        for _ in range(20):
            profiler.step_thermal_ode(dt_sec=dt, frame_latency_ms=16.0, precision="FP32")
            total_time += dt

            # Analytical expected temperature
            expected_temp = t_eq + (t0 - t_eq) * np.exp(-total_time / tau)
            assert abs(profiler.current_temp_c - expected_temp) < 0.05

        # At t = tau (10s), temperature should reach ~63.2% of total rise:
        # Delta = 57.3 - 42.0 = 15.3 C -> Rise = 15.3 * (1 - 1/e) = 9.67 C -> T ~ 51.67 C
        assert abs(profiler.current_temp_c - 51.67) < 0.2

    def test_precision_dependent_equilibrium(self):
        """Verify asymptotic equilibrium temperature scales with precision energy efficiency."""
        profiler = HardwareEnergyProfiler()

        # Equilibrium temperatures at full duty cycle:
        t_eq_fp32 = profiler.get_equilibrium_temp("FP32", duty_cycle=1.0)
        t_eq_fp16 = profiler.get_equilibrium_temp("FP16", duty_cycle=1.0)
        t_eq_int8 = profiler.get_equilibrium_temp("INT8", duty_cycle=1.0)

        # FP32: 42 + 1.8 * 8.5 = 57.3 C
        # FP16: 42 + 1.8 * (8.5 * 0.65) = 42 + 9.945 = 51.945 C
        # INT8: 42 + 1.8 * (8.5 * 0.38) = 42 + 5.814 = 47.814 C
        assert abs(t_eq_fp32 - 57.300) < 1e-3
        assert abs(t_eq_fp16 - 51.945) < 1e-3
        assert abs(t_eq_int8 - 47.814) < 1e-3

        # Asymptotic temperature hierarchy: FP32 > FP16 > INT8
        assert t_eq_fp32 > t_eq_fp16 > t_eq_int8

        # Run long simulation (100 seconds = 10 * tau) to reach steady-state
        profiler.reset(42.0)
        for _ in range(100):
            profiler.step_thermal_ode(dt_sec=1.0, frame_latency_ms=16.0, precision="INT8")

        assert abs(profiler.current_temp_c - t_eq_int8) < 0.01

    def test_duty_cycle_thermal_scaling(self):
        """Verify that lighter frame processing latency reduces thermal heating."""
        profiler = HardwareEnergyProfiler()

        # 1.4 ms real-time frame latency -> duty cycle = 1.4 / 16.0 = 0.0875
        profiler.reset(42.0)
        t_eq_low_duty = profiler.get_equilibrium_temp("FP32", duty_cycle=1.4 / 16.0)

        # Expected: 42.0 + 1.8 * 8.5 * 0.0875 = 42.0 + 1.339 = 43.34 C
        assert abs(t_eq_low_duty - 43.339) < 0.05

        # After 100 seconds at light load, temperature barely rises above ambient:
        for _ in range(100):
            profiler.step_thermal_ode(dt_sec=1.0, frame_latency_ms=1.4, precision="FP32")

        assert abs(profiler.current_temp_c - t_eq_low_duty) < 0.05

    def test_thermal_throttling_trigger_and_recovery(self):
        """Verify thermal throttling flag triggers at >= 85.0 C and clears when cooled."""
        profiler = HardwareEnergyProfiler()

        # Reset below threshold
        profiler.reset(80.0)
        assert profiler.is_throttled is False

        # Heat up past 85.0 C by forcing high temperature
        profiler.current_temp_c = 85.5
        profiler.step_thermal_ode(dt_sec=0.1, frame_latency_ms=16.0, precision="FP32")
        assert profiler.is_throttled is True

        # Cool down towards 42.0 C
        profiler.current_temp_c = 84.0
        profiler.step_thermal_ode(dt_sec=0.1, frame_latency_ms=1.4, precision="INT8")
        assert profiler.is_throttled is False

    def test_prometheus_exposition_hardware_metrics(self):
        """Verify Prometheus metrics exporter generates valid text format with thermal gauges."""
        registry = CollectorRegistry()
        exporter = PrometheusMetricsExporter(registry=registry)

        # Update gauges
        exporter.update_hardware_telemetry(temp_c=52.4, power_w=5.52, throttled=False)
        exporter.update_simd_tier(2)

        output_bytes = exporter.generate_metrics()
        output_text = output_bytes.decode("utf-8")

        assert "audio_pipeline_hardware_temperature_celsius 52.4" in output_text
        assert "audio_pipeline_hardware_power_watts 5.52" in output_text
        assert "audio_pipeline_hardware_throttled 0.0" in output_text
        assert "audio_pipeline_simd_active_tier 2.0" in output_text

        # Update with throttling active
        exporter.update_hardware_telemetry(temp_c=86.2, power_w=8.5, throttled=True)
        updated_text = exporter.generate_metrics().decode("utf-8")
        assert "audio_pipeline_hardware_temperature_celsius 86.2" in updated_text
        assert "audio_pipeline_hardware_throttled 1.0" in updated_text

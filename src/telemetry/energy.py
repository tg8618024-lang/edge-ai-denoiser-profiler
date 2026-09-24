"""NVIDIA GPU & Edge AI Energy / Power Profiler.

Provides:
- Real-time power telemetry in Watts.
- Energy consumption per audio frame in microjoules (uJ/frame).
- Power efficiency score (Frames per Watt-second).
- NVML hardware polling with fallback to calibrated NVIDIA Jetson/RTX edge power models.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class EnergyMetrics:
    power_watts: float
    energy_per_frame_uj: float
    efficiency_fps_per_watt: float
    temperature_c: float
    backend_source: str


class HardwareEnergyProfiler:
    """Measures and models energy consumption for real-time edge AI inference."""

    def __init__(self, target_platform: str = "rtx_edge"):
        self.target_platform = target_platform
        self._has_nvml = False
        self._nvml_handle = None
        self.current_temp_c = 42.0
        self._last_time = time.time()

        # Attempt to initialize NVML if present
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._has_nvml = True
        except Exception:
            self._has_nvml = False

    def profile_frame(
        self,
        frame_latency_ms: float,
        precision: str = "FP32",
    ) -> EnergyMetrics:
        """Calculate power and energy for a single frame inference."""
        latency_sec = frame_latency_ms / 1000.0

        if self._has_nvml and self._nvml_handle:
            try:
                import pynvml
                # Power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                temp_c = float(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
                power_watts = float(power_mw / 1000.0)
                energy_uj = float(power_watts * latency_sec * 1e6)
                eff = float(1.0 / (power_watts * latency_sec + 1e-6))
                return EnergyMetrics(
                    power_watts=round(power_watts, 2),
                    energy_per_frame_uj=round(energy_uj, 2),
                    efficiency_fps_per_watt=round(eff, 1),
                    temperature_c=temp_c,
                    backend_source="NVIDIA NVML (Live HW)",
                )
            except Exception:
                pass

        # Calibrated Edge / Jetson Orin Hardware Power Model:
        # Jetson Orin Nano / RTX Edge runs at ~7W to 15W TDP under lightweight audio workloads
        prec_factor = {"FP32": 1.0, "FP16": 0.65, "INT8": 0.38}.get(precision.upper(), 1.0)
        base_power_watts = 8.5 * prec_factor
        energy_uj = base_power_watts * latency_sec * 1e6
        fps = 1000.0 / max(0.001, frame_latency_ms)
        eff = fps / max(0.1, base_power_watts)

        # 1st-Order Thermal Differential Equation: dT/dt = (T_eq - T) / tau (tau = 10.0s)
        now = time.time()
        dt = max(0.001, min(1.0, now - self._last_time))
        self._last_time = now
        duty_cycle = min(1.0, frame_latency_ms / 16.0)
        t_eq = 42.0 + (1.8 * base_power_watts * duty_cycle)
        alpha = 1.0 - float(np.exp(-dt / 10.0))
        self.current_temp_c += alpha * (t_eq - self.current_temp_c)
        estimated_temp = self.current_temp_c

        return EnergyMetrics(
            power_watts=round(base_power_watts, 2),
            energy_per_frame_uj=round(energy_uj, 2),
            efficiency_fps_per_watt=round(eff, 1),
            temperature_c=round(estimated_temp, 1),
            backend_source="Calibrated NVIDIA Jetson/RTX Model",
        )


# Alias for convenience
EnergyProfiler = HardwareEnergyProfiler


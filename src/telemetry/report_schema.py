"""Standardized telemetry report schema and release compliance validator for Phase 0.

Conforms to SPEC-ARCH-00-BOUNDS and Google Antigravity SDK Multi-Agent Telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


@dataclass(slots=True)
class ModelMetadata:
    """Metadata describing the neural or DSP model architecture."""
    model_id: str
    architecture: str
    param_count: int
    binary_size_bytes: int
    quantization: str  # "FP32", "FP16", "INT8-QAT", "INT8-PTQ", "Q15"


@dataclass(slots=True)
class DSPParameters:
    """Core audio framing, sampling rate, and latency parameters."""
    sampling_rate_hz: int  # 16000 or 48000
    frame_size_samples: int  # 512 (16k) or 960 (48k)
    hop_size_samples: int  # 256 (16k) or 480 (48k)
    lookahead_samples: int  # Must be 0 for strictly causal
    algorithmic_delay_ms: float


@dataclass(slots=True)
class HardwareTelemetry:
    """Profiling metrics gathered on target silicon."""
    target_silicon: str  # "NVIDIA Jetson Orin", "Apple Silicon", "x86_64 VNNI", "Web Audio / AudioWorklet"
    backend: str  # "TensorRT", "CoreML-ANE", "x86-VNNI", "Wasm-SIMD", "WebGPU"
    rtf_p50: float  # Real-Time Factor median
    rtf_p95: float  # Real-Time Factor 95th percentile
    rtf_p99: float  # Real-Time Factor 99th percentile
    peak_ram_mb: float
    power_draw_watts: float = 0.0
    thermal_throttle_detected: bool = False


@dataclass(slots=True)
class QualityMetrics:
    """Objective perceptual and psychoacoustic quality metrics."""
    dnsmos_ovrl: float  # Target >= 3.65
    dnsmos_sig: float  # Target >= 4.05
    dnsmos_bak: float  # Target >= 4.00
    wb_pesq: float  # Target >= 3.15
    stoi: float  # Target >= 0.92
    delta_stoi: float  # Target >= 0.00
    delta_sisdr_db: float  # Target >= 8.50 dB
    static_noise_attenuation_db: float = 0.0


@dataclass(slots=True)
class AudioEnhancementTelemetryReport:
    """Complete multi-agent telemetry report schema satisfying Phase 0 bounds."""
    model_metadata: ModelMetadata
    dsp_parameters: DSPParameters
    hardware_telemetry: HardwareTelemetry
    quality_metrics: QualityMetrics
    report_id: str = "report_latest"
    timestamp_iso: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report hierarchy to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioEnhancementTelemetryReport:
        """Instantiate report from nested dictionary."""
        return cls(
            model_metadata=ModelMetadata(**data["model_metadata"]),
            dsp_parameters=DSPParameters(**data["dsp_parameters"]),
            hardware_telemetry=HardwareTelemetry(**data["hardware_telemetry"]),
            quality_metrics=QualityMetrics(**data["quality_metrics"]),
            report_id=data.get("report_id", "report_latest"),
            timestamp_iso=data.get("timestamp_iso", ""),
        )

    def verify_release_contract(self) -> Dict[str, Any]:
        """Enforces all Phase 0 frozen thresholds and SLA contracts.

        Returns:
            Dict containing 'status' ('PASSED' or 'QUARANTINED'),
            'passed' (bool), and 'violations' (List[str]).
        """
        violations: List[str] = []

        # 1. Latency & Causality Contracts (ITU-T G.114)
        if self.dsp_parameters.lookahead_samples > 0:
            violations.append(
                f"Strict Causality Violation: lookahead_samples={self.dsp_parameters.lookahead_samples} > 0"
            )
        if self.dsp_parameters.algorithmic_delay_ms > 20.0:
            violations.append(
                f"ITU-T G.114 Violation: algorithmic_delay_ms={self.dsp_parameters.algorithmic_delay_ms} > 20.0 ms"
            )
        if self.hardware_telemetry.rtf_p99 > 0.35:
            violations.append(
                f"RTF Budget Violation: rtf_p99={self.hardware_telemetry.rtf_p99} > 0.35 threshold"
            )
        if self.hardware_telemetry.thermal_throttle_detected:
            violations.append("Thermal Guard Violation: Thermal throttling was detected on target silicon")

        # 2. Quality & Distortion Contracts (ITU-T P.835 / DNSMOS)
        if self.quality_metrics.dnsmos_ovrl < 3.65:
            violations.append(
                f"DNSMOS OVRL Gate Failed: {self.quality_metrics.dnsmos_ovrl:.2f} < 3.65 required"
            )
        if self.quality_metrics.dnsmos_sig < 4.05:
            violations.append(
                f"DNSMOS SIG Gate Failed: {self.quality_metrics.dnsmos_sig:.2f} < 4.05 required"
            )
        if self.quality_metrics.dnsmos_bak < 4.00:
            violations.append(
                f"DNSMOS BAK Gate Failed: {self.quality_metrics.dnsmos_bak:.2f} < 4.00 required"
            )

        # 3. Objective Quality & Intelligibility Contracts
        if self.quality_metrics.wb_pesq < 3.15:
            violations.append(
                f"WB-PESQ Gate Failed: {self.quality_metrics.wb_pesq:.2f} < 3.15 required"
            )
        if self.quality_metrics.stoi < 0.92:
            violations.append(
                f"STOI Gate Failed: {self.quality_metrics.stoi:.2f} < 0.92 required"
            )
        if self.quality_metrics.delta_stoi < 0.00:
            violations.append(
                f"Intelligibility Degradation Violation: delta_stoi={self.quality_metrics.delta_stoi:.3f} < 0.00"
            )
        if self.quality_metrics.delta_sisdr_db < 8.50:
            violations.append(
                f"SI-SDR Gate Failed: delta_sisdr_db={self.quality_metrics.delta_sisdr_db:.2f} dB < 8.50 dB"
            )

        # 4. Memory Footprint Bounds (L2 Cache Residency)
        if self.model_metadata.binary_size_bytes > 2_500_000:
            violations.append(
                f"L2 Cache Residency Violation: binary_size_bytes={self.model_metadata.binary_size_bytes} > 2,500,000"
            )

        passed = len(violations) == 0
        return {
            "status": "PASSED" if passed else "QUARANTINED",
            "passed": passed,
            "violations": violations,
        }

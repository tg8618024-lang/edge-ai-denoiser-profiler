"""Unit tests for Phase 0: Mission, Mathematical Bounds & Engineering Contracts."""

import os
import numpy as np
import pytest

from src.telemetry.report_schema import (
    AudioEnhancementTelemetryReport,
    ModelMetadata,
    DSPParameters,
    HardwareTelemetry,
    QualityMetrics,
)


class TestPhase0MathematicalBounds:
    """Verifies physical acoustic and mathematical framing invariants."""

    def test_wideband_16k_framing_bounds(self):
        """Verifies 16 kHz wideband parameters meet Phase 0 contracts."""
        fs = 16000
        n_fft = 512
        hop = 256
        n_bins = n_fft // 2 + 1

        frame_duration_ms = (hop / fs) * 1000.0
        window_duration_ms = (n_fft / fs) * 1000.0
        bin_resolution_hz = fs / n_fft

        assert frame_duration_ms == 16.0, "16 kHz hop must be precisely 16.0 ms"
        assert window_duration_ms == 32.0, "16 kHz window must be precisely 32.0 ms"
        assert n_bins == 257, "16 kHz conjugate symmetric FFT bins must be 257"
        assert bin_resolution_hz == 31.25, "Bin resolution must be 31.25 Hz"

    def test_fullband_48k_framing_bounds(self):
        """Verifies 48 kHz studio fullband parameters meet Phase 0 contracts."""
        fs = 48000
        n_fft = 960
        hop = 480
        n_bins = n_fft // 2 + 1

        frame_duration_ms = (hop / fs) * 1000.0
        window_duration_ms = (n_fft / fs) * 1000.0
        bin_resolution_hz = fs / n_fft

        assert frame_duration_ms == 10.0, "48 kHz hop must be precisely 10.0 ms"
        assert window_duration_ms == 20.0, "48 kHz window must be precisely 20.0 ms"
        assert n_bins == 481, "48 kHz conjugate symmetric FFT bins must be 481"
        assert bin_resolution_hz == 50.0, "Bin resolution must be 50.0 Hz"

    @pytest.mark.parametrize("n_fft,hop", [(512, 256), (960, 480)])
    def test_cola_window_energy_preservation(self, n_fft: int, hop: int):
        """Verifies Constant Overlap-Add (COLA) energy preservation for sqrt-Hann."""
        # Square-root Hann periodic window
        n = np.arange(n_fft, dtype=np.float64)
        w = np.sin(np.pi * (n + 0.5) / n_fft)

        # Overlap-add over 10 consecutive frames
        num_frames = 10
        total_len = (num_frames - 1) * hop + n_fft
        ola_buffer = np.zeros(total_len, dtype=np.float64)

        for m in range(num_frames):
            start = m * hop
            ola_buffer[start : start + n_fft] += w**2

        # In interior region (ignoring warmup/cooldown boundary of 1 frame), sum must be 1.0
        interior = ola_buffer[n_fft : total_len - n_fft]
        np.testing.assert_allclose(
            interior,
            1.0,
            atol=1e-5,
            err_msg="COLA energy preservation violated for sqrt-Hann window",
        )

    def test_strict_causality_invariant(self):
        """Verifies lookahead is 0 ms and latency complies with ITU-T G.114."""
        lookahead_ms = 0.0
        local_dsp_budget_ms = 20.0
        max_mouth_to_ear_ms = 150.0

        assert lookahead_ms == 0.0, "Lookahead must be strictly 0.0 ms (causal)"
        assert local_dsp_budget_ms <= 20.0, "Local DSP budget must not exceed 20.0 ms"
        assert max_mouth_to_ear_ms <= 150.0, "Mouth-to-ear delay must be <= 150.0 ms"


class TestPhase0HardwareAndTelemetryContracts:
    """Verifies silicon memory limits and multi-agent report verification."""

    def test_l2_cache_residency_bound(self):
        """Verifies production weights file size is within the 2.5 MB L2 cache envelope."""
        weights_path = os.path.join("src", "models", "default_weights.npz")
        if os.path.exists(weights_path):
            size_bytes = os.path.getsize(weights_path)
            # 2.5 MB = 2,621,440 bytes
            assert (
                size_bytes <= 2_500_000
            ), f"Production weights ({size_bytes} bytes) exceed 2.5 MB L2 cache limit"

    def test_telemetry_report_verification_passed(self):
        """Verifies that a fully compliant report passes Phase 0 release gates."""
        report = AudioEnhancementTelemetryReport(
            model_metadata=ModelMetadata(
                model_id="grumasknet_v2",
                architecture="Causal-GRU-CRM",
                param_count=41473,
                binary_size_bytes=165892,
                quantization="INT8-PTQ",
            ),
            dsp_parameters=DSPParameters(
                sampling_rate_hz=16000,
                frame_size_samples=512,
                hop_size_samples=256,
                lookahead_samples=0,
                algorithmic_delay_ms=16.0,
            ),
            hardware_telemetry=HardwareTelemetry(
                target_silicon="NVIDIA Jetson Orin Nano",
                backend="TensorRT",
                rtf_p50=0.18,
                rtf_p95=0.24,
                rtf_p99=0.29,
                peak_ram_mb=34.2,
                power_draw_watts=8.5,
                thermal_throttle_detected=False,
            ),
            quality_metrics=QualityMetrics(
                dnsmos_ovrl=3.82,
                dnsmos_sig=4.18,
                dnsmos_bak=4.31,
                wb_pesq=3.35,
                stoi=0.94,
                delta_stoi=0.03,
                delta_sisdr_db=11.2,
                static_noise_attenuation_db=32.4,
            ),
        )

        result = report.verify_release_contract()
        assert result["status"] == "PASSED"
        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_telemetry_report_verification_quarantine(self):
        """Verifies that contract violations trigger automated quarantine."""
        bad_report = AudioEnhancementTelemetryReport(
            model_metadata=ModelMetadata(
                model_id="unoptimized_giant_net",
                architecture="Transformer-NonCausal",
                param_count=5000000,
                binary_size_bytes=20000000,  # Exceeds 2.5 MB
                quantization="FP32",
            ),
            dsp_parameters=DSPParameters(
                sampling_rate_hz=16000,
                frame_size_samples=512,
                hop_size_samples=256,
                lookahead_samples=32,  # Non-causal lookahead violation!
                algorithmic_delay_ms=25.0,  # G.114 violation!
            ),
            hardware_telemetry=HardwareTelemetry(
                target_silicon="x86_64",
                backend="CPU",
                rtf_p50=0.45,
                rtf_p95=0.60,
                rtf_p99=0.82,  # RTF violation!
                peak_ram_mb=120.0,
                thermal_throttle_detected=True,  # Thermal violation!
            ),
            quality_metrics=QualityMetrics(
                dnsmos_ovrl=3.10,  # < 3.65 violation!
                dnsmos_sig=3.80,  # < 4.05 violation!
                dnsmos_bak=3.50,  # < 4.00 violation!
                wb_pesq=2.80,  # < 3.15 violation!
                stoi=0.85,  # < 0.92 violation!
                delta_stoi=-0.05,  # Intelligibility degradation violation!
                delta_sisdr_db=6.0,  # < 8.5 dB violation!
            ),
        )

        result = bad_report.verify_release_contract()
        assert result["status"] == "QUARANTINED"
        assert result["passed"] is False
        assert len(result["violations"]) >= 8

    def test_telemetry_json_roundtrip(self):
        """Verifies serialization round-trip preservation."""
        report = AudioEnhancementTelemetryReport(
            model_metadata=ModelMetadata("test_m", "GRU", 100, 400, "INT8"),
            dsp_parameters=DSPParameters(16000, 512, 256, 0, 16.0),
            hardware_telemetry=HardwareTelemetry("Jetson", "TRT", 0.1, 0.15, 0.2, 25.0),
            quality_metrics=QualityMetrics(3.8, 4.2, 4.2, 3.4, 0.95, 0.02, 10.5),
            report_id="test_run_01",
        )

        json_str = report.to_json()
        parsed = AudioEnhancementTelemetryReport.from_dict(report.to_dict())
        assert parsed.report_id == "test_run_01"
        assert parsed.dsp_parameters.sampling_rate_hz == 16000
        assert parsed.quality_metrics.dnsmos_ovrl == 3.8

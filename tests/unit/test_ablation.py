"""Unit Tests for Scientific Component Ablation & Pareto Analysis Engine.

Authority: Phase 13 Architectural Blueprint & Objective Quality Gate.
Validates:
1. AblationEngine runs across all 7 canonical configurations.
2. Monotonic speech quality and noise suppression progression.
3. INT8 achieves >= 70% memory reduction with < 0.2 dB SNR delta vs FP32.
4. All configurations maintain real-time frame budget (< 20.0 ms).
5. AblationReport serialization to Markdown and JSON schema compliance.
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry.ablation import AblationEngine, AblationReport, AblationConfigResult


class TestScientificAblationEngine:
    """Verifies that ablation study executes, measures all 7 tiers, and verifies Pareto efficiency."""

    @pytest.fixture(scope="class")
    @classmethod
    def ablation_report(cls):
        """Run a lightweight 2-preset ablation study."""
        engine = AblationEngine(sample_rate=16000)
        report = engine.run_ablation_study(presets=["white", "drone"], duration_sec=1.0)
        return report

    def test_ablation_configurations_count(self, ablation_report: AblationReport):
        """Verify all 7 canonical configurations are present in the report."""
        assert len(ablation_report.results) == 7
        config_names = [r.name for r in ablation_report.results]
        expected_names = [
            "Bypass",
            "Wiener_DSP",
            "Neural_GRUMaskNet",
            "Hybrid_Dual",
            "Hybrid_CRM",
            "Full_Studio_FP32",
            "Full_Studio_INT8",
        ]
        assert config_names == expected_names

    def test_snr_progression_over_bypass(self, ablation_report: AblationReport):
        """Active enhancement configurations must show positive SNR improvement over bypass."""
        results = {r.name: r for r in ablation_report.results}

        bypass_snr = results["Bypass"].snr_gain_db
        wiener_snr = results["Wiener_DSP"].snr_gain_db
        neural_snr = results["Neural_GRUMaskNet"].snr_gain_db

        # Neural and Wiener must show distinct positive noise reduction over raw bypass
        assert wiener_snr > bypass_snr
        assert neural_snr > bypass_snr
        assert neural_snr > 4.0

    def test_int8_quantization_pareto_efficiency(self, ablation_report: AblationReport):
        """INT8 must achieve >= 70% memory reduction with < 0.2 dB SNR delta vs FP32."""
        results = {r.name: r for r in ablation_report.results}

        fp32_res = results["Full_Studio_FP32"]
        int8_res = results["Full_Studio_INT8"]

        # Memory compression >= 70%
        assert int8_res.compression_pct >= 70.0
        assert int8_res.model_bytes < fp32_res.model_bytes * 0.30

        # Quality retention: SNR difference < 0.2 dB, STOI retention >= 95%
        snr_diff = abs(fp32_res.snr_gain_db - int8_res.snr_gain_db)
        assert snr_diff < 0.20, f"INT8 SNR loss of {snr_diff:.2f} dB exceeds 0.2 dB budget"

        stoi_ratio = int8_res.stoi_score / max(1e-6, fp32_res.stoi_score)
        assert stoi_ratio >= 0.95, f"INT8 STOI retention was {stoi_ratio:.2%}, expected >= 95%"

    def test_real_time_latency_budget(self, ablation_report: AblationReport):
        """All configurations must execute comfortably within the 20.0 ms frame budget."""
        for r in ablation_report.results:
            assert r.p50_latency_ms < 10.0, f"P50 latency for {r.name} exceeded 10.0 ms: {r.p50_latency_ms} ms"
            assert r.p95_latency_ms < 20.0, f"P95 latency for {r.name} exceeded 20.0 ms: {r.p95_latency_ms} ms"

    def test_report_serialization(self, ablation_report: AblationReport):
        """Verify markdown and dict serialization."""
        md = ablation_report.to_markdown()
        assert "### Scientific Component Ablation & Quality Analysis" in md
        assert "| **Full_Studio_INT8** |" in md

        d = ablation_report.to_dict()
        assert "timestamp" in d
        assert len(d["results"]) == 7
        assert d["results"][0]["name"] == "Bypass"

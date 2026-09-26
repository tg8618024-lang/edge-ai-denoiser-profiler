"""Standalone CLI Runner for Scientific Component Ablation & Pareto Analysis."""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry.ablation import AblationEngine


def main():
    print("=" * 80)
    print("  NVIDIA-STYLE EDGE AI AUDIO DENOISER: SCIENTIFIC COMPONENT ABLATION")
    print("=" * 80)
    print("Running 7-configuration ablation study across stationary and non-stationary noise...")
    engine = AblationEngine(sample_rate=16000)
    report = engine.run_ablation_study(presets=["white", "drone", "rf_static"], duration_sec=1.5)

    print("\n" + report.to_markdown() + "\n")
    print("=" * 80)
    print("  ABLATION BENCHMARK COMPLETED SUCCESSFULLY (Exit code: 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

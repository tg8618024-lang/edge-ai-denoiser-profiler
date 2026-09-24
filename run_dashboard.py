"""Runner script for Edge AI Audio Denoiser & Profiler Web Dashboard."""

import sys
import os
import argparse
import uvicorn

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run Edge AI Audio Denoiser & Profiler Interactive Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default="FP32", help="Initial precision")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING NVIDIA-STYLE EDGE AI AUDIO DENOISER & PROFILER DASHBOARD")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Initial Precision: {args.precision}")
    print("=" * 80)

    from src.dashboard.app import state
    state.pipeline.set_precision(args.precision)

    uvicorn.run("src.dashboard.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

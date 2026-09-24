"""Benchmark and verification script for INT8 SIMD Kernel Exploration.
Tests current precision.py performance, SQNR, memory footprint, and explores SIMD kernels.
"""

import os
import sys
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.precision import PrecisionEngine


def benchmark_current_precision():
    print("=== Current PrecisionEngine Benchmarks ===")
    engine = PrecisionEngine()
    
    # Check sizes
    engine.set_precision("FP32")
    fp32_size = engine.get_model_size_bytes()
    
    engine.set_precision("FP16")
    fp16_size = engine.get_model_size_bytes()
    
    engine.set_precision("INT8")
    int8_size = engine.get_model_size_bytes()
    
    print(f"FP32 Model Size: {fp32_size:,} bytes")
    print(f"FP16 Model Size: {fp16_size:,} bytes ({fp16_size / fp32_size:.2%})")
    print(f"INT8 Model Size: {int8_size:,} bytes ({int8_size / fp32_size:.2%})")
    
    # Warmup
    x = np.random.randn(257).astype(np.float32)
    for _ in range(100):
        engine.set_precision("FP32")
        engine.forward_frame(x)
        engine.set_precision("INT8")
        engine.forward_frame(x)
        
    N_FRAMES = 1000
    
    # Benchmark FP32
    engine.set_precision("FP32")
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        engine.forward_frame(x)
    t_fp32 = (time.perf_counter() - t0) / N_FRAMES * 1000.0  # ms per frame
    
    # Benchmark FP16
    engine.set_precision("FP16")
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        engine.forward_frame(x)
    t_fp16 = (time.perf_counter() - t0) / N_FRAMES * 1000.0
    
    # Benchmark INT8
    engine.set_precision("INT8")
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        engine.forward_frame(x)
    t_int8 = (time.perf_counter() - t0) / N_FRAMES * 1000.0
    
    print(f"\nForward Frame Latency ({N_FRAMES} iterations):")
    print(f"  FP32: {t_fp32:.4f} ms ({1000.0 / t_fp32:.1f} fps)")
    print(f"  FP16: {t_fp16:.4f} ms ({1000.0 / t_fp16:.1f} fps)")
    print(f"  INT8: {t_int8:.4f} ms ({1000.0 / t_int8:.1f} fps)")
    print(f"  INT8 / FP32 ratio: {t_int8 / t_fp32:.2f}x (INT8 is currently SLOWER due to numpy.astype(int32) overhead!)")
    
    # Measure SQNR on synthetic audio frame
    engine.set_precision("FP32")
    out_fp32 = engine.forward_frame(x)
    
    engine.set_precision("INT8")
    out_int8 = engine.forward_frame(x)
    
    sqnr = engine.calculate_sqnr(out_fp32, out_int8)
    print(f"\nINT8 SQNR against FP32 reference: {sqnr:.2f} dB (Requirement: >= 35.0 dB)")


if __name__ == "__main__":
    benchmark_current_precision()

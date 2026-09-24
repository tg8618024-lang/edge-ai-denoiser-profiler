import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.precision import PrecisionEngine

def test_layer_sqnr():
    engine_fp32 = PrecisionEngine(precision="FP32")
    engine_int8 = PrecisionEngine(precision="INT8")
    
    rng = np.random.RandomState(42)
    x = rng.randn(257).astype(np.float32)
    
    # Layer 1
    w1_fp32 = engine_fp32.fp32_weights["W1"]
    b1_fp32 = engine_fp32.fp32_weights["b1"]
    z1_fp32 = x @ w1_fp32 + b1_fp32
    
    z1_int8 = engine_int8._gemm_int8(x, "W1", "b1")
    
    sqnr_w1 = PrecisionEngine.calculate_sqnr(w1_fp32, PrecisionEngine.dequantize_tensor(engine_int8.int8_weights["W1"], engine_int8.int8_scales["W1"]))
    sqnr_z1 = PrecisionEngine.calculate_sqnr(z1_fp32, z1_int8)
    
    print(f"Weight W1 Quantization SQNR: {sqnr_w1:.2f} dB")
    print(f"Layer 1 Activation (z1) SQNR: {sqnr_z1:.2f} dB")
    
    # Check end-to-end forward_frame with clean reset
    engine_fp32.reset()
    engine_int8.reset()
    
    out_fp32 = engine_fp32.forward_frame(x)
    out_int8 = engine_int8.forward_frame(x)
    
    sqnr_out = PrecisionEngine.calculate_sqnr(out_fp32, out_int8)
    print(f"Full forward_frame output mask SQNR: {sqnr_out:.2f} dB")
    print(f"Max abs diff in mask: {np.max(np.abs(out_fp32 - out_int8)):.6f}")
    print(f"Mean abs diff in mask: {np.mean(np.abs(out_fp32 - out_int8)):.6f}")

if __name__ == "__main__":
    test_layer_sqnr()

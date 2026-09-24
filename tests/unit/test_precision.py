"""Unit tests for Multi-Precision Quantization Engine and SIMD INT8 Kernel.

Authority:
- ORIGINAL_REQUEST.md Requirement R2 & R3
- 2026-09-23 Gen 2 Hardening Acceptance Criteria
- Memory footprint <= 30.0% of FP32 baseline (e.g. ~42,656 B vs 165,892 B = 25.71%)
- INT8 Quantization SQNR >= 35.0 dB
- Zero per-frame heap allocations during INT8 inference
- Multi-tier SIMD dispatch (Native ctypes, Numba LLVM JIT, Cached NumPy)
- Measurable speedup over FP32 without GIL serialization
"""

import os
import time
import tracemalloc
import unittest
import numpy as np

from src.models.precision import PrecisionEngine
from src.models.kernels.simd_dispatch import (
    SIMDDispatcher,
    get_simd_dispatcher,
    int8_gemv_dispatch,
)
from src.audio.pipeline import AudioDenoisingPipeline


class TestPrecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PrecisionEngine()

    def test_precision_modes_get_set(self):
        """Verify FP32, FP16, and INT8 modes can be toggled cleanly."""
        for mode in ["FP32", "FP16", "INT8"]:
            self.engine.set_precision(mode)
            self.assertEqual(self.engine.get_precision(), mode)

        with self.assertRaises(ValueError):
            self.engine.set_precision("INT4")

    def test_memory_reduction_ratios(self):
        """Verify INT8 is <= 30.0% of FP32 baseline and FP16 is exactly 50%."""
        self.engine.set_precision("FP32")
        fp32_bytes = self.engine.get_model_size_bytes()
        self.assertEqual(fp32_bytes, 165892)

        self.engine.set_precision("FP16")
        fp16_bytes = self.engine.get_model_size_bytes()
        self.assertEqual(fp16_bytes / fp32_bytes, 0.50)

        self.engine.set_precision("INT8")
        int8_bytes = self.engine.get_model_size_bytes()
        int8_ratio = int8_bytes / fp32_bytes
        self.assertLessEqual(int8_ratio, 0.30, f"INT8 memory ratio {int8_ratio:.4f} > 0.30 threshold")
        self.assertAlmostEqual(int8_ratio, 0.257, places=2)

    def test_int8_quantization_sqnr_ge_35db(self):
        """Verify INT8 quantization fidelity achieves SQNR >= 35.0 dB."""
        rng = np.random.RandomState(42)
        w = rng.normal(0, 0.5, (64, 257)).astype(np.float32)
        w_q, scale = PrecisionEngine.quantize_tensor(w)
        w_deq = PrecisionEngine.dequantize_tensor(w_q, scale)
        sqnr = PrecisionEngine.calculate_sqnr(w, w_deq)
        self.assertGreaterEqual(sqnr, 35.0)

    def test_all_model_layers_sqnr_ge_35db(self):
        """Verify each model layer tensor individually achieves SQNR >= 35.0 dB."""
        for key in ["W1", "W2", "W_rec", "W3"]:
            orig = self.engine.fp32_weights[key]
            q_arr = self.engine.int8_weights[key]
            scale = self.engine.int8_scales[key]
            deq = PrecisionEngine.dequantize_tensor(q_arr, scale)
            sqnr = PrecisionEngine.calculate_sqnr(orig, deq)
            self.assertGreaterEqual(
                sqnr,
                35.0,
                f"Layer {key} INT8 SQNR {sqnr:.2f} dB is below 35.0 dB threshold",
            )

    def test_fp16_sqnr_ge_50db(self):
        """Verify FP16 maintains > 50 dB SQNR."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 0.5, (1, 257)).astype(np.float32)
        x_fp16 = x.astype(np.float16).astype(np.float32)
        sqnr = PrecisionEngine.calculate_sqnr(x, x_fp16)
        self.assertGreaterEqual(sqnr, 50.0)

    def test_weights_preservation_roundtrip(self):
        """Verify switching precisions does not mutate master FP32 weights."""
        orig_w1 = self.engine.fp32_weights["W1"].copy()
        self.engine.set_precision("INT8")
        self.engine.set_precision("FP16")
        self.engine.set_precision("FP32")
        np.testing.assert_array_equal(self.engine.fp32_weights["W1"], orig_w1)

    def test_silence_input_handling(self):
        """Verify INT8 forward inference on zero/silent inputs produces valid gain masks."""
        self.engine.set_precision("INT8")
        zero_input = np.zeros(257, dtype=np.float32)
        mask = self.engine.forward_frame(zero_input)
        self.assertFalse(np.any(np.isnan(mask)))
        self.assertFalse(np.any(np.isinf(mask)))
        self.assertEqual(len(mask), 257)
        self.assertTrue(np.all(mask >= 0.0))
        self.assertTrue(np.all(mask <= 1.0))

    def test_pipeline_precision_switching(self):
        """Verify end-to-end audio pipeline supports runtime switching without glitches."""
        pipeline = AudioDenoisingPipeline(precision="FP32")
        frame = np.random.randn(256).astype(np.float32)
        for mode in ["FP32", "INT8", "FP16", "FP32"]:
            pipeline.set_precision(mode)
            out = pipeline.process_frame(frame)
            self.assertEqual(len(out), 256)
            self.assertFalse(np.any(np.isnan(out)))


class TestSIMDDispatcher(unittest.TestCase):
    """Test suite for Tier 1 / Tier 2 / Tier 3 SIMD dispatch architectures."""

    def setUp(self):
        self.dispatcher = SIMDDispatcher()

    def test_dispatcher_tiers_available(self):
        """Verify dispatcher reports active tier with description."""
        tier, tier_name = self.dispatcher.get_tier()
        self.assertIn(tier, [1, 2, 3])
        self.assertTrue(len(tier_name) > 0)

    def test_tier2_tier3_bit_exactness(self):
        """Verify Tier 2 (Numba JIT) and Tier 3 (NumPy fallback) yield identical results."""
        rng = np.random.RandomState(123)
        M, K = 64, 257
        x_u8 = rng.randint(0, 256, size=K, dtype=np.uint8)
        w_t = rng.randint(-128, 127, size=(M, K), dtype=np.int8)
        bias = rng.randint(-10000, 10000, size=M, dtype=np.int32)

        out_tier2 = np.zeros(M, dtype=np.int32)
        out_tier3 = np.zeros(M, dtype=np.int32)

        # Force Tier 2
        self.dispatcher.set_tier(SIMDDispatcher.TIER_2_NUMBA)
        self.dispatcher.gemv(x_u8, w_t, bias, out_tier2, is_unsigned_folded=True)

        # Force Tier 3
        self.dispatcher.set_tier(SIMDDispatcher.TIER_3_NUMPY)
        self.dispatcher.gemv(x_u8, w_t, bias, out_tier3, is_unsigned_folded=True)

        # Exact match
        np.testing.assert_array_equal(out_tier2, out_tier3)

    def test_unsigned_folded_vs_signed_mathematical_equivalence(self):
        """Verify unsigned GEMV with folded bias is mathematically equivalent to signed GEMV."""
        rng = np.random.RandomState(456)
        M, K = 64, 128
        x_s8 = rng.randint(-128, 128, size=K, dtype=np.int8)
        w_t = rng.randint(-128, 128, size=(M, K), dtype=np.int8)
        b_i32 = rng.randint(-5000, 5000, size=M, dtype=np.int32)

        # 1. Direct mathematical signed GEMV: W_t @ x_s8 + b
        expected = np.dot(w_t.astype(np.int64), x_s8.astype(np.int64)) + b_i32.astype(np.int64)

        # 2. Unsigned GEMV with folded bias
        x_u8 = (x_s8.astype(np.int16) + 128).astype(np.uint8)
        offset = (128 * np.sum(w_t, axis=1)).astype(np.int32)
        folded_bias = b_i32 - offset

        out_simd = np.zeros(M, dtype=np.int32)
        self.dispatcher.gemv(x_u8, w_t, folded_bias, out_simd, is_unsigned_folded=True)

        np.testing.assert_array_equal(out_simd, expected.astype(np.int32))

    def test_native_c_files_exist_and_well_formed(self):
        """Verify C SIMD kernel and header files are present with required symbols."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kernel_dir = os.path.join(project_root, "src", "models", "kernels")
        c_file = os.path.join(kernel_dir, "edge_simd_int8.c")
        h_file = os.path.join(kernel_dir, "edge_simd_int8.h")

        self.assertTrue(os.path.isfile(c_file), f"Missing {c_file}")
        self.assertTrue(os.path.isfile(h_file), f"Missing {h_file}")

        with open(c_file, "r", encoding="utf-8") as f:
            c_content = f.read()

        # Check required SIMD ISA implementations
        self.assertIn("vpdpbusd", c_content)
        self.assertIn("maddubs", c_content)
        self.assertIn("int8_gemv_vnni", c_content)
        self.assertIn("int8_gemv_avx2", c_content)
        self.assertIn("int8_gemv_sse41", c_content)
        self.assertIn("int8_gemv_scalar", c_content)
        self.assertIn("int8_gemv_auto", c_content)


class TestInferencePerformanceAndMemory(unittest.TestCase):
    """Performance benchmarks and zero per-frame allocation verification."""

    def setUp(self):
        self.engine = PrecisionEngine()

    def test_zero_per_frame_heap_allocations(self):
        """Verify INT8 inference achieves zero heap allocations during streaming."""
        self.engine.set_precision("INT8")
        x = np.random.randn(257).astype(np.float32)

        # Warmup (compiles Numba JIT and initializes internal views)
        for _ in range(20):
            self.engine.forward_frame(x, copy_output=False)

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Run 100 consecutive frames in streaming mode
        for _ in range(100):
            self.engine.forward_frame(x, copy_output=False)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        # Filter for allocations inside precision.py or simd_dispatch.py
        internal_allocs = [
            s for s in stats
            if "precision.py" in s.traceback[0].filename or "simd_dispatch.py" in s.traceback[0].filename
        ]

        total_internal_bytes = sum(s.size_diff for s in internal_allocs if s.size_diff > 0)
        self.assertEqual(
            total_internal_bytes,
            0,
            f"Expected 0 bytes allocated during INT8 streaming, but found {total_internal_bytes} bytes: {internal_allocs}",
        )

    def test_int8_speedup_over_fp32(self):
        """Verify INT8 execution achieves measurable latency speedup over FP32."""
        x = np.random.randn(257).astype(np.float32)

        # 1. Benchmark FP32
        self.engine.set_precision("FP32")
        for _ in range(50):  # warmup
            self.engine.forward_frame(x)

        n_iters = 500
        t0 = time.perf_counter()
        for _ in range(n_iters):
            self.engine.forward_frame(x)
        t_fp32 = (time.perf_counter() - t0) / n_iters

        # 2. Benchmark INT8
        self.engine.set_precision("INT8")
        for _ in range(50):  # warmup
            self.engine.forward_frame(x, copy_output=False)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            self.engine.forward_frame(x, copy_output=False)
        t_int8 = (time.perf_counter() - t0) / n_iters

        print(f"\n[BENCHMARK] FP32 Latency: {t_fp32 * 1000:.4f} ms/frame ({1.0 / t_fp32:.1f} fps)")
        print(f"[BENCHMARK] INT8 Latency: {t_int8 * 1000:.4f} ms/frame ({1.0 / t_int8:.1f} fps)")
        speedup = t_fp32 / max(t_int8, 1e-9)
        print(f"[BENCHMARK] INT8 vs FP32 Speedup: {speedup:.2f}x")

        # INT8 must be faster than FP32 (speedup > 1.0)
        self.assertGreater(
            speedup,
            1.0,
            f"INT8 ({t_int8*1000:.4f} ms) must be faster than FP32 ({t_fp32*1000:.4f} ms)",
        )


if __name__ == "__main__":
    unittest.main()

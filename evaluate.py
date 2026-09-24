"""
Standalone Automated Non-Interactive Evaluation Benchmark.
Authority: ORIGINAL_REQUEST.md AC, PROJECT.md Milestone 4, TEST_INFRA.md

Executes end-to-end against bundled synthetic benchmark audio, asserting:
1. Noise suppression: Delta SNR >= 10.0 dB across all 4 noise types (White, Pink, Drone, RF static).
2. Real-time latency budget: per-frame processing latency <= 20.0 ms.
3. 3-stage latency profiler isolation: Pre-processing, Tensor Compute, Output Synthesis > 0.
4. Multi-precision comparison: FP32, FP16, and INT8 modes with INT8 memory <= 0.35x FP32 and SQNR >= 35 dB.
5. Zero digital clipping, finite outputs (no NaNs / Infs).

Exits with code 0 on success, non-zero on failure.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.dataset import SyntheticAudioGenerator, create_mixture, compute_snr_gain
from src.audio.pipeline import AudioDenoisingPipeline
from src.telemetry.profiler import StageProfiler
from src.telemetry.ring_buffer import RollingMetricsBuffer
from src.telemetry.memory import get_process_memory
from src.models.precision import PrecisionEngine
from tests.conftest import ReferenceSyntheticGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge AI Real-Time Neural Audio Denoiser & Profiler - Automated Evaluation"
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "white", "pink", "drone", "rf"],
        default="all",
        help="Benchmark noise profile to evaluate (default: all)"
    )
    parser.add_argument(
        "--precision",
        choices=["all", "FP32", "FP16", "INT8"],
        default="all",
        help="Precision mode to evaluate (default: all)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of test audio in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose per-frame telemetry logging"
    )
    return parser.parse_args()


def print_banner():
    print("=" * 80)
    print("  NVIDIA-STYLE EDGE AI AUDIO DENOISER & LATENCY PROFILER")
    print("  AUTOMATED EVALUATION BENCHMARK SUITE")
    print("=" * 80)


def evaluate_noise_benchmarks(benchmark_filter: str, duration_sec: float, verbose: bool):
    print("\n>>> [1/3] EVALUATING AUDIO DENOISING PERFORMANCE & LATENCY BUDGETS...")
    print("-" * 80)

    gen = SyntheticAudioGenerator(sample_rate=16000)
    ref = ReferenceSyntheticGenerator(sample_rate=16000, seed=42)

    noise_targets = [
        ("white", "REF", 0.0),
        ("pink", "REF", 0.0),
        ("drone", "REF", 5.0),
        ("rf", "REF", -5.0),
        ("white", "GEN", 0.0),
        ("pink", "GEN", 0.0),
        ("drone", "GEN", 0.0),
        ("rf_static", "GEN", 0.0),
    ]

    if benchmark_filter != "all":
        noise_targets = [
            t for t in noise_targets
            if t[0] == benchmark_filter or (benchmark_filter == "rf" and t[0] == "rf_static")
        ]

    results = []
    all_passed = True

    print(f"{'Noise Type':<12} | {'Suite':<5} | {'In SNR':<8} | {'Out SNR':<8} | {'Delta SNR':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'Status':<6}")
    print("-" * 80)

    for ntype, suite, target_snr in noise_targets:
        if suite == "REF":
            clean = ref.generate_speech(duration_sec=duration_sec)
            noise_type_key = "rf" if ntype in ("rf", "rf_static") else ntype
            noise = ref.generate_noise(noise_type_key, duration_sec=duration_sec)
            c, n, mix = ref.generate_mixture(clean, noise, target_snr_db=target_snr)
        else:
            clean = gen.generate_speech(duration_sec=duration_sec, seed=42)
            noise_type_key = "rf_static" if ntype in ("rf", "rf_static") else ntype
            noise = gen.generate_noise(noise_type_key, duration_sec=duration_sec, seed=42)
            mix, c, n = create_mixture(clean, noise, target_snr_db=target_snr)

        pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        profiler = StageProfiler(budget_ms=20.0)
        ring_buffer = RollingMetricsBuffer(capacity=200, budget_ms=20.0)

        num_frames = len(mix) // 256
        denoised_frames = []

        for i in range(num_frames):
            frame = mix[i * 256 : (i + 1) * 256]
            profiler.start_frame()
            out_frame = pipeline.process_frame(frame, profiler=profiler)
            t_pre, t_tensor, t_synth, t_total = profiler.mark_synthesis_done()
            ring_buffer.append(t_pre, t_tensor, t_synth, t_total)
            denoised_frames.append(out_frame)

            # Assert real-time per-frame budget
            assert t_total <= 20.0, f"Frame {i} exceeded 20ms real-time budget: {t_total:.2f}ms"
            # Assert 3-stage isolation
            assert t_pre > 0.0, f"Frame {i}: Pre-processing stage timing not isolated (> 0)"
            assert t_tensor > 0.0, f"Frame {i}: Tensor compute stage timing not isolated (> 0)"
            assert t_synth > 0.0, f"Frame {i}: Output synthesis stage timing not isolated (> 0)"

        denoised_audio = np.concatenate(denoised_frames)

        # Numerical integrity checks
        assert not np.isnan(denoised_audio).any(), f"Denoised audio for {ntype} contains NaN!"
        assert not np.isinf(denoised_audio).any(), f"Denoised audio for {ntype} contains Inf!"
        max_peak = float(np.max(np.abs(denoised_audio)))
        assert max_peak <= 1.05, f"Denoised audio clipped at {max_peak:.3f} > 1.05"

        # SNR calculation
        in_snr = ref.calculate_snr(c, mix, delay_samples=0)
        out_snr = ref.calculate_snr(c, denoised_audio, delay_samples=256)
        delta_snr = out_snr - in_snr

        stats = ring_buffer.compute_stats()

        passed = delta_snr >= 10.0
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"{ntype:<12} | {suite:<5} | {in_snr:6.2f} dB | {out_snr:6.2f} dB | {delta_snr:7.2f} dB  | {stats.p50_total_ms:6.3f} ms | {stats.p95_total_ms:6.3f} ms | {status_str}")
        results.append({
            "noise": ntype,
            "suite": suite,
            "delta_snr": delta_snr,
            "passed": passed,
            "stats": stats,
        })

    return all_passed, results


def evaluate_precision_modes(precision_filter: str):
    print("\n>>> [2/3] EVALUATING MULTI-PRECISION ENGINE (FP32 vs FP16 vs INT8)...")
    print("-" * 80)

    modes = ["FP32", "FP16", "INT8"] if precision_filter == "all" else [precision_filter]
    engine = PrecisionEngine()
    fp32_bytes = 165892

    print(f"{'Precision':<10} | {'Size (Bytes)':<12} | {'Compression':<12} | {'SQNR (dB)':<10} | {'Delta SNR':<10} | {'Status':<6}")
    print("-" * 80)

    all_passed = True
    ref = ReferenceSyntheticGenerator(sample_rate=16000, seed=42)
    clean = ref.generate_speech(duration_sec=2.0)
    white = ref.generate_noise("white", duration_sec=2.0)
    c, n, mix = ref.generate_mixture(clean, white, target_snr_db=0.0)

    for mode in modes:
        engine.set_precision(mode)
        mode_bytes = engine.get_model_size_bytes()
        ratio = mode_bytes / fp32_bytes

        # SQNR check
        rng = np.random.RandomState(42)
        test_w = rng.normal(0, 0.5, (64, 257)).astype(np.float32)
        if mode == "INT8":
            w_q, scale = PrecisionEngine.quantize_tensor(test_w)
            w_deq = PrecisionEngine.dequantize_tensor(w_q, scale)
            sqnr = PrecisionEngine.calculate_sqnr(test_w, w_deq)
            sqnr_pass = sqnr >= 35.0
            ratio_pass = ratio <= 0.35
        elif mode == "FP16":
            w_fp16 = test_w.astype(np.float16).astype(np.float32)
            sqnr = PrecisionEngine.calculate_sqnr(test_w, w_fp16)
            sqnr_pass = sqnr >= 50.0
            ratio_pass = abs(ratio - 0.50) < 0.05
        else:
            sqnr = 100.0
            sqnr_pass = True
            ratio_pass = ratio == 1.0

        # Denoising performance in this precision mode
        pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000, precision=mode)
        num_frames = len(mix) // 256
        denoised_frames = [pipeline.process_frame(mix[i * 256 : (i + 1) * 256]) for i in range(num_frames)]
        denoised_audio = np.concatenate(denoised_frames)

        in_snr = ref.calculate_snr(c, mix, delay_samples=0)
        out_snr = ref.calculate_snr(c, denoised_audio, delay_samples=256)
        delta_snr = out_snr - in_snr
        snr_pass = delta_snr >= 10.0

        mode_passed = sqnr_pass and ratio_pass and snr_pass
        if not mode_passed:
            all_passed = False

        status_str = "PASS" if mode_passed else "FAIL"
        print(f"{mode:<10} | {mode_bytes:<12} | {ratio * 100.0:5.1f}%       | {sqnr:7.1f} dB | {delta_snr:7.2f} dB  | {status_str}")

    return all_passed


def evaluate_telemetry_isolation():
    print("\n>>> [3/3] VERIFYING 3-STAGE TELEMETRY ISOLATION & PROCESS MEMORY...")
    print("-" * 80)

    profiler = StageProfiler(budget_ms=20.0)
    pipeline = AudioDenoisingPipeline()
    dummy_frame = np.random.randn(256).astype(np.float32) * 0.1

    profiler.start_frame()
    pipeline.process_frame(dummy_frame, profiler=profiler)
    t_pre, t_tensor, t_synth, t_total = profiler.mark_synthesis_done()

    mem = get_process_memory()
    print(f"Stage 1 (Pre-processing):    {t_pre:6.3f} ms")
    print(f"Stage 2 (Tensor Compute):    {t_tensor:6.3f} ms")
    print(f"Stage 3 (Output Synthesis):  {t_synth:6.3f} ms")
    print(f"Total Frame Latency:         {t_total:6.3f} ms  (Budget: 20.000 ms, Headroom: {20.0 - t_total:6.3f} ms)")
    print(f"Process Memory RSS:          {mem.rss_mb:6.2f} MB")

    assert t_pre > 0, "Stage 1 timing not isolated"
    assert t_tensor > 0, "Stage 2 timing not isolated"
    assert t_synth > 0, "Stage 3 timing not isolated"
    assert t_total <= 20.0, "Frame latency exceeded real-time budget"
    assert mem.rss_mb > 0, "Process memory could not be queried"
    print("\nTelemetry isolation and hardware timing verification: PASS")
    return True


def main():
    args = parse_args()
    print_banner()

    start_time = time.time()
    pass_benchmarks, _ = evaluate_noise_benchmarks(args.benchmark, args.duration, args.verbose)
    pass_precision = evaluate_precision_modes(args.precision)
    pass_telemetry = evaluate_telemetry_isolation()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    if pass_benchmarks and pass_precision and pass_telemetry:
        print(f"  ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY (Elapsed: {total_time:.2f}s)")
        print("  Exit code: 0")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"  EVALUATION FAILED (Elapsed: {total_time:.2f}s)")
        print("  Exit code: 1")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

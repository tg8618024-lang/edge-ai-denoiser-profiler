"""
Edge AI Audio Denoiser & Profiler — Baseline Benchmark Capture Tool (Phase 0).

Executes end-to-end against all benchmark noise profiles and multi-precision configurations,
capturing exact pre-enhancement stage latencies (t_pre, t_tensor, t_synth), memory RSS footprints,
and SNR suppression improvements.

Outputs:
- docs/benchmarks/baseline_telemetry.json (Machine-readable immutable baseline)
- docs/benchmarks/BASELINE_REPORT.md (Formatted engineering report)
"""

from __future__ import annotations
import os
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.dataset import SyntheticAudioGenerator, create_mixture
from src.audio.pipeline import AudioDenoisingPipeline
from src.telemetry.profiler import StageProfiler
from src.telemetry.ring_buffer import RollingMetricsBuffer
from src.telemetry.memory import get_process_memory
from src.models.precision import PrecisionEngine
from tests.conftest import ReferenceSyntheticGenerator


def run_baseline_benchmarks(duration_sec: float = 3.0):
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

    benchmark_results = []
    print("=" * 80)
    print("  CAPTURING PRE-ENHANCEMENT BASELINE METRICS (PHASE 0)")
    print("=" * 80)
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

        denoised_audio = np.concatenate(denoised_frames)
        in_snr = ref.calculate_snr(c, mix, delay_samples=0)
        out_snr = ref.calculate_snr(c, denoised_audio, delay_samples=256)
        delta_snr = out_snr - in_snr
        stats = ring_buffer.compute_stats()

        status = "PASS" if delta_snr >= 10.0 and stats.p95_total_ms <= 20.0 else "FAIL"
        print(f"{ntype:<12} | {suite:<5} | {in_snr:6.2f} dB | {out_snr:6.2f} dB | {delta_snr:7.2f} dB  | {stats.p50_total_ms:6.3f} ms | {stats.p95_total_ms:6.3f} ms | {status}")

        benchmark_results.append({
            "noise_type": ntype,
            "suite": suite,
            "in_snr_db": round(float(in_snr), 2),
            "out_snr_db": round(float(out_snr), 2),
            "delta_snr_db": round(float(delta_snr), 2),
            "p50_ms": round(float(stats.p50_total_ms), 3),
            "p95_ms": round(float(stats.p95_total_ms), 3),
            "p99_ms": round(float(stats.p99_total_ms), 3),
            "p50_pre_ms": round(float(stats.p50_pre_ms), 3),
            "p50_tensor_ms": round(float(stats.p50_tensor_ms), 3),
            "p50_synth_ms": round(float(stats.p50_synth_ms), 3),
            "jitter_ms": round(float(stats.jitter_ms), 3),
            "headroom_pct": round(float(stats.headroom_pct), 1),
            "fps_capacity": round(float(stats.throughput_fps), 1),
            "status": status,
        })

    # Precision Modes
    print("\n" + "-" * 80)
    print(f"{'Precision':<10} | {'Size (Bytes)':<12} | {'Compression':<12} | {'SQNR (dB)':<10} | {'Delta SNR':<10} | {'Status':<6}")
    print("-" * 80)

    precision_results = []
    engine = PrecisionEngine()
    fp32_bytes = 165892
    clean_speech = ref.generate_speech(duration_sec=2.0)
    white_noise = ref.generate_noise("white", duration_sec=2.0)
    c, n, mix_prec = ref.generate_mixture(clean_speech, white_noise, target_snr_db=0.0)

    for mode in ["FP32", "FP16", "INT8"]:
        engine.set_precision(mode)
        mode_bytes = engine.get_model_size_bytes()
        ratio = mode_bytes / fp32_bytes

        rng = np.random.RandomState(42)
        test_w = rng.normal(0, 0.5, (64, 257)).astype(np.float32)
        if mode == "INT8":
            w_q, scale = PrecisionEngine.quantize_tensor(test_w)
            w_deq = PrecisionEngine.dequantize_tensor(w_q, scale)
            sqnr = PrecisionEngine.calculate_sqnr(test_w, w_deq)
        elif mode == "FP16":
            w_fp16 = test_w.astype(np.float16).astype(np.float32)
            sqnr = PrecisionEngine.calculate_sqnr(test_w, w_fp16)
        else:
            sqnr = 100.0

        pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000, precision=mode)
        num_frames = len(mix_prec) // 256
        denoised_frames = [pipeline.process_frame(mix_prec[i * 256 : (i + 1) * 256]) for i in range(num_frames)]
        denoised_audio = np.concatenate(denoised_frames)
        in_snr = ref.calculate_snr(c, mix_prec, delay_samples=0)
        out_snr = ref.calculate_snr(c, denoised_audio, delay_samples=256)
        delta_snr = out_snr - in_snr
        status = "PASS" if delta_snr >= 10.0 else "FAIL"

        print(f"{mode:<10} | {mode_bytes:<12} | {ratio * 100.0:5.1f}%       | {sqnr:7.1f} dB | {delta_snr:7.2f} dB  | {status}")
        precision_results.append({
            "mode": mode,
            "size_bytes": mode_bytes,
            "compression_pct": round((1.0 - ratio) * 100.0, 1),
            "sqnr_db": round(float(sqnr), 1),
            "delta_snr_db": round(float(delta_snr), 2),
            "status": status,
        })

    # Isolated 3-Stage Hardware Timing
    profiler = StageProfiler(budget_ms=20.0)
    pipeline = AudioDenoisingPipeline()
    dummy_frame = np.random.randn(256).astype(np.float32) * 0.1
    profiler.start_frame()
    pipeline.process_frame(dummy_frame, profiler=profiler)
    t_pre, t_tensor, t_synth, t_total = profiler.mark_synthesis_done()
    mem = get_process_memory()

    print("\n" + "-" * 80)
    print("ISOLATED 3-STAGE LATENCY & MEMORY PROFILE:")
    print(f"  Stage 1 (Pre-processing):    {t_pre:.3f} ms")
    print(f"  Stage 2 (Tensor Compute):    {t_tensor:.3f} ms")
    print(f"  Stage 3 (Output Synthesis):  {t_synth:.3f} ms")
    print(f"  Total Per-Frame Latency:     {t_total:.3f} ms (Budget: 20.000 ms, Headroom: {20.0 - t_total:.3f} ms)")
    print(f"  Process Memory RSS:          {mem.rss_mb:.2f} MB")
    print("=" * 80)

    baseline_payload = {
        "metadata": {
            "title": "Edge AI Audio Denoiser Baseline Telemetry",
            "phase": "Phase 0 (Pre-Enhancement Baseline)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "python_version": sys.version.split()[0],
            "sample_rate_hz": 16000,
            "hop_length_samples": 256,
            "frame_duration_ms": 16.0,
            "frame_budget_ms": 20.0,
        },
        "isolated_stages_ms": {
            "t_pre_ms": round(float(t_pre), 3),
            "t_tensor_ms": round(float(t_tensor), 3),
            "t_synth_ms": round(float(t_synth), 3),
            "t_total_ms": round(float(t_total), 3),
            "headroom_ms": round(float(20.0 - t_total), 3),
            "headroom_pct": round(float((20.0 - t_total) / 20.0 * 100.0), 1),
        },
        "memory": {
            "rss_mb": round(float(mem.rss_mb), 2),
            "peak_rss_mb": round(float(mem.peak_rss_mb), 2),
        },
        "noise_benchmarks": benchmark_results,
        "precision_benchmarks": precision_results,
    }

    # Write JSON
    docs_benchmarks_dir = os.path.join(PROJECT_ROOT, "docs", "benchmarks")
    os.makedirs(docs_benchmarks_dir, exist_ok=True)
    json_path = os.path.join(docs_benchmarks_dir, "baseline_telemetry.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)

    # Write Markdown
    md_path = os.path.join(docs_benchmarks_dir, "BASELINE_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Telemetry & Benchmark Report (Phase 0)\n\n")
        f.write(f"**Captured**: {baseline_payload['metadata']['timestamp']} | **Python**: {baseline_payload['metadata']['python_version']}\n\n")
        f.write("This report captures the pre-enhancement performance baseline of the **Real-Time Edge AI Audio Denoiser & Profiler** pipeline before subsequent Phase 1–5 optimizations. All future enhancements are evaluated against these numbers.\n\n")
        f.write("## 1. 3-Stage Hardware Latency Breakdown & Memory\n\n")
        f.write("| Pipeline Stage | Latency | Real-Time Frame Budget | Headroom |\n")
        f.write("|:---|:---|:---|:---|\n")
        f.write(f"| **Stage 1: Pre-processing (STFT, Windowing)** | `{t_pre:.3f} ms` | 20.000 ms | - |\n")
        f.write(f"| **Stage 2: Tensor Compute (GRUMaskNet / Wiener)** | `{t_tensor:.3f} ms` | 20.000 ms | - |\n")
        f.write(f"| **Stage 3: Output Synthesis (iSTFT, Overlap-Add)** | `{t_synth:.3f} ms` | 20.000 ms | - |\n")
        f.write(f"| **Total Frame Processing Time** | **`{t_total:.3f} ms`** | **20.000 ms** | **`{20.0 - t_total:.3f} ms` ({((20.0 - t_total)/20.0)*100.0:.1f}%)** |\n\n")
        f.write(f"- **Process Memory (RSS)**: `{mem.rss_mb:.2f} MB`\n")
        f.write(f"- **Nominal Frame Hop**: `16.0 ms` (`256 samples @ 16 kHz`)\n")
        f.write(f"- **Algorithmic Delay**: `16.0 ms` (`256 samples` deterministic delay)\n\n")
        f.write("## 2. Noise Suppression Performance (SNR Gain)\n\n")
        f.write("| Noise Profile | Suite | In SNR (dB) | Out SNR (dB) | Delta SNR Gain (dB) | P50 Latency | P95 Latency | Status |\n")
        f.write("|:---|:---|:---|:---|:---|:---|:---|:---|\n")
        for b in benchmark_results:
            f.write(f"| `{b['noise_type']}` | {b['suite']} | `{b['in_snr_db']:.2f}` | `{b['out_snr_db']:.2f}` | **`+{b['delta_snr_db']:.2f} dB`** | `{b['p50_ms']:.3f} ms` | `{b['p95_ms']:.3f} ms` | {b['status']} |\n")
        f.write("\n## 3. Multi-Precision Engine (FP32 vs FP16 vs INT8)\n\n")
        f.write("| Mode | Model Size | RAM Savings | SQNR (dB) | Denoising SNR Gain | Validation |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        for p in precision_results:
            f.write(f"| **{p['mode']}** | `{p['size_bytes']:,} B` | `{p['compression_pct']}%` | `{p['sqnr_db']} dB` | `+{p['delta_snr_db']} dB` | {p['status']} |\n")
        f.write("\n---\n*Report generated automatically by `benchmark_baseline.py`.* \n")

    print(f"\n[SUCCESS] Baseline captured to:\n  - {json_path}\n  - {md_path}")
    return baseline_payload


if __name__ == "__main__":
    run_baseline_benchmarks()

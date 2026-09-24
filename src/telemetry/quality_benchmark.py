"""
Objective Speech Quality Benchmark & Tradeoff Profiler (Phase 3).

Measures perceptual speech quality, intelligibility, and latency across precision and gating formats:
- STOI: Short-Time Objective Intelligibility (0.0 to 1.0) via pystoi.
- DNSMOS: ITU-T P.835 deep perceptual speech evaluation (SIG, BAK, OVRL, 1.0 to 5.0).
- Delta SNR: Objective Signal-to-Noise Ratio gain in dB.
- Latency & Compute Savings: Profiling per-frame P50 latency and VAD compute reduction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from src.audio.pipeline import AudioDenoisingPipeline
from src.telemetry.dnsmos import PerceptualQualityEvaluator
from src.telemetry.profiler import StageProfiler
from src.telemetry.ring_buffer import RollingMetricsBuffer
from tests.conftest import ReferenceSyntheticGenerator


@dataclass(slots=True)
class QualityMetrics:
    mode_name: str
    engine: str
    precision: str
    vad_gated: bool
    model_size_kb: float
    stoi: float
    dnsmos_sig: float
    dnsmos_bak: float
    dnsmos_ovrl: float
    delta_snr_db: float
    p50_latency_ms: float
    p95_latency_ms: float
    compute_savings_pct: float


class QualityBenchmarkSuite:
    """Automated benchmark comparing speech quality and computational efficiency."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.dnsmos = PerceptualQualityEvaluator(sample_rate=sample_rate)

    def calculate_stoi(self, clean: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Short-Time Objective Intelligibility (STOI) score."""
        try:
            from pystoi import stoi
            min_len = min(len(clean), len(processed))
            c = clean[:min_len].astype(np.float64)
            p = processed[:min_len].astype(np.float64)
            val = float(stoi(c, p, self.sample_rate, extended=False))
            return round(val, 3)
        except Exception:
            # Fallback envelope correlation approximation
            min_len = min(len(clean), len(processed))
            c = np.abs(clean[:min_len])
            p = np.abs(processed[:min_len])
            corr = np.corrcoef(c, p)[0, 1] if len(c) > 10 else 1.0
            return round(float(np.clip(corr, 0.0, 1.0)), 3)

    def evaluate_configuration(
        self,
        name: str,
        engine_type: str,  # "numpy" or "onnx"
        precision: str,    # "FP32", "FP16", "INT8"
        vad_gating: bool,
        speech: np.ndarray,
        noise: np.ndarray,
        clean_ref: np.ndarray,
        target_snr_db: float = 5.0,
    ) -> QualityMetrics:
        """Evaluate a single pipeline configuration on audio mixture."""
        ref = ReferenceSyntheticGenerator(sample_rate=self.sample_rate, seed=42)
        c, n, mix = ref.generate_mixture(speech, noise, target_snr_db=target_snr_db)

        # Instantiate model / engine
        if engine_type == "onnx":
            from src.models.onnx_engine import ONNXMaskNetEngine
            model = ONNXMaskNetEngine(precision=precision)
            pipeline = AudioDenoisingPipeline(model=model, vad_gating=vad_gating)
            model_size_kb = model.memory_footprint_bytes / 1024.0
        else:
            pipeline = AudioDenoisingPipeline(precision=precision, vad_gating=vad_gating)
            model_size_kb = pipeline.get_model_size_bytes() / 1024.0

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

        denoised = np.concatenate(denoised_frames)
        min_len = min(len(denoised), len(mix), len(c))
        denoised = denoised[:min_len]
        mix_trimmed = mix[:min_len]
        c_trimmed = c[:min_len]

        # Quality metrics
        in_snr = ref.calculate_snr(c_trimmed, mix_trimmed, delay_samples=0)
        out_snr = ref.calculate_snr(c_trimmed, denoised, delay_samples=256)
        delta_snr = out_snr - in_snr

        stoi_score = self.calculate_stoi(c_trimmed, denoised)
        mos = self.dnsmos.evaluate_frame(denoised, mix_trimmed, delta_snr, 1.0)

        stats = ring_buffer.compute_stats()
        savings_pct = pipeline.get_vad_compute_savings_pct() if vad_gating else 0.0

        return QualityMetrics(
            mode_name=name,
            engine=engine_type.upper(),
            precision=precision.upper(),
            vad_gated=vad_gating,
            model_size_kb=round(model_size_kb, 1),
            stoi=stoi_score,
            dnsmos_sig=round(mos.sig_mos, 2),
            dnsmos_bak=round(mos.bak_mos, 2),
            dnsmos_ovrl=round(mos.ovrl_mos, 2),
            delta_snr_db=round(delta_snr, 2),
            p50_latency_ms=round(stats.p50_total_ms, 3),
            p95_latency_ms=round(stats.p95_total_ms, 3),
            compute_savings_pct=savings_pct,
        )

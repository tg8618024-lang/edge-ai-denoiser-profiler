"""Systematic Scientific Component Ablation & Pareto Efficiency Analysis Engine.

Authority: Phase 13 Architectural Blueprint & Objective Quality Gate.
Orchestrates systematic evaluation of individual pipeline subsystems:
1. Bypass (Unprocessed baseline)
2. Classical Decision-Directed Wiener Filter
3. Recurrent Neural GRUMaskNet
4. Hybrid Dual-Pipeline (Crossfaded Neural + Wiener)
5. Complex Ratio Masking (CRM Cartesian Phase Alignment)
6. Full Studio Pipeline (FP32 Precision + Harmonic Formants + Vocal Suite + Parametric EQ)
7. Full Studio Pipeline (INT8 Quantization + SIMD Integer Dot-Product)
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import AudioDenoisingPipeline
from src.audio.dataset import SyntheticAudioGenerator, create_mixture, compute_snr_gain
from src.telemetry.dnsmos import PerceptualQualityEstimator, compute_third_octave_stoi


@dataclass
class AblationConfigResult:
    """Evaluation telemetry for a single ablation configuration."""
    name: str
    description: str
    snr_gain_db: float
    stoi_score: float
    sig_mos: float
    bak_mos: float
    ovrl_mos: float
    p50_latency_ms: float
    p95_latency_ms: float
    model_bytes: int
    compression_pct: float
    status: str = "PASS"


@dataclass
class AblationReport:
    """Full multi-configuration ablation study report."""
    timestamp: float
    presets_evaluated: List[str]
    duration_sec: float
    results: List[AblationConfigResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format ablation results as a clean Markdown table with Pareto highlights."""
        lines = [
            "### Scientific Component Ablation & Quality Analysis",
            "",
            "| Configuration | Description | Delta SNR (dB) | STOI | SIG | BAK | OVRL | P50 (ms) | Memory | Status |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            mem_str = f"{round(r.model_bytes / 1024.0, 1)} KB" if r.model_bytes > 0 else "0 KB (DSP)"
            status_tag = "[PASS]" if r.status == "PASS" else "[WARN]"
            lines.append(
                f"| **{r.name}** | {r.description} | {r.snr_gain_db:+.2f} dB | "
                f"{r.stoi_score:.3f} | {r.sig_mos:.2f} | {r.bak_mos:.2f} | {r.ovrl_mos:.2f} | "
                f"{r.p50_latency_ms:.2f} ms | {mem_str} ({r.compression_pct:.1f}%) | {status_tag} {r.status} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "presets_evaluated": self.presets_evaluated,
            "duration_sec": self.duration_sec,
            "results": [
                {
                    "name": r.name,
                    "description": r.description,
                    "snr_gain_db": r.snr_gain_db,
                    "stoi_score": r.stoi_score,
                    "sig_mos": r.sig_mos,
                    "bak_mos": r.bak_mos,
                    "ovrl_mos": r.ovrl_mos,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "model_bytes": r.model_bytes,
                    "compression_pct": r.compression_pct,
                    "status": r.status,
                }
                for r in self.results
            ],
        }


class AblationEngine:
    """Scientific ablation engine isolating algorithmic subsystem contributions."""

    CONFIGURATIONS = [
        {
            "name": "Bypass",
            "desc": "Unprocessed noisy audio baseline",
            "mode": "bypass",
            "crm": False,
            "prec": "FP32",
            "vocal": False,
            "eq": False,
        },
        {
            "name": "Wiener_DSP",
            "desc": "Classical Decision-Directed Wiener Filter",
            "mode": "wiener",
            "crm": False,
            "prec": "FP32",
            "vocal": False,
            "eq": False,
        },
        {
            "name": "Neural_GRUMaskNet",
            "desc": "Recurrent Causal GRU Magnitude Masking",
            "mode": "neural",
            "crm": False,
            "prec": "FP32",
            "vocal": False,
            "eq": False,
        },
        {
            "name": "Hybrid_Dual",
            "desc": "Crossfaded Neural + Classical Wiener",
            "mode": "hybrid",
            "crm": False,
            "prec": "FP32",
            "vocal": False,
            "eq": False,
        },
        {
            "name": "Hybrid_CRM",
            "desc": "Hybrid + Complex Ratio Masking (Phase Aware)",
            "mode": "hybrid",
            "crm": True,
            "prec": "FP32",
            "vocal": False,
            "eq": False,
        },
        {
            "name": "Full_Studio_FP32",
            "desc": "Full Suite (Harmonics + Vocal Suite + EQ) [FP32]",
            "mode": "hybrid",
            "crm": True,
            "prec": "FP32",
            "vocal": True,
            "eq": True,
        },
        {
            "name": "Full_Studio_INT8",
            "desc": "Full Suite (Quantized SIMD GEMV) [INT8]",
            "mode": "hybrid",
            "crm": True,
            "prec": "INT8",
            "vocal": True,
            "eq": True,
        },
    ]

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.gen = SyntheticAudioGenerator(sample_rate=sample_rate)
        self.quality_evaluator = PerceptualQualityEstimator(sample_rate=sample_rate)

    def run_ablation_study(
        self,
        presets: Optional[List[str]] = None,
        duration_sec: float = 2.0,
        target_snr_db: float = 0.0,
    ) -> AblationReport:
        """Execute systematic ablation benchmark across all 7 configurations."""
        presets = presets or ["white", "drone", "rf_static"]
        fp32_baseline_bytes = 165892

        # 1. Generate test signals
        speech = self.gen.generate_speech(duration_sec=duration_sec, seed=42)
        test_mixtures: List[Dict[str, np.ndarray]] = []
        for p in presets:
            noise = self.gen.generate_noise(p, duration_sec=duration_sec, seed=42)
            mix, c, n = create_mixture(speech, noise, target_snr_db=target_snr_db)
            test_mixtures.append({"preset": p, "mix": mix, "clean": c, "noise": n})

        results: List[AblationConfigResult] = []

        # 2. Iterate through each ablation configuration
        for cfg in self.CONFIGURATIONS:
            snr_deltas: List[float] = []
            stois: List[float] = []
            sigs: List[float] = []
            baks: List[float] = []
            ovrls: List[float] = []
            latencies_ms: List[float] = []
            model_bytes = 0

            is_bypass = cfg["mode"] == "bypass"

            pipeline = None
            if not is_bypass:
                pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=self.sample_rate)
                pipeline.set_mode(cfg["mode"])
                pipeline.set_precision(cfg["prec"])
                pipeline.crm_enabled = cfg["crm"]
                pipeline.vocal_suite_enabled = cfg["vocal"]
                pipeline.eq.set_enabled(cfg["eq"])
                model_bytes = pipeline.get_model_size_bytes() if cfg["mode"] != "wiener" else 0

            # Evaluate across test audio mixtures
            for item in test_mixtures:
                mix = item["mix"]
                clean = item["clean"]
                hop = 256

                if is_bypass:
                    processed = mix.copy()
                    dt_frame = 0.001
                    for _ in range(len(mix) // hop):
                        latencies_ms.append(dt_frame)
                else:
                    pipeline.reset()
                    out_chunks = []
                    for i in range(len(mix) // hop):
                        chunk = mix[i * hop : (i + 1) * hop]
                        t0 = time.perf_counter()
                        clean_chunk = pipeline.process_frame(chunk)
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        latencies_ms.append(dt_ms)
                        out_chunks.append(clean_chunk)
                    processed = np.concatenate(out_chunks)

                min_len = min(len(clean), len(processed))
                c_eval = clean[:min_len]
                p_eval = processed[:min_len]
                m_eval = mix[:min_len]

                delta_snr = compute_snr_gain(c_eval, m_eval, p_eval)
                stoi_val = compute_third_octave_stoi(c_eval, p_eval, sample_rate=self.sample_rate)
                score = self.quality_evaluator.evaluate(p_eval, m_eval, vad_active=True)

                snr_deltas.append(delta_snr)
                stois.append(stoi_val)
                sigs.append(score.sig_mos)
                baks.append(score.bak_mos)
                ovrls.append(score.ovrl_mos)

            avg_snr = float(np.mean(snr_deltas))
            avg_stoi = float(np.mean(stois))
            avg_sig = float(np.mean(sigs))
            avg_bak = float(np.mean(baks))
            avg_ovrl = float(np.mean(ovrls))
            p50_lat = float(np.median(latencies_ms))
            p95_lat = float(np.percentile(latencies_ms, 95))

            compression_pct = (1.0 - (model_bytes / fp32_baseline_bytes)) * 100.0 if model_bytes > 0 else (100.0 if is_bypass or cfg["mode"] == "wiener" else 0.0)

            status = "PASS"
            if not is_bypass and avg_snr < 5.0:
                status = "WARN"

            results.append(
                AblationConfigResult(
                    name=cfg["name"],
                    description=cfg["desc"],
                    snr_gain_db=round(avg_snr, 2),
                    stoi_score=round(avg_stoi, 3),
                    sig_mos=round(avg_sig, 2),
                    bak_mos=round(avg_bak, 2),
                    ovrl_mos=round(avg_ovrl, 2),
                    p50_latency_ms=round(p50_lat, 3),
                    p95_latency_ms=round(p95_lat, 3),
                    model_bytes=model_bytes,
                    compression_pct=round(compression_pct, 1),
                    status=status,
                )
            )

        return AblationReport(
            timestamp=time.time(),
            presets_evaluated=presets,
            duration_sec=duration_sec,
            results=results,
        )


if __name__ == "__main__":
    engine = AblationEngine()
    print("Executing Edge AI Denoiser Component Ablation Study...")
    report = engine.run_ablation_study(presets=["white", "drone"], duration_sec=1.5)
    print("\n" + report.to_markdown() + "\n")

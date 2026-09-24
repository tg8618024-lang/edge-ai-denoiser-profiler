"""Dual-Model Concurrent A/B Audio Denoising Pipeline.

Runs two distinct speech enhancement models in parallel on every 16.0 ms frame:
- Model A: Deep Recurrent Neural Masking Network (GRUMaskNet / Hybrid)
- Model B: Classical Decision-Directed Wiener Filter (Pure DSP)

Provides:
- Independent 3-stage latency telemetry for both models.
- Real-time continuous crossfading: y_mix[n] = (1 - alpha) * y_A[n] + alpha * y_B[n].
- Side-by-side comparative metrics (SNR gain, model size, inference latency, compute headroom).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from src.audio.stft import StreamingSTFT
from src.models.denoiser import HybridDenoiser, DecisionDirectedWienerFilter
from src.telemetry.profiler import StageProfiler


@dataclass
class ModelMetrics:
    """Performance telemetry for a single model."""
    name: str
    pre_processing_ms: float = 0.0
    tensor_compute_ms: float = 0.0
    synthesis_ms: float = 0.0
    total_latency_ms: float = 0.0
    model_size_bytes: int = 0
    snr_gain_db: float = 0.0
    headroom_pct: float = 0.0


@dataclass
class DualPipelineResult:
    """Output bundle returned by DualModelPipeline for every processed frame."""
    audio_a: np.ndarray
    audio_b: np.ndarray
    audio_mix: np.ndarray
    crossfade_alpha: float
    telemetry_a: ModelMetrics
    telemetry_b: ModelMetrics
    concurrent_latency_ms: float
    spec_in_mag: np.ndarray
    spec_a_mag: np.ndarray
    spec_b_mag: np.ndarray


class DualModelPipeline:
    """Orchestrates concurrent execution of Model A (Neural) and Model B (Classical Wiener DSP)."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        sample_rate: int = 16000,
        model_a_mode: str = "hybrid",
        crossfade_alpha: float = 0.5,
        weights_path: Optional[str] = None,
        precision: str = "FP32",
    ) -> None:
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.crossfade_alpha = float(np.clip(crossfade_alpha, 0.0, 1.0))
        self.frame_budget_ms = (self.hop_length / self.sample_rate) * 1000.0

        # Shared analysis STFT
        self.stft_analysis = StreamingSTFT(
            n_fft=self.n_fft, hop_length=self.hop_length, sample_rate=self.sample_rate
        )

        # Independent synthesis STFT engines for overlap-add reconstruction
        self.stft_synth_a = StreamingSTFT(
            n_fft=self.n_fft, hop_length=self.hop_length, sample_rate=self.sample_rate
        )
        self.stft_synth_b = StreamingSTFT(
            n_fft=self.n_fft, hop_length=self.hop_length, sample_rate=self.sample_rate
        )

        # Model A: Neural / Hybrid
        self.model_a = HybridDenoiser(
            num_bins=self.stft_analysis.num_bins,
            mode=model_a_mode,
            weights_path=weights_path,
        )
        self.model_a.set_precision(precision)

        # Model B: Classical Decision-Directed Wiener Filter (Pure DSP)
        self.model_b = DecisionDirectedWienerFilter(
            num_bins=self.stft_analysis.num_bins
        )

        # Signal power trackers for rolling SNR estimation
        self._noise_power_a: float = 1e-6
        self._speech_power_a: float = 1e-6
        self._noise_power_b: float = 1e-6
        self._speech_power_b: float = 1e-6

        self.total_frames_processed = 0

    def set_crossfade(self, alpha: float) -> None:
        """Set crossfade ratio: 0.0 = 100% Model A, 1.0 = 100% Model B."""
        self.crossfade_alpha = float(np.clip(alpha, 0.0, 1.0))

    def get_crossfade(self) -> float:
        """Return active crossfade ratio."""
        return self.crossfade_alpha

    def set_precision(self, precision: str) -> None:
        """Set numerical precision for Model A neural net."""
        self.model_a.set_precision(precision)

    def reset(self) -> None:
        """Reset state across both models and synthesis engines."""
        self.stft_analysis.reset()
        self.stft_synth_a.reset()
        self.stft_synth_b.reset()
        self.model_a.reset()
        self.model_b.reset()
        self._noise_power_a = 1e-6
        self._speech_power_a = 1e-6
        self._noise_power_b = 1e-6
        self._speech_power_b = 1e-6
        self.total_frames_processed = 0

    def process_frame(
        self,
        frame_pcm: np.ndarray,
        crossfade_alpha: Optional[float] = None,
    ) -> DualPipelineResult:
        """Process a single hop frame concurrently through Model A and Model B.

        Parameters
        ----------
        frame_pcm : np.ndarray
            Input audio chunk of length hop_length (256 samples).
        crossfade_alpha : Optional[float]
            Optional per-frame crossfade override in [0.0, 1.0].

        Returns
        -------
        DualPipelineResult
            Complete bundle containing audio_a, audio_b, audio_mix, and dual telemetry.
        """
        if crossfade_alpha is not None:
            alpha = float(np.clip(crossfade_alpha, 0.0, 1.0))
        else:
            alpha = self.crossfade_alpha

        frame_pcm = np.asarray(frame_pcm, dtype=np.float32).ravel()
        t_start_concurrent = time.perf_counter()

        # =====================================================================
        # Stage 1: Pre-processing (Shared STFT)
        # =====================================================================
        t_pre_0 = time.perf_counter()
        spec_complex = self.stft_analysis.analyze(frame_pcm)
        t_pre_ms = (time.perf_counter() - t_pre_0) * 1000.0

        mag_in = np.abs(spec_complex).astype(np.float32)

        # =====================================================================
        # Stage 2: Tensor Compute - Model A (Neural) vs Model B (Wiener DSP)
        # =====================================================================
        # Model A Compute
        t_t_a_0 = time.perf_counter()
        gain_a = self.model_a.compute_gain(spec_complex)
        t_t_a_ms = (time.perf_counter() - t_t_a_0) * 1000.0

        # Model B Compute
        t_t_b_0 = time.perf_counter()
        gain_b = self.model_b.compute_gain(mag_in)
        t_t_b_ms = (time.perf_counter() - t_t_b_0) * 1000.0

        # Apply gains
        spec_out_a = spec_complex * gain_a
        spec_out_b = spec_complex * gain_b

        # =====================================================================
        # Stage 3: Output Synthesis (Independent Overlap-Add)
        # =====================================================================
        # Model A Synthesis
        t_s_a_0 = time.perf_counter()
        out_pcm_a = self.stft_synth_a.synthesize(spec_out_a)
        t_s_a_ms = (time.perf_counter() - t_s_a_0) * 1000.0

        # Model B Synthesis
        t_s_b_0 = time.perf_counter()
        out_pcm_b = self.stft_synth_b.synthesize(spec_out_b)
        t_s_b_ms = (time.perf_counter() - t_s_b_0) * 1000.0

        # =====================================================================
        # Crossfaded Blend
        # =====================================================================
        out_pcm_mix = (1.0 - alpha) * out_pcm_a + alpha * out_pcm_b
        out_pcm_mix = np.clip(out_pcm_mix, -1.0, 1.0).astype(np.float32)

        concurrent_total_ms = (time.perf_counter() - t_start_concurrent) * 1000.0
        total_a_ms = t_pre_ms + t_t_a_ms + t_s_a_ms
        total_b_ms = t_pre_ms + t_t_b_ms + t_s_b_ms

        # Track SNR improvements
        noise_a = frame_pcm - out_pcm_a
        noise_b = frame_pcm - out_pcm_b
        self._speech_power_a = 0.95 * self._speech_power_a + 0.05 * float(np.mean(out_pcm_a ** 2))
        self._noise_power_a = 0.95 * self._noise_power_a + 0.05 * float(np.mean(noise_a ** 2))
        self._speech_power_b = 0.95 * self._speech_power_b + 0.05 * float(np.mean(out_pcm_b ** 2))
        self._noise_power_b = 0.95 * self._noise_power_b + 0.05 * float(np.mean(noise_b ** 2))

        snr_gain_a = float(10.0 * np.log10(max(self._speech_power_a, 1e-9) / max(self._noise_power_a, 1e-9)))
        snr_gain_b = float(10.0 * np.log10(max(self._speech_power_b, 1e-9) / max(self._noise_power_b, 1e-9)))

        headroom_a = max(0.0, (1.0 - total_a_ms / self.frame_budget_ms) * 100.0)
        headroom_b = max(0.0, (1.0 - total_b_ms / self.frame_budget_ms) * 100.0)

        metrics_a = ModelMetrics(
            name="Model A: GRUMaskNet Neural",
            pre_processing_ms=round(t_pre_ms, 3),
            tensor_compute_ms=round(t_t_a_ms, 3),
            synthesis_ms=round(t_s_a_ms, 3),
            total_latency_ms=round(total_a_ms, 3),
            model_size_bytes=self.model_a.get_model_size_bytes(),
            snr_gain_db=round(snr_gain_a, 2),
            headroom_pct=round(headroom_a, 1),
        )

        metrics_b = ModelMetrics(
            name="Model B: Classical Wiener DSP",
            pre_processing_ms=round(t_pre_ms, 3),
            tensor_compute_ms=round(t_t_b_ms, 3),
            synthesis_ms=round(t_s_b_ms, 3),
            total_latency_ms=round(total_b_ms, 3),
            model_size_bytes=0,  # Pure analytical algorithmic filter
            snr_gain_db=round(snr_gain_b, 2),
            headroom_pct=round(headroom_b, 1),
        )

        self.total_frames_processed += 1

        return DualPipelineResult(
            audio_a=out_pcm_a,
            audio_b=out_pcm_b,
            audio_mix=out_pcm_mix,
            crossfade_alpha=alpha,
            telemetry_a=metrics_a,
            telemetry_b=metrics_b,
            concurrent_latency_ms=round(concurrent_total_ms, 3),
            spec_in_mag=mag_in,
            spec_a_mag=np.abs(spec_out_a).astype(np.float32),
            spec_b_mag=np.abs(spec_out_b).astype(np.float32),
        )

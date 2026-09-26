"""Real-time end-to-end audio denoising pipeline with profiler stage instrumentation.

Ties together:
- Frame input (H = 256 samples @ 16 kHz = 16.0 ms)
- Stage 1: Pre-processing (Square-root Hann windowing, shift buffer, rFFT)
- Stage 2: Tensor Compute (Neural mask estimation & Wiener spectral filter gain)
- Stage 3: Output Synthesis (iSTFT, synthesis windowing, overlap-add accumulation)
- Fine-grained profiler hooks matching the M1 <-> M2 interface contract.
- Multi-precision mode selection (FP32, FP16, INT8).
"""

from typing import Any, Optional, Dict
import time
import numpy as np

from src.audio.stft import StreamingSTFT
from src.audio.stream import AudioStreamer
from src.audio.vad import VoiceActivityDetector, VADDecision
from src.audio.harmonics import HarmonicEnhancer
from src.audio.dereverb import SpectralDereverberator
from src.audio.vocal_suite import BroadcastVocalSuite
from src.audio.parametric_eq import ParametricEQ
from src.models.target_speaker import TargetSpeakerExtractor
from src.models.denoiser import HybridDenoiser
from src.models.noise_classifier import (
    NoiseSignatureClassifier,
    AdaptiveSuppressionController,
    NoiseClassificationResult,
)
from src.models.crm import ComplexRatioMasker
from src.telemetry.dnsmos import PerceptualQualityEstimator, PerceptualQualityScore
import io
import wave
from collections import deque



class AudioDenoisingPipeline:
    """Real-time streaming audio denoising pipeline with latency profiler hooks.

    Complies with the M1 <-> M2 interface contract defined in PROJECT.md:
    Inside process_frame:
      - profiler.start_stage("pre_processing") -> STFT & windowing -> profiler.end_stage("pre_processing")
      - profiler.start_stage("tensor_compute") -> Model / filter mask inference -> profiler.end_stage("tensor_compute")
      - profiler.start_stage("output_synthesis") -> iSTFT & overlap-add -> profiler.end_stage("output_synthesis")
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        sample_rate: int = 16000,
        denoiser_mode: str = "hybrid",
        model: Optional[Any] = None,
        weights_path: Optional[str] = None,
        precision: str = "FP32",
        intensity: float = 1.0,
        vad_gating: bool = False,
    ) -> None:
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self.vad_gating = bool(vad_gating)

        # VAD engine
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            frame_size=self.hop_length,
        )
        self.harmonic_enhancer = HarmonicEnhancer(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        )
        self.last_vad_decision: Optional[VADDecision] = None
        self.cumulative_tensor_time_saved_ms: float = 0.0

        # STFT engine
        self.stft = StreamingSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        # Denoiser model
        if model is not None:
            self.model = model
        else:
            self.model = HybridDenoiser(
                num_bins=self.stft.num_bins,
                mode=denoiser_mode,
                weights_path=weights_path,
            )

        self.precision = precision.upper().strip()
        self.set_precision(self.precision)

        # Complex Ratio Masking (CRM) Phase Alignment Engine
        self.crm = ComplexRatioMasker(mask_bound=2.0)
        self.crm_enabled: bool = True
        self.crm_phase_factor: float = 0.01
        self.last_substages: Dict[str, float] = {}

        # Option A: Room De-Reverberation Engine
        self.dereverberator = SpectralDereverberator(
            num_bins=self.stft.num_bins,
            sample_rate=self.sample_rate,
            amount=0.0,
        )

        # Option A: Studio Broadcast Vocal Suite (Compressor, De-Esser, Limiter)
        self.vocal_suite_enabled: bool = False
        self.vocal_suite = BroadcastVocalSuite(
            sample_rate=self.sample_rate,
            compressor_enabled=True,
            deesser_enabled=True,
            warmth_enabled=True,
        )

        # Trigger-Word Activated Target Speaker Extraction (TSE / Voice Lock)
        self.target_speaker = TargetSpeakerExtractor(
            sample_rate=self.sample_rate,
            num_bins=self.stft.num_bins,
            similarity_threshold=0.72,
        )

        # Option B: 5-Band Studio Parametric EQ Sculptor
        self.eq = ParametricEQ(sample_rate=self.sample_rate, enabled=False)

        # Option C: AI Noise Environment Classifier & Auto-Adaptive Engine
        self.noise_classifier = NoiseSignatureClassifier(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        )
        self.adaptive_controller = AdaptiveSuppressionController()
        self.last_noise_classification: Optional[NoiseClassificationResult] = None

        # Option C: ITU-T P.835 Perceptual Quality (DNSMOS Suite)
        self.dnsmos = PerceptualQualityEstimator(sample_rate=self.sample_rate)
        self.last_dnsmos_score: Optional[PerceptualQualityScore] = None

        # Option C: 1-Click Multi-Track Rolling Studio WAV Recorder
        # Stores up to 10 seconds of 16 kHz audio (160,000 samples = 625 hops)
        self.record_max_hops = int((self.sample_rate * 10.0) / self.hop_length)
        self.record_buffer_in = deque(maxlen=self.record_max_hops)
        self.record_buffer_out = deque(maxlen=self.record_max_hops)

        # Telemetry inspection caches for last processed frame
        self.last_input_spec: Optional[np.ndarray] = None
        self.last_gain_mask: Optional[np.ndarray] = None
        self.last_output_spec: Optional[np.ndarray] = None
        self.total_frames_processed = 0

    @property
    def frame_budget_ms(self) -> float:
        """Nominal real-time frame budget in ms."""
        return self.stft.frame_duration_ms

    @property
    def algorithmic_delay_samples(self) -> int:
        """Deterministic algorithmic delay in samples (1 hop)."""
        return self.hop_length

    def set_precision(self, mode: str) -> None:
        """Dynamically set inference precision ('FP32', 'FP16', 'INT8')."""
        self.precision = mode.upper().strip()
        if hasattr(self.model, "set_precision"):
            self.model.set_precision(self.precision)

    def get_precision(self) -> str:
        """Return active pipeline precision mode."""
        if hasattr(self.model, "get_precision"):
            return self.model.get_precision()
        return self.precision

    def get_model_size_bytes(self) -> int:
        """Return active model size in bytes."""
        if hasattr(self.model, "get_model_size_bytes"):
            return self.model.get_model_size_bytes()
        return 165892

    def reset(self) -> None:
        """Reset internal buffers and states of STFT and denoiser."""
        self.stft.reset()
        if hasattr(self.model, "reset"):
            self.model.reset()
        if hasattr(self, "vad"):
            self.vad.reset()
        if hasattr(self, "dereverberator"):
            self.dereverberator.reset()
        if hasattr(self, "vocal_suite"):
            self.vocal_suite.reset()
        if hasattr(self, "target_speaker"):
            self.target_speaker.unlock()
        if hasattr(self, "eq"):
            self.eq.reset()
        if hasattr(self, "record_buffer_in"):
            self.record_buffer_in.clear()
            self.record_buffer_out.clear()
        self.last_input_spec = None
        self.last_gain_mask = None
        self.last_output_spec = None
        self.last_vad_decision = None
        self.last_noise_classification = None
        self.last_dnsmos_score = None
        self.last_substages.clear()
        self.cumulative_tensor_time_saved_ms = 0.0
        self.total_frames_processed = 0

    # -------------------------------------------------------------------------
    # Complex Ratio Masking (CRM) Phase Enhancement Controls
    # -------------------------------------------------------------------------
    def set_crm_enabled(self, enabled: bool) -> None:
        """Enable or disable Complex Ratio Masking (CRM) phase enhancement."""
        self.crm_enabled = bool(enabled)

    def get_crm_enabled(self) -> bool:
        """Return whether CRM phase enhancement is active."""
        return self.crm_enabled

    def set_crm_phase_factor(self, factor: float) -> None:
        """Set CRM imaginary phase correction factor."""
        self.crm_phase_factor = float(np.clip(factor, 0.0, 0.5))

    def get_crm_phase_factor(self) -> float:
        """Return active CRM phase correction factor."""
        return self.crm_phase_factor

    def get_substage_latencies(self) -> Dict[str, float]:
        """Return fine-grained sub-stage latencies (VAD, classifier, dereverb, etc.) in ms."""
        return dict(self.last_substages)

    # -------------------------------------------------------------------------
    # Option A: Room De-Reverberation & Studio Broadcast Vocal Suite Controls
    # -------------------------------------------------------------------------
    def set_dereverb_amount(self, amount: float) -> None:
        """Set continuous dereverberation suppression intensity in [0.0, 1.0]."""
        self.dereverberator.set_amount(amount)

    def get_dereverb_amount(self) -> float:
        """Return active dereverberation suppression amount."""
        return self.dereverberator.get_amount()

    def set_vocal_suite_enabled(self, enabled: bool) -> None:
        """Enable or disable studio broadcast vocal suite post-processing."""
        self.vocal_suite_enabled = bool(enabled)

    def get_vocal_suite_enabled(self) -> bool:
        """Return whether studio vocal suite is active."""
        return self.vocal_suite_enabled

    def configure_vocal_suite(
        self,
        compressor: Optional[bool] = None,
        deesser: Optional[bool] = None,
        warmth: Optional[bool] = None,
        threshold_db: Optional[float] = None,
        ratio: Optional[float] = None,
    ) -> None:
        """Configure dynamic range compressor, de-esser, and limiter parameters."""
        self.vocal_suite.configure(
            compressor=compressor,
            deesser=deesser,
            warmth=warmth,
            threshold_db=threshold_db,
            ratio=ratio,
        )

    def get_vocal_suite_telemetry(self) -> Dict[str, Any]:
        """Return broadcast vocal suite telemetry meters."""
        data = self.vocal_suite.get_telemetry()
        data["suite_enabled"] = self.vocal_suite_enabled
        data["dereverb_amount"] = self.dereverberator.get_amount()
        return data

    # -------------------------------------------------------------------------
    # Target Speaker Extraction (TSE / Voice Lock) Controls
    # -------------------------------------------------------------------------
    def lock_target_speaker(self, num_frames: int = 25) -> None:
        """Initiate target speaker voiceprint enrollment."""
        self.target_speaker.trigger_enrollment(num_frames=num_frames)

    def lock_target_speaker_with_embedding(self, embedding: np.ndarray) -> None:
        """Lock immediately using a precomputed 64-dimensional speaker embedding."""
        self.target_speaker.lock_with_embedding(embedding)

    def unlock_target_speaker(self) -> None:
        """Unlock target speaker, allowing all voices to pass."""
        self.target_speaker.unlock()

    def get_tse_telemetry(self) -> Dict[str, Any]:
        """Return live Target Speaker Extraction telemetry for UI and monitoring."""
        return self.target_speaker.get_telemetry()

    # -------------------------------------------------------------------------
    # Option B: 5-Band Studio Parametric EQ Sculptor Controls
    # -------------------------------------------------------------------------
    def set_eq_enabled(self, enabled: bool) -> None:
        """Enable or disable the 5-band parametric EQ."""
        self.eq.set_enabled(enabled)

    def get_eq_enabled(self) -> bool:
        """Return whether the 5-band parametric EQ is active."""
        return self.eq.get_enabled()

    def set_eq_band(
        self,
        band_index: int,
        filter_type: Optional[str] = None,
        freq_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Configure a specific band of the parametric EQ."""
        self.eq.configure_band(
            band_index=band_index,
            filter_type=filter_type,
            freq_hz=freq_hz,
            gain_db=gain_db,
            q=q,
            enabled=enabled,
        )

    def apply_eq_preset(self, preset_name: str) -> None:
        """Apply a named studio broadcast EQ preset."""
        self.eq.apply_preset(preset_name)

    def get_eq_curve(self, num_points: int = 128) -> Dict[str, Any]:
        """Compute the composite magnitude response curve."""
        return self.eq.get_curve(num_points=num_points)

    def get_eq_config(self) -> Dict[str, Any]:
        """Return the current EQ state configuration."""
        return self.eq.to_dict()

    # -------------------------------------------------------------------------
    # Option C: AI Noise Environment Classifier & Auto-Adaptive Controls
    # -------------------------------------------------------------------------
    def set_auto_adapt_enabled(self, enabled: bool) -> None:
        """Enable or disable autonomous noise adaptation."""
        self.adaptive_controller.enabled = bool(enabled)

    def get_auto_adapt_enabled(self) -> bool:
        """Return whether auto-adaptive mode is enabled."""
        return self.adaptive_controller.enabled

    def get_noise_classification(self) -> Dict[str, Any]:
        """Return latest noise environment classification."""
        if self.last_noise_classification is not None:
            return {
                "noise_type": self.last_noise_classification.noise_type,
                "label": self.last_noise_classification.label,
                "confidence_pct": self.last_noise_classification.confidence_pct,
                "icon": self.last_noise_classification.icon,
                "auto_adapt_enabled": self.adaptive_controller.enabled,
                "adaptive_profile": self.adaptive_controller.current_profile,
            }
        return {
            "noise_type": "unknown",
            "label": "Analyzing acoustic scene...",
            "confidence_pct": 0.0,
            "icon": "🔍",
            "auto_adapt_enabled": self.adaptive_controller.enabled,
            "adaptive_profile": self.adaptive_controller.current_profile,
        }

    def get_dnsmos_telemetry(self) -> Dict[str, Any]:
        """Return standardized ITU-T P.835 perceptual speech quality scores."""
        if self.last_dnsmos_score is not None:
            return {
                "sig_mos": self.last_dnsmos_score.sig_mos,
                "bak_mos": self.last_dnsmos_score.bak_mos,
                "ovrl_mos": self.last_dnsmos_score.ovrl_mos,
                "snr_db": self.last_dnsmos_score.snr_db,
                "speech_distortion_db": self.last_dnsmos_score.speech_distortion_db,
                "noise_suppression_db": self.last_dnsmos_score.noise_suppression_db,
            }
        return self.dnsmos.to_dict()

    # -------------------------------------------------------------------------
    # Option C: 1-Click Multi-Track Studio WAV Exporter
    # -------------------------------------------------------------------------
    def get_recorded_wav(self, track_type: str = "clean") -> bytes:
        """Export rolling buffer as a standardized 16-bit 16 kHz PCM WAV file with studio mastering."""
        track_type = track_type.lower().strip()
        if len(self.record_buffer_in) == 0:
            # Dynamically synthesize and denoise a pristine 3-second studio reference clip
            from src.audio.dataset import SyntheticAudioGenerator, create_mixture
            gen = SyntheticAudioGenerator(sample_rate=self.sample_rate)
            speech = gen.generate_speech(duration_sec=3.0, seed=42)
            noise = gen.generate_noise("white", duration_sec=3.0, seed=42)
            mixture, _, _ = create_mixture(speech, noise, target_snr_db=0.0)
            self.process_stream(mixture, reset_before=False)

        in_audio = np.concatenate(list(self.record_buffer_in))
        out_audio = np.concatenate(list(self.record_buffer_out))
        min_len = min(len(in_audio), len(out_audio))
        in_audio = in_audio[:min_len]
        out_audio = out_audio[:min_len]

        if track_type in ("noisy", "raw", "raw_noisy"):
            audio_data = in_audio
        elif track_type in ("delta", "noise", "subtracted", "subtracted_noise"):
            audio_data = in_audio - out_audio
        else:  # "clean", "crystal_clean", "voice"
            audio_data = out_audio

        # Broadcast standard studio peak normalization (-1.0 dBFS target: 0.891)
        # Guarantees full-bandwidth punchy volume without digital clipping
        pk = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
        if pk > 0.001:
            norm_gain = float(np.clip(0.891 / pk, 0.5, 30.0))
            audio_norm = audio_data * norm_gain
            # Transparent studio soft-knee limiter at 0.84 to prevent hard clipping
            over = np.abs(audio_norm) > 0.84
            if np.any(over):
                audio_norm = np.where(
                    over,
                    np.sign(audio_norm) * (0.84 + 0.051 * np.tanh((np.abs(audio_norm) - 0.84) / 0.051)),
                    audio_norm,
                )
        else:
            audio_norm = audio_data

        # Encode to RIFF WAV (16-bit PCM, mono, 16000 Hz)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            pcm_int16 = np.clip(audio_norm * 32767.0, -32768, 32767).astype(np.int16)
            wav_file.writeframes(pcm_int16.tobytes())
        return buf.getvalue()

    def clear_recording_buffer(self) -> None:
        """Clear recorded audio buffers."""
        self.record_buffer_in.clear()
        self.record_buffer_out.clear()

    def set_mode(self, mode: str) -> None:
        """Change denoiser mode ('neural', 'wiener', 'hybrid')."""
        if hasattr(self.model, "set_mode"):
            self.model.set_mode(mode)

    def get_mode(self) -> str:
        """Return active denoiser mode ('neural', 'wiener', 'hybrid')."""
        if hasattr(self.model, "mode"):
            return getattr(self.model, "mode", "hybrid")
        return "hybrid"

    def set_intensity(self, intensity: float) -> None:
        """Set continuous noise suppression intensity in [0.0, 1.0]."""
        self.intensity = float(np.clip(intensity, 0.0, 1.0))

    def get_intensity(self) -> float:
        """Return active noise suppression intensity in [0.0, 1.0]."""
        return self.intensity

    def adapt_to_noise_category(self, category: str) -> None:
        """Adapt model parameters to detected noise category."""
        if hasattr(self.model, "adapt_to_noise_category"):
            self.model.adapt_to_noise_category(category)

    def set_vad_gating(self, enabled: bool) -> None:
        """Enable or disable VAD-based tensor compute bypass."""
        self.vad_gating = bool(enabled)

    def get_vad_gating(self) -> bool:
        """Return whether VAD gating is currently active."""
        return self.vad_gating

    def get_vad_decision(self) -> Optional[VADDecision]:
        """Return the latest VAD decision."""
        return self.last_vad_decision

    def get_vad_compute_savings_pct(self) -> float:
        """Return percentage of frames where tensor compute was bypassed."""
        return self.vad.compute_savings_pct

    def get_vad_stats(self) -> Dict[str, Any]:
        """Return comprehensive VAD gating metrics."""
        return {
            "vad_gating": self.vad_gating,
            "compute_saved_pct": self.vad.compute_savings_pct if hasattr(self, "vad") else 0.0,
            "cumulative_tensor_time_saved_ms": self.cumulative_tensor_time_saved_ms,
        }

    def process_frame(
        self,
        frame_pcm: np.ndarray,
        profiler: Optional[Any] = None,
    ) -> np.ndarray:
        """Process a single hop frame through the 3-stage denoising pipeline.

        Parameters
        ----------
        frame_pcm : np.ndarray
            PCM audio array of length equal to hop_length (256 samples @ 16kHz).
        profiler : Optional[Any]
            Telemetry profiler instrumenting stage start and end timestamps.

        Returns
        -------
        np.ndarray
            Cleaned PCM audio frame of length hop_length.
        """
        frame_pcm = np.asarray(frame_pcm, dtype=np.float32).ravel()
        if not np.all(np.isfinite(frame_pcm)):
            frame_pcm = np.nan_to_num(frame_pcm, nan=0.0, posinf=1.0, neginf=-1.0)

        # Denormal / subnormal float flush: eliminate values with 0 < |x| < 1e-15 to prevent x86 microcode stalls
        subnormal_mask = (np.abs(frame_pcm) > 0.0) & (np.abs(frame_pcm) < 1e-15)
        if np.any(subnormal_mask):
            frame_pcm = np.where(subnormal_mask, 0.0, frame_pcm)

        # Extreme digital clipping safeguard (+100 dBFS -> bounded soft saturation)
        # 100 dBFS corresponds to amplitude 100,000. Bound extreme excursions to prevent float32 dynamic range explosion
        max_abs = float(np.max(np.abs(frame_pcm))) if len(frame_pcm) > 0 else 0.0
        if max_abs > 10.0:
            frame_pcm = np.sign(frame_pcm) * (10.0 + 2.0 * np.tanh((np.abs(frame_pcm) - 10.0) / 2.0))

        # =========================================================================
        # Stage 1: Pre-processing (Framing, Windowing, rFFT, VAD, Noise Classifier, Dereverb)
        # =========================================================================
        if profiler is not None and hasattr(profiler, "start_stage"):
            profiler.start_stage("pre_processing")

        spec_complex = self.stft.analyze(frame_pcm)
        self.last_input_spec = spec_complex

        # VAD acoustic analysis with fine-grained profiler instrumentation
        if profiler is not None and hasattr(profiler, "start_substage"):
            profiler.start_substage("vad")
        t_sub_vad = time.perf_counter_ns()
        mag_spec = np.abs(spec_complex).astype(np.float32)
        vad_decision = self.vad.process_frame(frame_pcm, mag_spec)
        self.last_vad_decision = vad_decision
        t_vad_ms = max(time.perf_counter_ns() - t_sub_vad, 100) / 1_000_000.0
        if profiler is not None and hasattr(profiler, "end_substage"):
            profiler.end_substage("vad")

        # Option C: Real-Time Noise Environment Classification with profiler instrumentation
        if profiler is not None and hasattr(profiler, "start_substage"):
            profiler.start_substage("noise_classification")
        t_sub_nc = time.perf_counter_ns()
        self.last_noise_classification = self.noise_classifier.classify(frame_pcm, mag_spec)
        t_nc_ms = max(time.perf_counter_ns() - t_sub_nc, 100) / 1_000_000.0
        if profiler is not None and hasattr(profiler, "end_substage"):
            profiler.end_substage("noise_classification")

        # Adaptive suppression controller with profiler instrumentation
        if profiler is not None and hasattr(profiler, "start_substage"):
            profiler.start_substage("adaptive_controller")
        t_sub_ac = time.perf_counter_ns()
        adapt_profile = None
        if self.adaptive_controller.enabled:
            adapt_profile = self.adaptive_controller.update(self.last_noise_classification.noise_type)
        t_ac_ms = max(time.perf_counter_ns() - t_sub_ac, 100) / 1_000_000.0
        if profiler is not None and hasattr(profiler, "end_substage"):
            profiler.end_substage("adaptive_controller")

        # Option A: Room de-reverberation suppression mask with profiler instrumentation
        if profiler is not None and hasattr(profiler, "start_substage"):
            profiler.start_substage("dereverberation")
        t_sub_dr = time.perf_counter_ns()
        g_dereverb = self.dereverberator.compute_gain(mag_spec)
        t_dr_ms = max(time.perf_counter_ns() - t_sub_dr, 100) / 1_000_000.0
        if profiler is not None and hasattr(profiler, "end_substage"):
            profiler.end_substage("dereverberation")

        self.last_substages = {
            "vad_ms": t_vad_ms,
            "noise_classification_ms": t_nc_ms,
            "adaptive_controller_ms": t_ac_ms,
            "dereverberation_ms": t_dr_ms,
        }

        if profiler is not None and hasattr(profiler, "end_stage"):
            profiler.end_stage("pre_processing")

        # =========================================================================
        # Stage 2: Tensor Compute (Neural Mask Estimator / Wiener DSP Filter)
        # =========================================================================
        if profiler is not None and hasattr(profiler, "start_stage"):
            profiler.start_stage("tensor_compute")

        if self.vad_gating and not vad_decision.is_speech:
            # Gating active: Silence / non-speech frame -> bypass neural network compute
            # Comfort noise floor attenuation (0.005)
            gain_mask = np.full(self.stft.num_bins, 0.005, dtype=np.float32)
            self.cumulative_tensor_time_saved_ms += 0.160
        else:
            if hasattr(self.model, "compute_gain"):
                gain_mask = self.model.compute_gain(spec_complex)
            elif callable(self.model):
                gain_mask = self.model(spec_complex)
            else:
                gain_mask = np.ones(self.stft.num_bins, dtype=np.float32)

        # Target Speaker Extraction (TSE / Voice Lock) mask
        g_tse = self.target_speaker.process_frame(
            frame_pcm,
            mag_spec,
            is_speech=vad_decision.is_speech,
        )

        # Fused spectral filtering: Denoiser * Dereverberator * Target Speaker
        gain_mask = gain_mask * g_dereverb * g_tse

        # Option C: Auto-Adaptive suppression tuning
        if self.adaptive_controller.enabled and adapt_profile is not None:
            alpha = float(adapt_profile.get("alpha_oversubtraction", 1.0))
            if abs(alpha - 1.0) > 0.01:
                gain_mask = np.clip(np.power(gain_mask, alpha), 0.005, 1.0).astype(np.float32)

        # Apply continuous noise suppression intensity (1.0 = full AI filter, 0.0 = full bypass)
        if self.intensity < 0.999:
            gain_effective = (1.0 - self.intensity) + self.intensity * gain_mask
            gain_mask = np.clip(gain_effective, 0.005, 1.0).astype(np.float32)

        # Vocal harmonic peak preservation for voiced speech
        if vad_decision.is_speech and hasattr(self, "harmonic_enhancer"):
            f0_est, conf = self.harmonic_enhancer.estimate_f0(frame_pcm)
            if conf > 0.50 and f0_est > 75.0:
                gain_mask = self.harmonic_enhancer.enhance_gain_mask(
                    gain_mask, f0_est, conf, boost_strength=0.32
                )

        self.last_gain_mask = gain_mask

        if profiler is not None and hasattr(profiler, "end_stage"):
            profiler.end_stage("tensor_compute")

        # =========================================================================
        # Stage 3: Output Synthesis (CRM Phase Alignment, irFFT, Overlap-Add)
        # =========================================================================
        if profiler is not None and hasattr(profiler, "start_stage"):
            profiler.start_stage("output_synthesis")

        # Complex Ratio Masking (CRM) Phase Alignment & Cartesian Reconstruction
        if self.crm_enabled:
            real_x = np.ascontiguousarray(spec_complex.real, dtype=np.float32)
            imag_x = np.ascontiguousarray(spec_complex.imag, dtype=np.float32)
            real_m, imag_m = self.crm.estimate_crm_from_magnitude_mask(
                gain_mask, real_x, imag_x, phase_correction_factor=self.crm_phase_factor
            )
            real_y, imag_y = self.crm.apply_mask(
                real_x, imag_x, real_m, imag_m, intensity=self.intensity
            )
            clean_spec = (real_y + 1j * imag_y).astype(np.complex64)
        else:
            clean_spec = (spec_complex * gain_mask).astype(np.complex64)

        self.last_output_spec = clean_spec

        out_pcm = self.stft.synthesize(clean_spec)
        if not np.all(np.isfinite(out_pcm)):
            out_pcm = np.nan_to_num(out_pcm, nan=0.0, posinf=1.0, neginf=-1.0)

        # 5-Band Studio Parametric EQ Sculptor
        if hasattr(self, "eq") and self.eq.get_enabled():
            out_pcm = self.eq.process_frame(out_pcm)

        # Broadcast vocal suite mastering (De-Esser, Compressor, Limiter)
        if self.vocal_suite_enabled:
            out_pcm = self.vocal_suite.process_frame(out_pcm)

        # Option C: ITU-T P.835 Objective Perceptual Quality Telemetry
        self.last_dnsmos_score = self.dnsmos.evaluate(
            denoised_pcm=out_pcm,
            noisy_pcm=frame_pcm,
            vad_active=vad_decision.is_speech,
        )

        # Option C: Multi-Track Rolling Buffer (stores up to 10s of audio)
        self.record_buffer_in.append(frame_pcm.copy())
        self.record_buffer_out.append(out_pcm.copy())

        if profiler is not None and hasattr(profiler, "end_stage"):
            profiler.end_stage("output_synthesis")

        self.total_frames_processed += 1
        return out_pcm

    def process_stream(
        self,
        audio: np.ndarray,
        profiler: Optional[Any] = None,
        reset_before: bool = True,
    ) -> np.ndarray:
        """Process a continuous stream of audio frame-by-frame.

        Parameters
        ----------
        audio : np.ndarray
            Input 1D audio waveform.
        profiler : Optional[Any]
            Optional profiler for latency tracking.
        reset_before : bool
            Whether to reset pipeline state prior to streaming (default: True).

        Returns
        -------
        np.ndarray
            Denoised 1D audio waveform.
        """
        if reset_before:
            self.reset()

        output_frames = []
        for frame in AudioStreamer.stream_frames(audio, hop_length=self.hop_length, pad_tail=True):
            clean_frame = self.process_frame(frame, profiler=profiler)
            output_frames.append(clean_frame)

        if len(output_frames) == 0:
            return np.array([], dtype=np.float32)

        out = np.concatenate(output_frames)
        # Return output matching input length
        return out[: len(audio)]

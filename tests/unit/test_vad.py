"""
Unit tests for Dynamic Voice Activity Detection (VAD) Engine (Phase 3).
"""

import numpy as np
import pytest

from src.audio.vad import VoiceActivityDetector, VADDecision
from src.audio.pipeline import AudioDenoisingPipeline
from src.audio.dataset import SyntheticAudioGenerator


class TestVoiceActivityDetector:
    """Test suite for VoiceActivityDetector standalone module."""

    def test_vad_silence_detection(self):
        """Pure silence frames should be classified as non-speech."""
        vad = VoiceActivityDetector(sample_rate=16000, frame_size=256)
        silence_frame = np.zeros(256, dtype=np.float32)

        for _ in range(10):
            decision = vad.process_frame(silence_frame)

        assert not decision.is_speech
        assert decision.speech_probability < 0.2
        assert vad.compute_savings_pct == 100.0

    def test_vad_speech_detection(self):
        """Active speech signal frames should trigger positive speech detection."""
        vad = VoiceActivityDetector(sample_rate=16000, frame_size=256)
        gen = SyntheticAudioGenerator(sample_rate=16000)
        speech = gen.generate_speech(duration_sec=1.0, seed=42)

        speech_decisions = []
        num_frames = len(speech) // 256
        for i in range(num_frames):
            frame = speech[i * 256 : (i + 1) * 256]
            decision = vad.process_frame(frame)
            speech_decisions.append(decision.is_speech)

        # In natural speech with pauses, at least 60% of frames should be detected as speech
        speech_ratio = np.mean(speech_decisions)
        assert speech_ratio > 0.50, f"Expected speech ratio > 0.50, got {speech_ratio:.2f}"

    def test_vad_hangover_prevents_immediate_truncation(self):
        """VAD hangover should keep is_speech True for hangover_frames after speech stops."""
        vad = VoiceActivityDetector(sample_rate=16000, frame_size=256, hangover_frames=4)

        # High energy frame (burst)
        speech_frame = np.sin(2 * np.pi * 440 * np.arange(256) / 16000).astype(np.float32) * 0.5
        silence_frame = np.zeros(256, dtype=np.float32)

        # Feed speech frame
        d_speech = vad.process_frame(speech_frame)
        assert d_speech.is_speech

        # Feed silence frames during hangover window
        hangover_results = []
        for _ in range(4):
            d = vad.process_frame(silence_frame)
            hangover_results.append((d.is_speech, d.in_hangover))

        # All 4 hangover frames should remain active
        for is_sp, in_ho in hangover_results:
            assert is_sp, "Frame during hangover should be active"
            assert in_ho, "Frame during hangover should flag in_hangover=True"

        # Frame 5 (after hangover expires) should be inactive
        d_post = vad.process_frame(silence_frame)
        assert not d_post.is_speech
        assert not d_post.in_hangover

    def test_pipeline_vad_gating_compute_savings(self):
        """Pipeline with vad_gating=True should bypass tensor compute during silence."""
        pipeline = AudioDenoisingPipeline(vad_gating=True)
        silence_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence = ~62 frames

        denoised = pipeline.process_stream(silence_audio, reset_before=True)
        assert len(denoised) == len(silence_audio)
        savings = pipeline.get_vad_compute_savings_pct()
        assert savings > 80.0, f"Expected compute savings > 80%, got {savings}%"
        assert pipeline.cumulative_tensor_time_saved_ms > 0.0

    def test_pipeline_vad_toggle(self):
        """Pipeline should allow enabling and disabling VAD gating dynamically."""
        pipeline = AudioDenoisingPipeline(vad_gating=False)
        assert not pipeline.get_vad_gating()

        pipeline.set_vad_gating(True)
        assert pipeline.get_vad_gating()

        pipeline.set_vad_gating(False)
        assert not pipeline.get_vad_gating()

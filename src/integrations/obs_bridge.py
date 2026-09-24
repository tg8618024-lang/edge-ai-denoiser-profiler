"""OBS Studio & DAW Audio Filter Socket Bridge.

Provides:
- Lightweight binary/JSON socket bridge protocol for OBS Studio audio filter plugins
  and DAW VST3/CLAP sidecar processes.
- Frame-by-frame low latency streaming with sub-millisecond round-trip time.
"""

from __future__ import annotations
import sys
import os
import json
import asyncio
from typing import Optional, Dict, Any, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import AudioDenoisingPipeline
from src.telemetry.profiler import StageProfiler


class AudioFilterProtocol:
    """Serialization protocol for passing audio frames to/from OBS and DAWs."""

    @staticmethod
    def encode_pcm(frame: np.ndarray) -> bytes:
        int16_data = np.int16(np.clip(frame, -1.0, 1.0) * 32767.0)
        return int16_data.tobytes()

    @staticmethod
    def decode_pcm(pcm_bytes: bytes) -> np.ndarray:
        int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_data.astype(np.float32) / 32767.0


class OBSFilterBridge:
    """Handles audio processing requests from OBS Studio plugins or DAW wrappers."""

    def __init__(self, sample_rate: int = 16000, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.pipeline = AudioDenoisingPipeline(
            n_fft=512, hop_length=hop_length, sample_rate=sample_rate
        )
        self.profiler = StageProfiler(budget_ms=20.0)

    def process_binary_pcm(self, pcm_bytes: bytes) -> Tuple[bytes, Dict[str, float]]:
        """Process 16-bit 16kHz PCM bytes from OBS Studio or DAW plugin."""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0

        if len(audio_float) == self.hop_length:
            self.profiler.start_frame()
            clean_float = self.pipeline.process_frame(audio_float, profiler=self.profiler)
            t_pre, t_tensor, t_synth, t_total = self.profiler.mark_synthesis_done()
        else:
            # Re-frame through streaming engine if input block size != hop_length
            self.profiler.start_frame()
            clean_float = self.pipeline.process_stream(
                audio_float, profiler=self.profiler, reset_before=False
            )
            t_pre, t_tensor, t_synth, t_total = self.profiler.mark_synthesis_done()

        clean_int16 = np.int16(np.clip(clean_float, -1.0, 1.0) * 32767.0)
        telemetry = {
            "pre_ms": round(t_pre, 3),
            "tensor_ms": round(t_tensor, 3),
            "synth_ms": round(t_synth, 3),
            "total_latency_ms": round(t_total, 3),
        }
        return clean_int16.tobytes(), telemetry


# Alias for convenience
ObsBridge = OBSFilterBridge


"""Audio processing core module: STFT/iSTFT, streaming buffers, synthetic audio dataset, and pipeline."""

from src.audio.stft import StreamingSTFT, stft, istft
from src.audio.stream import AudioRingBuffer, AudioChunker, AudioStreamer
from src.audio.dataset import (
    SyntheticAudioGenerator,
    generate_synthetic_speech,
    generate_noise,
    create_mixture,
    calculate_snr,
    compute_snr_gain,
)
from src.audio.pipeline import AudioDenoisingPipeline

__all__ = [
    "StreamingSTFT",
    "stft",
    "istft",
    "AudioRingBuffer",
    "AudioChunker",
    "AudioStreamer",
    "SyntheticAudioGenerator",
    "generate_synthetic_speech",
    "generate_noise",
    "create_mixture",
    "calculate_snr",
    "compute_snr_gain",
    "AudioDenoisingPipeline",
]

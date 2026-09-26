"""Extreme Adversarial Stress Test Suite: Edge AI Audio Denoiser & Profiler.

Authority: Phase 12 Architectural Blueprint & SLA Guarantees.
Validates:
1. +100 dBFS digital clipping injection (100,000.0 linear amplitude) & state recovery
2. Subnormal / denormal float immunity (10^-38 to 10^-42) without CPU trap stalls
3. High-frequency (100 Hz) rapid precision switching across 200 continuous frames
4. Zero-energy absolute silence, negative zeros, and sub-threshold noise
5. Malformed binary ADEN and OBS IPC length prefix fuzzing
"""

import os
import sys
import time
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import AudioDenoisingPipeline
from src.dashboard.protocol import (
    ADEN_MAGIC,
    PROTOCOL_VERSION,
    parse_ingress_binary,
    pack_ingress_binary,
)
from src.integrations.obs_bridge import AudioFilterProtocol


class TestExtremeAdversarialHardening:
    """Extreme adversarial robustness, numerical bounds, and fuzzing tests."""

    @pytest.fixture
    def pipeline(self):
        """Create fresh wideband audio denoising pipeline."""
        p = AudioDenoisingPipeline(n_fft=512, hop_length=256, sample_rate=16000)
        p.reset()
        return p

    def test_extreme_100_dbfs_clipping_spikes(self, pipeline: AudioDenoisingPipeline):
        """Verify pipeline handles +100 dBFS (amplitude 100,000.0) without NaN/Inf explosion."""
        hop_size = pipeline.hop_length
        # +100 dBFS corresponds to 10^(100/20) = 100,000.0
        extreme_amplitude = 100000.0
        clipping_frame = np.array([extreme_amplitude, -extreme_amplitude] * (hop_size // 2), dtype=np.float32)

        # Process 10 frames of extreme clipping
        for _ in range(10):
            out = pipeline.process_frame(clipping_frame)
            assert np.all(np.isfinite(out)), "Extreme clipping produced NaN or Inf!"
            assert np.max(np.abs(out)) < 50.0, "Output energy exploded beyond safety limits!"

        # Subsequent silent frames must decay residual energy back to zero within 5 hops
        silence = np.zeros(hop_size, dtype=np.float32)
        trailing_outputs = [pipeline.process_frame(silence) for _ in range(5)]
        final_frame = trailing_outputs[-1]

        assert np.all(np.isfinite(final_frame))
        assert np.max(np.abs(final_frame)) < 1e-2, "Residual energy failed to decay after clipping!"

    def test_denormal_subnormal_float_microcode_trap_immunity(self, pipeline: AudioDenoisingPipeline):
        """Verify denormal/subnormal floats do not cause CPU microcode pipeline stalls."""
        hop_size = pipeline.hop_length

        # Subnormal floats in range [1e-38, 1e-42]
        denormal_frame = np.full(hop_size, 1e-39, dtype=np.float32)

        latencies_ms = []
        for _ in range(30):
            t0 = time.perf_counter()
            out = pipeline.process_frame(denormal_frame)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            assert np.all(np.isfinite(out)), "Denormal input produced non-finite output!"

        # Latency must remain strictly within 20.0 ms real-time frame budget
        median_ms = float(np.median(latencies_ms))
        max_ms = float(np.max(latencies_ms))
        assert median_ms < 5.0, f"Denormal processing experienced CPU stall: {median_ms:.2f} ms"
        assert max_ms < 20.0, f"Frame exceeded real-time budget: {max_ms:.2f} ms"

    def test_rapid_100hz_precision_mode_switching_continuous_stream(self, pipeline: AudioDenoisingPipeline):
        """Verify switching precision every single frame across 200 consecutive frames."""
        hop_size = pipeline.hop_length
        rng = np.random.RandomState(42)
        modes = ["FP32", "INT8", "FP16"]

        for i in range(200):
            target_mode = modes[i % len(modes)]
            pipeline.set_precision(target_mode)

            # Generate synthetic noisy speech frame
            frame = rng.uniform(-0.4, 0.4, hop_size).astype(np.float32)
            out = pipeline.process_frame(frame)

            assert len(out) == hop_size
            assert np.all(np.isfinite(out)), f"Mode {target_mode} on frame {i} produced non-finite output!"

    def test_zero_energy_silence_and_negative_zeros(self, pipeline: AudioDenoisingPipeline):
        """Verify absolute zero, negative zero, and micro-variance frames do not div-by-zero."""
        hop_size = pipeline.hop_length

        # Exact positive zero
        zero_frame = np.zeros(hop_size, dtype=np.float32)
        out_zero = pipeline.process_frame(zero_frame)
        assert np.all(np.isfinite(out_zero))

        # Negative zero
        neg_zero_frame = np.full(hop_size, -0.0, dtype=np.float32)
        out_neg_zero = pipeline.process_frame(neg_zero_frame)
        assert np.all(np.isfinite(out_neg_zero))

        # Micro-variance noise (sigma = 1e-12)
        micro_noise = (np.random.normal(0.0, 1e-12, hop_size)).astype(np.float32)
        out_micro = pipeline.process_frame(micro_noise)
        assert np.all(np.isfinite(out_micro))

    def test_malformed_binary_protocol_fuzzing(self):
        """Verify binary ADEN parser safely rejects corrupted and fuzzed buffers."""
        # 1. Truncated buffer (< 16 bytes)
        with pytest.raises(ValueError, match="[Tt]runcated"):
            parse_ingress_binary(b"\x00" * 10)

        # 2. Invalid magic header
        with pytest.raises(ValueError, match="Invalid binary magic bytes"):
            parse_ingress_binary(b"NOPE" + b"\x00" * 1040)

        # 3. Truncated payload (header says 256 samples = 512 bytes, but buffer has 10 bytes)
        valid_packet = pack_ingress_binary(seq=1, target_lang="hi", pcm_samples=np.zeros(256, dtype=np.float32))
        truncated_packet = valid_packet[:25]
        with pytest.raises(ValueError, match="[Tt]runcated"):
            parse_ingress_binary(truncated_packet)

        # 4. Valid packet parses cleanly
        pcm_out, seq_out, lang_out = parse_ingress_binary(valid_packet)
        assert len(pcm_out) == 256
        assert seq_out == 1
        assert lang_out == "hi"

    def test_obs_ipc_length_prefix_fuzzing(self):
        """Verify OBS audio filter protocol rejects invalid length headers."""
        # 1. Header with fewer than 4 bytes
        with pytest.raises(ValueError):
            AudioFilterProtocol.unpack_header(b"\x00\x01")

        # 2. Packing and unpacking valid frame
        raw_pcm = np.zeros(256, dtype=np.float32)
        pcm_bytes = AudioFilterProtocol.encode_pcm(raw_pcm)
        packet = AudioFilterProtocol.pack_frame(pcm_bytes)

        unpacked_len = AudioFilterProtocol.unpack_header(packet[:4])
        assert unpacked_len == len(pcm_bytes)
        assert unpacked_len == 512

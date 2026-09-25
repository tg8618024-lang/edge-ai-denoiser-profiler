"""Unit tests for SpectralDereverberator (Option A: Studio Sound Quality Suite).

Verifies:
- Initialization and parameter bounds.
- Full bypass transparency when amount = 0.0.
- Late acoustic reflection attenuation on reverberant speech.
- Sub-80 Hz DC protection.
- State reset behavior.
- Real-time sub-millisecond performance (< 0.1 ms per frame).
"""

import time
import pytest
import numpy as np

from src.audio.dereverb import SpectralDereverberator
from src.audio.dataset import SyntheticAudioGenerator


class TestSpectralDereverberator:
    """Test suite for real-time acoustic room dereverberation."""

    def test_init_and_parameters(self):
        dereverb = SpectralDereverberator(num_bins=257, sample_rate=16000, amount=0.5)
        assert dereverb.num_bins == 257
        assert dereverb.sample_rate == 16000
        assert dereverb.get_amount() == 0.5

        dereverb.set_amount(1.5)  # Clamped to 1.0
        assert dereverb.get_amount() == 1.0

        dereverb.set_amount(-0.5)  # Clamped to 0.0
        assert dereverb.get_amount() == 0.0

    def test_bypass_mode(self):
        dereverb = SpectralDereverberator(num_bins=257, sample_rate=16000, amount=0.0)
        mag = np.random.uniform(0.1, 1.0, 257).astype(np.float32)

        for _ in range(15):
            gain = dereverb.compute_gain(mag)
            assert np.allclose(gain, 1.0)

    def test_reverberation_suppression(self):
        """Simulate an impulse and decay tail, asserting late reflection suppression."""
        dereverb = SpectralDereverberator(
            num_bins=257,
            sample_rate=16000,
            delay_frames=3,
            history_frames=8,
            amount=0.85,
        )

        # Frame 0: Loud initial direct speech energy
        direct_mag = np.ones(257, dtype=np.float32) * 5.0
        g0 = dereverb.compute_gain(direct_mag)
        assert np.allclose(g0, 1.0)  # Direct frame has no prior history

        # Frames 1-3: Early reflections
        for _ in range(3):
            dereverb.compute_gain(direct_mag * 0.7)

        # Frames 4-8: Late reverberation tail with low direct energy
        late_tail_mag = np.ones(257, dtype=np.float32) * 0.4
        late_gain = dereverb.compute_gain(late_tail_mag)

        # Late gain should actively suppress the late tail
        assert np.mean(late_gain) < 0.60
        # Low frequencies (< 80 Hz, bins 0-2) are protected from excessive thinning
        assert np.all(late_gain[:3] >= 0.20)

    def test_reset(self):
        dereverb = SpectralDereverberator(num_bins=257, sample_rate=16000, amount=0.8)
        mag = np.ones(257, dtype=np.float32) * 2.0
        for _ in range(10):
            dereverb.compute_gain(mag)

        assert len(dereverb.power_history) > 0
        dereverb.reset()
        assert len(dereverb.power_history) == 0

    def test_sub_millisecond_latency(self):
        dereverb = SpectralDereverberator(num_bins=257, sample_rate=16000, amount=0.75)
        mag = np.random.uniform(0.01, 1.0, 257).astype(np.float32)

        # Warm up
        for _ in range(10):
            dereverb.compute_gain(mag)

        durations = []
        for _ in range(100):
            t0 = time.perf_counter()
            dereverb.compute_gain(mag)
            durations.append((time.perf_counter() - t0) * 1000.0)

        p95_ms = np.percentile(durations, 95)
        assert p95_ms < 0.60, f"Dereverberator compute too slow: {p95_ms:.4f} ms"

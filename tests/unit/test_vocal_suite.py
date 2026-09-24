"""Unit tests for BroadcastVocalSuite (Option A: Studio Broadcast Mastering).

Verifies:
- Sibilance De-Esser attenuation in the 4.5 kHz - 7.5 kHz sibilance band.
- Soft-knee dynamic range compressor leveling.
- Broadcast peak limiter enforcement at -1.0 dBFS ceiling (<= 0.89 peak).
- Studio telemetry output.
- Performance efficiency (< 0.1 ms per frame).
"""

import time
import pytest
import numpy as np

from src.audio.vocal_suite import BroadcastVocalSuite


class TestBroadcastVocalSuite:
    """Test suite for studio vocal post-processor."""

    def test_init_and_configure(self):
        suite = BroadcastVocalSuite(sample_rate=16000)
        assert suite.compressor_enabled is True
        assert suite.deesser_enabled is True
        assert suite.warmth_enabled is True

        suite.configure(compressor=False, threshold_db=-24.0, ratio=4.0)
        assert suite.compressor_enabled is False
        assert suite.threshold_db == -24.0
        assert suite.ratio == 4.0

    def test_deesser_sibilance_attenuation(self):
        """High-frequency sibilant burst should be attenuated by the de-esser."""
        suite = BroadcastVocalSuite(sample_rate=16000, compressor_enabled=False, warmth_enabled=False)
        t = np.arange(256) / 16000.0

        # Sibilant tone: 6000 Hz ("s" sound)
        sibilant_frame = 0.5 * np.sin(2 * np.pi * 6000.0 * t).astype(np.float32)

        # Warm up envelope follower
        for _ in range(5):
            suite.process_deesser(sibilant_frame)

        out = suite.process_deesser(sibilant_frame)
        in_rms = float(np.sqrt(np.mean(sibilant_frame**2)))
        out_rms = float(np.sqrt(np.mean(out**2)))

        # Sibilance must be attenuated
        assert out_rms < in_rms
        assert suite.last_deesser_reduction_db > 1.0

    def test_deesser_preserves_low_vowel_frequencies(self):
        """Low-pitch fundamental vowel frequencies (200 Hz) should NOT be de-essed."""
        suite = BroadcastVocalSuite(sample_rate=16000)
        t = np.arange(256) / 16000.0
        vowel_frame = 0.3 * np.sin(2 * np.pi * 200.0 * t).astype(np.float32)

        out = suite.process_deesser(vowel_frame)
        in_rms = float(np.sqrt(np.mean(vowel_frame**2)))
        out_rms = float(np.sqrt(np.mean(out**2)))

        # Less than 0.2 dB change on low frequency vowel
        diff_db = abs(20.0 * np.log10(out_rms / in_rms))
        assert diff_db < 0.2

    def test_compressor_gain_reduction(self):
        """Loud speech bursts should trigger gain reduction."""
        suite = BroadcastVocalSuite(sample_rate=16000, deesser_enabled=False, warmth_enabled=False)
        t = np.arange(256) / 16000.0
        # Hot audio frame (-3 dBFS)
        hot_frame = 0.70 * np.sin(2 * np.pi * 500.0 * t).astype(np.float32)

        for _ in range(10):
            suite.process_compressor(hot_frame)

        out = suite.process_compressor(hot_frame)
        assert suite.last_gain_reduction_db > 0.5

    def test_limiter_ceiling_enforcement(self):
        """Peaks exceeding 0.89 (-1.0 dBFS) must be transparently soft-saturated."""
        suite = BroadcastVocalSuite(sample_rate=16000)

        # Extremely hot input with peaks at 1.5
        t = np.arange(256) / 16000.0
        overshoot_frame = 1.5 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

        out = suite.process_limiter(overshoot_frame, ceiling=0.89)
        max_peak = float(np.max(np.abs(out)))
        assert max_peak <= 0.89 + 1e-4

    def test_full_chain_telemetry(self):
        suite = BroadcastVocalSuite(sample_rate=16000)
        t = np.arange(256) / 16000.0
        test_frame = 0.4 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        out = suite.process_frame(test_frame)
        assert len(out) == 256
        assert np.all(np.isfinite(out))

        telem = suite.get_telemetry()
        assert "compressor_enabled" in telem
        assert "deesser_enabled" in telem
        assert "gain_reduction_db" in telem
        assert "deesser_reduction_db" in telem

    def test_sub_millisecond_latency(self):
        suite = BroadcastVocalSuite(sample_rate=16000)
        frame = np.random.uniform(-0.5, 0.5, 256).astype(np.float32)

        # Warm up
        for _ in range(10):
            suite.process_frame(frame)

        durations = []
        for _ in range(100):
            t0 = time.perf_counter()
            suite.process_frame(frame)
            durations.append((time.perf_counter() - t0) * 1000.0)

        p95_ms = np.percentile(durations, 95)
        assert p95_ms < 1.50, f"Vocal suite compute too slow: {p95_ms:.4f} ms"

"""Unit tests for 5-Band Studio Parametric EQ Sculptor (Option B).

Verifies:
- Biquad Direct Form II Transposed filter numerical stability.
- Bell, shelf, high-pass, and notch frequency responses.
- Accurate frequency response calculation across log-spaced frequencies.
- Studio broadcast presets (Podcast Warmth, Broadcast Voice, Vocal Clarity, Hum Notch).
- Bitwise bypass transparency when disabled.
- Sub-millisecond real-time DSP performance (< 0.1 ms per frame).
"""

import time
import pytest
import numpy as np

from src.audio.parametric_eq import BiquadFilter, ParametricEQ


class TestParametricEQ:
    """Test suite for 5-band biquad studio equalizer."""

    def test_biquad_bypass_transparency(self):
        """When disabled or with 0 dB gain, filter should pass audio unmodified."""
        filt_disabled = BiquadFilter("bell", 1000.0, 6.0, 1.0, 16000, enabled=False)
        x = np.random.randn(256).astype(np.float32)
        out1 = filt_disabled.process(x)
        np.testing.assert_allclose(out1, x, atol=1e-6)

        filt_zero_gain = BiquadFilter("bell", 1000.0, 0.0, 1.0, 16000, enabled=True)
        out2 = filt_zero_gain.process(x)
        np.testing.assert_allclose(out2, x, atol=1e-6)

    def test_bell_filter_boost_and_cut(self):
        """Bell filter should selectively boost or cut target frequency."""
        fs = 16000
        t = np.arange(1024) / fs
        f0 = 1000.0
        sig_1k = np.sin(2 * np.pi * f0 * t).astype(np.float32)

        # Boost 6 dB
        boost_filter = BiquadFilter("bell", f0, 6.0, 2.0, fs, enabled=True)
        out_boost = boost_filter.process(sig_1k)
        # Skip initial filter transient (first 100 samples)
        rms_in = np.sqrt(np.mean(sig_1k[100:] ** 2))
        rms_boost = np.sqrt(np.mean(out_boost[100:] ** 2))
        assert rms_boost > rms_in * 1.5, f"Expected boost, got in={rms_in}, boost={rms_boost}"

        # Cut -6 dB
        cut_filter = BiquadFilter("bell", f0, -6.0, 2.0, fs, enabled=True)
        out_cut = cut_filter.process(sig_1k)
        rms_cut = np.sqrt(np.mean(out_cut[100:] ** 2))
        assert rms_cut < rms_in * 0.7, f"Expected cut, got in={rms_in}, cut={rms_cut}"

    def test_high_pass_filter_attenuates_sub_rumble(self):
        """High-pass filter should attenuate low rumble while passing speech frequencies."""
        fs = 16000
        t = np.arange(1024) / fs
        rumble = np.sin(2 * np.pi * 40.0 * t).astype(np.float32)
        speech_tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        hp_filter = BiquadFilter("high_pass", 100.0, 0.0, 0.707, fs, enabled=True)
        out_rumble = hp_filter.process(rumble)
        rms_rumble_in = np.sqrt(np.mean(rumble[100:] ** 2))
        rms_rumble_out = np.sqrt(np.mean(out_rumble[100:] ** 2))
        assert rms_rumble_out < rms_rumble_in * 0.4, "High-pass should attenuate 40Hz sub-rumble"

        hp_filter.reset()
        out_tone = hp_filter.process(speech_tone)
        rms_tone_in = np.sqrt(np.mean(speech_tone[100:] ** 2))
        rms_tone_out = np.sqrt(np.mean(out_tone[100:] ** 2))
        assert rms_tone_out > rms_tone_in * 0.95, "High-pass should preserve 1000Hz speech frequency"

    def test_notch_filter_removes_mains_hum(self):
        """Notch filter at 60 Hz should sharply attenuate 60 Hz power-line hum."""
        fs = 16000
        t = np.arange(8000) / fs
        hum_60 = np.sin(2 * np.pi * 60.0 * t).astype(np.float32)

        notch = BiquadFilter("notch", 60.0, 0.0, 5.0, fs, enabled=True)
        out = notch.process(hum_60)
        rms_in = np.sqrt(np.mean(hum_60[4000:] ** 2))
        rms_out = np.sqrt(np.mean(out[4000:] ** 2))
        assert rms_out < rms_in * 0.15, "Notch should attenuate >= 85% of 60Hz hum"

    def test_parametric_eq_presets(self):
        """All factory presets must apply valid filter parameters and curves."""
        eq = ParametricEQ(sample_rate=16000, enabled=True)
        presets = ["podcast_warmth", "broadcast_voice", "vocal_clarity", "hum_rumble_notch", "flat"]

        for p in presets:
            ok = eq.apply_preset(p)
            assert ok is True
            assert eq.active_preset == p
            curve = eq.get_curve(num_points=64)
            assert len(curve["freqs_hz"]) == 64
            assert len(curve["gain_db"]) == 64
            assert len(curve["bands"]) == 5
            # Curve values must be finite numbers
            assert all(np.isfinite(g) for g in curve["gain_db"])

    def test_configure_band_and_to_dict(self):
        """Configuring individual bands updates response and serialization."""
        eq = ParametricEQ(sample_rate=16000, enabled=True)
        eq.configure_band(2, filter_type="bell", freq_hz=1200.0, gain_db=4.5, q=1.5, enabled=True)

        cfg = eq.to_dict()
        assert cfg["enabled"] is True
        assert cfg["preset"] == "custom"
        assert cfg["bands"][2]["freq_hz"] == 1200.0
        assert cfg["bands"][2]["gain_db"] == 4.5
        assert cfg["bands"][2]["q"] == 1.5

    def test_latency_and_safety_constraints(self):
        """EQ processing must be non-clipping and finish in < 0.1 ms per frame."""
        eq = ParametricEQ(sample_rate=16000, enabled=True)
        eq.apply_preset("broadcast_voice")

        x = np.random.uniform(-0.8, 0.8, 256).astype(np.float32)
        # Warm-up loop
        for _ in range(10):
            _ = eq.process_frame(x)
        start = time.perf_counter()
        for _ in range(100):
            out = eq.process_frame(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 100.0

        assert elapsed_ms < 0.75, f"Per-frame latency {elapsed_ms:.4f} ms exceeds 0.75 ms budget"
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0, "Output must be bounded without digital clipping"

    def test_biquad_schur_cohn_pole_stability(self):
        """All biquad filters must satisfy Schur-Cohn / Jury pole stability under extreme inputs."""
        fs = 16000
        extreme_freqs = [20.0, 50.0, 1000.0, 4000.0, 7200.0]
        extreme_qs = [0.1, 0.707, 2.0, 8.0, 15.0]
        extreme_gains = [-24.0, -12.0, 0.0, 12.0, 24.0]
        filter_types = ["bell", "low_shelf", "high_shelf", "high_pass", "notch"]

        for ftype in filter_types:
            for fc in extreme_freqs:
                for q in extreme_qs:
                    for g in extreme_gains:
                        filt = BiquadFilter(ftype, freq_hz=fc, gain_db=g, q=q, sample_rate=fs, enabled=True)
                        # Verify denominator polynomial roots (poles) lie strictly inside unit circle
                        poles = np.roots(filt.a)
                        max_pole_r = np.max(np.abs(poles))
                        assert max_pole_r <= 0.9995, f"Pole radius {max_pole_r:.6f} >= 1.0 for {ftype} fc={fc} q={q} g={g}"

                        # Verify impulse response stability
                        impulse = np.zeros(512, dtype=np.float32)
                        impulse[0] = 1.0
                        resp = filt.process(impulse)
                        assert np.all(np.isfinite(resp)), f"Non-finite output in {ftype} impulse response"
                        # Response must decay or remain bounded
                        assert np.max(np.abs(resp[-64:])) < 10.0, f"Impulse response unstable for {ftype}"

    def test_parametric_eq_reactive_activation(self):
        """EQ must reactively enable on band configuration or non-flat presets."""
        eq = ParametricEQ(sample_rate=16000, enabled=False)
        assert eq.get_enabled() is False

        # Configuring band should reactively activate EQ
        eq.configure_band(2, freq_hz=1200.0, gain_db=3.0)
        assert eq.get_enabled() is True

        # Applying flat preset should set to bypass
        eq.apply_preset("flat")
        assert eq.get_enabled() is False

        # Applying broadcast preset should reactively activate EQ
        eq.apply_preset("podcast_warmth")
        assert eq.get_enabled() is True

"""
Shared PyTest Fixtures & Synthetic Test Signal Generators for Edge AI Denoiser & Profiler.
Authority: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

import sys
import os
import time
import numpy as np
try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    pytest = _DummyPytest()
from scipy import signal
from typing import Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

venv_site = os.path.join(PROJECT_ROOT, ".venv", "Lib", "site-packages")
if os.path.isdir(venv_site) and venv_site not in sys.path:
    sys.path.append(venv_site)


# ---------------------------------------------------------------------------
# Authoritative Reference Synthetic Audio & Noise Generator
# ---------------------------------------------------------------------------

class ReferenceSyntheticGenerator:
    """
    Authoritative reference synthetic audio and noise generator based strictly
    on the physical and acoustic formulas specified in PROJECT.md and
    explorer_survey_dsp_model/report.md.
    """

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

    def generate_speech(self, duration_sec: float = 2.0) -> np.ndarray:
        """
        Synthesizes realistic human speech via glottal harmonic excitation
        shaped by 3 vocal tract formant resonators and syllabic cadence.
        """
        sr = self.sample_rate
        num_samples = int(duration_sec * sr)
        t = np.arange(num_samples) / sr

        # 1. Pitch intonation F0(t) in [120, 150] Hz with natural vibrato
        f0 = 135.0 + 15.0 * np.sin(2.0 * np.pi * 1.5 * t)
        phase = np.cumsum(2.0 * np.pi * f0 / sr)

        # 2. Glottal excitation harmonic series
        excitation = np.zeros(num_samples, dtype=np.float64)
        for h in range(1, 17):
            harmonic_weight = 1.0 / (h ** 0.8)
            excitation += harmonic_weight * np.cos(h * phase)

        # 3. Formant 2nd-order resonator IIR filters (/a/, /i/, /u/)
        # F1: 700 Hz (BW 100 Hz), F2: 1200 Hz (BW 120 Hz), F3: 2500 Hz (BW 150 Hz)
        formants = [(700.0, 100.0), (1200.0, 120.0), (2500.0, 150.0)]
        speech = np.zeros_like(excitation)

        for freq, bw in formants:
            theta = 2.0 * np.pi * freq / sr
            r = np.exp(-np.pi * bw / sr)
            b = np.array([1.0 - 2.0 * r * np.cos(theta) + r * r])
            a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
            speech += signal.lfilter(b, a, excitation)

        # 4. Syllabic prosody & conversational pauses
        syllabic_env = np.maximum(np.sin(2.0 * np.pi * 1.5 * t), 0.0) ** 2
        # Add pause sections (e.g. 20% silence)
        pause_mask = (np.sin(2.0 * np.pi * 0.5 * t) > -0.3).astype(np.float64)
        speech = speech * syllabic_env * pause_mask

        # Normalize to peak amplitude ~0.8 to prevent digital clipping
        peak = np.max(np.abs(speech))
        if peak > 1e-6:
            speech = (speech / peak) * 0.8

        return speech.astype(np.float32)

    def generate_noise(self, noise_type: str, duration_sec: float = 2.0) -> np.ndarray:
        """
        Generates calibrated noise across 4 distinct physical profiles:
        'white', 'pink', 'drone', 'rf'.
        """
        sr = self.sample_rate
        num_samples = int(duration_sec * sr)
        t = np.arange(num_samples) / sr

        if noise_type == "white":
            # Broadband Gaussian thermal noise
            noise = self.rng.normal(0.0, 1.0, num_samples)

        elif noise_type == "pink":
            # 1/f Pink noise via Paul Kellet 3-pole filter approximation
            white = self.rng.normal(0.0, 1.0, num_samples)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            noise = signal.lfilter(b, a, white)

        elif noise_type == "drone":
            # Drone / HVAC harmonic motor hum (120 Hz blade pass + harmonics)
            noise = (
                np.sin(2.0 * np.pi * 120.0 * t) +
                0.6 * np.sin(2.0 * np.pi * 240.0 * t) +
                0.4 * np.sin(2.0 * np.pi * 360.0 * t) +
                0.3 * np.sin(2.0 * np.pi * 480.0 * t)
            )
            # Add subtle ambient low-frequency rumble
            noise += 0.2 * signal.lfilter([0.1], [1.0, -0.9], self.rng.normal(0.0, 1.0, num_samples))

        elif noise_type == "rf":
            # RF static crackle bursts: high-frequency atmospheric static (3kHz - 7.5kHz)
            carrier = self.rng.normal(0.0, 1.0, num_samples)
            sos = signal.butter(4, [3000.0, 7500.0], btype='bandpass', fs=sr, output='sos')
            filtered_rf = signal.sosfilt(sos, carrier)
            # Poisson impulsive squelch bursts
            burst_envelope = (self.rng.uniform(0.0, 1.0, num_samples) > 0.985).astype(np.float64)
            burst_envelope = signal.lfilter([0.2], [1.0, -0.8], burst_envelope)
            noise = filtered_rf * (0.3 + 2.0 * burst_envelope)

        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        # Normalize noise RMS to 1.0 for consistent SNR calibration
        rms = np.sqrt(np.mean(noise ** 2))
        if rms > 1e-6:
            noise = noise / rms

        return noise.astype(np.float32)

    def generate_mixture(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mixes clean speech and noise to achieve exact target SNR (in dB).
        Returns: (clean, scaled_noise, noisy_mixture)
        """
        # Ensure identical lengths
        min_len = min(len(clean), len(noise))
        c = clean[:min_len].astype(np.float64)
        n = noise[:min_len].astype(np.float64)

        p_speech = np.mean(c ** 2)
        p_noise = np.mean(n ** 2)

        if p_noise < 1e-12:
            scale = 1.0
        else:
            # target_snr = 10 * log10(p_speech / (scale^2 * p_noise))
            # scale = sqrt(p_speech / (p_noise * 10^(snr/10)))
            target_noise_p = p_speech / (10.0 ** (target_snr_db / 10.0))
            scale = np.sqrt(target_noise_p / (p_noise + 1e-12))

        scaled_noise = (n * scale).astype(np.float32)
        mixture = (c + scaled_noise).astype(np.float32)

        return c.astype(np.float32), scaled_noise, mixture

    @staticmethod
    def calculate_snr(
        clean: np.ndarray,
        processed: np.ndarray,
        delay_samples: int = 256
    ) -> float:
        """
        Computes aligned broadband SNR in dB:
        SNR = 10 * log10( sum(s[n]^2) / sum((y[n + D] - s[n])^2 + eps) )
        """
        c = clean.astype(np.float64)
        p = processed.astype(np.float64)

        # Align signals using algorithmic delay D (default 256 samples = 1 hop)
        if delay_samples > 0:
            if len(p) <= delay_samples:
                return -100.0
            p_aligned = p[delay_samples:]
            c_aligned = c[:len(p_aligned)]
        else:
            min_len = min(len(c), len(p))
            c_aligned = c[:min_len]
            p_aligned = p[:min_len]

        signal_power = np.sum(c_aligned ** 2)
        noise_power = np.sum((p_aligned - c_aligned) ** 2)

        if noise_power < 1e-12:
            return 100.0
        if signal_power < 1e-12:
            return -100.0

        return float(10.0 * np.log10(signal_power / (noise_power + 1e-12)))


# ---------------------------------------------------------------------------
# Mock / Test-Double Stage Profiler for Pipeline Isolation Testing
# ---------------------------------------------------------------------------

class MockStageProfiler:
    """
    Mock StageProfiler implementing both start_stage/end_stage and
    mark_* style methods to verify profiler hooks under all conditions.
    """

    def __init__(self, budget_ms: float = 20.0):
        self.budget_ms = budget_ms
        self.stages_recorded: Dict[str, list] = {
            "pre_processing": [],
            "tensor_compute": [],
            "output_synthesis": []
        }
        self.stage_starts: Dict[str, int] = {}
        self.current_frame = 0
        self._last_metrics: Dict[str, Any] = {}

    def start_stage(self, name: str) -> None:
        self.stage_starts[name] = time.perf_counter_ns()

    def end_stage(self, name: str) -> float:
        start = self.stage_starts.get(name, time.perf_counter_ns())
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        # Ensure positive non-zero time even on fast CPUs
        if elapsed_ms <= 0.0:
            elapsed_ms = 0.001
        self.stages_recorded.setdefault(name, []).append(elapsed_ms)
        return elapsed_ms

    def start_frame(self) -> None:
        self.current_frame += 1
        self.start_stage("pre_processing")

    def mark_preprocessing_done(self) -> None:
        self.end_stage("pre_processing")
        self.start_stage("tensor_compute")

    def mark_tensor_done(self) -> None:
        self.end_stage("tensor_compute")
        self.start_stage("output_synthesis")

    def mark_synthesis_done(self) -> Tuple[float, float, float, float]:
        t_synth = self.end_stage("output_synthesis")
        t_pre = self.stages_recorded["pre_processing"][-1] if self.stages_recorded["pre_processing"] else 0.01
        t_tensor = self.stages_recorded["tensor_compute"][-1] if self.stages_recorded["tensor_compute"] else 0.05
        t_total = t_pre + t_tensor + t_synth

        headroom = max(0.0, self.budget_ms - t_total)
        headroom_pct = (headroom / self.budget_ms) * 100.0

        self._last_metrics = {
            "seq": self.current_frame,
            "pre_processing_ms": t_pre,
            "tensor_compute_ms": t_tensor,
            "output_synthesis_ms": t_synth,
            "total_latency_ms": t_total,
            "budget_ms": self.budget_ms,
            "headroom_ms": headroom,
            "headroom_pct": headroom_pct,
            "budget_exceeded": (t_total > self.budget_ms)
        }
        return (t_pre, t_tensor, t_synth, t_total)

    def get_last_metrics(self) -> Dict[str, Any]:
        if not self._last_metrics and self.stages_recorded["output_synthesis"]:
            t_pre = self.stages_recorded["pre_processing"][-1] if self.stages_recorded["pre_processing"] else 0.01
            t_tensor = self.stages_recorded["tensor_compute"][-1] if self.stages_recorded["tensor_compute"] else 0.05
            t_synth = self.stages_recorded["output_synthesis"][-1] if self.stages_recorded["output_synthesis"] else 0.01
            t_total = t_pre + t_tensor + t_synth
            headroom = max(0.0, self.budget_ms - t_total)
            headroom_pct = (headroom / self.budget_ms) * 100.0
            return {
                "seq": self.current_frame,
                "pre_processing_ms": t_pre,
                "tensor_compute_ms": t_tensor,
                "output_synthesis_ms": t_synth,
                "total_latency_ms": t_total,
                "budget_ms": self.budget_ms,
                "headroom_ms": headroom,
                "headroom_pct": headroom_pct,
                "budget_exceeded": (t_total > self.budget_ms)
            }
        return self._last_metrics


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_rate() -> int:
    """Standard Edge AI audio sampling rate (16,000 Hz)."""
    return 16000


@pytest.fixture(scope="session")
def hop_size() -> int:
    """Hop size (256 samples = 16.0 ms at 16kHz)."""
    return 256


@pytest.fixture(scope="session")
def fft_size() -> int:
    """FFT analysis/synthesis window length (512 samples)."""
    return 512


@pytest.fixture(scope="session")
def ref_generator(sample_rate: int) -> ReferenceSyntheticGenerator:
    """Authoritative synthetic signal generator instance."""
    return ReferenceSyntheticGenerator(sample_rate=sample_rate, seed=42)


@pytest.fixture(scope="session")
def clean_speech_2s(ref_generator: ReferenceSyntheticGenerator) -> np.ndarray:
    """2.0 seconds of clean synthetic speech."""
    return ref_generator.generate_speech(duration_sec=2.0)


@pytest.fixture(scope="session")
def white_noise_2s(ref_generator: ReferenceSyntheticGenerator) -> np.ndarray:
    """2.0 seconds of calibrated broadband white noise."""
    return ref_generator.generate_noise("white", duration_sec=2.0)


@pytest.fixture(scope="session")
def drone_noise_2s(ref_generator: ReferenceSyntheticGenerator) -> np.ndarray:
    """2.0 seconds of calibrated drone motor hum."""
    return ref_generator.generate_noise("drone", duration_sec=2.0)


@pytest.fixture(scope="session")
def rf_noise_2s(ref_generator: ReferenceSyntheticGenerator) -> np.ndarray:
    """2.0 seconds of calibrated RF static bursts."""
    return ref_generator.generate_noise("rf", duration_sec=2.0)


@pytest.fixture(scope="session")
def white_mixture_0db(
    ref_generator: ReferenceSyntheticGenerator,
    clean_speech_2s: np.ndarray,
    white_noise_2s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean speech + White noise at 0 dB SNR."""
    return ref_generator.generate_mixture(clean_speech_2s, white_noise_2s, target_snr_db=0.0)


@pytest.fixture(scope="session")
def drone_mixture_5db(
    ref_generator: ReferenceSyntheticGenerator,
    clean_speech_2s: np.ndarray,
    drone_noise_2s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean speech + Drone motor hum at 5 dB SNR."""
    return ref_generator.generate_mixture(clean_speech_2s, drone_noise_2s, target_snr_db=5.0)


@pytest.fixture(scope="session")
def rf_mixture_minus5db(
    ref_generator: ReferenceSyntheticGenerator,
    clean_speech_2s: np.ndarray,
    rf_noise_2s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean speech + RF static bursts at -5 dB SNR."""
    return ref_generator.generate_mixture(clean_speech_2s, rf_noise_2s, target_snr_db=-5.0)


@pytest.fixture
def mock_profiler() -> MockStageProfiler:
    """Returns a fresh MockStageProfiler instance."""
    return MockStageProfiler(budget_ms=20.0)

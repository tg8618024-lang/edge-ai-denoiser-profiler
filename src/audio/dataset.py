"""Synthetic acoustic benchmark generator and broadband SNR evaluation metrics.

Provides:
- Physics-based speech simulation: glottal harmonic excitation (120-220 Hz pitch),
  3 resonant vocal tract formant bandpass filters (500, 1500, 2500 Hz),
  and syllabic amplitude envelope with natural pauses.
- 4 calibrated noise generators:
  1. White Gaussian noise (flat broadband acoustic static)
  2. Pink noise (1/f spectral roll-off via 3-pole IIR filter)
  3. Drone / HVAC motor humming (60Hz, 120Hz, 240Hz, 360Hz harmonics)
  4. RF static crackle bursts (Poisson impulsive burst gating)
- Target SNR mixture synthesis and calibrated broadband SNR gain calculation.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import signal as sp_signal


def generate_synthetic_speech(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    f0_mean: float = 140.0,
    f0_variation: float = 20.0,
    formant_freqs: Tuple[float, float, float] = (500.0, 1500.0, 2500.0),
    formant_bandwidths: Tuple[float, float, float] = (80.0, 100.0, 120.0),
    syllable_rate: float = 2.0,
    pause_interval: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate realistic synthetic speech with glottal harmonics, formants, and prosody.

    Parameters
    ----------
    duration_sec : float
        Total duration in seconds.
    sample_rate : int
        Audio sampling rate in Hz (default: 16000).
    f0_mean : float
        Mean fundamental pitch frequency in Hz (120-220 Hz range).
    f0_variation : float
        Pitch contour modulation amplitude in Hz.
    formant_freqs : Tuple[float, float, float]
        Center frequencies of first 3 formants (F1, F2, F3) in Hz.
    formant_bandwidths : Tuple[float, float, float]
        Bandwidths of first 3 formants in Hz.
    syllable_rate : float
        Cadence of speech syllables in Hz (default: 2.0 Hz).
    pause_interval : Optional[Tuple[float, float]]
        Start and end times of an intentional speech pause (in seconds).
        Defaults to a pause midway through the utterance if duration >= 2.5s.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Float32 normalized speech waveform in range [-0.8, 0.8].
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate

    # 1. Pitch contour with natural vibrato / intonation
    f0 = f0_mean + f0_variation * np.sin(2.0 * np.pi * 1.5 * t)
    phase = np.cumsum(2.0 * np.pi * f0 / sample_rate)

    # 2. Glottal harmonic excitation (harmonics h=1..24 with 1/h^0.8 roll-off)
    excitation = np.zeros_like(t)
    for h in range(1, 24):
        excitation += (1.0 / (h**0.8)) * np.cos(h * phase)

    # 3. Formant bandpass resonator filtering
    speech_filtered = np.zeros_like(t)
    for fk, bk in zip(formant_freqs, formant_bandwidths):
        theta = 2.0 * np.pi * fk / sample_rate
        r = np.exp(-np.pi * bk / sample_rate)
        # 2nd-order digital resonator normalized for unity peak resonance
        b = [1.0 - 2.0 * r * np.cos(theta) + r**2]
        a = [1.0, -2.0 * r * np.cos(theta), r**2]
        speech_filtered += sp_signal.lfilter(b, a, excitation)

    # 4. Syllabic amplitude envelope (raised-cosine cadence simulating syllables)
    envelope = np.maximum(np.sin(2.0 * np.pi * syllable_rate * t), 0.0) ** 2.0

    # 5. Conversational pause
    if pause_interval is None and duration_sec >= 2.5:
        p_start = duration_sec * 0.45
        p_end = duration_sec * 0.65
        pause_interval = (p_start, p_end)

    if pause_interval is not None:
        p_start, p_end = pause_interval
        envelope[(t >= p_start) & (t <= p_end)] = 0.0

    speech = speech_filtered * envelope

    # Normalize speech amplitude to peak 0.75 (avoid clipping)
    peak = np.max(np.abs(speech))
    if peak > 1e-7:
        speech = (speech / peak) * 0.75

    return speech.astype(np.float32)


def generate_noise(
    noise_type: str,
    num_samples: int,
    sample_rate: int = 16000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate calibrated synthetic noise of specified type.

    Supported noise types:
    - 'white': Flat spectrum Gaussian white noise.
    - 'pink': 1/f spectral roll-off (-3 dB/octave) via 3-pole IIR filter.
    - 'drone': Low-frequency HVAC/drone motor harmonic hum (60, 120, 240, 360 Hz).
    - 'rf_static': Bursty RF carrier static with Poisson-distributed impulses.

    Parameters
    ----------
    noise_type : str
        Type of noise ('white', 'pink', 'drone', 'rf_static').
    num_samples : int
        Number of output samples.
    sample_rate : int
        Audio sampling rate in Hz.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Float32 noise waveform with unit standard deviation.
    """
    rng = np.random.default_rng(seed)
    noise_type = noise_type.lower().strip()

    if noise_type in ("white", "gaussian"):
        noise = rng.standard_normal(num_samples)

    elif noise_type == "pink":
        # Paul Kellet 3-pole pinking IIR filter (-3 dB/octave roll-off)
        white = rng.standard_normal(num_samples + 2000)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        pink_full = sp_signal.lfilter(b, a, white)
        noise = pink_full[2000:]

    elif noise_type in ("drone", "hum", "hvac"):
        t = np.arange(num_samples, dtype=np.float64) / sample_rate
        # Fundamental motor blade pass + harmonics
        drone = (
            1.00 * np.sin(2.0 * np.pi * 60.0 * t)
            + 0.80 * np.sin(2.0 * np.pi * 120.0 * t)
            + 0.50 * np.sin(2.0 * np.pi * 240.0 * t)
            + 0.30 * np.sin(2.0 * np.pi * 360.0 * t)
        )
        # Small random motor jitter
        jitter = rng.standard_normal(num_samples) * 0.05
        noise = drone + jitter

    elif noise_type in ("rf_static", "rf", "static"):
        # Poisson-distributed burst gating on high-frequency atmospheric noise
        carrier = rng.standard_normal(num_samples)
        poisson_rate = 0.015
        events = (rng.random(num_samples) < poisson_rate).astype(np.float64)
        burst_envelope = sp_signal.lfilter([1.0], [1.0, -0.995], events)
        burst_envelope = burst_envelope / (np.max(burst_envelope) + 1e-8)
        # Combine steady low background hiss with burst events
        noise = carrier * (0.10 + 0.90 * burst_envelope)

    elif noise_type in ("cafe", "cafe_babble", "babble", "chatter"):
        # Acoustic cafe background: multi-talker speech babble + room murmur
        t = np.arange(num_samples, dtype=np.float64) / sample_rate
        babble = np.zeros(num_samples, dtype=np.float64)
        # Simulate 4 overlapping voices with different pitch and formant bands
        speakers = [
            (115.0, (480.0, 1400.0), 2.2, 0),
            (165.0, (550.0, 1750.0), 2.8, 1),
            (210.0, (620.0, 2100.0), 1.9, 2),
            (255.0, (700.0, 2400.0), 2.5, 3),
        ]
        for f0, (f1, f2), cadence, spk_idx in speakers:
            phase = 2.0 * np.pi * f0 * t + rng.uniform(0, 2.0 * np.pi)
            harmonics = np.sin(phase) + 0.5 * np.sin(2.0 * phase) + 0.25 * np.sin(3.0 * phase)
            # Syllabic envelope modulation
            env = np.maximum(0.0, np.sin(2.0 * np.pi * cadence * t + spk_idx * 1.5)) ** 2
            # 2-pole resonant formant filter for f1
            r = 0.94
            theta = 2.0 * np.pi * f1 / sample_rate
            b = [1.0 - r]
            a = [1.0, -2.0 * r * np.cos(theta), r**2]
            vocal_track = sp_signal.lfilter(b, a, harmonics * env)
            babble += vocal_track

        # Diffuse background room murmur
        murmur = sp_signal.lfilter([0.05], [1.0, -0.95], rng.standard_normal(num_samples))
        noise = babble * 0.75 + murmur * 0.25

    elif noise_type in ("rain", "rainfall", "shower"):
        # Acoustic rainfall: high-frequency pink wash + stochastic droplet impacts
        t = np.arange(num_samples, dtype=np.float64) / sample_rate
        # Broadband continuous rain wash (high-pass filtered pink noise)
        white = rng.standard_normal(num_samples + 1000)
        b_pink = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a_pink = [1.0, -2.494956002, 2.017265875, -0.522189400]
        pink = sp_signal.lfilter(b_pink, a_pink, white)[1000:]
        # Highpass above 800 Hz to simulate water hiss
        b_hp, a_hp = sp_signal.butter(2, 800.0 / (sample_rate / 2), btype="highpass")
        rain_wash = sp_signal.lfilter(b_hp, a_hp, pink)

        # Stochastic droplet impact transients
        droplet_rate = 0.012
        droplet_triggers = (rng.random(num_samples) < droplet_rate).astype(np.float64)
        # Resonant droplet ping filter (damped high-Q resonance at 3200 Hz)
        r_drop = 0.96
        theta_drop = 2.0 * np.pi * 3200.0 / sample_rate
        b_drop = [0.2]
        a_drop = [1.0, -2.0 * r_drop * np.cos(theta_drop), r_drop**2]
        droplet_pings = sp_signal.lfilter(b_drop, a_drop, droplet_triggers * rng.uniform(0.5, 1.5, num_samples))

        noise = rain_wash * 0.65 + droplet_pings * 0.35

    elif noise_type in ("keyboard", "mechanical_keyboard", "typing", "click"):
        # Tactile mechanical keyboard switch typing: sharp transient clicks + chassis thuds
        t = np.arange(num_samples, dtype=np.float64) / sample_rate
        noise = np.zeros(num_samples, dtype=np.float64)
        # Keypress arrival times (~5 keystrokes/sec with natural rhythm)
        key_rate = 0.00035  # Poisson key event probability per sample at 16kHz (~5.6 keys/sec)
        key_events = np.where(rng.random(num_samples) < key_rate)[0]

        # Template for single mechanical switch keystroke
        # 1. High-frequency switch click (4000 Hz damped sine, 2ms)
        # 2. Low-frequency keycap bottom-out thud (220 Hz damped sine, 15ms)
        len_click = int(0.040 * sample_rate)  # 40ms window
        t_click = np.arange(len_click, dtype=np.float64) / sample_rate
        click_hi = np.sin(2.0 * np.pi * 4200.0 * t_click) * np.exp(-t_click / 0.002)
        thud_lo = np.sin(2.0 * np.pi * 240.0 * t_click) * np.exp(-t_click / 0.012)
        key_template = click_hi * 0.70 + thud_lo * 0.50

        for pos in key_events:
            avail = min(len_click, num_samples - pos)
            vol = rng.uniform(0.7, 1.2)
            noise[pos : pos + avail] += key_template[:avail] * vol

        # Subtle ambient office / room floor
        room_floor = rng.standard_normal(num_samples) * 0.04
        noise += room_floor

    elif noise_type in ("air_conditioner", "ac", "ac_hum", "ventilation"):
        # Acoustic HVAC / Air Conditioner: 60Hz compressor fundamental + harmonics + airflow turbulence
        t = np.arange(num_samples, dtype=np.float64) / sample_rate
        hum = (
            1.00 * np.sin(2.0 * np.pi * 60.0 * t)
            + 0.65 * np.sin(2.0 * np.pi * 120.0 * t)
            + 0.35 * np.sin(2.0 * np.pi * 180.0 * t)
            + 0.20 * np.sin(2.0 * np.pi * 240.0 * t)
        )
        # Duct air turbulence: 2-pole lowpass filtered noise around 400 Hz with slow surging modulation
        white = rng.standard_normal(num_samples)
        b_air, a_air = sp_signal.butter(2, 450.0 / (sample_rate / 2), btype="lowpass")
        air_rush = sp_signal.lfilter(b_air, a_air, white)
        # Slow breathing modulation (0.25 Hz airflow surge)
        air_mod = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.25 * t)
        noise = hum * 0.40 + (air_rush * air_mod) * 0.60

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Must be one of: 'white', 'pink', 'drone', 'rf_static', 'cafe', 'rain', 'keyboard', 'air_conditioner'"
        )

    # Standardize to zero mean and unit variance
    std = np.std(noise)
    if std > 1e-9:
        noise = (noise - np.mean(noise)) / std

    return noise.astype(np.float32)


def create_mixture(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix clean speech and noise to achieve exact target broadband SNR.

    Parameters
    ----------
    speech : np.ndarray
        Clean speech waveform.
    noise : np.ndarray
        Noise waveform (must be at least as long as speech).
    target_snr_db : float
        Target input SNR in dB (e.g. 0.0 dB).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mixture, scaled_clean_speech, scaled_noise)
    """
    speech = np.asarray(speech, dtype=np.float32).ravel()
    noise = np.asarray(noise, dtype=np.float32).ravel()

    n = min(len(speech), len(noise))
    s = speech[:n]
    noise_seg = noise[:n]

    p_speech = np.mean(s**2)
    p_noise = np.mean(noise_seg**2)

    if p_speech < 1e-12 or p_noise < 1e-12:
        return s.copy(), s.copy(), noise_seg.copy()

    # Desired noise power: P_noise_target = P_speech / 10^(target_snr / 10)
    desired_p_noise = p_speech / (10.0 ** (target_snr_db / 10.0))
    scale_noise = np.sqrt(desired_p_noise / p_noise)
    scaled_noise = noise_seg * scale_noise

    mixture = s + scaled_noise

    # Ensure mixture doesn't hard clip; scale down proportionally if necessary
    peak = np.max(np.abs(mixture))
    if peak > 0.95:
        norm_factor = 0.90 / peak
        mixture *= norm_factor
        s = s * norm_factor
        scaled_noise *= norm_factor

    return mixture.astype(np.float32), s.astype(np.float32), scaled_noise.astype(np.float32)


def calculate_snr(
    clean: np.ndarray,
    degraded: np.ndarray,
    delay_samples: int = 0,
) -> float:
    """Calculate broadband Signal-to-Noise Ratio (SNR) in decibels.

    Equation:
        SNR = 10 * log10( sum(clean^2) / sum((degraded - clean)^2) )

    Parameters
    ----------
    clean : np.ndarray
        Reference clean audio signal.
    degraded : np.ndarray
        Degraded (noisy or denoised) audio signal.
    delay_samples : int
        Deterministic algorithmic delay to compensate (e.g. 256 samples for 1-hop STFT delay).

    Returns
    -------
    float
        Broadband SNR in dB.
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    degraded = np.asarray(degraded, dtype=np.float64).ravel()

    if delay_samples > 0:
        clean_aligned = clean[:-delay_samples] if len(clean) > delay_samples else clean
        degraded_aligned = degraded[delay_samples : delay_samples + len(clean_aligned)]
    elif delay_samples < 0:
        abs_delay = abs(delay_samples)
        degraded_aligned = degraded[:-abs_delay]
        clean_aligned = clean[abs_delay : abs_delay + len(degraded_aligned)]
    else:
        clean_aligned = clean
        degraded_aligned = degraded

    n = min(len(clean_aligned), len(degraded_aligned))
    if n == 0:
        return 0.0

    s = clean_aligned[:n]
    d = degraded_aligned[:n]

    signal_energy = np.sum(s**2)
    noise_energy = np.sum((d - s) ** 2)

    if noise_energy < 1e-15:
        return 100.0  # Numerical infinite SNR
    if signal_energy < 1e-15:
        return -100.0

    return float(10.0 * np.log10(signal_energy / noise_energy))


def compute_snr_gain(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    delay_samples: int = 256,
) -> Tuple[float, float, float]:
    """Calculate input SNR, output SNR, and SNR improvement gain delta.

    Parameters
    ----------
    clean : np.ndarray
        Clean reference speech.
    noisy : np.ndarray
        Input noisy mixture.
    denoised : np.ndarray
        Processed denoised audio from streaming pipeline.
    delay_samples : int
        Compensated 1-hop algorithmic delay (default: 256 samples = 1 hop @ 16kHz).

    Returns
    -------
    Tuple[float, float, float]
        (snr_in_db, snr_out_db, snr_gain_db)
    """
    snr_in = calculate_snr(clean, noisy, delay_samples=0)
    snr_out = calculate_snr(clean, denoised, delay_samples=delay_samples)
    snr_gain = snr_out - snr_in
    return snr_in, snr_out, snr_gain


class SyntheticAudioGenerator:
    """Comprehensive synthetic testbench audio suite generator."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def generate_speech(self, duration_sec: float = 3.0, seed: Optional[int] = 42) -> np.ndarray:
        return generate_synthetic_speech(
            duration_sec=duration_sec,
            sample_rate=self.sample_rate,
            seed=seed,
        )

    def generate_noise(
        self,
        noise_type: str,
        duration_sec: float = 3.0,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        num_samples = int(duration_sec * self.sample_rate)
        return generate_noise(
            noise_type=noise_type,
            num_samples=num_samples,
            sample_rate=self.sample_rate,
            seed=seed,
        )

    def generate_benchmark_suite(
        self,
        duration_sec: float = 3.0,
        target_snr_db: float = 0.0,
        seed: Optional[int] = 42,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate a complete dictionary of benchmark mixtures for all 4 noise types.

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Map of noise_name -> {'clean': ..., 'noise': ..., 'mixture': ..., 'target_snr_db': ...}
        """
        speech = self.generate_speech(duration_sec=duration_sec, seed=seed)
        noise_types = ["white", "pink", "drone", "rf_static"]
        suite = {}

        for idx, ntype in enumerate(noise_types):
            noise_seed = (seed + idx * 100) if seed is not None else None
            noise = self.generate_noise(ntype, duration_sec=duration_sec, seed=noise_seed)
            mix, s_scaled, n_scaled = create_mixture(speech, noise, target_snr_db=target_snr_db)
            suite[ntype] = {
                "clean": s_scaled,
                "noise": n_scaled,
                "mixture": mix,
                "target_snr_db": target_snr_db,
            }

        return suite

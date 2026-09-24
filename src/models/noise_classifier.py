"""
AI Noise Environment Classifier & Auto-Adaptive Denoiser Engine.

Extracts real-time acoustic features (spectral centroid, rolloff, flux, flatness,
zero-crossing rate, sub-band energies, and periodicity) to classify background
ambient noise into 8 distinct environments:
  1. Mechanical Keyboard (transient impulsive clicks)
  2. Heavy Rainfall (high-frequency broadband)
  3. Cafe Murmur (diffuse babble / vocal formants)
  4. Drone / Engine Hum (strong harmonic resonant tones)
  5. RF Static / Buzz (colored high-frequency hiss)
  6. Air Conditioner / Fan (low-frequency stationary rumble)
  7. Gaussian White Noise (flat power spectral density)
  8. Pink / Acoustic 1/f Noise (-3 dB/octave decay)

Provides an Auto-Adaptive Suppression Controller that autonomously tunes
denoising parameters (spectral over-subtraction, harmonic comb weighting,
and transient attack times) based on the detected acoustic scene.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AcousticFeatures:
    """Acoustic features computed from an audio frame and magnitude spectrum."""
    spectral_centroid_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    spectral_flux: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    low_band_ratio: float = 0.0     # 0 - 500 Hz
    mid_band_ratio: float = 0.0     # 500 - 3000 Hz
    high_band_ratio: float = 0.0    # 3000 - 8000 Hz
    periodicity: float = 0.0        # Autocorrelation peak ratio [0, 1]
    rms_energy_db: float = -100.0


@dataclass
class NoiseClassificationResult:
    """Result of noise signature classification."""
    noise_type: str = "unknown"
    label: str = "Analyzing..."
    confidence_pct: float = 0.0
    icon: str = "🔍"
    features: AcousticFeatures = field(default_factory=AcousticFeatures)
    adaptive_params: Dict[str, float] = field(default_factory=dict)


class AcousticFeatureExtractor:
    """
    Sub-millisecond acoustic feature extractor operating on 16 kHz audio frames
    and 257-bin single-sided magnitude spectra.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_bins = n_fft // 2 + 1
        self.freqs = np.linspace(0, sample_rate / 2, self.num_bins, dtype=np.float32)
        self.prev_norm_spec: Optional[np.ndarray] = None
        self.low_mask = self.freqs <= 500.0
        self.mid_mask = (self.freqs > 500.0) & (self.freqs <= 3000.0)
        self.high_mask = self.freqs > 3000.0

    def extract(self, frame_pcm: np.ndarray, mag_spec: np.ndarray) -> AcousticFeatures:
        """
        Extract acoustic features from PCM frame and magnitude spectrum.
        Execution budget: < 0.03 ms.
        """
        eps = 1e-12
        frame_len = len(frame_pcm)

        # 1. Zero Crossing Rate (ZCR)
        if frame_len > 1:
            signs = np.sign(frame_pcm)
            signs[signs == 0] = 1
            zcr = float(np.sum(np.abs(np.diff(signs))) / (2 * frame_len))
        else:
            zcr = 0.0

        # 2. RMS Energy (dBFS)
        rms = float(np.sqrt(np.mean(frame_pcm ** 2) + eps))
        rms_db = float(20.0 * np.log10(rms + eps))

        # Magnitude spectrum properties
        mag = np.maximum(mag_spec, eps).astype(np.float32)
        total_mag = float(np.sum(mag))

        # 3. Spectral Centroid
        centroid = float(np.sum(self.freqs * mag) / (total_mag + eps))

        # 4. Spectral Rolloff (85% of total spectral energy)
        cum_energy = np.cumsum(mag ** 2)
        rolloff_idx = np.searchsorted(cum_energy, 0.85 * cum_energy[-1])
        rolloff_idx = min(rolloff_idx, self.num_bins - 1)
        rolloff = float(self.freqs[rolloff_idx])

        # 5. Spectral Flux (Euclidean distance between normalized successive spectra)
        norm_spec = mag / (total_mag + eps)
        if self.prev_norm_spec is not None and len(self.prev_norm_spec) == len(norm_spec):
            flux = float(np.sqrt(np.sum((norm_spec - self.prev_norm_spec) ** 2)))
        else:
            flux = 0.0
        self.prev_norm_spec = norm_spec.copy()

        # 6. Spectral Flatness (Wiener Entropy: geometric mean / arithmetic mean)
        # We compute in log-domain to avoid numeric underflow: exp(mean(log(P))) / mean(P)
        pwr = mag ** 2
        log_mean = float(np.mean(np.log(pwr + eps)))
        geom_mean = float(np.exp(log_mean))
        arith_mean = float(np.mean(pwr) + eps)
        flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # 7. Sub-Band Energy Densities (Normalized by band width in Hz)
        # Low: 0 - 500 Hz (500 Hz span)
        # Mid: 500 - 3000 Hz (2500 Hz span)
        # High: 3000 - 8000 Hz (5000 Hz span)
        pwr_total = float(np.sum(pwr) + eps)
        low_ratio = float(np.sum(pwr[self.low_mask]) / pwr_total)
        mid_ratio = float(np.sum(pwr[self.mid_mask]) / pwr_total)
        high_ratio = float(np.sum(pwr[self.high_mask]) / pwr_total)

        # Spectral energy densities (energy per Hz)
        low_density = low_ratio / 500.0
        mid_density = mid_ratio / 2500.0
        high_density = high_ratio / 5000.0
        dens_sum = low_density + mid_density + high_density + eps
        norm_low_dens = low_density / dens_sum
        norm_mid_dens = mid_density / dens_sum
        norm_high_dens = high_density / dens_sum

        # 8. Periodicity via Autocorrelation Peak
        # Fast direct lag search in pitch range [50 Hz, 500 Hz] -> lag [32, 190]
        periodicity = 0.0
        if frame_len >= 128 and rms > 1e-4:
            r0 = float(np.sum(frame_pcm ** 2) + eps)
            max_lag = min(190, frame_len - 16)
            if max_lag > 32:
                auto_corr = np.correlate(frame_pcm, frame_pcm, mode="full")
                center = frame_len - 1
                lags = np.arange(32, max_lag, 6)
                corr_vals = auto_corr[center + lags]
                if len(corr_vals) > 0:
                    periodicity = float(np.clip(float(np.max(corr_vals)) / r0, 0.0, 1.0))

        return AcousticFeatures(
            spectral_centroid_hz=centroid,
            spectral_rolloff_hz=rolloff,
            spectral_flux=flux,
            spectral_flatness=flatness,
            zero_crossing_rate=zcr,
            low_band_ratio=norm_low_dens,
            mid_band_ratio=norm_mid_dens,
            high_band_ratio=norm_high_dens,
            periodicity=periodicity,
            rms_energy_db=rms_db,
        )


class NoiseSignatureClassifier:
    """
    Real-Time Noise Signature Classifier.
    Evaluates acoustic feature vectors against signature fingerprints
    for 8 standard noise environments.
    """

    NOISE_ENVIRONMENTS = {
        "mechanical_keyboard": {
            "label": "Mechanical Keyboard",
            "icon": "⌨️",
            "desc": "Impulsive high-flux transient key clicks",
        },
        "heavy_rainfall": {
            "label": "Heavy Rainfall",
            "icon": "🌧️",
            "desc": "High-frequency broadband precipitation hiss",
        },
        "cafe_murmur": {
            "label": "Cafe Murmur",
            "icon": "☕",
            "desc": "Diffuse vocal formants & restaurant chatter",
        },
        "drone_hum": {
            "label": "Drone / Engine Hum",
            "icon": "✈️",
            "desc": "High-periodicity low/mid resonant harmonics",
        },
        "rf_static": {
            "label": "RF Static / Buzz",
            "icon": "⚡",
            "desc": "Electromagnetic interference & colored static",
        },
        "air_conditioner": {
            "label": "Air Conditioner / Fan",
            "icon": "🌬️",
            "desc": "Low-frequency stationary rumble & airflow",
        },
        "white_noise": {
            "label": "Gaussian White Noise",
            "icon": "⚪",
            "desc": "Equal energy per frequency, flat spectrum",
        },
        "pink_noise": {
            "label": "Pink Acoustic Noise",
            "icon": "🌸",
            "desc": "1/f acoustic decay (-3 dB per octave)",
        },
    }

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512):
        self.extractor = AcousticFeatureExtractor(sample_rate=sample_rate, n_fft=n_fft)
        # Smoothing state
        self.prob_history: Dict[str, float] = {k: 1.0 / 8.0 for k in self.NOISE_ENVIRONMENTS}
        self.smooth_alpha = 0.30  # Exponential moving average factor

    def classify(self, frame_pcm: np.ndarray, mag_spec: np.ndarray) -> NoiseClassificationResult:
        """
        Classify an audio frame into one of the 8 acoustic environments.
        Returns the top predicted class, confidence %, and extracted features.
        """
        feats = self.extractor.extract(frame_pcm, mag_spec)

        # Compute likelihood scores based on physical acoustics
        scores: Dict[str, float] = {}

        # 1. Mechanical Keyboard: High flux, impulsive, low periodicity
        scores["mechanical_keyboard"] = (
            (feats.spectral_flux * 5.0) +
            (1.5 if feats.spectral_centroid_hz > 2000 else 0.0) -
            (feats.periodicity * 3.0)
        )

        # 2. Heavy Rainfall: High high-density, elevated centroid, moderate flatness
        scores["heavy_rainfall"] = (
            (feats.high_band_ratio * 4.0) +
            (2.0 if feats.spectral_centroid_hz > 3000 else 0.0) -
            (feats.low_band_ratio * 3.0)
        )

        # 3. Cafe Murmur: Dominant mid-band (formants), moderate flux, moderate periodicity
        scores["cafe_murmur"] = (
            (feats.mid_band_ratio * 4.0) +
            (feats.periodicity * 2.0) +
            (1.5 if 800 < feats.spectral_centroid_hz < 2200 else 0.0) -
            (feats.spectral_flatness * 2.5)
        )

        # 4. Drone Hum: High periodicity, strong low-density, low flux
        scores["drone_hum"] = (
            (feats.periodicity * 6.0) +
            (feats.low_band_ratio * 3.0) +
            (1.5 if feats.spectral_flux < 0.2 else 0.0) -
            (feats.high_band_ratio * 2.5)
        )

        # 5. RF Static: High ZCR, elevated centroid, buzzing character
        scores["rf_static"] = (
            (feats.zero_crossing_rate * 5.0) +
            (feats.spectral_flatness * 2.0) +
            (1.5 if feats.spectral_centroid_hz > 2500 else 0.0) -
            (feats.low_band_ratio * 2.5)
        )

        # 6. Air Conditioner: Low-density dominant, low centroid (<1200 Hz), very low flux
        scores["air_conditioner"] = (
            (feats.low_band_ratio * 5.0) +
            (2.5 if feats.spectral_centroid_hz < 1500 else 0.0) +
            (1.5 if feats.spectral_flux < 0.15 else 0.0) -
            (feats.high_band_ratio * 3.0)
        )

        # 7. White Noise: High flatness (>0.40), balanced low/mid/high energy density
        # For white noise, all 3 normalized densities are close to 0.33
        density_balance = 1.0 - (
            abs(feats.low_band_ratio - 0.33) +
            abs(feats.mid_band_ratio - 0.33) +
            abs(feats.high_band_ratio - 0.33)
        )
        scores["white_noise"] = (
            (feats.spectral_flatness * 6.0) +
            (density_balance * 4.0) +
            (feats.zero_crossing_rate * 2.0) -
            (feats.periodicity * 3.0)
        )

        # 8. Pink Noise: Decaying spectrum (-3dB/oct), low-to-mid dominance with moderate flatness
        scores["pink_noise"] = (
            (feats.spectral_flatness * 3.0) +
            (feats.low_band_ratio * 2.5) +
            (feats.mid_band_ratio * 2.0) -
            (feats.high_band_ratio * 2.0)
        )

        # Softmax normalization with temperature
        score_arr = np.array([scores[k] for k in self.NOISE_ENVIRONMENTS], dtype=np.float32)
        score_arr = np.nan_to_num(score_arr, nan=0.0)
        temp = 1.2
        exp_scores = np.exp((score_arr - np.max(score_arr)) / temp)
        probs = exp_scores / (np.sum(exp_scores) + 1e-12)

        # Exponential moving average smoothing across frames
        keys = list(self.NOISE_ENVIRONMENTS.keys())
        for idx, k in enumerate(keys):
            self.prob_history[k] = float(
                (1.0 - self.smooth_alpha) * self.prob_history[k] + self.smooth_alpha * probs[idx]
            )

        # Select winner
        best_noise = max(self.prob_history.items(), key=lambda item: item[1])
        noise_type = best_noise[0]
        confidence_pct = float(best_noise[1] * 100.0)

        meta = self.NOISE_ENVIRONMENTS[noise_type]
        return NoiseClassificationResult(
            noise_type=noise_type,
            label=meta["label"],
            confidence_pct=round(confidence_pct, 1),
            icon=meta["icon"],
            features=feats,
        )


class AdaptiveSuppressionController:
    """
    Dynamically tunes denoising and DSP parameters when Auto-Adapt Mode is active.
    Calibrates over-subtraction, harmonic comb weighting, and transient attack speeds
    to the specific acoustic physics of the detected noise environment.
    """

    ADAPTATION_PROFILES = {
        "mechanical_keyboard": {
            "alpha_oversubtraction": 1.15,
            "transient_preserve_gain": 0.85,
            "harmonic_comb_weight": 0.10,
            "high_band_attenuation_db": -18.0,
            "action": "Transient click suppression with speech onset protection",
        },
        "heavy_rainfall": {
            "alpha_oversubtraction": 1.70,
            "transient_preserve_gain": 0.50,
            "harmonic_comb_weight": 0.20,
            "high_band_attenuation_db": -24.0,
            "action": "Broadband high-frequency spectral subtraction boost",
        },
        "cafe_murmur": {
            "alpha_oversubtraction": 1.45,
            "transient_preserve_gain": 0.70,
            "harmonic_comb_weight": 0.35,
            "high_band_attenuation_db": -16.0,
            "action": "Speech formant separation & mid-band spatial notch",
        },
        "drone_hum": {
            "alpha_oversubtraction": 1.85,
            "transient_preserve_gain": 0.40,
            "harmonic_comb_weight": 0.60,
            "high_band_attenuation_db": -12.0,
            "action": "Harmonic resonant comb notch on fundamental tones",
        },
        "rf_static": {
            "alpha_oversubtraction": 1.95,
            "transient_preserve_gain": 0.45,
            "harmonic_comb_weight": 0.15,
            "high_band_attenuation_db": -26.0,
            "action": "Aggressive high-frequency notch & static de-buzzing",
        },
        "air_conditioner": {
            "alpha_oversubtraction": 2.10,
            "transient_preserve_gain": 0.30,
            "harmonic_comb_weight": 0.25,
            "high_band_attenuation_db": -10.0,
            "action": "Deep stationary low-frequency rumble elimination",
        },
        "white_noise": {
            "alpha_oversubtraction": 1.60,
            "transient_preserve_gain": 0.50,
            "harmonic_comb_weight": 0.20,
            "high_band_attenuation_db": -18.0,
            "action": "Flat broadband spectral gate with musical noise suppression",
        },
        "pink_noise": {
            "alpha_oversubtraction": 1.65,
            "transient_preserve_gain": 0.50,
            "harmonic_comb_weight": 0.25,
            "high_band_attenuation_db": -16.0,
            "action": "1/f acoustic slope matched spectral subtraction",
        },
    }

    def __init__(self):
        self.enabled: bool = False
        self.current_profile_name: str = "white_noise"
        self.current_profile: Dict[str, float] = self.ADAPTATION_PROFILES["white_noise"].copy()

    def update(self, noise_type: str) -> Dict[str, float]:
        """Update adaptation parameters based on detected noise type."""
        if noise_type in self.ADAPTATION_PROFILES:
            self.current_profile_name = noise_type
            self.current_profile = self.ADAPTATION_PROFILES[noise_type].copy()
        return self.current_profile

    def get_status(self) -> Dict[str, object]:
        """Return status dictionary for UI and telemetry."""
        return {
            "auto_adapt_enabled": self.enabled,
            "profile_name": self.current_profile_name,
            "profile": self.current_profile,
        }

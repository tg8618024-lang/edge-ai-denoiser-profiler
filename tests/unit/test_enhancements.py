"""
Unit and Integration Tests for Neural DSP & Edge Telemetry Enhancements.
Tests:
- Complex Ratio Masking (CRM) and Phase Recovery (src.models.crm)
- Psychoacoustic ERB Sub-band Auditory Filterbank (src.audio.erb)
- Pitch-Synchronous Harmonic Comb Filter (src.audio.harmonics)
- Real-Time Acoustic Noise Signature Classifier (src.models.classifier)
- ITU-T P.835 Perceptual Voice Quality Evaluator (src.telemetry.dnsmos)
- NVIDIA GPU / Edge Energy & Efficiency Profiler (src.telemetry.energy)
- Virtual Audio Microphone & OBS Studio Bridges (src.integrations)
- Telemetry REST API Endpoints (/api/telemetry/quality, /api/telemetry/energy)
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.crm import ComplexRatioMasker
from src.audio.erb import ERBFilterbank, hz_to_erb, erb_to_hz
from src.audio.harmonics import HarmonicEnhancer
from src.models.classifier import AcousticNoiseClassifier, NoiseClassifier
from src.telemetry.dnsmos import PerceptualQualityEvaluator, VoiceQualityEvaluator
from src.telemetry.energy import HardwareEnergyProfiler, EnergyProfiler
from src.integrations.virtual_mic import VirtualMicBridge
from src.integrations.obs_bridge import OBSFilterBridge, ObsBridge, AudioFilterProtocol


# ===========================================================================
# 1. Complex Ratio Masker (CRM) Tests
# ===========================================================================

def test_complex_ratio_masker():
    """Verify CRM mask estimation, complex multiplication, and bounded output."""
    crm = ComplexRatioMasker(mask_bound=2.0)

    # Create test complex spectrum components
    real_x = np.random.randn(257).astype(np.float32)
    imag_x = np.random.randn(257).astype(np.float32)
    mag_mask = np.full(257, 0.7, dtype=np.float32)

    # Estimate CRM from magnitude mask and phase gradient
    real_m, imag_m = crm.estimate_crm_from_magnitude_mask(mag_mask, real_x, imag_x)
    assert real_m.shape == (257,)
    assert imag_m.shape == (257,)
    assert np.all(np.abs(real_m) <= 2.0)
    assert np.all(np.abs(imag_m) <= 2.0)

    # Apply complex mask
    real_y, imag_y = crm.apply_mask(real_x, imag_x, real_m, imag_m, intensity=1.0)
    assert real_y.shape == (257,)
    assert imag_y.shape == (257,)
    assert real_y.dtype == np.float32
    assert imag_y.dtype == np.float32

    # Compute polar magnitude and phase
    mag_y, phase_y = crm.compute_magnitude_and_phase(real_y, imag_y)
    assert mag_y.shape == (257,)
    assert phase_y.shape == (257,)
    assert np.all(mag_y >= 0.0)
    assert np.all(phase_y >= -np.pi)
    assert np.all(phase_y <= np.pi)


# ===========================================================================
# 2. ERB Filterbank Tests
# ===========================================================================

def test_erb_filterbank():
    """Verify ERB compression from 257 linear bins to 32 ERB bands and reconstruction."""
    erb = ERBFilterbank(num_bands=32, n_fft=512, sample_rate=16000)

    assert erb.filter_matrix.shape == (32, 257)
    assert erb.inverse_matrix.shape == (257, 32)
    assert len(erb.bin_freqs) == 257

    # Test conversion utilities
    assert hz_to_erb(1000.0) > hz_to_erb(100.0)
    assert abs(erb_to_hz(hz_to_erb(1000.0)) - 1000.0) < 1e-3

    # Test forward spectral compression
    linear_mag = np.ones(257, dtype=np.float32)
    erb_energy = erb.compress(linear_mag)
    assert erb_energy.shape == (32,)
    assert np.all(erb_energy > 0)

    # Test inverse expansion back to linear bins
    reconstructed_mag = erb.expand(erb_energy)
    assert reconstructed_mag.shape == (257,)
    assert np.all(reconstructed_mag >= 0)
    assert np.all(reconstructed_mag <= 1.0)


# ===========================================================================
# 3. Pitch-Synchronous Harmonic Comb Filter Tests
# ===========================================================================

def test_harmonic_enhancer():
    """Verify pitch estimation and harmonic enhancement on a periodic voiced signal."""
    enhancer = HarmonicEnhancer(sample_rate=16000)

    # Synthesize 150 Hz harmonic tone
    t = np.arange(256) / 16000.0
    f0_true = 150.0
    voiced = (np.sin(2 * np.pi * f0_true * t) + 0.5 * np.sin(2 * np.pi * 2 * f0_true * t)).astype(np.float32)

    # Estimate pitch and confidence
    f0_est, confidence = enhancer.estimate_f0(voiced)
    assert abs(f0_est - f0_true) < 20.0, f"Expected ~150 Hz, got {f0_est:.1f} Hz"
    assert confidence > 0.5

    # Gain mask enhancement at harmonic multiples
    base_mask = np.full(257, 0.5, dtype=np.float32)
    boosted_mask = enhancer.enhance_gain_mask(base_mask, f0_hz=f0_est, confidence=confidence)
    assert boosted_mask.shape == (257,)
    assert np.all(boosted_mask >= 0.0)
    assert np.all(boosted_mask <= 1.0)
    # Harmonics should have higher gain than base_mask
    assert np.max(boosted_mask) > 0.5


# ===========================================================================
# 4. Real-Time Acoustic Noise Classifier Tests
# ===========================================================================

def test_noise_signature_classifier():
    """Verify classification of acoustic noise signatures into correct categories."""
    classifier = NoiseClassifier(sample_rate=16000)

    # 1. White thermal noise
    white_frame = np.random.randn(256).astype(np.float32)
    white_mag = np.abs(np.fft.rfft(white_frame, n=512))
    sig_white = classifier.classify(white_frame, white_mag)
    assert sig_white.category in ("white", "pink", "rf_static")
    assert sig_white.confidence_pct >= 40.0
    assert len(sig_white.name) > 0
    assert len(sig_white.icon) > 0

    # 2. Drone low-frequency motor hum (120 Hz tone)
    t = np.linspace(0, 256 / 16000, 256, endpoint=False)
    drone_frame = np.sin(2 * np.pi * 120 * t).astype(np.float32)
    drone_mag = np.abs(np.fft.rfft(drone_frame, n=512))
    sig_drone = classifier.classify(drone_frame, drone_mag)
    assert sig_drone.category in ("drone", "air_conditioner")

    # 3. Features dictionary
    features = classifier.extract_features(white_frame, white_mag)
    assert "centroid" in features
    assert "flatness" in features
    assert "rolloff_hz" in features
    assert "zcr" in features


# ===========================================================================
# 5. ITU-T P.835 Perceptual Voice Quality Evaluator Tests
# ===========================================================================

def test_voice_quality_evaluator():
    """Verify ITU-T P.835 DNSMOS score bounds and responsiveness to clean audio."""
    evaluator = VoiceQualityEvaluator(sample_rate=16000)

    clean_frame = np.random.randn(256).astype(np.float32) * 0.1
    noisy_frame = clean_frame + np.random.randn(256).astype(np.float32) * 0.2

    # High SNR gain -> High MOS
    high_qual = evaluator.evaluate_frame(
        clean_frame=clean_frame,
        noisy_frame=noisy_frame,
        snr_delta_db=15.0,
        intensity=1.0,
    )
    assert 1.0 <= high_qual.sig_mos <= 5.0
    assert 1.0 <= high_qual.bak_mos <= 5.0
    assert 1.0 <= high_qual.ovrl_mos <= 5.0
    assert 0.0 <= high_qual.stoi_score <= 1.0
    assert -0.5 <= high_qual.pesq_score <= 4.5

    # Lower SNR gain -> Lower BAK MOS
    low_qual = evaluator.evaluate_frame(
        clean_frame=noisy_frame,
        noisy_frame=noisy_frame,
        snr_delta_db=0.0,
        intensity=0.1,
    )
    assert low_qual.bak_mos < high_qual.bak_mos
    assert low_qual.ovrl_mos < high_qual.ovrl_mos


# ===========================================================================
# 6. NVIDIA GPU / Edge Energy & Efficiency Profiler Tests
# ===========================================================================

def test_energy_profiler():
    """Verify power, energy per frame, and efficiency across precision tiers."""
    profiler = EnergyProfiler(target_platform="jetson_orin")

    # FP32 frame
    fp32_met = profiler.profile_frame(frame_latency_ms=0.30, precision="FP32")
    assert fp32_met.power_watts > 0.0
    assert fp32_met.energy_per_frame_uj > 0.0
    assert fp32_met.efficiency_fps_per_watt > 0.0

    # INT8 frame (lower power and lower energy per frame)
    int8_met = profiler.profile_frame(frame_latency_ms=0.15, precision="INT8")
    assert int8_met.energy_per_frame_uj < fp32_met.energy_per_frame_uj
    assert int8_met.efficiency_fps_per_watt > fp32_met.efficiency_fps_per_watt


# ===========================================================================
# 7. Virtual Mic & OBS Bridge Tests
# ===========================================================================

def test_virtual_mic_bridge():
    """Verify virtual microphone bridge initialization and simulated streaming loop."""
    bridge = VirtualMicBridge(sample_rate=16000, hop_length=256)
    devices = bridge.list_audio_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0

    # Process single frame
    test_frame = (np.sin(np.linspace(0, 10, 256)) * 0.5).astype(np.float32)
    out_frame = bridge.process_frame(test_frame)
    assert len(out_frame) == 256
    assert not np.isnan(out_frame).any()

    # Diagnostic streaming loop
    stats = bridge.run_simulated_stream(duration_sec=0.1)
    assert stats["frames_processed"] > 0
    assert stats["avg_latency_ms"] >= 0.0


def test_obs_bridge_protocol():
    """Verify audio filter protocol serialization between edge model and DAW/OBS."""
    protocol = AudioFilterProtocol()
    frame = np.array([0.1, -0.2, 0.5, -0.8], dtype=np.float32)

    # Encode to 16-bit PCM bytes
    pcm_bytes = protocol.encode_pcm(frame)
    assert len(pcm_bytes) == len(frame) * 2

    # Decode back to float32
    decoded_frame = protocol.decode_pcm(pcm_bytes)
    assert len(decoded_frame) == len(frame)
    assert np.allclose(frame, decoded_frame, atol=1e-3)

    # Test bridge processing
    obs = OBSFilterBridge(sample_rate=16000, hop_length=256)
    dummy_input_pcm = protocol.encode_pcm(np.zeros(256, dtype=np.float32))
    out_pcm, telem = obs.process_binary_pcm(dummy_input_pcm)
    assert len(out_pcm) == len(dummy_input_pcm)
    assert "total_latency_ms" in telem


# ===========================================================================
# 8. Dashboard Telemetry Endpoints Integration Test
# ===========================================================================

def test_dashboard_telemetry_endpoints():
    """Verify /api/telemetry/quality and /api/telemetry/energy REST endpoints."""
    import asyncio
    from src.dashboard.app import get_voice_quality_telemetry, get_energy_telemetry

    qual_resp = asyncio.run(get_voice_quality_telemetry())
    assert "sig_mos" in qual_resp
    assert "bak_mos" in qual_resp
    assert "ovrl_mos" in qual_resp
    assert "stoi" in qual_resp

    energy_resp = asyncio.run(get_energy_telemetry())
    assert "power_watts" in energy_resp
    assert "energy_per_frame_uj" in energy_resp
    assert "efficiency_fps_per_watt" in energy_resp

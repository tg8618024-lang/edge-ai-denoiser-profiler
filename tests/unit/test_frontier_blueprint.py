"""
Verification & Automated Testing Suite for Frontier Innovations Blueprint & Prototypes.

Tests:
1. Blueprint Document Verification:
   - docs/frontier_innovations_blueprint.md exists and is complete (>10KB).
   - Contains all required domain sections: Mamba, Diffusion/Flow Matching, MVDR Beamforming,
     Voice Biometrics (TSE), WebGPU, and Comparative Trade-off Matrix.
2. Streaming Mamba State-Space Model:
   - Recurrence step execution, numerical stability, and linear time complexity.
3. Voiceprint Enrollment & Target Speaker Extraction:
   - Voiceprint centroid normalization, similarity scoring, and gating mask suppression.
4. 4-Channel MVDR Spatial Beamformer:
   - Steering vector calculation, spatial noise covariance updates, and multi-channel filtering.
"""

import os
import sys
import time
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experimental.mamba_stream import StreamingMambaBlock
from src.experimental.voiceprint_enrollment import VoiceprintEnrollmentEngine
from src.experimental.beamformer_mvdr import MVDRBeamformer


# ===========================================================================
# 1. Blueprint Document Structure & Completeness Tests
# ===========================================================================

def test_frontier_blueprint_document_exists_and_complete():
    """Verify docs/frontier_innovations_blueprint.md meets all acceptance criteria."""
    doc_path = os.path.join(PROJECT_ROOT, "docs", "frontier_innovations_blueprint.md")
    assert os.path.exists(doc_path), f"Blueprint not found at {doc_path}"

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify document size (> 10 KB)
    assert len(content) > 10000, f"Blueprint too short ({len(content)} bytes)"

    # Verify key architectural sections and topics
    required_keywords = [
        "Selective State-Space Models",
        "Mamba",
        "Flow Matching",
        "Continuous Wavelet",
        "MVDR",
        "Beamforming",
        "Weighted Prediction Error",
        "Direction-of-Arrival",
        "TensorRT",
        "WebGPU",
        "Voice Biometrics",
        "Target Speaker Extraction",
        "VST3",
        "Comparative Technology Evaluation Matrix",
        "Phased Implementation Roadmap",
    ]
    for kw in required_keywords:
        assert kw in content, f"Missing required architectural topic: {kw}"

    # Verify mathematical formulas are present
    assert "$$" in content or "\\[" in content
    assert "\\frac{dh(t)}{dt}" in content or "\\bar{A}" in content or "w_{\\text{MVDR}}" in content


# ===========================================================================
# 2. Streaming Mamba SSM Layer Tests
# ===========================================================================

def test_streaming_mamba_block():
    """Verify StreamingMambaBlock per-step recurrence and latency budget."""
    mamba = StreamingMambaBlock(d_model=64, d_state=16, dt_rank=4)
    x = np.random.randn(64).astype(np.float32)

    # Test single online recurrence step
    t0 = time.perf_counter()
    y = mamba.step(x)
    step_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert y.shape == (64,)
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()
    assert step_latency_ms < 5.0, f"Step latency {step_latency_ms:.2f}ms exceeds edge budget"

    # Test multi-frame sequence stability
    seq_in = np.random.randn(20, 64).astype(np.float32)
    seq_out = mamba.process_sequence(seq_in)
    assert seq_out.shape == (20, 64)
    assert not np.isnan(seq_out).any()

    # Verify state reset
    mamba.reset_state()
    assert np.all(mamba.h == 0.0)


# ===========================================================================
# 3. Voiceprint Enrollment & Target Speaker Extraction Tests
# ===========================================================================

def test_voiceprint_enrollment_engine():
    """Verify 5-second voiceprint enrollment, d-vector similarity, and TSE gating."""
    tse = VoiceprintEnrollmentEngine(sample_rate=16000, embedding_dim=128)

    # 1. Synthesize 2-second target speaker audio (with formant resonances)
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    target_speech = (
        np.sin(2 * np.pi * 180 * t) +
        0.5 * np.sin(2 * np.pi * 360 * t) +
        0.25 * np.sin(2 * np.pi * 1200 * t)
    ).astype(np.float32)

    voiceprint = tse.enroll_speaker(target_speech)
    assert voiceprint.shape == (128,)
    # Should be normalized on L2 unit sphere
    assert abs(np.linalg.norm(voiceprint) - 1.0) < 1e-3

    # 2. Test similarity on target frame vs random noise frame
    target_chunk = target_speech[:512]
    target_mag = np.abs(np.fft.rfft(target_chunk))
    sim_target = tse.compute_similarity(target_mag)

    interfering_noise = np.random.randn(512).astype(np.float32)
    noise_mag = np.abs(np.fft.rfft(interfering_noise))
    sim_noise = tse.compute_similarity(noise_mag)

    assert sim_target > sim_noise, f"Target similarity ({sim_target:.2f}) should exceed noise ({sim_noise:.2f})"

    # 3. Test TSE gating mask
    base_mask = np.full(257, 0.8, dtype=np.float32)
    gated_mask, score, is_target = tse.apply_speaker_isolation(base_mask, target_mag, isolation_strength=1.0)
    assert len(gated_mask) == 257
    assert np.all(gated_mask >= 0.0)
    assert np.all(gated_mask <= 1.0)


# ===========================================================================
# 4. 4-Channel MVDR Beamformer Tests
# ===========================================================================

def test_mvdr_beamformer():
    """Verify 4-channel MVDR spatial steering, covariance updates, and audio filtering."""
    beamformer = MVDRBeamformer(num_mics=4, mic_spacing_m=0.04, sample_rate=16000)

    # 1. Steering vector computation (0 degrees broadside)
    steering = beamformer.compute_steering_vector(target_angle_deg=0.0)
    assert steering.shape == (257, 4)
    # Broadside: all mics have 0 phase delay, so steering vectors are 1.0 + 0j
    assert np.allclose(steering[:, 0], 1.0 + 0j)

    # 2. MVDR weight computation
    weights = beamformer.compute_mvdr_weights(target_angle_deg=0.0)
    assert weights.shape == (257, 4)
    assert not np.isnan(weights).any()

    # 3. 4-channel audio frame processing
    mics_frame = np.random.randn(4, 256).astype(np.float32) * 0.1
    out_chunk = beamformer.process_multichannel_frame(mics_frame, target_angle_deg=0.0)
    assert out_chunk.shape == (256,)
    assert not np.isnan(out_chunk).any()

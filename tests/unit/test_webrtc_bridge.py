"""Unit tests for WebRTC Audio Bridge, RFC 3550 Jitter Estimator, and PLC.

Authority: Phase 10 Studio/WebRTC Integrations, Task 9 Specification.
"""

import time
import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.integrations.webrtc_bridge import (
    RTPPacket,
    RFC3550JitterEstimator,
    PacketLossConcealer,
    AdaptiveJitterBuffer,
    WebRTCAudioBridge,
    SDPHandler,
    webrtc_bridge,
)
from src.dashboard.app import app
from src.audio.dataset import SyntheticAudioGenerator, create_mixture, compute_snr_gain


def test_rfc3550_jitter_estimation():
    """Verifies that interarrival jitter tracks transit variation per RFC 3550."""
    estimator = RFC3550JitterEstimator(clock_rate=16000)

    # Packet 1 at t=0, ts=0
    j1 = estimator.update(rtp_timestamp=0, arrival_time_s=0.0)
    assert j1 == 0.0

    # Constant transit delay: packet 2 at t=0.016 (256 ticks), ts=256
    # Transit1 = 0 - 0 = 0. Transit2 = 256 - 256 = 0. D = 0.
    j2 = estimator.update(rtp_timestamp=256, arrival_time_s=0.016)
    assert j2 == 0.0

    # Jitter step: Packet 3 arrives late (delay increased by 160 ticks = 10 ms)
    # arrival_ticks = 0.042 * 16000 = 672. timestamp = 512. Transit3 = 160.
    # D = |160 - 0| = 160 ticks.
    # J = 0 + (160 - 0) / 16 = 10 ticks = 10 / 16000 * 1000 = 0.625 ms.
    j3 = estimator.update(rtp_timestamp=512, arrival_time_s=0.042)
    assert math.isclose(j3, 0.625, abs_tol=0.01)

    # Several constant delay packets should decay jitter back toward 0
    t_curr = 0.042
    ts_curr = 512
    for _ in range(50):
        t_curr += 0.016
        ts_curr += 256
        j_decay = estimator.update(rtp_timestamp=ts_curr, arrival_time_s=t_curr)
    assert j_decay < 0.1


def test_packet_loss_concealment_energy_decay():
    """Verifies that PLC synthesizes smooth pitch-decayed replacement waveforms."""
    plc = PacketLossConcealer(sample_rate=16000, history_length=512, decay_factor=0.85)

    # Seed history with a 200 Hz sine wave
    t = np.arange(256) / 16000.0
    seed_pcm = (np.sin(2.0 * np.pi * 200.0 * t) * 0.5).astype(np.float32)
    plc.update_history(seed_pcm)

    # 1st lost frame
    frame1 = plc.conceal(num_samples=256, pitch_lag=80)
    assert len(frame1) == 256
    rms_seed = float(np.sqrt(np.mean(seed_pcm**2)))
    rms_frame1 = float(np.sqrt(np.mean(frame1**2)))
    assert math.isclose(rms_frame1 / rms_seed, 0.85, rel_tol=0.05)

    # 2nd consecutive lost frame
    frame2 = plc.conceal(num_samples=256, pitch_lag=80)
    rms_frame2 = float(np.sqrt(np.mean(frame2**2)))
    assert math.isclose(rms_frame2 / rms_seed, 0.85**2, rel_tol=0.05)
    assert rms_frame2 < rms_frame1

    # Recovery with valid frame resets consecutive loss counter
    plc.update_history(seed_pcm)
    assert plc.consecutive_lost_frames == 0
    frame_rec = plc.conceal(num_samples=256, pitch_lag=80)
    rms_rec = float(np.sqrt(np.mean(frame_rec**2)))
    assert math.isclose(rms_rec / rms_seed, 0.85, rel_tol=0.05)


def test_adaptive_jitter_buffer_reordering_and_wraparound():
    """Verifies that out-of-order packets and 16-bit sequence wraparounds are correctly ordered."""
    jb = AdaptiveJitterBuffer(sample_rate=16000)

    def make_packet(seq: int, val: float) -> RTPPacket:
        pcm = np.full(256, val, dtype=np.float32)
        int16_bytes = np.int16(pcm * 32767.0).tobytes()
        return RTPPacket(
            seq=seq,
            timestamp=seq * 256,
            payload=int16_bytes,
            arrival_time_s=time.perf_counter(),
        )

    # Push out of order: seq 2, 0, 1
    jb.push_packet(make_packet(2, 0.3))
    jb.push_packet(make_packet(0, 0.1))
    jb.push_packet(make_packet(1, 0.2))

    # Should pop in order: 0, 1, 2
    f0, c0 = jb.pop_frame(256)
    f1, c1 = jb.pop_frame(256)
    f2, c2 = jb.pop_frame(256)

    assert not c0 and not c1 and not c2
    assert math.isclose(float(np.mean(f0)), 0.1, abs_tol=0.01)
    assert math.isclose(float(np.mean(f1)), 0.2, abs_tol=0.01)
    assert math.isclose(float(np.mean(f2)), 0.3, abs_tol=0.01)

    # Test 16-bit sequence number wraparound: 65534, 65535, 0, 1
    jb.reset()
    jb.push_packet(make_packet(65535, 0.5))
    jb.push_packet(make_packet(65534, 0.4))
    jb.push_packet(make_packet(1, 0.7))
    jb.push_packet(make_packet(0, 0.6))

    w0, _ = jb.pop_frame(256)
    w1, _ = jb.pop_frame(256)
    w2, _ = jb.pop_frame(256)
    w3, _ = jb.pop_frame(256)

    assert math.isclose(float(np.mean(w0)), 0.4, abs_tol=0.01)
    assert math.isclose(float(np.mean(w1)), 0.5, abs_tol=0.01)
    assert math.isclose(float(np.mean(w2)), 0.6, abs_tol=0.01)
    assert math.isclose(float(np.mean(w3)), 0.7, abs_tol=0.01)


def test_adaptive_jitter_buffer_loss_detection_and_plc():
    """Verifies that sequence gaps automatically trigger packet loss concealment."""
    jb = AdaptiveJitterBuffer(sample_rate=16000)

    def make_packet(seq: int, val: float) -> RTPPacket:
        pcm = np.full(256, val, dtype=np.float32)
        int16_bytes = np.int16(pcm * 32767.0).tobytes()
        return RTPPacket(
            seq=seq,
            timestamp=seq * 256,
            payload=int16_bytes,
            arrival_time_s=time.perf_counter(),
        )

    # Push packet 0 and packet 2 (gap: packet 1 lost)
    jb.push_packet(make_packet(0, 0.5))
    jb.push_packet(make_packet(2, 0.5))

    # Pop 1: Packet 0 (valid)
    f0, c0 = jb.pop_frame(256)
    assert not c0

    # Pop 2: Packet 1 (lost -> PLC triggered)
    f1, c1 = jb.pop_frame(256)
    assert c1 is True

    # Pop 3: Packet 2 (valid)
    f2, c2 = jb.pop_frame(256)
    assert not c2

    stats = jb.get_stats()
    assert stats["total_lost_packets"] == 1
    assert stats["total_plc_frames"] == 1
    assert stats["packet_loss_rate_pct"] > 0.0


def test_webrtc_audio_bridge_end_to_end_denoising():
    """Verifies that WebRTCAudioBridge denoises streaming audio packets and outputs valid egress RTP."""
    bridge = WebRTCAudioBridge(sample_rate=16000, hop_length=256)
    gen = SyntheticAudioGenerator(sample_rate=16000)

    speech = gen.generate_speech(duration_sec=1.0, seed=42)
    noise = gen.generate_noise("white", duration_sec=1.0, seed=42)
    mix, clean_ref, _ = create_mixture(speech, noise, target_snr_db=0.0)

    num_frames = len(mix) // 256
    clean_audio_collected = []

    for i in range(num_frames):
        chunk = mix[i * 256 : (i + 1) * 256]
        int16_bytes = np.int16(np.clip(chunk, -1.0, 1.0) * 32767.0).tobytes()
        pkt = RTPPacket(
            seq=i,
            timestamp=i * 256,
            payload=int16_bytes,
            arrival_time_s=time.perf_counter() + (i * 0.016),
        )
        bridge.ingest_rtp_packet(pkt)
        egress_pkt, telemetry = bridge.process_next_frame()

        assert egress_pkt.seq == i + 1
        assert "stages_ms" in telemetry
        assert telemetry["stages_ms"]["total_ms"] <= 20.0
        assert "network" in telemetry

        clean_pcm = np.frombuffer(egress_pkt.payload, dtype=np.int16).astype(np.float32) / 32767.0
        clean_audio_collected.append(clean_pcm)

    clean_total = np.concatenate(clean_audio_collected)
    _, _, snr_gain = compute_snr_gain(clean_ref, mix, clean_total, delay_samples=256)
    assert snr_gain >= 5.0


def test_sdp_offer_answer_negotiation():
    """Verifies RFC-compliant SDP answer synthesis matching peer offer."""
    sample_offer = (
        "v=0\r\n"
        "o=- 123456789 2 IN IP4 192.168.1.10\r\n"
        "s=Browser Client\r\n"
        "t=0 0\r\n"
        "m=audio 5004 UDP/TLS/RTP/SAVPF 111 97 0\r\n"
        "c=IN IP4 192.168.1.10\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
        "a=rtpmap:97 L16/16000/1\r\n"
        "a=mid:audio\r\n"
    )

    answer = SDPHandler.create_answer(sample_offer)
    assert "v=0" in answer
    assert "m=audio" in answer
    assert "a=sendrecv" in answer
    assert "a=rtpmap:97 L16/16000/1" in answer
    assert "a=rtpmap:111 opus/48000/2" in answer


def test_api_webrtc_endpoints():
    """Verifies FastAPI REST endpoints for WebRTC SDP signaling and packet ingestion."""
    client = TestClient(app)

    # 1. Successful SDP offer
    valid_offer = {
        "type": "offer",
        "sdp": "v=0\r\nm=audio 5004 UDP/TLS/RTP/SAVPF 97\r\na=mid:audio\r\n",
    }
    resp_offer = client.post("/api/webrtc/offer", json=valid_offer)
    assert resp_offer.status_code == 200
    data_offer = resp_offer.json()
    assert data_offer["type"] == "answer"
    assert "m=audio" in data_offer["sdp"]

    # 2. Invalid SDP offer
    invalid_offer = {"type": "offer", "sdp": "v=0\r\ns=NoAudio\r\n"}
    resp_bad = client.post("/api/webrtc/offer", json=invalid_offer)
    assert resp_bad.status_code == 400

    # 3. WebRTC Stats endpoint
    resp_stats = client.get("/api/webrtc/stats")
    assert resp_stats.status_code == 200
    data_stats = resp_stats.json()
    assert "network" in data_stats
    assert "jitter_ms" in data_stats["network"]
    assert "packet_loss_rate_pct" in data_stats["network"]

    # 4. WebRTC Packet ingestion endpoint
    dummy_pcm = (np.sin(2.0 * np.pi * 300.0 * np.arange(256) / 16000.0) * 0.4).tolist()
    packet_req = {
        "seq": 100,
        "timestamp": 25600,
        "pcm_samples": dummy_pcm,
    }
    resp_packet = client.post("/api/webrtc/packet", json=packet_req)
    assert resp_packet.status_code == 200
    data_packet = resp_packet.json()
    assert "egress_packet" in data_packet
    assert "telemetry" in data_packet
    assert len(data_packet["egress_packet"]["pcm_samples"]) == 256


def test_adaptive_jitter_buffer_bounded_capacity_and_clamping():
    """Verifies that the jitter buffer strictly caps queue depth at max_delay_ms without memory growth."""
    jb = AdaptiveJitterBuffer(sample_rate=16000, min_delay_ms=16.0, max_delay_ms=100.0)

    def make_pkt(seq: int) -> RTPPacket:
        pcm = np.full(256, 0.1, dtype=np.float32)
        return RTPPacket(
            seq=seq,
            timestamp=seq * 256,
            payload=np.int16(pcm * 32767.0).tobytes(),
            arrival_time_s=seq * 0.016,
        )

    # Push 30 consecutive packets
    for i in range(30):
        jb.push_packet(make_pkt(i))

    stats = jb.get_stats()
    # Queue length must be bounded to <= 9 packets (~144ms max), not 30
    assert stats["buffered_packets"] <= 9
    assert stats["total_lost_packets"] > 0  # Dropped stale/overflow packets

    # Test duplicate rejection
    buffered_before = stats["buffered_packets"]
    jb.push_packet(make_pkt(29))  # Already in queue
    assert jb.get_stats()["buffered_packets"] == buffered_before

    # Verify target delay clamping under extreme jitter
    jb.jitter_estimator.jitter_ticks = 16000.0 * 2.0  # 2000 ms jitter
    assert jb.get_target_delay_ms() == 100.0


def test_adaptive_jitter_buffer_large_burst_gap_resync():
    """Verifies that massive packet gaps trigger immediate re-sync instead of hundreds of PLC stalls."""
    jb = AdaptiveJitterBuffer(sample_rate=16000)

    def make_pkt(seq: int) -> RTPPacket:
        pcm = np.full(256, 0.2, dtype=np.float32)
        return RTPPacket(
            seq=seq,
            timestamp=seq * 256,
            payload=np.int16(pcm * 32767.0).tobytes(),
            arrival_time_s=time.perf_counter(),
        )

    # Start with packet 0
    jb.push_packet(make_pkt(0))
    f0, c0 = jb.pop_frame(256)
    assert not c0

    # Massive gap: packet 50 arrives (packets 1..49 dropped)
    jb.push_packet(make_pkt(50))
    f_resync, c_resync = jb.pop_frame(256)

    # Playout must re-sync and pop packet 50 immediately without 49 stalls
    assert not c_resync
    assert math.isclose(float(np.mean(f_resync)), 0.2, abs_tol=0.01)


def test_packet_loss_concealer_edge_cases():
    """Verifies that empty history updates and zero-sample conceal requests do not throw slicing errors."""
    plc = PacketLossConcealer(sample_rate=16000)

    # Empty history update should not throw ValueError
    plc.update_history(np.array([], dtype=np.float32))

    # Conceal 0 samples should return empty array
    zero_frame = plc.conceal(0)
    assert len(zero_frame) == 0


def test_sdp_offer_preserves_mid():
    """Verifies that SDP answer preserves the offer's media identifier (RFC 8829 Section 5.3)."""
    offer = (
        "v=0\r\n"
        "o=- 999 1 IN IP4 10.0.0.1\r\n"
        "s=Test\r\n"
        "m=audio 5004 UDP/TLS/RTP/SAVPF 97\r\n"
        "a=mid:track-vocal\r\n"
    )
    answer = SDPHandler.create_answer(offer)
    assert "a=mid:track-vocal" in answer
    assert "a=group:BUNDLE track-vocal" in answer


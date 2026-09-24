"""Unit & Integration Tests for High-Throughput Binary WebSocket Protocol (ADEN).

Authority: Task 2 Architecture Specification (Phase 6 Real-Time Ingress/Egress).
Tests:
- Bit-exactness of ingress audio framing (1,040 bytes).
- Bit-exactness of egress audio and visualizer framing (3,872 bytes).
- Error handling on corrupt, truncated, or invalid magic headers.
- End-to-end WebSocket binary streaming via FastAPI TestClient.
- Dual multiplexing: binary audio streaming alongside JSON control messages.
"""

import struct
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import app
from src.dashboard.protocol import (
    ADEN_MAGIC,
    PROTOCOL_VERSION,
    MSG_INGRESS_AUDIO,
    MSG_EGRESS_AUDIO,
    INGRESS_TOTAL_SIZE,
    EGRESS_TOTAL_SIZE,
    parse_ingress_binary,
    pack_egress_binary,
)


class TestBinaryProtocolUnit:
    """Unit tests for ADEN packet packing, parsing, and data integrity."""

    def test_ingress_frame_roundtrip(self):
        """Verify binary ingress framing packs and parses bit-exactly."""
        # 256 samples of 440 Hz test tone
        t = np.arange(256) / 16000.0
        pcm_in = (0.75 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

        # Manually pack ingress packet
        lang_code = "ta"
        lang_bytes = lang_code.encode("ascii").ljust(4, b"\x00")
        lang_int = struct.unpack("<I", lang_bytes)[0]
        seq_num = 1337

        header = struct.pack(
            "<4sBBHII",
            ADEN_MAGIC,
            PROTOCOL_VERSION,
            MSG_INGRESS_AUDIO,
            256,
            seq_num,
            lang_int,
        )
        packet = header + pcm_in.tobytes()
        assert len(packet) == INGRESS_TOTAL_SIZE == 1040

        # Parse packet
        pcm_out, seq_out, lang_out = parse_ingress_binary(packet)

        assert seq_out == seq_num
        assert lang_out == lang_code
        assert len(pcm_out) == 256
        assert np.allclose(pcm_out, pcm_in, atol=1e-7)

    def test_egress_frame_structure(self):
        """Verify egress binary frame structure, header offsets, and channel payloads."""
        seq = 101
        t_total = 1.250
        snr_delta = 14.85
        noise_erased = 96.5
        purity = 98.2
        ovrl_mos = 4.35

        denoised = np.full(256, 0.45, dtype=np.float32)
        noisy = np.full(256, 0.85, dtype=np.float32)
        diff = noisy - denoised

        in_vis = [0.1 * i for i in range(64)]
        out_vis = [0.05 * i for i in range(64)]
        mask_vis = [0.9] * 64

        packed = pack_egress_binary(
            seq=seq,
            total_latency_ms=t_total,
            snr_delta_db=snr_delta,
            noise_erased_pct=noise_erased,
            purity_score=purity,
            ovrl_mos=ovrl_mos,
            denoised_pcm=denoised,
            raw_noisy_pcm=noisy,
            diff_pcm=diff,
            in_vis=in_vis,
            out_vis=out_vis,
            mask_vis=mask_vis,
        )

        assert len(packed) == EGRESS_TOTAL_SIZE == 3872

        # Unpack header
        magic, ver, mtype, samples, u_seq, u_lat, u_snr, u_erased, u_purity, u_mos = struct.unpack_from(
            "<4sBBHIfffff", packed, 0
        )
        assert magic == ADEN_MAGIC
        assert ver == PROTOCOL_VERSION
        assert mtype == MSG_EGRESS_AUDIO
        assert samples == 256
        assert u_seq == seq
        assert abs(u_lat - t_total) < 1e-4
        assert abs(u_snr - snr_delta) < 1e-4
        assert abs(u_erased - noise_erased) < 1e-4
        assert abs(u_purity - purity) < 1e-4
        assert abs(u_mos - ovrl_mos) < 1e-4

        # Verify audio channels offset unpacking
        u_denoised = np.frombuffer(packed, dtype=np.float32, count=256, offset=32)
        u_noisy = np.frombuffer(packed, dtype=np.float32, count=256, offset=1056)
        u_diff = np.frombuffer(packed, dtype=np.float32, count=256, offset=2080)
        u_in_vis = np.frombuffer(packed, dtype=np.float32, count=64, offset=3104)
        u_out_vis = np.frombuffer(packed, dtype=np.float32, count=64, offset=3360)
        u_mask_vis = np.frombuffer(packed, dtype=np.float32, count=64, offset=3616)

        assert np.allclose(u_denoised, denoised)
        assert np.allclose(u_noisy, noisy)
        assert np.allclose(u_diff, diff)
        assert np.allclose(u_in_vis, in_vis)
        assert np.allclose(u_out_vis, out_vis)
        assert np.allclose(u_mask_vis, mask_vis)

    def test_corrupt_and_truncated_packets(self):
        """Verify robust rejection of malformed or invalid packets."""
        # Truncated packet
        with pytest.raises(ValueError, match="truncated"):
            parse_ingress_binary(b"ADEN\x01\x01\x00")

        # Wrong magic bytes
        bad_magic = b"NOPE" + b"\x00" * 1036
        with pytest.raises(ValueError, match="Invalid binary magic"):
            parse_ingress_binary(bad_magic)

        # Unsupported version
        bad_ver = struct.pack("<4sBBHII", ADEN_MAGIC, 99, MSG_INGRESS_AUDIO, 256, 0, 0) + b"\x00" * 1024
        with pytest.raises(ValueError, match="Unsupported protocol version"):
            parse_ingress_binary(bad_ver)


class TestBinaryWebSocketIntegration:
    """Integration tests for live WebSocket binary streaming."""

    def test_live_binary_audio_streaming(self):
        """Verify streaming binary audio frames through /ws/stream."""
        test_client = TestClient(app)

        t = np.arange(256) / 16000.0
        pcm_in = (0.5 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

        with test_client.websocket_connect("/ws/stream") as ws:
            # 1. Verify JSON control message still works
            ws.send_text(json.dumps({"type": "set_precision", "precision": "FP32"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "precision_updated"

            # 2. Stream binary audio frames
            for seq in range(1, 10):
                lang_int = struct.unpack("<I", b"hi\x00\x00")[0]
                header = struct.pack(
                    "<4sBBHII",
                    ADEN_MAGIC,
                    PROTOCOL_VERSION,
                    MSG_INGRESS_AUDIO,
                    256,
                    seq,
                    lang_int,
                )
                bin_packet = header + pcm_in.tobytes()

                ws.send_bytes(bin_packet)
                resp_bytes = ws.receive_bytes()

                assert len(resp_bytes) == EGRESS_TOTAL_SIZE == 3872
                assert resp_bytes[:4] == ADEN_MAGIC

                # Unpack sequence and latency
                _, _, _, _, ret_seq, lat, snr, _, _, _ = struct.unpack_from("<4sBBHIfffff", resp_bytes, 0)
                assert ret_seq == seq
                assert lat > 0.0

                # Unpack denoised PCM
                denoised = np.frombuffer(resp_bytes, dtype=np.float32, count=256, offset=32)
                assert len(denoised) == 256
                assert not np.isnan(denoised).any()
                assert not np.isinf(denoised).any()

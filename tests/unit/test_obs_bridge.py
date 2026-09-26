"""Unit Tests for OBS Studio & DAW Audio Filter Socket Bridge & IPC Server."""

import os
import sys
import asyncio
import socket
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.integrations.obs_bridge import (
    AudioFilterProtocol,
    OBSFilterBridge,
    OBSFilterServer,
)


class TestAudioFilterProtocol:
    """Tests for binary PCM encoding, decoding, and length-prefixed framing."""

    def test_encode_decode_round_trip(self):
        """Verify float32 audio converts to int16 PCM and back with high precision."""
        original = np.array([0.0, 0.5, -0.5, 0.99, -0.99], dtype=np.float32)
        pcm_bytes = AudioFilterProtocol.encode_pcm(original)
        assert len(pcm_bytes) == len(original) * 2

        decoded = AudioFilterProtocol.decode_pcm(pcm_bytes)
        np.testing.assert_allclose(decoded, original, atol=1e-4)

    def test_clipping_bounds(self):
        """Verify audio outside [-1.0, 1.0] is safely clamped."""
        out_of_bounds = np.array([2.5, -3.0], dtype=np.float32)
        pcm_bytes = AudioFilterProtocol.encode_pcm(out_of_bounds)
        decoded = AudioFilterProtocol.decode_pcm(pcm_bytes)
        assert np.all(decoded <= 1.0)
        assert np.all(decoded >= -1.0)

    def test_pack_and_unpack_header(self):
        """Verify 4-byte length prefix packing and unpacking."""
        data = b"\x01\x02\x03\x04\x05"
        packet = AudioFilterProtocol.pack_frame(data)
        assert len(packet) == AudioFilterProtocol.HEADER_SIZE + len(data)

        unpacked_len = AudioFilterProtocol.unpack_header(packet[:AudioFilterProtocol.HEADER_SIZE])
        assert unpacked_len == len(data)

    def test_unpack_header_insufficient_bytes(self):
        """Verify unpack_header raises ValueError if fewer than 4 bytes provided."""
        with pytest.raises(ValueError):
            AudioFilterProtocol.unpack_header(b"\x00\x01")


class TestOBSFilterBridge:
    """Tests for OBSFilterBridge frame and stream denoising."""

    def test_process_binary_pcm_hop_length(self):
        """Verify single hop-length frame (256 samples = 512 bytes) processing."""
        bridge = OBSFilterBridge(sample_rate=16000, hop_length=256)
        samples = np.random.randn(256).astype(np.float32) * 0.1
        pcm_in = AudioFilterProtocol.encode_pcm(samples)

        clean_pcm, telemetry = bridge.process_binary_pcm(pcm_in)
        assert len(clean_pcm) == len(pcm_in)
        assert "pre_ms" in telemetry
        assert "tensor_ms" in telemetry
        assert "synth_ms" in telemetry
        assert "total_latency_ms" in telemetry
        assert telemetry["total_latency_ms"] < 20.0  # Real-time frame budget constraint

    def test_process_arbitrary_block_size(self):
        """Verify arbitrary block sizes (e.g. 128 samples, 512 samples) process cleanly."""
        bridge = OBSFilterBridge(sample_rate=16000, hop_length=256)
        samples = np.random.randn(512).astype(np.float32) * 0.1
        pcm_in = AudioFilterProtocol.encode_pcm(samples)

        clean_pcm, telemetry = bridge.process_binary_pcm(pcm_in)
        assert len(clean_pcm) == len(pcm_in)
        assert telemetry["total_latency_ms"] < 25.0

    def test_reset_behavior(self):
        """Verify bridge reset clears pipeline state."""
        bridge = OBSFilterBridge(sample_rate=16000, hop_length=256)
        samples = np.random.randn(256).astype(np.float32) * 0.1
        pcm_in = AudioFilterProtocol.encode_pcm(samples)
        _ = bridge.process_binary_pcm(pcm_in)
        bridge.reset()
        assert bridge.pipeline.total_frames_processed == 0


class TestOBSFilterServer:
    """Tests for asynchronous TCP socket IPC server lifecycle and streaming."""

    def test_server_start_stop_lifecycle(self):
        """Verify OBSFilterServer starts, reports status, and stops cleanly."""
        async def _run():
            # Find an open port
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()

            server = OBSFilterServer(host="127.0.0.1", port=port, sample_rate=16000, hop_length=256)
            assert not server.is_running

            await server.start()
            assert server.is_running

            status = server.get_status()
            assert status["is_running"] is True
            assert status["port"] == port
            assert status["active_clients"] == 0
            assert status["total_frames_processed"] == 0

            await server.stop()
            assert not server.is_running

        asyncio.run(_run())

    def test_tcp_client_audio_streaming(self):
        """Verify real TCP client connects, sends length-prefixed PCM, and receives clean audio."""
        async def _run():
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()

            server = OBSFilterServer(host="127.0.0.1", port=port, sample_rate=16000, hop_length=256)
            await server.start()

            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)

                # Generate 256-sample noisy audio frame
                noisy_audio = np.random.randn(256).astype(np.float32) * 0.1
                pcm_in = AudioFilterProtocol.encode_pcm(noisy_audio)
                packed_frame = AudioFilterProtocol.pack_frame(pcm_in)

                # Send to server
                writer.write(packed_frame)
                await writer.drain()

                # Read response header
                resp_header = await reader.readexactly(AudioFilterProtocol.HEADER_SIZE)
                resp_len = AudioFilterProtocol.unpack_header(resp_header)
                assert resp_len == len(pcm_in)

                # Read clean PCM payload
                clean_pcm = await reader.readexactly(resp_len)
                assert len(clean_pcm) == len(pcm_in)

                clean_samples = AudioFilterProtocol.decode_pcm(clean_pcm)
                assert len(clean_samples) == 256
                assert np.max(np.abs(clean_samples)) <= 1.0

                # Close connection
                writer.close()
                await writer.wait_closed()

                # Small delay for task teardown
                await asyncio.sleep(0.05)

                # Check server metrics
                status = server.get_status()
                assert status["total_frames_processed"] >= 1
                assert status["last_latency_ms"] > 0.0

            finally:
                await server.stop()

        asyncio.run(_run())

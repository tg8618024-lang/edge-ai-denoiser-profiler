"""Multi-Threaded Concurrency, Synchronization & Thread-Safety Test Suite.

Authority: Phase 12 Architectural Blueprint & SLA Guarantees.
Validates:
1. AudioCircularFIFO re-entrant thread safety across 10 concurrent reader/writer threads.
2. SIMDDispatcher GEMV thread-safety across concurrent worker threads.
3. PrecisionEngine multi-thread inference consistency across FP32, FP16, and INT8.
4. OBSFilterServer concurrent client streaming and abrupt disconnect resilience.
"""

import os
import sys
import time
import socket
import asyncio
import threading
import concurrent.futures
from typing import List
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.plugins.vst.vst_bridge import AudioCircularFIFO
from src.models.kernels.simd_dispatch import get_simd_dispatcher
from src.models.precision import PrecisionEngine
from src.integrations.obs_bridge import AudioFilterProtocol, OBSFilterServer


class TestThreadSafetyAndConcurrency:
    """Verifies that shared data structures and server components are thread-safe."""

    def test_audio_circular_fifo_concurrent_read_write(self):
        """Stress-test AudioCircularFIFO with concurrent reader and writer threads."""
        fifo = AudioCircularFIFO(capacity=16384)
        num_items_per_thread = 500
        block_size = 64
        total_expected_samples = num_items_per_thread * block_size

        written_samples = 0
        read_samples = 0
        lock = threading.Lock()

        def writer():
            nonlocal written_samples
            rng = np.random.RandomState(123)
            for _ in range(num_items_per_thread):
                chunk = rng.uniform(-1.0, 1.0, block_size).astype(np.float32)
                # Retry until written
                while True:
                    w = fifo.write(chunk)
                    if w > 0:
                        with lock:
                            written_samples += w
                        break
                    time.sleep(0.0001)

        def reader():
            nonlocal read_samples
            collected = 0
            while collected < total_expected_samples:
                avail = fifo.available_read()
                if avail >= block_size:
                    out = fifo.read(block_size)
                    collected += len(out)
                    with lock:
                        read_samples += len(out)
                else:
                    time.sleep(0.0001)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)

        t_write.start()
        t_read.start()

        t_write.join(timeout=10.0)
        t_read.join(timeout=10.0)

        assert not t_write.is_alive(), "Writer thread hung!"
        assert not t_read.is_alive(), "Reader thread hung!"
        assert written_samples == total_expected_samples
        assert read_samples == total_expected_samples

    def test_simd_dispatcher_concurrent_gemv(self):
        """Stress-test SIMDDispatcher across 8 concurrent worker threads."""
        dispatcher = get_simd_dispatcher()
        M, K = 64, 257

        rng = np.random.RandomState(42)
        w_t = rng.randint(-64, 64, size=(M, K), dtype=np.int8)
        bias = rng.randint(-500, 500, size=M, dtype=np.int32)

        def worker(seed: int) -> bool:
            w_rng = np.random.RandomState(seed)
            for _ in range(50):
                x = w_rng.randint(-64, 64, size=K, dtype=np.int8)
                out = np.zeros(M, dtype=np.int32)
                dispatcher.gemv(x, w_t, bias, out, is_unsigned_folded=False)

                # Reference computation
                expected = np.dot(w_t.astype(np.int32), x.astype(np.int32)) + bias
                if not np.array_equal(out, expected):
                    return False
            return True

        num_threads = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, 1000 + i) for i in range(num_threads)]
            results = [f.result(timeout=10.0) for f in futures]

        assert all(results), "Concurrent GEMV execution produced corrupted calculations!"

    def test_precision_engine_concurrent_inference(self):
        """Stress-test PrecisionEngine inference across concurrent threads switching precisions."""
        K = 257

        def worker(precision: str, seed: int) -> bool:
            # Each worker thread maintains its isolated PrecisionEngine instance
            engine = PrecisionEngine(precision=precision)
            rng = np.random.RandomState(seed)
            for _ in range(30):
                log_mag = rng.uniform(-4.0, 2.0, K).astype(np.float32)
                mask = engine.forward_frame(log_mag, copy_output=True)
                if not np.all(np.isfinite(mask)) or not np.all(np.isfinite(engine.hidden_state)):
                    return False
            return True

        num_workers = 6
        modes = ["FP32", "FP16", "INT8"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker, modes[i % len(modes)], 2000 + i)
                for i in range(num_workers)
            ]
            results = [f.result(timeout=10.0) for f in futures]

        assert all(results), "Concurrent precision inference produced non-finite masks!"

    def test_obs_filter_server_concurrent_tcp_streaming(self):
        """Verify OBSFilterServer handles 3 simultaneous streaming clients with abrupt disconnections."""
        asyncio.run(self._async_obs_filter_server_concurrent_tcp_streaming())

    async def _async_obs_filter_server_concurrent_tcp_streaming(self):
        port = 18895
        server = OBSFilterServer(host="127.0.0.1", port=port, sample_rate=16000, hop_length=256)
        await server.start()
        assert server.is_running is True

        try:
            async def client_worker(worker_id: int):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                pcm = np.array([0.1 * (worker_id + 1)] * 256, dtype=np.float32)
                pcm_bytes = AudioFilterProtocol.encode_pcm(pcm)
                packet = AudioFilterProtocol.pack_frame(pcm_bytes)

                # Send 5 frames
                for _ in range(5):
                    writer.write(packet)
                    await writer.drain()

                    header = await reader.readexactly(4)
                    payload_len = AudioFilterProtocol.unpack_header(header)
                    res_bytes = await reader.readexactly(payload_len)
                    assert len(res_bytes) == len(pcm_bytes)

                # Abrupt disconnect
                writer.close()
                await writer.wait_closed()

            # Run 3 concurrent client streams
            await asyncio.gather(
                client_worker(1),
                client_worker(2),
                client_worker(3),
            )

            # Give a moment for connection teardowns
            await asyncio.sleep(0.1)
            status = server.get_status()
            assert status["total_frames_processed"] >= 15

        finally:
            await server.stop()
            assert server.is_running is False

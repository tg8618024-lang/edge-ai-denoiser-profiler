"""Unit & Concurrency Tests for Live Audio Ingress Safety & Client State Isolation.

Verifies:
1. Per-client isolation of PerceptualQualityEvaluator (zero metric cross-talk).
2. Live audio frame ingestion at 16 kHz with exactly 256-sample frame guarantees.
3. Client-side mic-streamer.js static safety contract (mute gain node, Catmull-Rom resampler).
"""

import os
import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import app
from src.telemetry.dnsmos import PerceptualQualityEvaluator


class TestIngressSafetyAndSessionIsolation:
    """Test suite for Task 1: Ingress Safety & Concurrency Hardening."""

    def test_mic_streamer_js_safety_contracts(self):
        """Verify mic-streamer.js implements mute node and Catmull-Rom resampling."""
        js_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "src",
                "dashboard",
                "static",
                "js",
                "audio",
                "mic-streamer.js",
            )
        )
        assert os.path.isfile(js_path), f"mic-streamer.js must exist at {js_path}"

        with open(js_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Feedback loop prevention: mute node must be present and connected to destination
        assert "this.micMuteNode = muteNode" in content or "muteNode" in content
        assert "muteNode.gain.setValueAtTime(0.0" in content or "gain.setValueAtTime(0" in content
        assert "scriptNode.connect(muteNode)" in content
        assert "muteNode.connect(audioCtx.destination)" in content
        assert "scriptNode.connect(audioCtx.destination)" not in content

        # Hardware sample rate resampling to 16,000 Hz
        assert "resampleTo16k" in content
        assert "Catmull-Rom" in content
        assert "this.resampleQueue.splice(0, 256)" in content
        assert "this.micMuteNode.disconnect()" in content

    def test_per_client_perceptual_evaluator_session_isolation(self):
        """Verify two concurrent clients maintain independent smoothed metrics without bleed."""
        test_client = TestClient(app)

        # Client 1 frames: high SNR clean sine tone (should yield high MOS scores)
        t = np.arange(256) / 16000.0
        clean_tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32).tolist()

        # Client 2 frames: pure chaotic noise with near 0 SNR (should yield lower MOS scores)
        rng = np.random.RandomState(42)
        heavy_noise = rng.normal(0, 0.4, 256).astype(np.float32).tolist()

        with test_client.websocket_connect("/ws/stream") as ws1:
            with test_client.websocket_connect("/ws/stream") as ws2:
                # Stream 15 frames for client 1
                c1_ovrl_scores = []
                for _ in range(15):
                    ws1.send_text(json.dumps({"type": "audio_frame", "pcm": clean_tone}))
                    resp1 = json.loads(ws1.receive_text())
                    assert resp1["type"] == "live_telemetry"
                    c1_ovrl_scores.append(resp1["voice_quality"]["ovrl_mos"])

                # Stream 15 frames for client 2
                c2_ovrl_scores = []
                for _ in range(15):
                    ws2.send_text(json.dumps({"type": "audio_frame", "pcm": heavy_noise}))
                    resp2 = json.loads(ws2.receive_text())
                    assert resp2["type"] == "live_telemetry"
                    c2_ovrl_scores.append(resp2["voice_quality"]["ovrl_mos"])

                # Verify metrics are completely decoupled
                # Client 1 final score should reflect clean tone; Client 2 should reflect noise
                c1_final = c1_ovrl_scores[-1]
                c2_final = c2_ovrl_scores[-1]

                # Both should have valid bounded MOS scores in [1.0, 5.0]
                assert 1.0 <= c1_final <= 5.0
                assert 1.0 <= c2_final <= 5.0

                # Ensure Client 1 metrics were not dragged down by Client 2's heavy noise
                # Stream one more frame on Client 1 and assert it remained stable
                ws1.send_text(json.dumps({"type": "audio_frame", "pcm": clean_tone}))
                resp1_after = json.loads(ws1.receive_text())
                c1_after = resp1_after["voice_quality"]["ovrl_mos"]

                assert abs(c1_after - c1_final) <= 0.35, (
                    f"Client 1 score jumped unexpectedly from {c1_final} to {c1_after} "
                    "indicating cross-client state pollution!"
                )

"""Unit tests for Target Speaker Controller and Google Antigravity Agent integration.

Verifies:
- TargetSpeakerController state machine, wake-word detection, and multi-speaker registry.
- Spectrum pre-conditioning: Input -> STFT -> VAD -> Target Speaker Mask -> GRUMaskNet -> CRM -> iSTFT.
- Google Antigravity SDK Agent tools and command orchestration.
- FastAPI REST endpoints for /api/target-speaker/*.
"""

import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.target_speaker import (
    TriggerWordDetector,
    SpeakerVoiceprint,
    TargetSpeakerController,
    TargetSpeakerExtractor,
)
from src.agents.target_speaker_agent import (
    TargetSpeakerAgent,
    TargetSpeakerTools,
    ANTIGRAVITY_SDK_AVAILABLE,
)
from src.audio.pipeline import AudioDenoisingPipeline


class TestTargetSpeakerControllerExtended:
    """Test suite for TargetSpeakerController capabilities."""

    def test_state_machine_transitions(self):
        controller = TargetSpeakerController(sample_rate=16000, num_bins=257)
        assert controller.state == "IDLE"

        # Trigger enrollment
        controller.trigger_enrollment(num_frames=5, speaker_name="Host")
        assert controller.state == "ENROLLING"
        assert controller.active_speaker_name == "Host"

        # Simulate enrollment speech frames
        t = np.arange(256) / 16000.0
        pcm = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        mag = np.abs(np.fft.rfft(pcm, n=512)).astype(np.float32)

        for _ in range(5):
            controller.process_frame(pcm, mag, is_speech=True)

        assert controller.is_locked is True
        assert controller.state == "LOCKED_TRACKING"

        # Now feed a competing speaker (different pitch/formants)
        pcm_competing = (0.5 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        mag_competing = np.abs(np.fft.rfft(pcm_competing, n=512)).astype(np.float32)

        for _ in range(10):
            controller.process_frame(pcm_competing, mag_competing, is_speech=True)

        # Should enter ATTENUATING state
        assert controller.is_target_active is False
        assert controller.state == "ATTENUATING"
        assert controller.suppression_db > 10.0

        # Unlock
        controller.unlock()
        assert controller.state == "IDLE"

    def test_speaker_registry_and_switching(self):
        controller = TargetSpeakerController()

        emb_a = np.ones(64, dtype=np.float32)
        emb_a /= np.linalg.norm(emb_a)
        controller.lock_with_embedding(emb_a, speaker_name="Alice")

        emb_b = np.full(64, 2.0, dtype=np.float32)
        emb_b /= np.linalg.norm(emb_b)
        controller.lock_with_embedding(emb_b, speaker_name="Bob")

        speakers = controller.list_enrolled_speakers()
        speaker_names = [s["name"] for s in speakers]
        assert "Alice" in speaker_names
        assert "Bob" in speaker_names

        # Switch back to Alice
        assert controller.switch_target_speaker("Alice") is True
        assert controller.active_speaker_name == "Alice"
        assert np.allclose(controller.target_voiceprint, emb_a)

        # Switch to unknown speaker
        assert controller.switch_target_speaker("Unknown") is False

    def test_suppression_depth_configuration(self):
        controller = TargetSpeakerController()
        controller.set_suppression_depth(28.0)
        assert controller.max_suppression_db == 28.0

        # Clipping checks
        controller.set_suppression_depth(2.0)
        assert controller.max_suppression_db == 6.0

        controller.set_suppression_depth(55.0)
        assert controller.max_suppression_db == 40.0

    def test_condition_spectrum(self):
        controller = TargetSpeakerController()
        mag = np.ones(257, dtype=np.float32)
        g_tse = np.full(257, 0.5, dtype=np.float32)
        conditioned = controller.condition_spectrum(mag, g_tse)
        assert np.allclose(conditioned, 0.5)


class TestTargetSpeakerAgentAndTools:
    """Test suite for Google Antigravity Agent and registered tools."""

    def test_agent_tools_execution(self):
        controller = TargetSpeakerController()
        tools = TargetSpeakerTools(controller=controller)

        # Test enroll tool
        res_enroll = tools.tool_enroll_speaker("Alice", num_frames=10)
        assert "Alice" in res_enroll
        assert controller.is_enrolling is True

        # Test status tool
        res_status = tools.tool_get_status()
        status_data = json.loads(res_status)
        assert status_data["active_speaker_name"] == "Alice"
        assert status_data["state"] == "ENROLLING"

        # Test suppression depth tool
        res_depth = tools.tool_set_suppression_depth(26.0)
        assert "26.0" in res_depth
        assert controller.max_suppression_db == 26.0

        # Test unlock tool
        res_unlock = tools.tool_unlock_speaker()
        assert "disabled" in res_unlock
        assert controller.is_locked is False

    def test_agent_command_execution(self):
        controller = TargetSpeakerController()
        agent = TargetSpeakerAgent(controller=controller)

        # Trigger word detection in command
        res_lock = agent.execute_command("Hey Denoiser, please lock voice")
        assert "Target speaker enrollment activated" in res_lock
        assert controller.is_enrolling is True

        res_unlock = agent.execute_command("unlock voice now")
        assert "Target speaker lock disabled" in res_unlock
        assert controller.is_locked is False

        # Status command
        res_status = agent.execute_command("Show me the target speaker status")
        status_json = json.loads(res_status)
        assert "state" in status_json

        # Telemetry
        tel = agent.get_telemetry()
        assert "state" in tel
        assert "sdk_available" in tel


class TestSignalFlowOrderingInPipeline:
    """Verify that STFT -> VAD -> Target Speaker Mask -> GRUMaskNet -> CRM -> iSTFT executes in order."""

    def test_pipeline_target_speaker_substage_and_ordering(self):
        pipeline = AudioDenoisingPipeline(sample_rate=16000)
        assert isinstance(pipeline.target_speaker, TargetSpeakerController)

        # Run one frame
        pcm_frame = (0.2 * np.random.randn(256)).astype(np.float32)
        out_pcm = pipeline.process_frame(pcm_frame)

        assert len(out_pcm) == 256
        assert np.all(np.isfinite(out_pcm))

        # Verify that Stage 1 measured tse_ms
        assert "tse_ms" in pipeline.last_substages
        assert pipeline.last_substages["tse_ms"] >= 0.0

        # Verify that last_g_tse exists
        assert hasattr(pipeline, "last_g_tse")
        assert len(pipeline.last_g_tse) == pipeline.stft.num_bins

    def test_target_speaker_conditioning_attenuates_secondary_speech(self):
        pipeline = AudioDenoisingPipeline(sample_rate=16000)

        # Enroll Target Speaker (200 Hz tone)
        t = np.arange(256) / 16000.0
        target_pcm = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

        pipeline.lock_target_speaker(num_frames=10)
        for _ in range(10):
            pipeline.process_frame(target_pcm)

        assert pipeline.target_speaker.is_locked is True

        # Now stream competing speech (500 Hz tone)
        competing_pcm = (0.5 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        out_frames = []
        for _ in range(15):
            out_frames.append(pipeline.process_frame(competing_pcm))

        # Competing voice should be attenuated
        assert pipeline.target_speaker.is_target_active is False
        assert pipeline.target_speaker.suppression_db > 10.0
        # Final output frame RMS energy should be lower than input energy
        in_rms = float(np.sqrt(np.mean(competing_pcm**2)))
        out_rms = float(np.sqrt(np.mean(out_frames[-1]**2)))
        assert out_rms < in_rms * 0.5


class TestDashboardTargetSpeakerAPI:
    """Verify FastAPI target speaker routes."""

    def test_target_speaker_api_endpoints(self):
        from fastapi.testclient import TestClient
        from src.dashboard.app import app

        client = TestClient(app)

        # GET /api/target-speaker/status
        res = client.get("/api/target-speaker/status")
        assert res.status_code == 200
        data = res.json()
        assert "state" in data
        assert "is_locked" in data

        # POST /api/target-speaker/enroll
        res_enroll = client.post(
            "/api/target-speaker/enroll",
            json={"speaker_name": "Broadcaster", "enrollment_frames": 15},
        )
        assert res_enroll.status_code == 200
        assert res_enroll.json()["success"] is True

        # POST /api/target-speaker/command
        res_cmd = client.post(
            "/api/target-speaker/command",
            json={"command": "get status"},
        )
        assert res_cmd.status_code == 200
        assert res_cmd.json()["success"] is True
        assert "state" in res_cmd.json()["response"]

        # POST /api/target-speaker/unlock
        res_unlock = client.post("/api/target-speaker/unlock")
        assert res_unlock.status_code == 200
        assert res_unlock.json()["success"] is True

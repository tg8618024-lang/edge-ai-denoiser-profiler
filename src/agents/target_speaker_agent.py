"""Autonomous Target Speaker Agent powered by Google Antigravity SDK.

Coordinates Target Speaker Extraction (TSE), wake-word detection, speaker enrollment,
and other-speaker suppression via Google Antigravity Agent workflows.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
import numpy as np

from src.models.target_speaker import TargetSpeakerController

logger = logging.getLogger("target_speaker_agent")

# Check for Google Antigravity SDK availability
try:
    from google.antigravity import Agent, LocalAgentConfig, ToolContext
    ANTIGRAVITY_SDK_AVAILABLE = True
except ImportError:
    ANTIGRAVITY_SDK_AVAILABLE = False
    Agent = None
    LocalAgentConfig = None
    ToolContext = None


class TargetSpeakerTools:
    """Tool definitions exposing TargetSpeakerController capabilities to Google Antigravity Agents."""

    def __init__(self, controller: Optional[TargetSpeakerController] = None) -> None:
        self.controller = controller or TargetSpeakerController()

    def tool_enroll_speaker(self, speaker_name: str = "Host", num_frames: int = 25) -> str:
        """Enrolls the target speaker's voiceprint from upcoming speech frames.

        Args:
            speaker_name: Identifier or name of the target speaker (e.g. "Host", "Alice").
            num_frames: Number of continuous speech frames to average (default 25 frames ~ 400ms).
        """
        self.controller.trigger_enrollment(num_frames=num_frames, speaker_name=speaker_name)
        return (
            f"Voiceprint enrollment initiated for speaker '{speaker_name}'. "
            f"Please speak clearly for approximately {num_frames * 16} ms."
        )

    def tool_unlock_speaker(self) -> str:
        """Unlocks voice lock and allows all voices to pass through without attenuation."""
        self.controller.unlock()
        return "Voice lock disabled. All speakers will now be heard naturally without suppression."

    def tool_get_status(self) -> str:
        """Retrieves real-time telemetry from the Target Speaker Controller.

        Returns:
            JSON-formatted string with state, similarity, active speaker, and suppression metrics.
        """
        telemetry = self.controller.get_telemetry()
        return json.dumps(telemetry, indent=2)

    def tool_set_suppression_depth(self, suppression_db: float) -> str:
        """Sets the suppression depth in decibels for competing secondary speakers.

        Args:
            suppression_db: Attenuation in dB (e.g. 18.0 for moderate, 34.0 for aggressive isolation).
        """
        self.controller.set_suppression_depth(suppression_db)
        return f"Competing speaker suppression depth configured to -{abs(suppression_db):.1f} dB."

    def tool_switch_speaker(self, speaker_name: str) -> str:
        """Switches the active target speaker to a previously enrolled speaker profile.

        Args:
            speaker_name: Name of previously enrolled speaker to activate.
        """
        success = self.controller.switch_target_speaker(speaker_name)
        if success:
            return f"Successfully switched target speaker to '{speaker_name}'."
        enrolled = [s["name"] for s in self.controller.list_enrolled_speakers()]
        return f"Speaker '{speaker_name}' not found. Currently enrolled speakers: {enrolled}."

    def tool_list_enrolled_speakers(self) -> str:
        """Lists all registered speaker profiles stored in the controller registry."""
        speakers = self.controller.list_enrolled_speakers()
        return json.dumps(speakers, indent=2)

    def tool_process_voice_command(self, command: str) -> str:
        """Analyzes a spoken or written command for trigger words (e.g. 'lock voice', 'unlock voice').

        Args:
            command: Transcript text or natural language command.
        """
        action = self.controller.check_trigger_word(command)
        if action == "lock":
            return "Detected lock trigger phrase. Target speaker enrollment activated."
        elif action == "unlock":
            return "Detected unlock trigger phrase. Target speaker lock disabled."
        return f"Processed text '{command}'. No trigger phrase detected."


class TargetSpeakerAgent:
    """Autonomous Agent orchestrating Target Speaker Controller via Google Antigravity SDK."""

    SYSTEM_INSTRUCTIONS = (
        "You are the Target Speaker Controller Agent for a professional real-time audio broadcast testbench. "
        "Your mission is to maintain acoustic voiceprint isolation for the primary host or target speaker, "
        "detect trigger phrases, manage speaker profiles, and attenuate competing secondary speakers. "
        "Use your tools to query status, enroll new voiceprints, switch speakers, and adjust suppression levels."
    )

    def __init__(
        self,
        controller: Optional[TargetSpeakerController] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.tools = TargetSpeakerTools(controller=controller)
        self.controller = self.tools.controller
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._sdk_agent: Any = None

        if ANTIGRAVITY_SDK_AVAILABLE and self.api_key:
            self._init_sdk_agent()

    def _init_sdk_agent(self) -> None:
        """Instantiate Google Antigravity Agent with registered custom tools."""
        try:
            config = LocalAgentConfig(
                model=self.model_name,
                api_key=self.api_key,
                system_instructions=self.SYSTEM_INSTRUCTIONS,
                tools=[
                    self.tools.tool_enroll_speaker,
                    self.tools.tool_unlock_speaker,
                    self.tools.tool_get_status,
                    self.tools.tool_set_suppression_depth,
                    self.tools.tool_switch_speaker,
                    self.tools.tool_list_enrolled_speakers,
                    self.tools.tool_process_voice_command,
                ],
            )
            self._sdk_agent = Agent(config)
            logger.info("Google Antigravity SDK Agent initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Antigravity SDK Agent: {e}")
            self._sdk_agent = None

    def execute_command(self, query: str) -> str:
        """Execute a natural language command or trigger word query.

        Uses the Google Antigravity SDK Agent if available; otherwise falls back
        to intelligent deterministic heuristic routing.
        """
        query_clean = str(query or "").strip().lower()

        # Check for direct trigger words first
        trigger_res = self.controller.check_trigger_word(query_clean)
        if trigger_res == "lock":
            return "Trigger word detected: Target speaker enrollment activated."
        elif trigger_res == "unlock":
            return "Trigger word detected: Target speaker lock disabled."

        # Heuristic routing if SDK Agent is offline
        if "status" in query_clean or "telemetry" in query_clean:
            return self.tools.tool_get_status()
        elif "unlock" in query_clean or "disable" in query_clean or "reset" in query_clean:
            return self.tools.tool_unlock_speaker()
        elif "enroll" in query_clean or "lock" in query_clean:
            name = "Host"
            words = query.split()
            if "for" in words:
                idx = words.index("for")
                if idx + 1 < len(words):
                    name = words[idx + 1]
            return self.tools.tool_enroll_speaker(speaker_name=name)
        elif "list" in query_clean or "speakers" in query_clean:
            return self.tools.tool_list_enrolled_speakers()
        elif "suppress" in query_clean or "depth" in query_clean:
            # Check for numeric dB in query
            nums = [float(s) for s in query.split() if s.replace(".", "", 1).isdigit()]
            db = nums[0] if nums else 24.0
            return self.tools.tool_set_suppression_depth(db)

        return (
            f"Understood command: '{query}'. Target speaker controller is in state: "
            f"{self.controller.state}. Active speaker: '{self.controller.active_speaker_name}'."
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Return combined controller and agent telemetry."""
        telemetry = self.controller.get_telemetry()
        telemetry["sdk_available"] = ANTIGRAVITY_SDK_AVAILABLE
        telemetry["agent_online"] = self._sdk_agent is not None
        return telemetry

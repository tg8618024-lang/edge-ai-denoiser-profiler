"""Trigger-Word Activated Target Speaker Extraction (TSE) & Voiceprint Lock Engine.

Provides:
- TriggerWordDetector: Keyword spotting for wake / lock commands ("lock voice", "hey denoiser", "unlock voice").
- SpeakerVoiceprint: Compact 64-dimensional acoustic speaker embedding extractor.
- TargetSpeakerExtractor: Real-time speaker verification and competing background voice suppression.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TriggerWordDetector:
    """Detects spoken trigger words and voice commands to lock or unlock target speaker."""

    LOCK_PHRASES = [
        "lock voice",
        "lock speaker",
        "hey denoiser",
        "focus voice",
        "focus speaker",
        "lock target",
        "enroll voice",
        "my voice only",
        "isolate voice",
        "voice lock",
    ]

    UNLOCK_PHRASES = [
        "unlock voice",
        "unlock speaker",
        "disable voice lock",
        "disable lock",
        "reset voice",
        "clear speaker",
        "all voices",
    ]

    def __init__(self) -> None:
        # Precompile regex patterns for boundary-aware matching
        self.lock_patterns = [
            re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            for phrase in self.LOCK_PHRASES
        ]
        self.unlock_patterns = [
            re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            for phrase in self.UNLOCK_PHRASES
        ]

    def detect(self, text: str) -> Optional[Tuple[str, str]]:
        """Check transcript for trigger phrases.

        Returns
        -------
        Optional[Tuple[str, str]]
            ("lock", matched_phrase) or ("unlock", matched_phrase) or None.
        """
        clean_text = str(text or "").strip().lower()
        if not clean_text:
            return None

        for pattern, phrase in zip(self.unlock_patterns, self.UNLOCK_PHRASES):
            if pattern.search(clean_text):
                return ("unlock", phrase)

        for pattern, phrase in zip(self.lock_patterns, self.LOCK_PHRASES):
            if pattern.search(clean_text):
                return ("lock", phrase)

        return None


class SpeakerVoiceprint:
    """Extracts compact 64-dimensional speaker acoustic embedding vector."""

    def __init__(self, sample_rate: int = 16000, num_bins: int = 257) -> None:
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.bin_hz = (self.sample_rate / 2.0) / (self.num_bins - 1)

        # 16-band Mel filterbank matrix: shape (16, num_bins)
        self.mel_fb = self._create_mel_filterbank(num_filters=16)

    def _create_mel_filterbank(self, num_filters: int = 16) -> np.ndarray:
        """Construct triangular Mel filterbank matrix."""
        low_mel = 0.0
        high_mel = 2595.0 * np.log10(1.0 + (self.sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((hz_points / (self.sample_rate / 2.0)) * (self.num_bins - 1)).astype(int)

        fb = np.zeros((num_filters, self.num_bins), dtype=np.float32)
        for m in range(1, num_filters + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]

            for k in range(f_left, f_center):
                if f_center > f_left:
                    fb[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fb[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fb

    def extract_embedding(self, time_frame: np.ndarray, mag_spec: np.ndarray) -> np.ndarray:
        """Compute normalized speaker embedding vector for a 256-sample frame."""
        pcm = np.asarray(time_frame, dtype=np.float32).ravel()
        mag = np.asarray(mag_spec, dtype=np.float32).ravel()

        if np.sum(pcm**2) < 1e-6:
            return np.zeros(64, dtype=np.float32)

        # 1. 16 Mel filterbank probability distribution (Vocal tract shape)
        mel_energies = np.dot(self.mel_fb, mag)
        mel_sum = np.sum(mel_energies) + 1e-7
        mel_prob = (mel_energies / mel_sum) * 2.0  # 16 dims

        # 2. Pitch autocorrelation profile (lags 35 to 195 covering 80 Hz to 450 Hz)
        ac = np.correlate(pcm, pcm, mode="full")
        mid = len(ac) // 2
        lags = np.maximum(0.0, ac[mid + 35 : mid + 195])
        lag_vec = lags[:160].reshape(16, 10).mean(axis=1).astype(np.float32)
        lag_sum = np.sum(lag_vec) + 1e-7
        lag_prob = (lag_vec / lag_sum) * 2.0  # 16 dims

        # 3. Spectral shape & Formant descriptors (16 dims)
        freqs = np.linspace(0.0, self.sample_rate / 2.0, self.num_bins)
        tot_e = np.sum(mag) + 1e-7
        centroid = float(np.sum(freqs * mag) / tot_e)
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / tot_e))
        rolloff_idx = np.searchsorted(np.cumsum(mag), 0.85 * tot_e)
        rolloff = float(freqs[min(rolloff_idx, self.num_bins - 1)])

        # Formant ratios: F1 (300-800Hz), F2 (800-2000Hz), F3 (2000-3500Hz)
        f1_e = float(np.sum(mag[9:26])) / tot_e
        f2_e = float(np.sum(mag[26:64])) / tot_e
        f3_e = float(np.sum(mag[64:112])) / tot_e
        mag_norm = mag / (np.max(mag) + 1e-6)
        shape_scalars = np.array([
            centroid / 4000.0,
            spread / 2000.0,
            rolloff / 4000.0,
            f1_e * 2.0,
            f2_e * 2.0,
            f3_e * 2.0,
        ], dtype=np.float32)
        shape_feats = np.concatenate([shape_scalars, mag_norm[4:14].astype(np.float32)])  # 16 dims

        # 4. Harmonic comb energy distribution (16 dims: bins 14 to 46 step 2)
        harm_feats = mag_norm[14:46:2].astype(np.float32)

        # Concatenate all components into 64-dimensional vector
        vec = np.concatenate([mel_prob, lag_prob, shape_feats, harm_feats])[:64]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec.astype(np.float32)


class TargetSpeakerExtractor:
    """Locks onto target speaker voice and extracts target speech while suppressing competing voices."""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_bins: int = 257,
        similarity_threshold: float = 0.72,
        enrollment_frames: int = 25,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.similarity_threshold = float(similarity_threshold)
        self.enrollment_frames_target = int(enrollment_frames)

        self.voiceprint_extractor = SpeakerVoiceprint(sample_rate=sample_rate, num_bins=num_bins)
        self.trigger_detector = TriggerWordDetector()

        # Target speaker state
        self.is_locked: bool = False
        self.is_enrolling: bool = False
        self.target_voiceprint: Optional[np.ndarray] = None
        self.enrollment_buffer: List[np.ndarray] = []

        # Real-time running telemetry
        self.current_similarity: float = 1.0
        self.smoothed_similarity: float = 1.0
        self.last_trigger_phrase: str = ""
        self.is_target_active: bool = True
        self.suppression_db: float = 0.0

    def trigger_enrollment(self, num_frames: int = 25) -> None:
        """Begin enrolling target speaker voiceprint from upcoming speech frames."""
        self.enrollment_frames_target = max(1, int(num_frames))
        self.enrollment_buffer.clear()
        self.is_enrolling = True
        self.is_locked = False
        self.target_voiceprint = None
        self.current_similarity = 1.0
        self.smoothed_similarity = 1.0

    def lock_with_embedding(self, embedding: np.ndarray) -> None:
        """Lock immediately with precomputed speaker embedding."""
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm
        self.target_voiceprint = emb
        self.is_locked = True
        self.is_enrolling = False
        self.enrollment_buffer.clear()

    def unlock(self) -> None:
        """Unlock voice lock, returning to pass-all-speakers mode."""
        self.is_locked = False
        self.is_enrolling = False
        self.target_voiceprint = None
        self.enrollment_buffer.clear()
        self.current_similarity = 1.0
        self.smoothed_similarity = 1.0
        self.suppression_db = 0.0
        self.is_target_active = True

    def check_trigger_word(self, transcript_text: str) -> Optional[str]:
        """Check transcript for trigger word and automatically lock or unlock."""
        match = self.trigger_detector.detect(transcript_text)
        if match:
            cmd, phrase = match
            self.last_trigger_phrase = phrase
            if cmd == "lock":
                self.trigger_enrollment()
                return "lock"
            elif cmd == "unlock":
                self.unlock()
                return "unlock"
        return None

    def process_frame(
        self,
        time_frame: np.ndarray,
        mag_spec: np.ndarray,
        is_speech: bool = True,
    ) -> np.ndarray:
        """Compute target speaker spectral gain mask G_TSE(f) in [0.02, 1.0]."""
        # If unlocked, all voices pass freely
        if not self.is_locked and not self.is_enrolling:
            self.current_similarity = 1.0
            self.smoothed_similarity = 1.0
            self.suppression_db = 0.0
            self.is_target_active = True
            return np.ones(self.num_bins, dtype=np.float32)

        # Extract current frame speaker embedding
        frame_emb = self.voiceprint_extractor.extract_embedding(time_frame, mag_spec)

        # State 1: Enrolling target voiceprint
        if self.is_enrolling:
            if is_speech:
                self.enrollment_buffer.append(frame_emb)
                if len(self.enrollment_buffer) >= self.enrollment_frames_target:
                    # Finalize voiceprint: mean vector normalized to unit sphere
                    mean_vec = np.mean(self.enrollment_buffer, axis=0)
                    norm = float(np.linalg.norm(mean_vec))
                    if norm > 1e-6:
                        mean_vec /= norm
                    self.target_voiceprint = mean_vec.astype(np.float32)
                    self.is_locked = True
                    self.is_enrolling = False
                    self.enrollment_buffer.clear()
            return np.ones(self.num_bins, dtype=np.float32)

        # State 2: Locked onto Target Speaker
        if self.is_locked and self.target_voiceprint is not None:
            pcm_energy = float(np.sum(time_frame**2))
            if not is_speech or pcm_energy < 1e-5:
                # Non-speech / silent pause frame -> maintain active state without decaying similarity
                self.suppression_db = 0.0
                return np.ones(self.num_bins, dtype=np.float32)

            # Compute cosine similarity
            sim = float(np.dot(frame_emb, self.target_voiceprint))
            self.current_similarity = float(np.clip(sim, -1.0, 1.0))

            # Temporal smoothing for smooth crossfading without clicks
            self.smoothed_similarity = 0.70 * self.smoothed_similarity + 0.30 * self.current_similarity
            smooth_s = self.smoothed_similarity

            # Target speaker verification decision
            if smooth_s >= self.similarity_threshold:
                # Target speaker verified! Crystal clear vocal transmission
                self.is_target_active = True
                self.suppression_db = 0.0
                return np.ones(self.num_bins, dtype=np.float32)
            else:
                # Competing secondary speaker detected!
                self.is_target_active = False
                # Suppression curve: smoothly attenuates down to 0.02 (-34 dB)
                atten = np.clip((smooth_s / self.similarity_threshold) ** 6, 0.02, 1.0)
                self.suppression_db = round(-20.0 * np.log10(max(1e-3, float(atten))), 1)

                # Generate speech-band suppression mask
                mask = np.full(self.num_bins, float(atten), dtype=np.float32)
                return mask

        return np.ones(self.num_bins, dtype=np.float32)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return comprehensive target speaker telemetry for UI and WebSocket."""
        return {
            "is_locked": self.is_locked,
            "is_enrolling": self.is_enrolling,
            "enrollment_progress_pct": round(
                (len(self.enrollment_buffer) / max(1, self.enrollment_frames_target)) * 100.0, 1
            ) if self.is_enrolling else (100.0 if self.is_locked else 0.0),
            "similarity_pct": round(max(0.0, min(100.0, self.smoothed_similarity * 100.0)), 1),
            "raw_similarity": round(self.current_similarity, 3),
            "similarity_threshold": self.similarity_threshold,
            "is_target_active": self.is_target_active,
            "secondary_speaker_suppression_db": self.suppression_db,
            "last_trigger_phrase": self.last_trigger_phrase,
        }

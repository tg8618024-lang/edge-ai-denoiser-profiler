"""WebRTC Audio Bridge, RFC 3550 Adaptive Jitter Buffer & Packet Loss Concealment (PLC).

Authority:
- RFC 3550: RTP: A Transport Protocol for Real-Time Applications (Jitter Estimation)
- ITU-T G.711 Appendix I: Packet Loss Concealment for Real-Time Audio
- Phase 10 Blueprint: WebRTC Direct Audio Transport & Latency Profiling
"""

from __future__ import annotations
import sys
import os
import time
import math
import heapq
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import AudioDenoisingPipeline
from src.telemetry.profiler import StageProfiler


@dataclass(order=True)
class RTPPacket:
    """Represents a single Real-time Transport Protocol (RTP) audio packet."""
    seq: int  # 16-bit sequence number (0..65535)
    timestamp: int = field(compare=False)  # 32-bit sampling timestamp
    payload: bytes = field(compare=False)  # PCM audio bytes (16-bit little endian)
    arrival_time_s: float = field(compare=False, default_factory=time.perf_counter)
    payload_type: int = field(compare=False, default=97)  # Dynamic payload type (e.g. L16)
    ssrc: int = field(compare=False, default=0x12345678)


class RFC3550JitterEstimator:
    """Estimates statistical interarrival jitter according to RFC 3550 Section 6.4.1.

    Formula:
      D(i, j) = (R_j - S_j) - (R_i - S_i)
      J(i) = J(i-1) + (|D(i-1, i)| - J(i-1)) / 16.0
    where:
      R_k: Local arrival time in timestamp units (e.g., 16 kHz ticks)
      S_k: Packet RTP timestamp in timestamp units
    """

    def __init__(self, clock_rate: int = 16000) -> None:
        self.clock_rate = int(clock_rate)
        self.jitter_ticks: float = 0.0
        self.last_transit: Optional[float] = None
        self._lock = threading.RLock()

    def update(self, rtp_timestamp: int, arrival_time_s: float) -> float:
        """Update jitter estimate with a newly arrived packet. Returns smoothed jitter in ms."""
        with self._lock:
            # Arrival time in RTP timestamp clock units
            arrival_ticks = arrival_time_s * self.clock_rate
            transit = arrival_ticks - rtp_timestamp

            if self.last_transit is not None:
                d = abs(transit - self.last_transit)
                self.jitter_ticks += (d - self.jitter_ticks) / 16.0
            else:
                self.jitter_ticks = 0.0

            self.last_transit = transit
            return self.get_jitter_ms()

    def get_jitter_ms(self) -> float:
        """Return current estimated interarrival jitter in milliseconds."""
        with self._lock:
            rate = max(1, self.clock_rate)
            return round((self.jitter_ticks / rate) * 1000.0, 3)

    def reset(self) -> None:
        with self._lock:
            self.jitter_ticks = 0.0
            self.last_transit = None


class PacketLossConcealer:
    """Synthesizes replacement audio upon packet loss per ITU-T G.711 App. I.

    Uses pitch-synchronous history replication with geometric energy attenuation (g = 0.85).
    """

    def __init__(self, sample_rate: int = 16000, history_length: int = 1024, decay_factor: float = 0.85) -> None:
        self.sample_rate = sample_rate
        self.history_length = history_length
        self.decay_factor = float(np.clip(decay_factor, 0.5, 0.95))
        self.history_buffer = np.zeros(history_length, dtype=np.float32)
        self.pitch_template = np.zeros(history_length, dtype=np.float32)
        self.consecutive_lost_frames = 0
        self._lock = threading.RLock()

    def update_history(self, pcm_samples: np.ndarray) -> None:
        """Update recent history buffer with received valid audio samples."""
        with self._lock:
            n = len(pcm_samples)
            if n == 0:
                return
            if n >= self.history_length:
                self.history_buffer[:] = pcm_samples[-self.history_length:]
            else:
                self.history_buffer[:-n] = self.history_buffer[n:]
                self.history_buffer[-n:] = pcm_samples
            self.pitch_template[:] = self.history_buffer[:]
            self.consecutive_lost_frames = 0

    def conceal(self, num_samples: int, pitch_lag: int = 160) -> np.ndarray:
        """Generate concealed audio samples when a packet is dropped or late."""
        with self._lock:
            if num_samples <= 0:
                return np.zeros(0, dtype=np.float32)

            self.consecutive_lost_frames += 1
            gain = float(self.decay_factor ** self.consecutive_lost_frames)

            # Ensure valid pitch lag within history buffer
            lag = int(np.clip(pitch_lag, 40, self.history_length // 2))

            # Vectorized pitch replication per Rule 7 (zero per-element Python loops)
            indices = (self.history_length - lag) + (np.arange(num_samples) % lag)
            concealed = (self.pitch_template[indices] * gain).astype(np.float32)

            # Update history with synthesized samples so subsequent losses smoothly fade
            if num_samples >= self.history_length:
                self.history_buffer[:] = concealed[-self.history_length:]
            else:
                self.history_buffer[:-num_samples] = self.history_buffer[num_samples:]
                self.history_buffer[-num_samples:] = concealed

            return concealed

    def reset(self) -> None:
        with self._lock:
            self.history_buffer.fill(0.0)
            self.pitch_template.fill(0.0)
            self.consecutive_lost_frames = 0


class AdaptiveJitterBuffer:
    """Thread-safe adaptive jitter buffer with RFC 3550 jitter tracking & PLC.

    - Sorts arriving RTP packets by 16-bit sequence number (handling wraparound 65535 -> 0).
    - Dynamically bounds playout target delay D_target in [min_delay_ms, max_delay_ms].
    - Detects missing/late packets and triggers concealment.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_delay_ms: float = 16.0,
        max_delay_ms: float = 100.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_delay_ms = float(min_delay_ms)
        self.max_delay_ms = float(max_delay_ms)

        self.jitter_estimator = RFC3550JitterEstimator(clock_rate=sample_rate)
        self.plc = PacketLossConcealer(sample_rate=sample_rate)

        self._queue: List[Tuple[int, RTPPacket]] = []  # Priority queue sorted by unwrapped seq
        self._lock = threading.RLock()

        self.last_popped_unwrapped_seq: Optional[int] = None
        self.max_unwrapped_seq: Optional[int] = None

        # Telemetry counters
        self.total_packets_received: int = 0
        self.total_packets_popped: int = 0
        self.total_lost_packets: int = 0
        self.total_plc_frames: int = 0

    def _unwrap_seq(self, seq: int) -> int:
        """Unwrap 16-bit sequence number to a monotonic 64-bit integer."""
        if self.max_unwrapped_seq is None:
            self.max_unwrapped_seq = seq
            return seq

        diff = (seq - (self.max_unwrapped_seq & 0xFFFF)) & 0xFFFF
        if diff < 32768:
            unwrapped = self.max_unwrapped_seq + diff
            self.max_unwrapped_seq = unwrapped
        else:
            unwrapped = self.max_unwrapped_seq - (65536 - diff)
        return max(0, unwrapped)

    def push_packet(self, packet: RTPPacket) -> None:
        """Push an incoming RTP packet into the jitter buffer."""
        with self._lock:
            self.total_packets_received += 1
            unwrapped_seq = self._unwrap_seq(packet.seq)

            # Update RFC 3550 jitter
            self.jitter_estimator.update(packet.timestamp, packet.arrival_time_s)

            # Drop packet if it arrived later than what has already been popped
            if self.last_popped_unwrapped_seq is not None and unwrapped_seq <= self.last_popped_unwrapped_seq:
                return

            # Drop in-flight duplicate if identical unwrapped_seq is already in queue
            if any(item[0] == unwrapped_seq for item in self._queue):
                return

            heapq.heappush(self._queue, (unwrapped_seq, packet))

            # Enforce max buffer delay bound (shed oldest packets if queue exceeds capacity)
            max_capacity = max(6, int(math.ceil(self.max_delay_ms / 16.0)) + 2)
            while len(self._queue) > max_capacity:
                dropped_seq, _ = heapq.heappop(self._queue)
                self.last_popped_unwrapped_seq = dropped_seq
                self.total_lost_packets += 1

    def get_target_delay_ms(self) -> float:
        """Compute target adaptive playout buffer delay based on 3 * jitter_ms."""
        jitter_ms = self.jitter_estimator.get_jitter_ms()
        target = max(self.min_delay_ms, min(self.max_delay_ms, 3.0 * jitter_ms))
        return round(target, 2)

    def pop_frame(self, frame_samples: int = 256) -> Tuple[np.ndarray, bool]:
        """Pop the next in-order audio frame from the jitter buffer.

        Returns:
            Tuple of (pcm_float_samples, was_concealed_boolean)
        """
        with self._lock:
            # Discard any stale or duplicate packets that have already been popped
            while self._queue and self.last_popped_unwrapped_seq is not None and self._queue[0][0] <= self.last_popped_unwrapped_seq:
                heapq.heappop(self._queue)

            if not self._queue:
                # Underrun: synthesize concealment frame
                self.total_plc_frames += 1
                self.total_lost_packets += 1
                concealed = self.plc.conceal(frame_samples)
                return (concealed, True)

            expected_seq = (self.last_popped_unwrapped_seq + 1) if self.last_popped_unwrapped_seq is not None else self._queue[0][0]
            next_unwrapped_seq, packet = self._queue[0]

            # Detect large sequence gaps (burst loss / pause) beyond buffer capacity: re-sync immediately
            max_gap = max(4, int(math.ceil(self.max_delay_ms / 16.0)))
            if next_unwrapped_seq - expected_seq > max_gap:
                self.total_lost_packets += (next_unwrapped_seq - expected_seq)
                self.last_popped_unwrapped_seq = next_unwrapped_seq - 1
                expected_seq = next_unwrapped_seq

            if next_unwrapped_seq > expected_seq:
                # Missing packet detected in sequence gap: synthesize PLC
                self.total_lost_packets += 1
                self.total_plc_frames += 1
                self.last_popped_unwrapped_seq = expected_seq
                concealed = self.plc.conceal(frame_samples)
                return (concealed, True)

            # In-order packet ready
            heapq.heappop(self._queue)
            self.last_popped_unwrapped_seq = next_unwrapped_seq
            self.total_packets_popped += 1

            # Convert bytes to float32
            int16_data = np.frombuffer(packet.payload, dtype=np.int16)
            pcm_float = int16_data.astype(np.float32) / 32767.0

            # Match exact requested frame size
            if len(pcm_float) < frame_samples:
                # Pad short packet with zero
                out = np.zeros(frame_samples, dtype=np.float32)
                out[:len(pcm_float)] = pcm_float
                pcm_float = out
            elif len(pcm_float) > frame_samples:
                pcm_float = pcm_float[:frame_samples]

            self.plc.update_history(pcm_float)
            return (pcm_float, False)

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive jitter buffer and network telemetry."""
        with self._lock:
            q_len = len(self._queue)
            jitter_ms = self.jitter_estimator.get_jitter_ms()
            target_delay_ms = self.get_target_delay_ms()
            loss_rate = (
                round((self.total_lost_packets / max(1, self.total_packets_received + self.total_lost_packets)) * 100.0, 2)
            )

            return {
                "buffered_packets": q_len,
                "jitter_ms": jitter_ms,
                "target_delay_ms": target_delay_ms,
                "total_packets_received": self.total_packets_received,
                "total_packets_popped": self.total_packets_popped,
                "total_lost_packets": self.total_lost_packets,
                "total_plc_frames": self.total_plc_frames,
                "packet_loss_rate_pct": loss_rate,
            }

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self.jitter_estimator.reset()
            self.plc.reset()
            self.last_popped_unwrapped_seq = None
            self.max_unwrapped_seq = None
            self.total_packets_received = 0
            self.total_packets_popped = 0
            self.total_lost_packets = 0
            self.total_plc_frames = 0


class WebRTCAudioBridge:
    """Manages WebRTC audio track streaming, adaptive jitter buffering, and neural denoising."""

    def __init__(self, sample_rate: int = 16000, hop_length: int = 256) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.jitter_buffer = AdaptiveJitterBuffer(sample_rate=sample_rate)
        self.pipeline = AudioDenoisingPipeline(n_fft=512, hop_length=hop_length, sample_rate=sample_rate)
        self.profiler = StageProfiler(budget_ms=20.0)
        self.egress_seq = 0
        self.egress_ts = 0
        self._lock = threading.RLock()

    def ingest_rtp_packet(self, packet: RTPPacket) -> None:
        """Push an incoming RTP packet into the jitter buffer."""
        self.jitter_buffer.push_packet(packet)

    def process_next_frame(self) -> Tuple[RTPPacket, Dict[str, Any]]:
        """Pop the next audio frame from the jitter buffer, denoise, and packetize."""
        with self._lock:
            # 1. Pop from jitter buffer (with automatic PLC if missing/delayed)
            in_frame, was_concealed = self.jitter_buffer.pop_frame(self.hop_length)

            # 2. Denoise through neural DSP pipeline
            self.profiler.start_frame()
            clean_frame = self.pipeline.process_frame(in_frame, profiler=self.profiler)
            t_pre, t_tensor, t_synth, t_total = self.profiler.mark_synthesis_done()

            # 3. Formulate egress RTP packet
            self.egress_seq = (self.egress_seq + 1) & 0xFFFF
            self.egress_ts = (self.egress_ts + self.hop_length) & 0xFFFFFFFF

            clean_int16 = np.int16(np.clip(clean_frame, -1.0, 1.0) * 32767.0)
            egress_packet = RTPPacket(
                seq=self.egress_seq,
                timestamp=self.egress_ts,
                payload=clean_int16.tobytes(),
                arrival_time_s=time.perf_counter(),
                payload_type=97,
            )

            # 4. Compute acoustic metrics
            rms_in = float(np.sqrt(np.mean(in_frame**2)))
            rms_out = float(np.sqrt(np.mean(clean_frame**2)))
            snr_gain = (
                round(10.0 * math.log10(max(1.0, (rms_in**2) / (rms_out**2 + 1e-9))), 2)
                if rms_out > 1e-6
                else 0.0
            )

            jb_stats = self.jitter_buffer.get_stats()
            telemetry = {
                "was_concealed": was_concealed,
                "stages_ms": {
                    "pre_ms": round(t_pre, 3),
                    "tensor_ms": round(t_tensor, 3),
                    "synth_ms": round(t_synth, 3),
                    "total_ms": round(t_total, 3),
                },
                "network": jb_stats,
                "snr_gain_db": snr_gain,
                "precision": self.pipeline.get_precision(),
            }
            return (egress_packet, telemetry)

    def reset(self) -> None:
        with self._lock:
            self.jitter_buffer.reset()
            self.pipeline.reset()
            self.profiler.reset()
            self.egress_seq = 0
            self.egress_ts = 0


class SDPHandler:
    """Handles RFC 3264 / RFC 8829 Session Description Protocol (SDP) negotiation."""

    @staticmethod
    def create_answer(offer_sdp: str) -> str:
        """Generate an RFC-compliant SDP answer matching the peer's audio offer."""
        session_id = str(int(time.time()))

        # Extract mid tag from offer if present, default to 'audio' per RFC 8829 Section 5.3
        mid = "audio"
        for line in offer_sdp.splitlines():
            line_str = line.strip()
            if line_str.startswith("a=mid:"):
                mid = line_str.split(":", 1)[1].strip()
                break

        lines = [
            "v=0",
            f"o=- {session_id} 2 IN IP4 127.0.0.1",
            "s=Edge AI Neural Audio Denoiser WebRTC Session",
            "t=0 0",
            f"a=group:BUNDLE {mid}",
            "m=audio 9 UDP/TLS/RTP/SAVPF 97 111 0",
            "c=IN IP4 127.0.0.1",
            "a=rtcp:9 IN IP4 127.0.0.1",
            "a=sendrecv",
            "a=rtpmap:97 L16/16000/1",
            "a=rtpmap:111 opus/48000/2",
            "a=rtpmap:0 PCMU/8000",
            "a=fmtp:111 minptime=10;useinbandfec=1",
            f"a=mid:{mid}",
        ]
        return "\r\n".join(lines) + "\r\n"


# Singleton instance for active WebRTC peer session
webrtc_bridge = WebRTCAudioBridge(sample_rate=16000, hop_length=256)

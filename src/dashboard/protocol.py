"""High-Throughput Binary WebSocket Audio Protocol (ADEN Frame Specification).

Authority: Task 2 Architecture Specification (Phase 6 Real-Time Ingress/Egress).

Packet Specifications:
1. Ingress Audio Frame (Client -> Server: 1,040 Bytes):
   - Offset 0..3:   Magic bytes: b"ADEN"
   - Offset 4:      Protocol Version: 0x01 (uint8)
   - Offset 5:      Message Type: 0x01 (uint8 = Ingress PCM)
   - Offset 6..7:   Sample Count: 256 (uint16, little-endian)
   - Offset 8..11:  Sequence Number: uint32 (little-endian)
   - Offset 12..15: Target Language: char[4] ASCII (e.g. "hi\\0\\0", "en\\0\\0")
   - Offset 16..1039: 256 samples IEEE 754 Float32 little-endian PCM (-1.0 to 1.0)

2. Egress Audio & Telemetry Frame (Server -> Client: 3,872 Bytes):
   - Offset 0..3:   Magic bytes: b"ADEN"
   - Offset 4:      Protocol Version: 0x01 (uint8)
   - Offset 5:      Message Type: 0x02 (uint8 = Egress Audio & Visualizer)
   - Offset 6..7:   Sample Count: 256 (uint16, little-endian)
   - Offset 8..11:  Sequence Number: uint32 (little-endian)
   - Offset 12..15: Total Latency: float32 (ms)
   - Offset 16..19: SNR Gain: float32 (dB)
   - Offset 20..23: Noise Erased: float32 (%)
   - Offset 24..27: Purity Score: float32 (%)
   - Offset 28..31: OVRL MOS: float32 [1.0, 5.0]
   - Offset 32..1055: Denoised PCM (256 floats = 1024 B)
   - Offset 1056..2079: Raw Noisy PCM (256 floats = 1024 B)
   - Offset 2080..3103: Subtracted Noise PCM (256 floats = 1024 B)
   - Offset 3104..3359: Ingress Spectrogram 64 visualizer bands (64 floats = 256 B)
   - Offset 3360..3615: Egress Spectrogram 64 visualizer bands (64 floats = 256 B)
   - Offset 3616..3871: Gain Mask 64 visualizer bands (64 floats = 256 B)
"""

from __future__ import annotations
import struct
from typing import Tuple, List, Sequence
import numpy as np

ADEN_MAGIC = b"ADEN"
PROTOCOL_VERSION = 1
MSG_INGRESS_AUDIO = 1
MSG_EGRESS_AUDIO = 2

INGRESS_HEADER_FORMAT = "<4sBBHII"  # magic(4), ver(1), type(1), samples(2), seq(4), lang_bytes(4)
INGRESS_HEADER_SIZE = 16
INGRESS_FRAME_SAMPLES = 256
INGRESS_PCM_BYTES = INGRESS_FRAME_SAMPLES * 4
INGRESS_TOTAL_SIZE = INGRESS_HEADER_SIZE + INGRESS_PCM_BYTES  # 1040 bytes

EGRESS_HEADER_FORMAT = "<4sBBHIfffff"  # magic(4), ver(1), type(1), samples(2), seq(4), lat(4), snr(4), erased(4), purity(4), mos(4)
EGRESS_HEADER_SIZE = 32
EGRESS_AUDIO_CHANNELS_BYTES = 3 * (INGRESS_FRAME_SAMPLES * 4)  # denoised, raw_noisy, noise_subtracted = 3072 B
EGRESS_SPEC_BANDS_BYTES = 3 * (64 * 4)  # spec_in, spec_out, gain_mask = 768 B
EGRESS_TOTAL_SIZE = EGRESS_HEADER_SIZE + EGRESS_AUDIO_CHANNELS_BYTES + EGRESS_SPEC_BANDS_BYTES  # 3872 bytes


def parse_ingress_binary(raw_bytes: bytes) -> Tuple[np.ndarray, int, str]:
    """Parse a binary ADEN ingress audio frame.
    
    Parameters
    ----------
    raw_bytes : bytes
        Raw binary payload received over WebSocket.
        
    Returns
    -------
    Tuple[np.ndarray, int, str]
        (pcm_data: Float32Array[256], sequence_number: int, target_lang: str)
        
    Raises
    ------
    ValueError
        If payload size, magic bytes, or version are invalid.
    """
    if len(raw_bytes) < INGRESS_TOTAL_SIZE:
        raise ValueError(
            f"Binary payload truncated: got {len(raw_bytes)} bytes, expected {INGRESS_TOTAL_SIZE}"
        )

    magic, ver, msg_type, num_samples, seq, lang_raw = struct.unpack_from(
        "<4sBBHII", raw_bytes, 0
    )

    if magic != ADEN_MAGIC:
        raise ValueError(f"Invalid binary magic bytes: {magic!r}, expected {ADEN_MAGIC!r}")
    if ver != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {ver}, expected {PROTOCOL_VERSION}")
    if msg_type != MSG_INGRESS_AUDIO:
        raise ValueError(f"Invalid message type: {msg_type}, expected {MSG_INGRESS_AUDIO}")
    if num_samples != INGRESS_FRAME_SAMPLES:
        raise ValueError(f"Unexpected sample count: {num_samples}, expected {INGRESS_FRAME_SAMPLES}")

    # Decode 4-byte language code
    lang_str = struct.pack("<I", lang_raw).decode("ascii", errors="ignore").rstrip("\x00") or "hi"

    # Zero-copy float32 array construction
    pcm = np.frombuffer(raw_bytes, dtype=np.float32, count=INGRESS_FRAME_SAMPLES, offset=INGRESS_HEADER_SIZE)
    # Ensure writeable copy if downstream modifies in-place
    if not pcm.flags.writeable:
        pcm = pcm.copy()

    return pcm, seq, lang_str


def pack_egress_binary(
    seq: int,
    total_latency_ms: float,
    snr_delta_db: float,
    noise_erased_pct: float,
    purity_score: float,
    ovrl_mos: float,
    denoised_pcm: np.ndarray,
    raw_noisy_pcm: np.ndarray,
    diff_pcm: np.ndarray,
    in_vis: Sequence[float],
    out_vis: Sequence[float],
    mask_vis: Sequence[float],
) -> bytes:
    """Pack an egress audio and visualizer frame into a high-throughput binary ADEN buffer.
    
    Parameters
    ----------
    seq : int
        Monotonic sequence number.
    total_latency_ms : float
        End-to-end frame processing latency.
    snr_delta_db : float
        Measured SNR improvement.
    noise_erased_pct : float
        Noise suppressed percentage [0.0, 99.9].
    purity_score : float
        Purity gauge score [50.0, 100.0].
    ovrl_mos : float
        Overall perceptual MOS [1.0, 5.0].
    denoised_pcm : np.ndarray
        256-sample cleaned audio.
    raw_noisy_pcm : np.ndarray
        256-sample input noisy audio.
    diff_pcm : np.ndarray
        256-sample subtracted noise audio.
    in_vis : Sequence[float]
        64-band input visualizer magnitudes.
    out_vis : Sequence[float]
        64-band output visualizer magnitudes.
    mask_vis : Sequence[float]
        64-band gain mask bands.

    Returns
    -------
    bytes
        Packed 3,872-byte binary payload ready for zero-copy transmission.
    """
    header = struct.pack(
        EGRESS_HEADER_FORMAT,
        ADEN_MAGIC,
        PROTOCOL_VERSION,
        MSG_EGRESS_AUDIO,
        INGRESS_FRAME_SAMPLES,
        seq,
        float(total_latency_ms),
        float(snr_delta_db),
        float(noise_erased_pct),
        float(purity_score),
        float(ovrl_mos),
    )

    d_bytes = np.asarray(denoised_pcm[:256], dtype=np.float32).tobytes()
    n_bytes = np.asarray(raw_noisy_pcm[:256], dtype=np.float32).tobytes()
    diff_bytes = np.asarray(diff_pcm[:256], dtype=np.float32).tobytes()

    in_v_bytes = np.asarray(in_vis[:64], dtype=np.float32).tobytes()
    out_v_bytes = np.asarray(out_vis[:64], dtype=np.float32).tobytes()
    m_v_bytes = np.asarray(mask_vis[:64], dtype=np.float32).tobytes()

    return header + d_bytes + n_bytes + diff_bytes + in_v_bytes + out_v_bytes + m_v_bytes

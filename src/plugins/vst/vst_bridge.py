"""VST3 / CLAP Audio Plugin Bridge & Variable Block Size Processing Engine.

Provides an industry-standard DAW plugin adapter for the Real-Time Edge AI Audio Denoiser:
- Adapts arbitrary DAW host block sizes (32, 64, 128, 256, 512, 1024 samples) via a
  double-buffered circular FIFO to the fixed 256-sample (16.0 ms) STFT frame hop.
- Parameter automation interface matching VST3 / CLAP specifications:
  * kParamBypass (0 = Active, 1 = Bypassed)
  * kParamDenoiseAmount (0.0 to 1.0 wet/dry ratio)
  * kParamModelSelect (0: Neural GRUMaskNet, 1: Wiener DSP, 2: Dual Crossfade)
  * kParamCrossfade (0.0 to 1.0 Model A vs Model B blend)
  * kParamPrecision (0: FP32, 1: FP16, 2: INT8)
- Sample-accurate Plugin Delay Compensation (PDC) reporting (256 samples / 16.0 ms at 16 kHz).
- Continuous parameter smoothing avoiding audible clicks and zipper noise during automation.
- C ABI interface helper and mock export hooks for JUCE / nih-plug integration.
"""

from __future__ import annotations
import ctypes
import threading
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.audio.dual_pipeline import DualModelPipeline

# Standard VST3 / CLAP Parameter IDs
PARAM_BYPASS = "kParamBypass"
PARAM_DENOISE_AMOUNT = "kParamDenoiseAmount"
PARAM_MODEL_SELECT = "kParamModelSelect"
PARAM_CROSSFADE = "kParamCrossfade"
PARAM_PRECISION = "kParamPrecision"

PRECISION_MAP = {0: "FP32", 1: "FP16", 2: "INT8"}
REVERSE_PRECISION_MAP = {"FP32": 0, "FP16": 1, "INT8": 2}


class AudioCircularFIFO:
    """Thread-safe single-channel circular ring buffer for variable block size audio adaptation."""

    def __init__(self, capacity: int = 8192) -> None:
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.read_pos = 0
        self.write_pos = 0
        self.count = 0
        self._lock = threading.Lock()

    def available_read(self) -> int:
        """Return number of samples available to read."""
        with self._lock:
            return self.count

    def available_write(self) -> int:
        """Return remaining space available for writing."""
        with self._lock:
            return self.capacity - self.count

    def write(self, data: np.ndarray) -> int:
        """Write 1D float32 samples into the circular buffer. Returns samples written."""
        arr = np.asarray(data, dtype=np.float32).ravel()
        n = len(arr)
        if n == 0:
            return 0
        with self._lock:
            to_write = min(n, self.capacity - self.count)
            if to_write == 0:
                return 0

            first_chunk = min(to_write, self.capacity - self.write_pos)
            self.buffer[self.write_pos : self.write_pos + first_chunk] = arr[:first_chunk]
            second_chunk = to_write - first_chunk
            if second_chunk > 0:
                self.buffer[:second_chunk] = arr[first_chunk:to_write]

            self.write_pos = (self.write_pos + to_write) % self.capacity
            self.count += to_write
            return to_write

    def read(self, num_samples: int) -> np.ndarray:
        """Read and advance read pointer by num_samples. Pads with zeros if underrun."""
        out = np.zeros(num_samples, dtype=np.float32)
        with self._lock:
            to_read = min(num_samples, self.count)
            if to_read == 0:
                return out

            first_chunk = min(to_read, self.capacity - self.read_pos)
            out[:first_chunk] = self.buffer[self.read_pos : self.read_pos + first_chunk]
            second_chunk = to_read - first_chunk
            if second_chunk > 0:
                out[first_chunk:to_read] = self.buffer[:second_chunk]

            self.read_pos = (self.read_pos + to_read) % self.capacity
            self.count -= to_read
            return out

    def peek(self, num_samples: int) -> np.ndarray:
        """Read num_samples without advancing the read pointer."""
        out = np.zeros(num_samples, dtype=np.float32)
        with self._lock:
            to_read = min(num_samples, self.count)
            if to_read == 0:
                return out

            pos = self.read_pos
            first_chunk = min(to_read, self.capacity - pos)
            out[:first_chunk] = self.buffer[pos : pos + first_chunk]
            second_chunk = to_read - first_chunk
            if second_chunk > 0:
                out[first_chunk:to_read] = self.buffer[:second_chunk]
            return out

    def prefill(self, num_samples: int, value: float = 0.0) -> None:
        """Prefill buffer with constant samples (used for PDC latency alignment)."""
        samples = np.full(num_samples, value, dtype=np.float32)
        self.write(samples)

    def clear(self) -> None:
        """Reset buffer pointers and zero memory."""
        with self._lock:
            self.buffer.fill(0.0)
            self.read_pos = 0
            self.write_pos = 0
            self.count = 0


class VST3PluginProcessor:
    """VST3 / CLAP Audio Plugin Bridge Processor.

    Bridges DAW variable host block sizes (32..1024) to the fixed 256-sample STFT hop size
    with full parameter automation, linear smoothing, sample-accurate PDC, and bypass support.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 256,
        n_fft: int = 512,
        fifo_capacity: int = 8192,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)

        # Underlying dual-model speech enhancement engine
        self.pipeline = DualModelPipeline(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            crossfade_alpha=0.5,
        )

        # FIFOs for block adaptation
        self.input_fifo = AudioCircularFIFO(capacity=fifo_capacity)
        self.output_fifo = AudioCircularFIFO(capacity=fifo_capacity)

        # Parameters
        self.params: Dict[str, float] = {
            PARAM_BYPASS: 0.0,            # 0.0: Active, 1.0: Bypassed
            PARAM_DENOISE_AMOUNT: 1.0,     # 0.0 (dry) to 1.0 (wet)
            PARAM_MODEL_SELECT: 0.0,       # 0: Neural, 1: Wiener DSP, 2: Dual Crossfade
            PARAM_CROSSFADE: 0.5,          # 0.0 (A) to 1.0 (B)
            PARAM_PRECISION: 0.0,          # 0: FP32, 1: FP16, 2: INT8
        }

        # Parameter smoothing state (prevents clicks during automation)
        self._smoothed_denoise_amount = 1.0
        self._smoothed_crossfade = 0.5
        self.smoothing_alpha = 0.90  # One-pole low-pass coefficient per hop

        # Latency & Delay Compensation (PDC)
        # STFT overlap-add introduces exactly 1 hop_length latency (256 samples = 16.0 ms)
        self.latency_samples = self.hop_length
        self._prefill_pdc_latency()

        # Telemetry
        self.total_blocks_processed = 0
        self.total_samples_processed = 0

    def _prefill_pdc_latency(self) -> None:
        """Prefill output FIFO with latency samples to ensure zero-starvation block returns."""
        self.output_fifo.prefill(self.latency_samples, 0.0)

    def get_latency_samples(self) -> int:
        """Return algorithmic latency in samples for DAW Plugin Delay Compensation (PDC)."""
        return self.latency_samples

    def get_latency_ms(self) -> float:
        """Return algorithmic latency in milliseconds."""
        return (self.latency_samples / self.sample_rate) * 1000.0

    def set_parameter(self, param_id: str, value: float) -> None:
        """Update an automated parameter value."""
        if param_id == PARAM_BYPASS:
            self.params[PARAM_BYPASS] = 1.0 if bool(round(value)) else 0.0
        elif param_id == PARAM_DENOISE_AMOUNT:
            self.params[PARAM_DENOISE_AMOUNT] = float(np.clip(value, 0.0, 1.0))
        elif param_id == PARAM_MODEL_SELECT:
            self.params[PARAM_MODEL_SELECT] = float(np.clip(int(round(value)), 0, 2))
        elif param_id == PARAM_CROSSFADE:
            self.params[PARAM_CROSSFADE] = float(np.clip(value, 0.0, 1.0))
        elif param_id == PARAM_PRECISION:
            prec_idx = int(np.clip(int(round(value)), 0, 2))
            self.params[PARAM_PRECISION] = float(prec_idx)
            prec_str = PRECISION_MAP.get(prec_idx, "FP32")
            self.pipeline.set_precision(prec_str)
        else:
            raise KeyError(f"Unknown parameter ID: {param_id}")

    def get_parameter(self, param_id: str) -> float:
        """Retrieve current value of an automated parameter."""
        if param_id not in self.params:
            raise KeyError(f"Unknown parameter ID: {param_id}")
        return self.params[param_id]

    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter metadata matching VST3 / CLAP parameter specifications."""
        return {
            PARAM_BYPASS: {
                "id": PARAM_BYPASS,
                "name": "Bypass",
                "type": "bool",
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
            },
            PARAM_DENOISE_AMOUNT: {
                "id": PARAM_DENOISE_AMOUNT,
                "name": "Denoise Amount",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 1.0,
            },
            PARAM_MODEL_SELECT: {
                "id": PARAM_MODEL_SELECT,
                "name": "Model Select",
                "type": "choice",
                "options": ["Neural GRUMaskNet", "Wiener DSP", "Dual Crossfade"],
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
            },
            PARAM_CROSSFADE: {
                "id": PARAM_CROSSFADE,
                "name": "Crossfade Blend",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
            },
            PARAM_PRECISION: {
                "id": PARAM_PRECISION,
                "name": "Quantization Precision",
                "type": "choice",
                "options": ["FP32", "FP16", "INT8"],
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
            },
        }

    def reset(self) -> None:
        """Reset internal DSP states and FIFOs (called on DAW transport restart or sample rate change)."""
        self.input_fifo.clear()
        self.output_fifo.clear()
        self._prefill_pdc_latency()
        self.pipeline.reset()
        self._smoothed_denoise_amount = self.params[PARAM_DENOISE_AMOUNT]
        self._smoothed_crossfade = self.params[PARAM_CROSSFADE]
        self.total_blocks_processed = 0
        self.total_samples_processed = 0

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process an arbitrary DAW host audio buffer block.

        Parameters
        ----------
        input_samples : np.ndarray
            1D float32 audio array of length N (e.g. 32, 64, 128, 256, 512, 1024).

        Returns
        -------
        np.ndarray
            1D float32 processed audio array of identical length N.
        """
        arr = np.asarray(input_samples, dtype=np.float32).ravel()
        block_len = len(arr)
        if block_len == 0:
            return np.zeros(0, dtype=np.float32)

        # 1. Push incoming block to input FIFO
        self.input_fifo.write(arr)

        # 2. Process all complete frames of hop_length
        while self.input_fifo.available_read() >= self.hop_length:
            frame_in = self.input_fifo.read(self.hop_length)

            # Smooth parameters per frame to eliminate zipper noise
            target_amount = self.params[PARAM_DENOISE_AMOUNT]
            target_crossfade = self.params[PARAM_CROSSFADE]
            self._smoothed_denoise_amount = (
                self.smoothing_alpha * self._smoothed_denoise_amount
                + (1.0 - self.smoothing_alpha) * target_amount
            )
            self._smoothed_crossfade = (
                self.smoothing_alpha * self._smoothed_crossfade
                + (1.0 - self.smoothing_alpha) * target_crossfade
            )

            is_bypassed = bool(round(self.params[PARAM_BYPASS]))

            if is_bypassed:
                # In bypass mode, input frame is passed through PDC delay cleanly
                # Also keep pipeline state updated
                _ = self.pipeline.process_frame(frame_in, crossfade_alpha=self._smoothed_crossfade)
                frame_out = frame_in
            else:
                model_sel = int(round(self.params[PARAM_MODEL_SELECT]))
                if model_sel == 0:
                    # Model A: Neural GRUMaskNet
                    res = self.pipeline.process_frame(frame_in, crossfade_alpha=0.0)
                    wet = res.audio_a
                elif model_sel == 1:
                    # Model B: Wiener DSP
                    res = self.pipeline.process_frame(frame_in, crossfade_alpha=1.0)
                    wet = res.audio_b
                else:
                    # 2: Dual Crossfade
                    res = self.pipeline.process_frame(frame_in, crossfade_alpha=self._smoothed_crossfade)
                    wet = res.audio_mix

                # Wet/Dry mix with smoothed denoise amount
                frame_out = (1.0 - self._smoothed_denoise_amount) * frame_in + self._smoothed_denoise_amount * wet

            # Write processed frame to output FIFO
            self.output_fifo.write(frame_out)

        # 3. Read exactly block_len samples from output FIFO
        out_block = self.output_fifo.read(block_len)

        self.total_blocks_processed += 1
        self.total_samples_processed += block_len
        return out_block


# ============================================================================
# C ABI Export Interface for JUCE & nih-plug Wrappers
# ============================================================================

_INSTANCES: Dict[int, VST3PluginProcessor] = {}
_NEXT_HANDLE: int = 1


def create_vst_instance(sample_rate: int = 16000) -> int:
    """Create a new VST3 processor instance and return unique integer handle."""
    global _NEXT_HANDLE
    handle = _NEXT_HANDLE
    _NEXT_HANDLE += 1
    _INSTANCES[handle] = VST3PluginProcessor(sample_rate=sample_rate)
    return handle


def destroy_vst_instance(handle: int) -> int:
    """Destroy VST3 processor instance by handle. Returns 0 on success, -1 if not found."""
    if handle in _INSTANCES:
        del _INSTANCES[handle]
        return 0
    return -1


def vst_set_param(handle: int, param_id: str, value: float) -> int:
    """Set parameter on instance by handle. Returns 0 on success, -1 on failure."""
    if handle not in _INSTANCES:
        return -1
    try:
        _INSTANCES[handle].set_parameter(param_id, value)
        return 0
    except KeyError:
        return -1


def vst_get_param(handle: int, param_id: str) -> float:
    """Get parameter from instance by handle. Returns -1.0 if not found."""
    if handle not in _INSTANCES:
        return -1.0
    try:
        return _INSTANCES[handle].get_parameter(param_id)
    except KeyError:
        return -1.0


def vst_get_latency(handle: int) -> int:
    """Get sample latency for PDC. Returns -1 if not found."""
    if handle not in _INSTANCES:
        return -1
    return _INSTANCES[handle].get_latency_samples()


def vst_process_buffer(handle: int, in_samples: np.ndarray) -> Optional[np.ndarray]:
    """Process an audio buffer through the instance. Returns None if invalid handle."""
    if handle not in _INSTANCES:
        return None
    return _INSTANCES[handle].process_block(in_samples)


def vst_reset_instance(handle: int) -> int:
    """Reset instance state. Returns 0 on success, -1 if not found."""
    if handle not in _INSTANCES:
        return -1
    _INSTANCES[handle].reset()
    return 0

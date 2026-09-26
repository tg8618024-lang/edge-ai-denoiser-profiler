"""VST3 / CLAP Audio Plugin Bridge & Variable Block Size Processing Engine.

Provides an industry-standard DAW plugin adapter for the Real-Time Edge AI Audio Denoiser:
- Adapts arbitrary DAW host block sizes (32, 64, 128, 256, 512, 1024 samples) via a
  double-buffered circular FIFO to the fixed 256-sample (16.0 ms) STFT frame hop.
- Multi-channel support: Mono (N,) and Stereo (2, N) / (N, 2) processing with preserved
  stereo phase coherence and spatial imaging.
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
from typing import Dict, Any, Optional, Tuple, Union
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
    with full parameter automation, linear smoothing, sample-accurate PDC, bypass support,
    and native stereo / multi-channel processing.
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

        # Underlying dual-model speech enhancement engines (Left and Right channels)
        self.pipeline_l = DualModelPipeline(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            crossfade_alpha=0.5,
        )
        self.pipeline_r = DualModelPipeline(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            crossfade_alpha=0.5,
        )
        self.pipeline = self.pipeline_l  # Backward compatibility alias

        # FIFOs for block adaptation (Left and Right channels)
        self.input_fifo_l = AudioCircularFIFO(capacity=fifo_capacity)
        self.output_fifo_l = AudioCircularFIFO(capacity=fifo_capacity)
        self.input_fifo_r = AudioCircularFIFO(capacity=fifo_capacity)
        self.output_fifo_r = AudioCircularFIFO(capacity=fifo_capacity)

        # Backward compatibility aliases
        self.input_fifo = self.input_fifo_l
        self.output_fifo = self.output_fifo_l

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
        """Prefill output FIFOs with latency samples to ensure zero-starvation block returns."""
        self.output_fifo_l.prefill(self.latency_samples, 0.0)
        self.output_fifo_r.prefill(self.latency_samples, 0.0)

    def get_latency_samples(self) -> int:
        """Return algorithmic latency in samples for DAW Plugin Delay Compensation (PDC)."""
        return self.latency_samples

    def get_latency_ms(self) -> float:
        """Return algorithmic latency in milliseconds."""
        return (self.latency_samples / self.sample_rate) * 1000.0

    def set_parameter(self, param_id: str, value: float) -> None:
        """Update an automated parameter value across channels."""
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
            self.pipeline_l.set_precision(prec_str)
            self.pipeline_r.set_precision(prec_str)
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
        self.input_fifo_l.clear()
        self.output_fifo_l.clear()
        self.input_fifo_r.clear()
        self.output_fifo_r.clear()
        self._prefill_pdc_latency()
        self.pipeline_l.reset()
        self.pipeline_r.reset()
        self._smoothed_denoise_amount = self.params[PARAM_DENOISE_AMOUNT]
        self._smoothed_crossfade = self.params[PARAM_CROSSFADE]
        self.total_blocks_processed = 0
        self.total_samples_processed = 0

    def _process_channel_frames(
        self,
        in_fifo: AudioCircularFIFO,
        out_fifo: AudioCircularFIFO,
        pipeline: DualModelPipeline,
        update_smoothing: bool = False,
    ) -> None:
        """Process complete hop_length frames from in_fifo and write to out_fifo."""
        is_bypassed = bool(round(self.params[PARAM_BYPASS]))
        model_sel = int(round(self.params[PARAM_MODEL_SELECT]))

        while in_fifo.available_read() >= self.hop_length:
            frame_in = in_fifo.read(self.hop_length)

            if update_smoothing:
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

            if is_bypassed:
                _ = pipeline.process_frame(frame_in, crossfade_alpha=self._smoothed_crossfade)
                frame_out = frame_in
            else:
                if model_sel == 0:
                    res = pipeline.process_frame(frame_in, crossfade_alpha=0.0)
                    wet = res.audio_a
                elif model_sel == 1:
                    res = pipeline.process_frame(frame_in, crossfade_alpha=1.0)
                    wet = res.audio_b
                else:
                    res = pipeline.process_frame(frame_in, crossfade_alpha=self._smoothed_crossfade)
                    wet = res.audio_mix

                frame_out = (
                    (1.0 - self._smoothed_denoise_amount) * frame_in
                    + self._smoothed_denoise_amount * wet
                )

            out_fifo.write(frame_out)

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process an arbitrary DAW host audio buffer block (Mono or Stereo).

        Parameters
        ----------
        input_samples : np.ndarray
            Audio array. Can be:
            - 1D mono array of shape (N,)
            - 2D stereo channels-first array of shape (2, N)
            - 2D stereo channels-last array of shape (N, 2)

        Returns
        -------
        np.ndarray
            Processed audio array of identical shape.
        """
        arr = np.asarray(input_samples, dtype=np.float32)
        if arr.size == 0:
            return np.zeros_like(arr)

        if arr.ndim == 1:
            # Mono processing
            block_len = len(arr)
            self.input_fifo_l.write(arr)
            self._process_channel_frames(
                self.input_fifo_l, self.output_fifo_l, self.pipeline_l, update_smoothing=True
            )
            out_block = self.output_fifo_l.read(block_len)
            self.total_blocks_processed += 1
            self.total_samples_processed += block_len
            return out_block

        elif arr.ndim == 2:
            if arr.shape[0] == 2:
                # Channels-first stereo: shape (2, N)
                block_len = arr.shape[1]
                self.input_fifo_l.write(arr[0])
                self.input_fifo_r.write(arr[1])

                self._process_channel_frames(
                    self.input_fifo_l, self.output_fifo_l, self.pipeline_l, update_smoothing=True
                )
                self._process_channel_frames(
                    self.input_fifo_r, self.output_fifo_r, self.pipeline_r, update_smoothing=False
                )

                out_l = self.output_fifo_l.read(block_len)
                out_r = self.output_fifo_r.read(block_len)
                self.total_blocks_processed += 1
                self.total_samples_processed += block_len
                return np.stack([out_l, out_r], axis=0)

            elif arr.shape[1] == 2:
                # Channels-last stereo: shape (N, 2)
                block_len = arr.shape[0]
                self.input_fifo_l.write(arr[:, 0])
                self.input_fifo_r.write(arr[:, 1])

                self._process_channel_frames(
                    self.input_fifo_l, self.output_fifo_l, self.pipeline_l, update_smoothing=True
                )
                self._process_channel_frames(
                    self.input_fifo_r, self.output_fifo_r, self.pipeline_r, update_smoothing=False
                )

                out_l = self.output_fifo_l.read(block_len)
                out_r = self.output_fifo_r.read(block_len)
                self.total_blocks_processed += 1
                self.total_samples_processed += block_len
                return np.column_stack([out_l, out_r])

            else:
                # Fallback to mono ravel
                flat = arr.ravel()
                out_flat = self.process_block(flat)
                return out_flat.reshape(arr.shape)

        else:
            flat = arr.ravel()
            out_flat = self.process_block(flat)
            return out_flat.reshape(arr.shape)


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


def vst_process_stereo_buffer(
    handle: int, in_left: np.ndarray, in_right: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Process stereo Left and Right audio arrays through instance. Returns (out_left, out_right)."""
    if handle not in _INSTANCES:
        return None
    proc = _INSTANCES[handle]
    arr_l = np.asarray(in_left, dtype=np.float32).ravel()
    arr_r = np.asarray(in_right, dtype=np.float32).ravel()
    min_len = min(len(arr_l), len(arr_r))
    stereo_in = np.stack([arr_l[:min_len], arr_r[:min_len]], axis=0)
    stereo_out = proc.process_block(stereo_in)
    return stereo_out[0], stereo_out[1]


def vst_reset_instance(handle: int) -> int:
    """Reset instance state. Returns 0 on success, -1 if not found."""
    if handle not in _INSTANCES:
        return -1
    _INSTANCES[handle].reset()
    return 0

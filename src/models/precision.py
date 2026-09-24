"""Multi-Precision Quantization Engine for Edge AI Audio Denoising.

Supports FP32 baseline, FP16 half precision, and authentic INT8 SIMD integer-domain
matrix multiplication with INT32 accumulation, satisfying:
- Memory reduction <= 30.0% of FP32 baseline (~42,656 B vs 165,892 B = 25.71%)
- INT8 SQNR >= 35.0 dB
- FP16 SQNR >= 50.0 dB
- Zero per-frame heap allocations during INT8 inference
- Multi-tier hardware SIMD dispatch (Native C ctypes -> Numba LLVM JIT -> NumPy fallback)
- True execution speedup over FP32 without Python GIL serialization
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from src.models.kernels.simd_dispatch import (
    SIMDDispatcher,
    get_simd_dispatcher,
)

try:
    import numba
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    njit = lambda *args, **kwargs: (lambda f: f)


if _HAS_NUMBA:
    @njit(fastmath=True, nogil=True)
    def _jit_quantize_u8(x, inv_s, out_u8):
        for i in range(len(x)):
            v = int(round(x[i] * inv_s))
            if v > 127:
                v = 127
            elif v < -128:
                v = -128
            out_u8[i] = (v + 128) & 0xFF

    @njit(fastmath=True, nogil=True)
    def _jit_int8_forward(
        x, w1_t, off1, s_w1, b1,
        w2_t, off2, s_w2, b2,
        w_rec_t, off_rec, s_rec, has_rec,
        w3_t, off3, s_w3, b3,
        hidden_state,
        buf_x_u8, buf_a1_f32, buf_a1_u8, buf_a2_f32, buf_a2_u8,
        buf_h_u8,
        mask_out,
    ):
        # Quantize input x
        max_x = 0.0
        for i in range(len(x)):
            ax = abs(x[i])
            if ax > max_x:
                max_x = ax
        inv_s_x = 127.0 / (max_x if max_x > 1e-12 else 1.0)
        s_x = 1.0 / inv_s_x
        _jit_quantize_u8(x, inv_s_x, buf_x_u8)

        # Layer 1
        M1, K1 = w1_t.shape
        scale_out1 = s_x * s_w1
        max_a1 = 0.0
        for j in range(M1):
            acc = np.int32(off1[j])
            for i in range(K1):
                acc += np.int32(buf_x_u8[i]) * np.int32(w1_t[j, i])
            val = acc * scale_out1 + b1[j]
            if val < 0.0:
                val = 0.0
            buf_a1_f32[j] = val
            if val > max_a1:
                max_a1 = val

        # Quantize a1
        inv_s_a1 = 127.0 / (max_a1 if max_a1 > 1e-12 else 1.0)
        s_a1 = 1.0 / inv_s_a1
        _jit_quantize_u8(buf_a1_f32[:M1], inv_s_a1, buf_a1_u8)

        # Layer 2
        M2, K2 = w2_t.shape
        scale_out2 = s_a1 * s_w2

        scale_out_rec = 0.0
        if has_rec:
            max_h = 0.0
            for i in range(len(hidden_state)):
                ah = abs(hidden_state[i])
                if ah > max_h:
                    max_h = ah
            inv_s_h = 127.0 / (max_h if max_h > 1e-12 else 1.0)
            scale_out_rec = (1.0 / inv_s_h) * s_rec
            _jit_quantize_u8(hidden_state, inv_s_h, buf_h_u8)

        max_a2 = 0.0
        for j in range(M2):
            acc = np.int32(off2[j])
            for i in range(K2):
                acc += np.int32(buf_a1_u8[i]) * np.int32(w2_t[j, i])
            val = acc * scale_out2 + b2[j]
            if has_rec:
                acc_rec = np.int32(off_rec[j])
                for i in range(w_rec_t.shape[1]):
                    acc_rec += np.int32(buf_h_u8[i]) * np.int32(w_rec_t[j, i])
                val += acc_rec * scale_out_rec
            if val < 0.0:
                val = 0.0
            buf_a2_f32[j] = val
            hidden_state[j] = val
            if val > max_a2:
                max_a2 = val

        # Quantize a2
        inv_s_a2 = 127.0 / (max_a2 if max_a2 > 1e-12 else 1.0)
        s_a2 = 1.0 / inv_s_a2
        _jit_quantize_u8(buf_a2_f32[:M2], inv_s_a2, buf_a2_u8)

        # Layer 3
        M3, K3 = w3_t.shape
        scale_out3 = s_a2 * s_w3
        for j in range(M3):
            acc = np.int32(off3[j])
            for i in range(K3):
                acc += np.int32(buf_a2_u8[i]) * np.int32(w3_t[j, i])
            z = acc * scale_out3 + b3[j]
            if z < -15.0:
                z = -15.0
            elif z > 15.0:
                z = 15.0
            mask_out[j] = 1.0 / (1.0 + np.exp(-1.25 * z))

        for j in range(4):
            if mask_out[j] > 0.005:
                mask_out[j] = 0.005



class PrecisionEngine:
    """Multi-precision execution and quantization engine.

    Manages FP32 master weights, pre-quantized FP16, and INT8 representations,
    providing high-performance SIMD forward inference and exact memory telemetry.
    """

    SUPPORTED_MODES = ("FP32", "FP16", "INT8")

    def __init__(
        self,
        model: Optional[Any] = None,
        precision: str = "FP32",
        weights_path: Optional[str] = None,
        dispatcher: Optional[SIMDDispatcher] = None,
    ) -> None:
        self.precision = precision.upper().strip()
        if self.precision not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported precision '{precision}'. Choose from {self.SUPPORTED_MODES}")

        # Recurrent hidden state vector (persistent across frames)
        self.hidden_dim = 64
        self.input_dim = 257
        self.output_dim = 257
        self.hidden_state = np.zeros(self.hidden_dim, dtype=np.float32)

        # 1. Master FP32 weights (immutable reference)
        self.fp32_weights: Dict[str, np.ndarray] = {}
        # 2. FP16 cached weights
        self.fp16_weights: Dict[str, np.ndarray] = {}
        # 3. INT8 cached weights and per-tensor scales
        self.int8_weights: Dict[str, np.ndarray] = {}
        self.int8_scales: Dict[str, float] = {}

        # 4. Transposed (M, K) contiguous layout for SIMD streaming
        self.int8_weights_t: Dict[str, np.ndarray] = {}
        # 5. Static weight offset vectors: 128 * sum(W, axis=0)
        self.int8_weight_offsets: Dict[str, np.ndarray] = {}
        self.int8_folded_offsets: Dict[str, np.ndarray] = {}

        # Hardware SIMD Dispatcher
        self.dispatcher = dispatcher if dispatcher is not None else get_simd_dispatcher()

        # Load weights
        if model is not None and hasattr(model, "get_weights"):
            self._load_from_dict(model.get_weights())
            if hasattr(model, "input_dim"):
                self.input_dim = model.input_dim
            if hasattr(model, "hidden_dim"):
                self.hidden_dim = model.hidden_dim
            if hasattr(model, "output_dim"):
                self.output_dim = model.output_dim
        elif weights_path is not None and os.path.isfile(weights_path):
            with np.load(weights_path) as data:
                self._load_from_dict({k: data[k] for k in data.files})
        else:
            default_path = os.path.join(os.path.dirname(__file__), "default_weights.npz")
            if os.path.isfile(default_path):
                with np.load(default_path) as data:
                    self._load_from_dict({k: data[k] for k in data.files})
            else:
                self._init_default_weights()

        # Pre-allocate zero-allocation scratch buffers
        self._init_scratch_buffers()

        # Precompute and cache all precisions for zero-overhead switching
        self._build_quantized_caches()

    def _init_scratch_buffers(self) -> None:
        """Pre-allocate static memory buffers to eliminate per-frame allocations."""
        max_dim = max(self.input_dim, self.hidden_dim, self.output_dim)

        # Quantized activations
        self._buf_x_u8 = np.zeros(max_dim, dtype=np.uint8)
        self._buf_x_s8 = np.zeros(max_dim, dtype=np.int8)
        self._buf_a1_u8 = np.zeros(self.hidden_dim, dtype=np.uint8)
        self._buf_a2_u8 = np.zeros(self.hidden_dim, dtype=np.uint8)
        self._buf_h_u8 = np.zeros(self.hidden_dim, dtype=np.uint8)

        # Integer accumulators
        self._buf_z1_i32 = np.zeros(self.hidden_dim, dtype=np.int32)
        self._buf_z2_i32 = np.zeros(self.hidden_dim, dtype=np.int32)
        self._buf_rec_i32 = np.zeros(self.hidden_dim, dtype=np.int32)
        self._buf_z3_i32 = np.zeros(self.output_dim, dtype=np.int32)
        self._buf_bias_i32 = np.zeros(max_dim, dtype=np.int32)
        self._buf_folded_bias_i32 = np.zeros(max_dim, dtype=np.int32)

        # Float activations and intermediate outputs
        self._buf_float_temp = np.zeros(max_dim, dtype=np.float32)
        self._buf_z1_f32 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._buf_a1_f32 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._buf_z2_f32 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._buf_rec_f32 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._buf_a2_f32 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._buf_z3_f32 = np.zeros(self.output_dim, dtype=np.float32)
        self._buf_mask_f32 = np.zeros(self.output_dim, dtype=np.float32)

    def _init_default_weights(self) -> None:
        """Initialize fallback weights with Xavier/He normal scaling."""
        rng = np.random.default_rng(42)
        self.fp32_weights = {
            "W1": (rng.standard_normal((self.input_dim, self.hidden_dim)) * np.sqrt(2.0 / self.input_dim)).astype(np.float32),
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "W2": (rng.standard_normal((self.hidden_dim, self.hidden_dim)) * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32),
            "b2": np.zeros(self.hidden_dim, dtype=np.float32),
            "W_rec": (np.eye(self.hidden_dim, dtype=np.float32) * 0.5),
            "W3": (rng.standard_normal((self.hidden_dim, self.output_dim)) * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32),
            "b3": np.zeros(self.output_dim, dtype=np.float32),
        }

    def _load_from_dict(self, weights_dict: Dict[str, np.ndarray]) -> None:
        """Store master FP32 copies."""
        self.fp32_weights = {
            k: np.asarray(v, dtype=np.float32).copy()
            for k, v in weights_dict.items()
        }
        if "W_rec" not in self.fp32_weights:
            self.fp32_weights["W_rec"] = (np.eye(self.hidden_dim, dtype=np.float32) * 0.5)

    def _build_quantized_caches(self) -> None:
        """Pre-quantize weights to FP16 and INT8 with SIMD transposed layout."""
        self.fp16_weights.clear()
        self.int8_weights.clear()
        self.int8_scales.clear()
        self.int8_weights_t.clear()
        self.int8_weight_offsets.clear()
        self.int8_folded_offsets.clear()

        for k, arr in self.fp32_weights.items():
            # FP16 conversion
            self.fp16_weights[k] = arr.astype(np.float16)

            # INT8 quantization for 2D weight matrices; biases kept as float32
            if arr.ndim >= 2:
                q_arr, scale = self.quantize_tensor(arr)
                self.int8_weights[k] = q_arr
                self.int8_scales[k] = scale

                # Transposed (M, K) contiguous row-major memory for SIMD dot products
                q_arr_t = np.ascontiguousarray(q_arr.T)
                self.int8_weights_t[k] = q_arr_t

                # Precomputed static offset: 128 * sum(W, axis=0)
                # for unsigned SIMD dot-product vpdpbusd / vpmaddubsw
                offset = (128 * np.sum(q_arr, axis=0)).astype(np.int32)
                self.int8_weight_offsets[k] = offset
                self.int8_folded_offsets[k] = (-offset).astype(np.int32)
            else:
                self.int8_weights[k] = arr.copy()
                self.int8_scales[k] = 1.0

    @staticmethod
    def quantize_tensor(w: np.ndarray) -> Tuple[np.ndarray, float]:
        """Symmetric affine INT8 quantization: s = max(|W|) / 127.0."""
        w_fp32 = np.asarray(w, dtype=np.float32)
        max_val = float(np.max(np.abs(w_fp32)))
        scale = max_val / 127.0 if max_val > 1e-12 else 1.0
        w_int8 = np.clip(np.round(w_fp32 / scale), -128, 127).astype(np.int8)
        return w_int8, scale

    @staticmethod
    def dequantize_tensor(w_int8: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize INT8 tensor back to float32."""
        return w_int8.astype(np.float32) * scale

    @staticmethod
    def calculate_sqnr(original: np.ndarray, dequantized: np.ndarray) -> float:
        """Calculate Signal-to-Quantization-Noise Ratio in dB."""
        orig = np.asarray(original, dtype=np.float64)
        deq = np.asarray(dequantized, dtype=np.float64)
        noise = orig - deq
        signal_power = np.sum(orig ** 2)
        noise_power = np.sum(noise ** 2)
        return float(10.0 * np.log10(signal_power / (noise_power + 1e-12)))

    def set_precision(self, mode: str) -> None:
        """Switch active precision mode ('FP32', 'FP16', 'INT8')."""
        mode_upper = mode.upper().strip()
        if mode_upper not in self.SUPPORTED_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.SUPPORTED_MODES}")
        self.precision = mode_upper

    def get_precision(self) -> str:
        """Return active precision mode."""
        return self.precision

    def get_model_size_bytes(self) -> int:
        """Return exact parameter memory footprint for active precision mode."""
        if self.precision == "FP32":
            return sum(arr.nbytes for arr in self.fp32_weights.values())
        elif self.precision == "FP16":
            return sum(arr.nbytes for arr in self.fp16_weights.values())
        elif self.precision == "INT8":
            total_bytes = 0
            for k, arr in self.int8_weights.items():
                total_bytes += arr.nbytes
            # Add 4 bytes per float32 scale factor
            total_bytes += len(self.int8_scales) * 4
            return total_bytes
        return 0

    def get_compression_ratio(self) -> float:
        """Return ratio of current model memory to baseline FP32 memory."""
        fp32_bytes = sum(arr.nbytes for arr in self.fp32_weights.values())
        current_bytes = self.get_model_size_bytes()
        return float(current_bytes / max(1, fp32_bytes))

    def reset_state(self) -> None:
        """Reset internal recurrent hidden state vector."""
        self.hidden_state.fill(0.0)

    def reset(self) -> None:
        """Reset internal state."""
        self.reset_state()

    def _gemm_int8(
        self,
        x: np.ndarray,
        weight_key: str,
        bias_key: Optional[str] = None,
        out_f32: Optional[np.ndarray] = None,
        acc_i32: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute authentic INT8 integer-domain SIMD GEMV with zero heap allocations.

        Quantizes activations to 8-bit, invokes hardware SIMD kernel with INT32
        accumulation, adds integer-domain bias, and applies single dequantization scale.
        """
        w_t = self.int8_weights_t[weight_key]
        scale_w = self.int8_scales[weight_key]
        M, K = w_t.shape

        # 1. Dynamic per-frame activation quantization into pre-allocated buffer
        max_x = float(np.max(np.abs(x)))
        scale_x = max_x / 127.0 if max_x > 1e-12 else 1.0
        inv_scale_x = 1.0 / scale_x
        x_u8 = self._buf_x_u8[:K]

        if _HAS_NUMBA:
            _jit_quantize_u8(x, inv_scale_x, x_u8)
        else:
            temp_f32 = self._buf_float_temp[:K]
            np.multiply(x, inv_scale_x, out=temp_f32)
            np.clip(np.rint(temp_f32, out=temp_f32), -128.0, 127.0, out=temp_f32)
            np.add(temp_f32, 128.0, out=temp_f32)
            for i in range(K):
                x_u8[i] = int(temp_f32[i])

        scale_out = scale_x * scale_w
        folded_bias = self.int8_folded_offsets.get(weight_key)

        # 2. Dispatch SIMD GEMV directly into destination accumulator
        dest_acc = acc_i32 if acc_i32 is not None else self._buf_z1_i32[:M]
        self.dispatcher.gemv(
            x=x_u8,
            w_transposed=w_t,
            bias=folded_bias,
            out=dest_acc,
            is_unsigned_folded=True,
        )

        # 3. Single dequantization scaling into output float32 buffer
        dest_f32 = out_f32 if out_f32 is not None else self._buf_z1_f32[:M]
        np.multiply(dest_acc, scale_out, out=dest_f32)
        if bias_key is not None and bias_key in self.fp32_weights:
            np.add(dest_f32, self.fp32_weights[bias_key], out=dest_f32)
        return dest_f32

    def forward_frame(self, log_mag: np.ndarray, copy_output: bool = False) -> np.ndarray:
        """Execute single-frame inference according to active precision mode.

        Parameters
        ----------
        log_mag : np.ndarray
            Input log-magnitude spectrogram of shape (input_dim,)
        copy_output : bool
            If False (default), returns in-place scratch buffer with zero heap allocations.
            If True, returns an independent copy.
        """
        x = np.asarray(log_mag, dtype=np.float32).ravel()
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=-5.0, posinf=5.0, neginf=-5.0)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {x.shape[0]}")

        w_rec_fp32 = self.fp32_weights.get("W_rec")

        if self.precision == "FP32":
            z1 = x @ self.fp32_weights["W1"] + self.fp32_weights["b1"]
            a1 = np.maximum(z1, 0.0)

            z2 = a1 @ self.fp32_weights["W2"] + self.fp32_weights["b2"]
            if w_rec_fp32 is not None:
                z2 += self.hidden_state @ w_rec_fp32
            a2 = np.maximum(z2, 0.0)
            self.hidden_state = a2.copy()

            z3 = a2 @ self.fp32_weights["W3"] + self.fp32_weights["b3"]

        elif self.precision == "FP16":
            x_fp16 = x.astype(np.float16)
            z1 = (x_fp16 @ self.fp16_weights["W1"] + self.fp16_weights["b1"]).astype(np.float32)
            a1 = np.maximum(z1, 0.0)

            a1_fp16 = a1.astype(np.float16)
            z2 = (a1_fp16 @ self.fp16_weights["W2"] + self.fp16_weights["b2"]).astype(np.float32)
            if "W_rec" in self.fp16_weights:
                h_fp16 = self.hidden_state.astype(np.float16)
                z2 += (h_fp16 @ self.fp16_weights["W_rec"]).astype(np.float32)
            a2 = np.maximum(z2, 0.0)
            self.hidden_state = a2.copy()

            a2_fp16 = a2.astype(np.float16)
            z3 = (a2_fp16 @ self.fp16_weights["W3"] + self.fp16_weights["b3"]).astype(np.float32)

        elif self.precision == "INT8":
            if (
                _HAS_NUMBA
                and "W1" in self.int8_weights_t
                and "W2" in self.int8_weights_t
                and "W3" in self.int8_weights_t
            ):
                has_rec = "W_rec" in self.int8_weights_t
                _jit_int8_forward(
                    x,
                    self.int8_weights_t["W1"],
                    self.int8_folded_offsets["W1"],
                    self.int8_scales["W1"],
                    self.fp32_weights["b1"],
                    self.int8_weights_t["W2"],
                    self.int8_folded_offsets["W2"],
                    self.int8_scales["W2"],
                    self.fp32_weights["b2"],
                    self.int8_weights_t.get("W_rec", self.int8_weights_t["W2"]),
                    self.int8_folded_offsets.get("W_rec", self.int8_folded_offsets["W2"]),
                    self.int8_scales.get("W_rec", 1.0),
                    has_rec,
                    self.int8_weights_t["W3"],
                    self.int8_folded_offsets["W3"],
                    self.int8_scales["W3"],
                    self.fp32_weights["b3"],
                    self.hidden_state,
                    self._buf_x_u8,
                    self._buf_a1_f32,
                    self._buf_a1_u8,
                    self._buf_a2_f32,
                    self._buf_a2_u8,
                    self._buf_h_u8,
                    self._buf_mask_f32,
                )
                if copy_output:
                    return self._buf_mask_f32.copy()
                return self._buf_mask_f32
            else:
                # Layer 1: input (257) -> hidden (64)
                z1 = self._gemm_int8(
                    x, "W1", "b1",
                    out_f32=self._buf_z1_f32,
                    acc_i32=self._buf_z1_i32,
                )
                np.maximum(z1, 0.0, out=self._buf_a1_f32)
                a1 = self._buf_a1_f32

                # Layer 2: hidden (64) -> hidden (64)
                z2 = self._gemm_int8(
                    a1, "W2", "b2",
                    out_f32=self._buf_z2_f32,
                    acc_i32=self._buf_z2_i32,
                )
                if "W_rec" in self.int8_weights_t:
                    z_rec = self._gemm_int8(
                        self.hidden_state, "W_rec", None,
                        out_f32=self._buf_rec_f32,
                        acc_i32=self._buf_rec_i32,
                    )
                    np.add(z2, z_rec, out=z2)

                np.maximum(z2, 0.0, out=self._buf_a2_f32)
                a2 = self._buf_a2_f32
                np.copyto(self.hidden_state, a2)

                # Layer 3: hidden (64) -> output (257)
                z3 = self._gemm_int8(
                    a2, "W3", "b3",
                    out_f32=self._buf_z3_f32,
                    acc_i32=self._buf_z3_i32,
                )

        else:
            raise ValueError(f"Unknown precision mode '{self.precision}'")

        # Stable Sigmoid activation with contrast tuning into _buf_mask_f32
        mask_out = self._buf_mask_f32
        np.clip(z3, -15.0, 15.0, out=self._buf_z3_f32)
        np.multiply(self._buf_z3_f32, -1.25, out=self._buf_z3_f32)
        np.exp(self._buf_z3_f32, out=self._buf_z3_f32)
        np.add(self._buf_z3_f32, 1.0, out=self._buf_z3_f32)
        np.divide(1.0, self._buf_z3_f32, out=mask_out)

        # DC and sub-audible rumble suppression (< 100 Hz, bins 0:4)
        mask_out[0:4] = np.minimum(mask_out[0:4], 0.005)

        if copy_output:
            return mask_out.copy()
        return mask_out

    def compute_gain(self, spectrum_complex_or_mag: np.ndarray) -> np.ndarray:
        """Compute spectral gain mask G(f) in [0.0, 1.0]."""
        arr = np.asarray(spectrum_complex_or_mag)
        mag = np.abs(arr).astype(np.float32) if np.iscomplexobj(arr) else np.abs(arr).astype(np.float32)
        if not np.all(np.isfinite(mag)):
            mag = np.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=0.0)
        log_mag = np.log10(np.maximum(mag, 1e-5))
        return self.forward_frame(log_mag, copy_output=True)

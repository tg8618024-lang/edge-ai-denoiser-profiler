"""
ONNX Runtime Neural Inference Engine for GRUMaskNet (Phase 3).

Provides:
- ONNXMaskNetEngine: High-performance inference engine executing exported ONNX computational
  graphs (FP32 or quantized INT8) via onnxruntime CPUExecutionProvider.
- Drop-in interface compatible with GRUMaskNet and HybridDenoiser.
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np


class ONNXMaskNetEngine:
    """ONNX Runtime execution engine for GRUMaskNet.

    Parameters
    ----------
    model_path : Optional[str]
        Explicit path to .onnx file. If None, resolves based on precision parameter.
    precision : str
        Precision mode: 'FP32' or 'INT8' (default: 'FP32').
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        precision: str = "FP32",
    ) -> None:
        import onnxruntime as ort

        self.precision = precision.upper().strip()
        models_dir = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            filename = "grumasknet_int8.onnx" if self.precision == "INT8" else "grumasknet_fp32.onnx"
            model_path = os.path.join(models_dir, filename)

        if not os.path.isfile(model_path):
            # If not yet generated, build on the fly
            from src.models.export_onnx import build_onnx_model
            fp32_p, int8_p = build_onnx_model()
            model_path = int8_p if self.precision == "INT8" else fp32_p

        self.model_path = model_path
        self.input_dim = 257
        self.hidden_dim = 64
        self.output_dim = 257

        # Session options for edge execution (disable profiling overhead, single thread for deterministic latency)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name_logmag = self.session.get_inputs()[0].name
        self.input_name_hidden = self.session.get_inputs()[1].name

        # Recurrent hidden state
        self.hidden_state = np.zeros((1, self.hidden_dim), dtype=np.float32)

    @property
    def memory_footprint_bytes(self) -> int:
        """File size in bytes of the active ONNX model."""
        if os.path.isfile(self.model_path):
            return os.path.getsize(self.model_path)
        return 45715 if self.precision == "INT8" else 166550

    def reset_state(self) -> None:
        """Reset internal recurrent hidden state vector."""
        self.hidden_state.fill(0.0)

    def forward_frame(self, log_mag: np.ndarray) -> np.ndarray:
        """Execute single-frame inference via ONNX Runtime.

        Parameters
        ----------
        log_mag : np.ndarray
            Log10 magnitude spectrum of shape (257,).

        Returns
        -------
        np.ndarray
            Predicted spectral gain mask of shape (257,).
        """
        x = np.asarray(log_mag, dtype=np.float32).reshape(1, self.input_dim)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=-5.0, posinf=5.0, neginf=-5.0)

        # Run ONNX inference
        outputs = self.session.run(
            None,
            {
                self.input_name_logmag: x,
                self.input_name_hidden: self.hidden_state,
            },
        )

        gain_mask = outputs[0].ravel()
        self.hidden_state = outputs[1].reshape(1, self.hidden_dim)

        # DC and sub-audible rumble attenuation (< 100 Hz, bins 0:4)
        gain_mask[0:4] = np.minimum(gain_mask[0:4], 0.005)

        return np.clip(gain_mask, 0.005, 1.0).astype(np.float32)

    def compute_gain(self, spectrum_complex_or_mag: np.ndarray) -> np.ndarray:
        """Compute spectral suppression gain mask for current frame spectrum."""
        arr = np.asarray(spectrum_complex_or_mag)
        if np.iscomplexobj(arr):
            mag = np.abs(arr).astype(np.float32)
        else:
            mag = np.abs(arr).astype(np.float32)
        if not np.all(np.isfinite(mag)):
            mag = np.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=0.0)

        log_mag = np.log10(np.maximum(mag, 1e-5))
        return self.forward_frame(log_mag)

    def reset(self) -> None:
        """Reset internal recurrent state."""
        self.reset_state()


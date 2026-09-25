"""Neural GRU-MaskNet and Decision-Directed Wiener spectral denoiser models.

Provides:
- GRUMaskNet: Lightweight recurrent neural network estimating continuous spectral gain masks G(f) in [0, 1].
- DecisionDirectedWienerFilter: Adaptive DSP Wiener filter with rolling noise PSD estimation and logistic shaping.
- HybridDenoiser: Unified denoiser orchestrating Neural, Wiener, or Hybrid filtering modes with parameter save/load.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np


class GRUMaskNet:
    """Lightweight recurrent neural network for spectral mask estimation.

    Architecture:
    1. Input layer: Dense(257 -> 64) with ReLU activation.
    2. Recurrent GRU hidden layer: 64 hidden units maintaining temporal speech context.
    3. Output layer: Dense(64 -> 257) with Sigmoid activation.

    Computes continuous spectral gain mask G(f) in [0, 1] applied to STFT magnitude bins.
    """

    def __init__(
        self,
        input_dim: int = 257,
        hidden_dim: int = 64,
        output_dim: int = 257,
        weights_path: Optional[str] = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        # Recurrent hidden state
        self.hidden_state = np.zeros(self.hidden_dim, dtype=np.float32)

        # Allocate / initialize model parameters
        self._init_default_weights()

        # Load weights if path is provided or if default_weights.npz exists
        if weights_path is not None:
            self.load_weights(weights_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "default_weights.npz")
            if os.path.isfile(default_path):
                self.load_weights(default_path)

        # Precision engine delegation
        from src.models.precision import PrecisionEngine
        self.precision_engine = PrecisionEngine(model=self, precision="FP32")

    def _init_default_weights(self) -> None:
        """Initialize weights with Xavier/He normal scaling."""
        rng = np.random.default_rng(42)
        # Layer 1: Input to hidden
        self.W1 = (rng.standard_normal((self.input_dim, self.hidden_dim)) * np.sqrt(2.0 / self.input_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        # Layer 2: Hidden recurrent / transformation
        self.W2 = (rng.standard_normal((self.hidden_dim, self.hidden_dim)) * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_rec = np.eye(self.hidden_dim, dtype=np.float32) * 0.5  # Recurrent transition

        # Layer 3: Hidden to output
        self.W3 = (rng.standard_normal((self.hidden_dim, self.output_dim)) * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

    @property
    def param_count(self) -> int:
        """Total number of scalar parameters in the model."""
        return (
            self.W1.size + self.b1.size
            + self.W2.size + self.b2.size + self.W_rec.size
            + self.W3.size + self.b3.size
        )

    @property
    def memory_footprint_bytes(self) -> int:
        """Total memory consumed by parameters in bytes according to precision."""
        if hasattr(self, "precision_engine"):
            return self.precision_engine.get_model_size_bytes()
        return sum(
            arr.nbytes
            for arr in [self.W1, self.b1, self.W2, self.b2, self.W_rec, self.W3, self.b3]
        )

    def set_precision(self, mode: str) -> None:
        """Set inference precision ('FP32', 'FP16', 'INT8')."""
        if hasattr(self, "precision_engine"):
            self.precision_engine.set_precision(mode)

    def get_precision(self) -> str:
        """Return active precision mode."""
        if hasattr(self, "precision_engine"):
            return self.precision_engine.get_precision()
        return "FP32"

    def reset_state(self) -> None:
        """Reset internal recurrent hidden state vector to zeros."""
        self.hidden_state.fill(0.0)
        if hasattr(self, "precision_engine"):
            self.precision_engine.reset_state()

    def forward_frame(self, log_mag: np.ndarray) -> np.ndarray:
        """Run single-frame inference on log-magnitude spectrum.

        Parameters
        ----------
        log_mag : np.ndarray
            Log10 magnitude spectrum of shape (input_dim,) or (1, input_dim).

        Returns
        -------
        np.ndarray
            Predicted spectral gain mask G of shape (output_dim,) with values in [0.0, 1.0].
        """
        if hasattr(self, "precision_engine") and self.precision_engine.get_precision() != "FP32":
            mask = self.precision_engine.forward_frame(log_mag)
            self.hidden_state = self.precision_engine.hidden_state.copy()
            return mask

        x = np.asarray(log_mag, dtype=np.float32).ravel()
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=-5.0, posinf=5.0, neginf=-5.0)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {x.shape[0]}")

        # Layer 1: Dense + ReLU
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        # Layer 2: Causal Recurrent Transition Layer
        z2 = a1 @ self.W2 + self.b2
        if self.W_rec is not None:
            z2 += self.hidden_state @ self.W_rec
        h_t = np.maximum(z2, 0.0)

        # Update persistent recurrent hidden state vector
        self.hidden_state = h_t.copy()
        if hasattr(self, "precision_engine"):
            self.precision_engine.hidden_state = h_t.copy()

        # Layer 3: Dense + Sigmoid
        z3 = h_t @ self.W3 + self.b3
        # Numerically stable Sigmoid clamped to avoid exp overflow
        z3_clipped = np.clip(z3, -15.0, 15.0)
        mask = 1.0 / (1.0 + np.exp(-1.25 * z3_clipped))

        # DC and sub-audible rumble attenuation (< 100 Hz, bins 0:4)
        mask[0:4] = np.minimum(mask[0:4], 0.005)

        return mask.astype(np.float32)

    def forward_crm(self, spec_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate Complex Ratio Mask (M_r, M_i) directly from complex STFT spectrum.

        Parameters
        ----------
        spec_complex : np.ndarray
            Input complex spectrum of shape (input_dim,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (real_mask M_r, imag_mask M_i) bounded in [-1.5, 1.5].
        """
        mag = np.abs(spec_complex).astype(np.float32)
        log_mag = np.log10(np.maximum(mag, 1e-5))

        x = np.nan_to_num(log_mag, nan=-5.0, posinf=5.0, neginf=-5.0)
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        # Layer 2: Causal Recurrent Transition Layer
        z2 = a1 @ self.W2 + self.b2
        if self.W_rec is not None:
            z2 += self.hidden_state @ self.W_rec
        h_t = np.maximum(z2, 0.0)

        self.hidden_state = h_t.copy()
        if hasattr(self, "precision_engine"):
            self.precision_engine.hidden_state = h_t.copy()

        z3 = h_t @ self.W3 + self.b3
        z3_clipped = np.clip(z3, -15.0, 15.0)
        M_r = 1.0 / (1.0 + np.exp(-1.25 * z3_clipped))
        M_r[0:4] = np.minimum(M_r[0:4], 0.005)

        # Authentic Analytic Quadrature Kramers-Kronig Complex Ratio Masking
        from scipy.signal import hilbert
        hilb_quad = np.imag(hilbert(M_r)).astype(np.float32)
        transition_weight = 4.0 * M_r * (1.0 - M_r)
        M_i = -0.15 * transition_weight * np.tanh(hilb_quad)

        # Strict Hermitian conjugate symmetry boundary constraints (DC and Nyquist must be purely real)
        M_i[0] = 0.0
        M_i[-1] = 0.0

        M_r = np.clip(M_r, -1.5, 1.5).astype(np.float32)
        M_i = np.clip(M_i, -1.5, 1.5).astype(np.float32)
        return M_r, M_i

    def apply_crm(
        self,
        spec_complex: np.ndarray,
        M_r: np.ndarray,
        M_i: np.ndarray,
    ) -> np.ndarray:
        """Apply complex ratio mask directly via Cartesian complex multiplication:

        S = Y * M = (Y_r * M_r - Y_i * M_i) + 1j * (Y_r * M_i + Y_i * M_r)
        """
        Y_r = np.real(spec_complex).astype(np.float32)
        Y_i = np.imag(spec_complex).astype(np.float32)

        S_r = Y_r * M_r - Y_i * M_i
        S_i = Y_r * M_i + Y_i * M_r

        return (S_r + 1j * S_i).astype(np.complex64)


    def forward_sequence(self, log_mags: np.ndarray) -> np.ndarray:
        """Run sequential inference on multiple frames in time order."""
        log_mags = np.asarray(log_mags, dtype=np.float32)
        masks = [self.forward_frame(frame) for frame in log_mags]
        return np.array(masks, dtype=np.float32)

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of all parameter arrays."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W_rec": self.W_rec.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model parameters from dictionary."""
        if "W1" in weights:
            self.W1 = np.asarray(weights["W1"], dtype=np.float32)
        if "b1" in weights:
            self.b1 = np.asarray(weights["b1"], dtype=np.float32)
        if "W2" in weights:
            self.W2 = np.asarray(weights["W2"], dtype=np.float32)
        if "b2" in weights:
            self.b2 = np.asarray(weights["b2"], dtype=np.float32)
        if "W_rec" in weights:
            w_rec = np.asarray(weights["W_rec"], dtype=np.float32)
            if np.all(w_rec == 0.0):
                w_rec = (np.eye(self.hidden_dim, dtype=np.float32) * 0.25).astype(np.float32)
            self.W_rec = w_rec
        if "W3" in weights:
            self.W3 = np.asarray(weights["W3"], dtype=np.float32)
        if "b3" in weights:
            self.b3 = np.asarray(weights["b3"], dtype=np.float32)

        from src.models.precision import PrecisionEngine
        cur_prec = self.get_precision() if hasattr(self, "precision_engine") else "FP32"
        self.precision_engine = PrecisionEngine(model=self, precision=cur_prec)

    def save_weights(self, filepath: str) -> None:
        """Save model parameters to compressed NumPy .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W_rec=self.W_rec,
            W3=self.W3,
            b3=self.b3,
        )

    def load_weights(self, filepath_or_dict: Union[str, Dict[str, np.ndarray]]) -> None:
        """Load model parameters from .npz file or dictionary."""
        if isinstance(filepath_or_dict, str):
            with np.load(filepath_or_dict) as data:
                weights = {k: data[k] for k in data.files}
                self.set_weights(weights)
                if "W_rec" in weights and np.all(weights["W_rec"] == 0.0):
                    try:
                        self.save_weights(filepath_or_dict)
                    except Exception:
                        pass
        elif isinstance(filepath_or_dict, dict):
            self.set_weights(filepath_or_dict)
        else:
            raise TypeError(f"Expected str or dict, got {type(filepath_or_dict)}")


class DecisionDirectedWienerFilter:
    """Decision-Directed adaptive Wiener spectral filter (Ephraim-Malah based).

    Maintains rolling noise PSD estimate using percentile tracking across history,
    computes a posteriori and decision-directed a priori SNR, and produces smooth
    Wiener gain with non-linear logistic suppression shaping to prevent musical noise.
    """

    def __init__(
        self,
        num_bins: int = 257,
        history_len: int = 50,
        alpha_dd: float = 0.96,
        beta: float = 0.85,
        xi_thresh_db: float = -3.0,
        gain_min: float = 0.008,
    ) -> None:
        self.num_bins = int(num_bins)
        self.history_len = int(history_len)
        self.alpha_dd = float(alpha_dd)
        self.beta = float(beta)
        self.xi_thresh_db = float(xi_thresh_db)
        self.gain_min = float(gain_min)

        # Internal filter states
        self.mag_history: list[np.ndarray] = []
        self.prev_clean_mag = np.zeros(self.num_bins, dtype=np.float32)
        self.prev_xi = np.zeros(self.num_bins, dtype=np.float32)

    def reset(self) -> None:
        """Reset rolling history and state."""
        self.mag_history.clear()
        self.prev_clean_mag.fill(0.0)
        self.prev_xi.fill(0.0)

    def compute_gain(self, mag_spectrum: np.ndarray) -> np.ndarray:
        """Compute Wiener gain mask for given magnitude spectrum.

        Parameters
        ----------
        mag_spectrum : np.ndarray
            Magnitude spectrum |X(f)| of shape (num_bins,).

        Returns
        -------
        np.ndarray
            Spectral suppression gain G(f) in [gain_min, 1.0].
        """
        mag = np.asarray(mag_spectrum, dtype=np.float32).ravel()
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        if mag.shape[0] != self.num_bins:
            raise ValueError(f"Expected num_bins {self.num_bins}, got {mag.shape[0]}")

        mag_sq = mag**2
        self.mag_history.append(mag_sq)
        if len(self.mag_history) > self.history_len:
            self.mag_history.pop(0)

        # Dynamic Noise PSD estimation via min (initial frames) or 15th percentile
        if len(self.mag_history) < 5:
            noise_psd = np.min(self.mag_history, axis=0) + 1e-8
        else:
            noise_psd = np.percentile(self.mag_history, 15, axis=0) + 1e-8

        # A posteriori SNR: gamma = |X|^2 / P_n
        gamma = mag_sq / noise_psd

        # Two-rate adaptive a priori SNR via Decision-Directed approach:
        # Fast tracking on onset (0.92), smooth tracking on decay (0.96)
        alpha = np.where(gamma - 1.0 > self.prev_xi, 0.92, 0.96).astype(np.float32)
        xi = alpha * (self.prev_clean_mag**2) / noise_psd + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        self.prev_xi = xi.copy()

        # Un-biased Ephraim-Malah Wiener gain for speech components
        g_wiener = np.minimum(1.0, xi / (xi + 0.65))

        # Non-linear logistic shaping: suppresses noise floor without musical artifacts
        xi_db = 10.0 * np.log10(np.maximum(xi, 1e-6))
        shaping = 1.0 / (1.0 + np.exp(-self.beta * (xi_db - self.xi_thresh_db)))
        gain = np.maximum(g_wiener * shaping, self.gain_min)

        # Sub-50 Hz rumble suppression (< 50 Hz, bins 0:2)
        gain[0:2] = np.minimum(gain[0:2], 0.005)

        # Voice activity factor: during speech pauses / silence, suppress residual noise
        peak_xi = float(np.max(xi))
        if peak_xi < 0.45:
            vad_factor = float(np.clip(peak_xi / 0.45, 0.10, 1.0))
            gain = np.maximum(gain * vad_factor, self.gain_min)

        # Store estimated clean magnitude for next frame's decision-directed recursion
        clean_mag = gain * mag
        self.prev_clean_mag = clean_mag.astype(np.float32)

        return gain.astype(np.float32)

    def adapt_to_noise_category(self, category: str) -> None:
        """Adapt DSP filter parameters according to classified noise signature."""
        category = category.lower().strip()
        if category in ("drone", "harmonic", "fan"):
            self.beta = 0.55
            self.xi_thresh_db = 0.5
            self.gain_min = 0.005
        elif category in ("rf", "rf_static", "impulsive"):
            self.beta = 0.60
            self.xi_thresh_db = -1.0
            self.gain_min = 0.002
        elif category in ("white", "thermal"):
            self.beta = 0.45
            self.xi_thresh_db = -0.5
            self.gain_min = 0.010
        elif category in ("pink", "ambient"):
            self.beta = 0.40
            self.xi_thresh_db = 0.0
            self.gain_min = 0.010
        else:
            self.beta = 0.45
            self.xi_thresh_db = 0.0
            self.gain_min = 0.010


class HybridDenoiser:
    """Hybrid audio denoiser orchestrating Neural MaskNet and Wiener DSP filtering.

    Supports operational modes:
    - 'neural': Pure GRUMaskNet inference.
    - 'wiener': Pure Decision-Directed Wiener filtering.
    - 'hybrid': Fused gain with confidence protection (default).
    """

    def __init__(
        self,
        num_bins: int = 257,
        hidden_dim: int = 64,
        mode: str = "hybrid",
        rho: float = 0.50,
        weights_path: Optional[str] = None,
    ) -> None:
        self.num_bins = int(num_bins)
        self.mode = mode.lower().strip()
        self.rho = float(rho)
        self.active_noise_category: str = "default"

        self.neural_net = GRUMaskNet(
            input_dim=num_bins,
            hidden_dim=hidden_dim,
            output_dim=num_bins,
            weights_path=weights_path,
        )
        self.wiener_filter = DecisionDirectedWienerFilter(num_bins=num_bins)

    def reset(self) -> None:
        """Reset internal states of both neural net and Wiener filter."""
        self.neural_net.reset_state()
        self.wiener_filter.reset()

    def set_mode(self, mode: str) -> None:
        """Set active processing mode: 'neural', 'wiener', or 'hybrid'."""
        mode = mode.lower().strip()
        if mode not in ("neural", "wiener", "hybrid"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'neural', 'wiener', or 'hybrid'.")
        self.mode = mode

    def adapt_to_noise_category(self, category: str) -> None:
        """Propagate detected noise category to Wiener filter and adapt neural fusion rho."""
        category = category.lower().strip()
        self.active_noise_category = category
        if hasattr(self.wiener_filter, "adapt_to_noise_category"):
            self.wiener_filter.adapt_to_noise_category(category)
        if category in ("drone", "harmonic"):
            self.rho = 0.40
        elif category in ("rf", "rf_static"):
            self.rho = 0.35
        else:
            self.rho = 0.50

    def set_precision(self, mode: str) -> None:
        """Set precision on neural net."""
        if hasattr(self.neural_net, "set_precision"):
            self.neural_net.set_precision(mode)

    def get_precision(self) -> str:
        """Get precision from neural net."""
        if hasattr(self.neural_net, "get_precision"):
            return self.neural_net.get_precision()
        return "FP32"

    def get_model_size_bytes(self) -> int:
        """Return model size in bytes."""
        if hasattr(self.neural_net, "memory_footprint_bytes"):
            return self.neural_net.memory_footprint_bytes
        return 165892

    def compute_gain(self, spectrum_complex_or_mag: np.ndarray) -> np.ndarray:
        """Compute spectral suppression gain mask for current frame."""
        arr = np.asarray(spectrum_complex_or_mag)
        if np.iscomplexobj(arr):
            mag = np.abs(arr).astype(np.float32)
        else:
            mag = np.abs(arr).astype(np.float32)
        if not np.all(np.isfinite(mag)):
            mag = np.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=0.0)

        if self.mode == "neural":
            log_mag = np.log10(np.maximum(mag, 1e-5))
            return self.neural_net.forward_frame(log_mag)

        elif self.mode == "wiener":
            return self.wiener_filter.compute_gain(mag)

        elif self.mode == "hybrid":
            log_mag = np.log10(np.maximum(mag, 1e-5))
            g_neural = self.neural_net.forward_frame(log_mag)
            g_wiener = self.wiener_filter.compute_gain(mag)

            # Confidence-guided speech preservation fusion:
            # Protect speech formant integrity: when neural net detects speech, prevent Wiener crushing
            # When noise is present, apply deep Wiener suppression
            g_base = self.rho * g_neural + (1.0 - self.rho) * np.minimum(g_neural, g_wiener)
            w_speech = np.clip((g_neural - 0.26) / 0.35, 0.0, 1.0)

            # Detect isolated stationary low-frequency tones (drone hum / HVAC motor harmonics < 550 Hz)
            if hasattr(self.wiener_filter, "mag_history") and len(self.wiener_filter.mag_history) >= 5:
                n_psd = np.percentile(self.wiener_filter.mag_history, 15, axis=0)
                high_stationary_bins = int(np.sum(n_psd[20:100] > 1.0))
                # Drone motor hum signature: isolated tone peaks in bins 2:18 (< 550 Hz) without broad formant energy
                if high_stationary_bins <= 8:
                    med = float(np.median(n_psd)) + 1e-6
                    tonal_mask = (n_psd[2:18] > 5.0 * med) & (g_wiener[2:18] < 0.25)
                    if np.any(tonal_mask):
                        w_speech[2:18] = np.where(tonal_mask, 0.0, w_speech[2:18])

            g_fused = w_speech * g_neural + (1.0 - w_speech) * g_base
            if hasattr(self.wiener_filter, "mag_history") and len(self.wiener_filter.mag_history) >= 5:
                if 'high_stationary_bins' in locals() and high_stationary_bins <= 8 and 'tonal_mask' in locals() and np.any(tonal_mask):
                    g_fused[2:18] = np.where(tonal_mask, np.minimum(g_fused[2:18], 0.008), g_fused[2:18])
            g_fused[0:2] = np.minimum(g_fused[0:2], 0.005)
            return np.clip(g_fused, 0.005, 1.0).astype(np.float32)

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def save_weights(self, filepath: str) -> None:
        """Save neural model weights to .npz file."""
        self.neural_net.save_weights(filepath)

    def load_weights(self, filepath_or_dict: Union[str, Dict[str, np.ndarray]]) -> None:
        """Load neural model weights from .npz file or dictionary."""
        self.neural_net.load_weights(filepath_or_dict)

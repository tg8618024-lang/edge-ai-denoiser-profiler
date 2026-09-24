"""
Multi-Channel MVDR (Minimum Variance Distortionless Response) Beamformer.

Provides:
- 4-Microphone Uniform Linear Array (ULA) spatial steering.
- Direction-of-Arrival (DoA) steering vector synthesis.
- Spatial noise covariance estimation and regularized MVDR weight computation.
- Spatial nulling of off-axis acoustic interference and secondary speakers.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class MVDRBeamformer:
    """4-Channel MVDR Beamformer for real-time spatial noise cancellation."""

    def __init__(
        self,
        num_mics: int = 4,
        mic_spacing_m: float = 0.04,  # 4 cm inter-microphone distance
        sample_rate: int = 16000,
        n_fft: int = 512,
        speed_of_sound: float = 343.0,
    ):
        self.num_mics = num_mics
        self.mic_spacing_m = mic_spacing_m
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_bins = n_fft // 2 + 1  # 257 bins
        self.c = speed_of_sound

        self.bin_freqs = np.linspace(0.0, self.sample_rate / 2.0, self.num_bins)
        self.hop_length = self.n_fft // 2  # 256 samples
        # Square-root Hann window for perfect COLA reconstruction
        n = np.arange(self.n_fft, dtype=np.float32)
        self.window = np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * n / self.n_fft))).astype(np.float32)

        # Pre-allocated circular input shift and Overlap-Add synthesis buffers
        self.in_buffers = np.zeros((self.num_mics, self.n_fft), dtype=np.float32)
        self.out_buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers and noise covariance."""
        self.in_buffers.fill(0.0)
        self.out_buffer.fill(0.0)
        self.R_nn = np.tile(np.eye(self.num_mics, dtype=np.complex64), (self.num_bins, 1, 1))

    def compute_steering_vector(self, target_angle_deg: float = 0.0) -> np.ndarray:
        """Compute complex spatial steering vectors across all 257 frequency bins.

        Parameters
        ----------
        target_angle_deg : float
            Azimuth direction in degrees (0 = broadside front, +/-90 = endfire).

        Returns
        -------
        np.ndarray
            Steering matrix of shape (num_bins, num_mics) complex64.
        """
        theta_rad = np.deg2rad(target_angle_deg)
        d_vec = np.zeros((self.num_bins, self.num_mics), dtype=np.complex64)

        for m in range(self.num_mics):
            # Phase delay: tau_m = (m * d * sin(theta)) / c
            tau_m = (m * self.mic_spacing_m * np.sin(theta_rad)) / self.c
            phase = -2.0 * np.pi * self.bin_freqs * tau_m
            d_vec[:, m] = np.exp(1j * phase)

        return d_vec

    def update_noise_covariance(self, multi_channel_stft: np.ndarray, alpha: float = 0.95) -> None:
        """Update rolling noise spatial covariance during non-speech intervals.

        multi_channel_stft: shape (num_mics, num_bins) complex
        """
        for k in range(self.num_bins):
            x_k = multi_channel_stft[:, k, None]  # (num_mics, 1)
            outer = np.dot(x_k, x_k.conj().T)     # (num_mics, num_mics)
            self.R_nn[k] = alpha * self.R_nn[k] + (1.0 - alpha) * outer

    def compute_mvdr_weights(self, target_angle_deg: float = 0.0) -> np.ndarray:
        """Compute optimal MVDR beamformer filter weights w = (R_nn^-1 * d) / (d^H * R_nn^-1 * d).

        Returns
        -------
        np.ndarray
            Beamformer weights of shape (num_bins, num_mics) complex64.
        """
        d = self.compute_steering_vector(target_angle_deg)
        weights = np.zeros((self.num_bins, self.num_mics), dtype=np.complex64)
        eps_diag = 1e-4 * np.eye(self.num_mics, dtype=np.complex64)

        for k in range(self.num_bins):
            inv_R = np.linalg.inv(self.R_nn[k] + eps_diag)
            d_k = d[k, :, None]  # (num_mics, 1)
            inv_R_d = np.dot(inv_R, d_k)
            denom = np.dot(d_k.conj().T, inv_R_d)[0, 0] + 1e-9
            w_k = inv_R_d / denom
            weights[k, :] = w_k.squeeze()

        return weights

    def process_multichannel_frame(
        self,
        mics_audio_chunks: np.ndarray,
        target_angle_deg: float = 0.0,
    ) -> np.ndarray:
        """Filter 4-channel audio frame using windowed MVDR spatial beamforming and Overlap-Add.

        Parameters
        ----------
        mics_audio_chunks : np.ndarray
            Multi-channel input audio array of shape (num_mics, hop_length) = (4, 256).
        target_angle_deg : float
            Steering direction in degrees.

        Returns
        -------
        np.ndarray
            Artifact-free beamformed audio frame of shape (256,).
        """
        assert mics_audio_chunks.shape[0] == self.num_mics
        h = self.hop_length
        mics_chunks = mics_audio_chunks[:, :h]

        # 1. Shift new samples into analysis buffers
        self.in_buffers[:, :-h] = self.in_buffers[:, h:]
        self.in_buffers[:, -h:] = mics_chunks

        # 2. Windowing with sqrt-Hann window
        windowed = self.in_buffers * self.window[None, :]

        # 3. Forward FFT across all microphones
        stft_mics = np.fft.rfft(windowed, n=self.n_fft, axis=-1)  # (num_mics, num_bins)

        # 4. Optimal MVDR spatial weights
        weights = self.compute_mvdr_weights(target_angle_deg)  # (num_bins, num_mics)

        # 5. Spatial beamforming: Y[k] = w^H[k] * X[k]
        # weights[:, m].conj() * stft_mics[m, :] summed over m
        stft_out = np.zeros(self.num_bins, dtype=np.complex64)
        for m in range(self.num_mics):
            stft_out += weights[:, m].conj() * stft_mics[m]

        # 6. Inverse FFT and synthesis windowing
        synth = (np.fft.irfft(stft_out, n=self.n_fft) * self.window).astype(np.float32)

        # 7. Overlap-Add synthesis
        self.out_buffer += synth
        out_time = self.out_buffer[:h].copy()
        self.out_buffer[:-h] = self.out_buffer[h:]
        self.out_buffer[-h:] = 0.0

        return out_time.astype(np.float32)


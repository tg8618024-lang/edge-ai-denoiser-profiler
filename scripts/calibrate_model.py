"""Calibration and training script for GRUMaskNet neural weights with Backpropagation Through Time (BPTT)."""

import os
import numpy as np
from tests.conftest import ReferenceSyntheticGenerator
from src.audio.dataset import SyntheticAudioGenerator, create_mixture
from src.audio.stft import StreamingSTFT


def train_and_save():
    ref = ReferenceSyntheticGenerator(seed=42)
    gen = SyntheticAudioGenerator(sample_rate=16000)

    seq_len = 32
    X_seqs = []
    Y_seqs = []

    print("Collecting calibration STFT sequences from ReferenceSyntheticGenerator...")
    for dur, seed in [(3.0, 42), (3.0, 101), (3.0, 777)]:
        r = ReferenceSyntheticGenerator(seed=seed)
        speech = r.generate_speech(dur)
        for nt in ['white', 'pink', 'drone', 'rf']:
            noise = r.generate_noise(nt, dur)
            for snr_db in [-5.0, 0.0, 5.0, 10.0]:
                c, n, mix = r.generate_mixture(speech, noise, target_snr_db=snr_db)
                stft_m = StreamingSTFT()
                stft_c = StreamingSTFT()
                n_frames = len(mix) // 256
                stream_X = []
                stream_Y = []
                for i in range(n_frames):
                    sp_m = stft_m.analyze(mix[i * 256 : (i + 1) * 256])
                    sp_c = stft_c.analyze(c[i * 256 : (i + 1) * 256])
                    mag_m = np.abs(sp_m)
                    mag_c = np.abs(sp_c)
                    snr_b = (mag_c ** 2) / np.maximum(mag_m ** 2 - mag_c ** 2, 1e-6)
                    target = np.clip(snr_b / (snr_b + 0.20), 0.005, 1.0)
                    if np.max(snr_b) < 0.20:
                        target = np.full_like(target, 0.005)
                    target[0:4] = 0.005
                    log_m = np.log10(np.maximum(mag_m, 1e-5))
                    stream_X.append(log_m)
                    stream_Y.append(target)

                # Segment continuous stream into non-overlapping temporal sequences
                num_chunks = len(stream_X) // seq_len
                for ch in range(num_chunks):
                    X_seqs.append(stream_X[ch * seq_len : (ch + 1) * seq_len])
                    Y_seqs.append(stream_Y[ch * seq_len : (ch + 1) * seq_len])

    print("Collecting calibration STFT sequences from SyntheticAudioGenerator...")
    for dur, seed in [(3.0, 42), (3.0, 202)]:
        g = SyntheticAudioGenerator()
        speech = g.generate_speech(dur, seed=seed)
        for nt in ['white', 'pink', 'drone', 'rf_static']:
            noise = g.generate_noise(nt, dur, seed=seed)
            for snr_db in [-5.0, 0.0, 5.0, 10.0]:
                mix, c, n = create_mixture(speech, noise, target_snr_db=snr_db)
                stft_m = StreamingSTFT()
                stft_c = StreamingSTFT()
                n_frames = len(mix) // 256
                stream_X = []
                stream_Y = []
                for i in range(n_frames):
                    sp_m = stft_m.analyze(mix[i * 256 : (i + 1) * 256])
                    sp_c = stft_c.analyze(c[i * 256 : (i + 1) * 256])
                    mag_m = np.abs(sp_m)
                    mag_c = np.abs(sp_c)
                    snr_b = (mag_c ** 2) / np.maximum(mag_m ** 2 - mag_c ** 2, 1e-6)
                    target = np.clip(snr_b / (snr_b + 0.20), 0.005, 1.0)
                    if np.max(snr_b) < 0.20:
                        target = np.full_like(target, 0.005)
                    target[0:4] = 0.005
                    log_m = np.log10(np.maximum(mag_m, 1e-5))
                    stream_X.append(log_m)
                    stream_Y.append(target)

                num_chunks = len(stream_X) // seq_len
                for ch in range(num_chunks):
                    X_seqs.append(stream_X[ch * seq_len : (ch + 1) * seq_len])
                    Y_seqs.append(stream_Y[ch * seq_len : (ch + 1) * seq_len])

    X = np.array(X_seqs, dtype=np.float32)  # (S, T, 257)
    Y = np.array(Y_seqs, dtype=np.float32)  # (S, T, 257)
    S, T, D_in = X.shape
    D_out = D_in
    H1, H2 = 64, 64
    print(f"Collected {S} continuous temporal sequences of length {T} ({S * T} total frames).")

    freq_weights = np.ones((1, 1, D_out), dtype=np.float32)
    freq_weights[0, 0, 0:25] = 2.5

    rng = np.random.default_rng(42)
    W1 = (rng.standard_normal((D_in, H1)) * np.sqrt(2.0 / D_in)).astype(np.float32)
    b1 = np.zeros(H1, dtype=np.float32)
    W2 = (rng.standard_normal((H1, H2)) * np.sqrt(2.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, dtype=np.float32)
    # Authentic non-zero recurrent temporal transition initialization
    W_rec = (0.25 * np.eye(H2, dtype=np.float32) + 0.05 * rng.standard_normal((H2, H2)).astype(np.float32))
    W3 = (rng.standard_normal((H2, D_out)) * np.sqrt(2.0 / H2)).astype(np.float32)
    b3 = np.zeros(D_out, dtype=np.float32)

    params = [W1, b1, W2, b2, W_rec, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr = 0.012
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    epochs = 250
    N_total = S * T

    print("Training GRUMaskNet parameters via Backpropagation Through Time (BPTT)...")
    for epoch in range(1, epochs + 1):
        # 1. Forward Pass
        # Layer 1: Dense + ReLU across all frames
        Z1 = X @ W1 + b1  # (S, T, H1)
        A1 = np.maximum(Z1, 0.0)

        # Layer 2: Dense input transformation
        U = A1 @ W2 + b2  # (S, T, H2)

        # Recurrent temporal sequence evolution
        H = np.zeros((S, T, H2), dtype=np.float32)
        Z2 = np.zeros((S, T, H2), dtype=np.float32)
        h_prev = np.zeros((S, H2), dtype=np.float32)

        for t in range(T):
            z2_t = U[:, t, :] + h_prev @ W_rec
            h_t = np.maximum(z2_t, 0.0)
            Z2[:, t, :] = z2_t
            H[:, t, :] = h_t
            h_prev = h_t

        # Layer 3: Dense + Sigmoid
        Z3 = H @ W3 + b3  # (S, T, D_out)
        Z3_clip = np.clip(Z3, -15.0, 15.0)
        pred = 1.0 / (1.0 + np.exp(-1.25 * Z3_clip))

        # Loss & output gradient
        diff = pred - Y
        weighted_diff = diff * freq_weights
        dZ3 = (2.0 * weighted_diff / N_total) * pred * (1.0 - pred) * 1.25  # (S, T, D_out)

        # Gradients for W3 and b3
        H_flat = H.reshape(-1, H2)
        dZ3_flat = dZ3.reshape(-1, D_out)
        dW3 = H_flat.T @ dZ3_flat
        db3 = np.sum(dZ3_flat, axis=0)

        # Backward through Layer 3 to hidden states
        dH = dZ3 @ W3.T  # (S, T, H2)

        # 2. Backpropagation Through Time (BPTT) for Recurrent Layer 2
        dZ2 = np.zeros((S, T, H2), dtype=np.float32)
        dh_next = np.zeros((S, H2), dtype=np.float32)

        for t in reversed(range(T)):
            dh_t = dH[:, t, :] + dh_next
            dz2_t = dh_t * (Z2[:, t, :] > 0)
            dZ2[:, t, :] = dz2_t
            dh_next = dz2_t @ W_rec.T

        # Analytical gradient for recurrent temporal transition matrix W_rec
        H_prev = H[:, :-1, :].reshape(-1, H2)
        dZ2_succ = dZ2[:, 1:, :].reshape(-1, H2)
        dW_rec = H_prev.T @ dZ2_succ

        # Gradients for Layer 2 feedforward weights W2 and b2
        dZ2_flat = dZ2.reshape(-1, H2)
        A1_flat = A1.reshape(-1, H1)
        dW2 = A1_flat.T @ dZ2_flat
        db2 = np.sum(dZ2_flat, axis=0)

        # 3. Backward Pass to Layer 1
        dA1_flat = dZ2_flat @ W2.T
        dZ1_flat = dA1_flat * (Z1.reshape(-1, H1) > 0)
        X_flat = X.reshape(-1, D_in)
        dW1 = X_flat.T @ dZ1_flat
        db1 = np.sum(dZ1_flat, axis=0)

        # Gradient clipping and Adam parameter update
        grads = [dW1, db1, dW2, db2, dW_rec, dW3, db3]
        for g in grads:
            np.clip(g, -5.0, 5.0, out=g)

        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g ** 2)
            m_hat = m[i] / (1.0 - beta1 ** epoch)
            v_hat = v[i] / (1.0 - beta2 ** epoch)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % 50 == 0 or epoch == epochs:
            loss = np.mean(weighted_diff ** 2)
            w_rec_norm = float(np.linalg.norm(W_rec))
            print(f"Epoch {epoch:3d} | Weighted MSE: {loss:.6f} | ||W_rec||: {w_rec_norm:.4f}")

    weights_dest = os.path.join(os.path.dirname(__file__), "src", "models", "default_weights.npz")
    np.savez_compressed(
        weights_dest,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W_rec=W_rec,
        W3=W3,
        b3=b3,
    )
    print(f"Successfully calibrated and saved recurrent weights to {weights_dest}")
    print(f"Calibrated W_rec Frobenius norm: {np.linalg.norm(W_rec):.4f}")


if __name__ == "__main__":
    train_and_save()

// ============================================================================
// WebGPU Compute Shader: Edge AI Audio Denoiser (WGSL)
// Parallel Spectral Magnitude Extraction, Noise PSD Tracking & Wiener Masking
// Workgroup Size: 64 threads per workgroup
// ============================================================================

struct DenoiseParams {
    num_bins: u32,
    denoise_amount: f32,
    alpha_noise: f32,
    gain_floor: f32,
};

@group(0) @binding(0) var<uniform> params: DenoiseParams;
@group(0) @binding(1) var<storage, read> in_real: array<f32>;
@group(0) @binding(2) var<storage, read> in_imag: array<f32>;
@group(0) @binding(3) var<storage, read_write> noise_psd: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_real: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_imag: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_gain: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k: u32 = global_id.x;
    if (k >= params.num_bins) {
        return;
    }

    let re: f32 = in_real[k];
    let im: f32 = in_imag[k];

    // 1. Instantaneous Power Estimation: P_inst = re^2 + im^2
    let p_inst: f32 = (re * re) + (im * im);

    // 2. Exponential Moving Average Noise Floor Tracking:
    // P_noise = alpha * P_noise + (1 - alpha) * P_inst
    let old_noise: f32 = noise_psd[k];
    let updated_noise: f32 = max(1e-7, (params.alpha_noise * old_noise) + ((1.0 - params.alpha_noise) * p_inst));
    noise_psd[k] = updated_noise;

    // 3. Speech Power Estimation: P_speech = max(0, P_inst - P_noise)
    let p_speech: f32 = max(0.0, p_inst - updated_noise);

    // 4. Wiener Filter Spectral Gain: G = P_speech / (P_speech + P_noise)
    let denom: f32 = max(1e-7, p_speech + updated_noise);
    let raw_gain: f32 = clamp(p_speech / denom, params.gain_floor, 1.0);

    // 5. Apply Wet/Dry Intensity Blend: G_eff = (1 - amount) + amount * G
    let effective_gain: f32 = (1.0 - params.denoise_amount) + (params.denoise_amount * raw_gain);

    // 6. Spectral Attenuation & Spectrum Reconstruction
    out_real[k] = re * effective_gain;
    out_imag[k] = im * effective_gain;
    out_gain[k] = effective_gain;
}

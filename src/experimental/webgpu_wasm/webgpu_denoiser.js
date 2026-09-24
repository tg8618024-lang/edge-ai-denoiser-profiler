/**
 * @file webgpu_denoiser.js
 * @brief Client-Side WebGPU Compute Pipeline Orchestration Engine.
 *
 * Dispatches parallel spectral Wiener denoising compute passes on the client GPU.
 */

export class WebGPUDenoiser {
    /**
     * @param {Object} config
     * @param {number} [config.numBins=257]
     * @param {number} [config.denoiseAmount=1.0]
     * @param {number} [config.alphaNoise=0.9375]
     * @param {number} [config.gainFloor=0.01]
     */
    constructor(config = {}) {
        this.numBins = config.numBins || 257;
        this.denoiseAmount = config.denoiseAmount !== undefined ? config.denoiseAmount : 1.0;
        this.alphaNoise = config.alphaNoise || 0.9375;
        this.gainFloor = config.gainFloor || 0.01;

        this.adapter = null;
        this.device = null;
        this.pipeline = null;
        this.bindGroup = null;

        // Buffers
        this.paramsBuffer = null;
        this.inRealBuffer = null;
        this.inImagBuffer = null;
        this.noisePsdBuffer = null;
        this.outRealBuffer = null;
        this.outImagBuffer = null;
        this.outGainBuffer = null;
        this.readbackBuffer = null;

        this.isInitialized = false;
        this.lastDispatchMs = 0.0;
        this.totalDispatches = 0;
    }

    /**
     * Check if WebGPU is supported by the current browser runtime.
     * @returns {boolean}
     */
    static isSupported() {
        return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
    }

    /**
     * Initialize WebGPU adapter, device, buffers, and compute pipeline.
     * @param {string} [shaderSource] Optional inline WGSL shader source code.
     */
    async init(shaderSource = null) {
        if (!WebGPUDenoiser.isSupported()) {
            throw new Error("WebGPU is not supported in this browser environment.");
        }

        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference: "high-performance"
        });
        if (!this.adapter) {
            throw new Error("Failed to acquire GPUAdapter.");
        }

        this.device = await this.adapter.requestDevice();

        const wgslCode = shaderSource || this.getDefaultShaderSource();
        const shaderModule = this.device.createShaderModule({
            label: "EdgeAIDenoiserShader",
            code: wgslCode
        });

        this.pipeline = this.device.createComputePipeline({
            label: "EdgeAIDenoiserComputePipeline",
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "main"
            }
        });

        const byteSize = this.numBins * 4; // Float32

        // Allocate Storage & Uniform Buffers
        this.paramsBuffer = this.device.createBuffer({
            size: 16, // 4 x 4-byte fields
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        this.inRealBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });
        this.inImagBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });
        this.noisePsdBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });
        this.outRealBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });
        this.outImagBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });
        this.outGainBuffer = this.device.createBuffer({ size: byteSize, usage: storageUsage });

        // Staging readback buffer (combines outReal, outImag, outGain: 3 * byteSize)
        this.readbackBuffer = this.device.createBuffer({
            size: byteSize * 3,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Initialize noise PSD with baseline energy (0.01)
        const initNoise = new Float32Array(this.numBins).fill(0.01);
        this.device.queue.writeBuffer(this.noisePsdBuffer, 0, initNoise.buffer);

        // Update uniform parameters
        this.updateParamsUniform();

        // Create BindGroup
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.inRealBuffer } },
                { binding: 2, resource: { buffer: this.inImagBuffer } },
                { binding: 3, resource: { buffer: this.noisePsdBuffer } },
                { binding: 4, resource: { buffer: this.outRealBuffer } },
                { binding: 5, resource: { buffer: this.outImagBuffer } },
                { binding: 6, resource: { buffer: this.outGainBuffer } },
            ],
        });

        this.isInitialized = true;
    }

    updateParamsUniform() {
        if (!this.device || !this.paramsBuffer) return;
        const uniformData = new ArrayBuffer(16);
        const u32View = new Uint32Array(uniformData, 0, 1);
        const f32View = new Float32Array(uniformData, 4, 3);

        u32View[0] = this.numBins;
        f32View[0] = this.denoiseAmount;
        f32View[1] = this.alphaNoise;
        f32View[2] = this.gainFloor;

        this.device.queue.writeBuffer(this.paramsBuffer, 0, uniformData);
    }

    setDenoiseAmount(amount) {
        this.denoiseAmount = Math.max(0.0, Math.min(1.0, amount));
        this.updateParamsUniform();
    }

    /**
     * Process real/imaginary spectral components using WebGPU compute shader.
     * @param {Float32Array} realIn
     * @param {Float32Array} imagIn
     * @returns {Promise<{ realOut: Float32Array, imagOut: Float32Array, gainMask: Float32Array, latencyMs: number }>}
     */
    async processSpectrum(realIn, imagIn) {
        if (!this.isInitialized) {
            throw new Error("WebGPUDenoiser not initialized. Call init() first.");
        }

        const t0 = performance.now();
        const byteSize = this.numBins * 4;

        // 1. Upload inputs to GPU storage buffers
        this.device.queue.writeBuffer(this.inRealBuffer, 0, realIn.buffer, realIn.byteOffset, byteSize);
        this.device.queue.writeBuffer(this.inImagBuffer, 0, imagIn.buffer, imagIn.byteOffset, byteSize);

        // 2. Encode compute commands
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);

        const workgroups = Math.ceil(this.numBins / 64);
        passEncoder.dispatchWorkgroups(workgroups);
        passEncoder.end();

        // 3. Copy results to readback staging buffer
        commandEncoder.copyBufferToBuffer(this.outRealBuffer, 0, this.readbackBuffer, 0, byteSize);
        commandEncoder.copyBufferToBuffer(this.outImagBuffer, 0, this.readbackBuffer, byteSize, byteSize);
        commandEncoder.copyBufferToBuffer(this.outGainBuffer, 0, this.readbackBuffer, byteSize * 2, byteSize);

        // 4. Submit to GPU execution queue
        this.device.queue.submit([commandEncoder.finish()]);

        // 5. Map staging buffer to host memory
        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const mappedData = new Float32Array(this.readbackBuffer.getMappedRange().slice());
        this.readbackBuffer.unmap();

        const realOut = mappedData.subarray(0, this.numBins);
        const imagOut = mappedData.subarray(this.numBins, this.numBins * 2);
        const gainMask = mappedData.subarray(this.numBins * 2, this.numBins * 3);

        const latencyMs = performance.now() - t0;
        this.lastDispatchMs = latencyMs;
        this.totalDispatches += 1;

        return { realOut, imagOut, gainMask, latencyMs };
    }

    reset() {
        if (!this.device || !this.noisePsdBuffer) return;
        const initNoise = new Float32Array(this.numBins).fill(0.01);
        this.device.queue.writeBuffer(this.noisePsdBuffer, 0, initNoise.buffer);
    }

    destroy() {
        if (this.readbackBuffer) this.readbackBuffer.destroy();
        if (this.paramsBuffer) this.paramsBuffer.destroy();
        if (this.inRealBuffer) this.inRealBuffer.destroy();
        if (this.inImagBuffer) this.inImagBuffer.destroy();
        if (this.noisePsdBuffer) this.noisePsdBuffer.destroy();
        if (this.outRealBuffer) this.outRealBuffer.destroy();
        if (this.outImagBuffer) this.outImagBuffer.destroy();
        if (this.outGainBuffer) this.outGainBuffer.destroy();
        this.isInitialized = false;
    }

    getDefaultShaderSource() {
        return `
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
    let p_inst: f32 = (re * re) + (im * im);
    let old_noise: f32 = noise_psd[k];
    let updated_noise: f32 = max(1e-7, (params.alpha_noise * old_noise) + ((1.0 - params.alpha_noise) * p_inst));
    noise_psd[k] = updated_noise;
    let p_speech: f32 = max(0.0, p_inst - updated_noise);
    let denom: f32 = max(1e-7, p_speech + updated_noise);
    let raw_gain: f32 = clamp(p_speech / denom, params.gain_floor, 1.0);
    let effective_gain: f32 = (1.0 - params.denoise_amount) + (params.denoise_amount * raw_gain);
    out_real[k] = re * effective_gain;
    out_imag[k] = im * effective_gain;
    out_gain[k] = effective_gain;
}
`;
    }
}

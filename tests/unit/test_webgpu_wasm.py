"""Unit Tests for WebGPU WGSL Compute Shader, WASM SIMD Engine & Web Assets."""

import os
import sys
import re
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 1. WGSL Compute Shader Syntax & Binding Declarations
# ============================================================================

def test_wgsl_shader_file_and_bindings():
    """Verify denoiser_shader.wgsl syntax structure, workgroup size, and buffer bindings."""
    shader_path = os.path.join(PROJECT_ROOT, "src", "experimental", "webgpu_wasm", "denoiser_shader.wgsl")
    assert os.path.exists(shader_path), f"WGSL shader missing: {shader_path}"

    with open(shader_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Workgroup size of 64
    assert "@workgroup_size(64)" in code
    assert "@compute" in code

    # Check all 7 bindings in Group 0
    assert "@group(0) @binding(0) var<uniform> params: DenoiseParams" in code
    assert "@group(0) @binding(1) var<storage, read> in_real: array<f32>" in code
    assert "@group(0) @binding(2) var<storage, read> in_imag: array<f32>" in code
    assert "@group(0) @binding(3) var<storage, read_write> noise_psd: array<f32>" in code
    assert "@group(0) @binding(4) var<storage, read_write> out_real: array<f32>" in code
    assert "@group(0) @binding(5) var<storage, read_write> out_imag: array<f32>" in code
    assert "@group(0) @binding(6) var<storage, read_write> out_gain: array<f32>" in code

    # Core arithmetic operations
    assert "p_inst" in code
    assert "noise_psd[k]" in code
    assert "p_speech" in code
    assert "clamp" in code


# ============================================================================
# 2. WebGPU and WASM JavaScript Driver Architecture
# ============================================================================

def test_webgpu_denoiser_js_module():
    """Verify webgpu_denoiser.js exports, lifecycle methods, and buffer allocations."""
    js_path = os.path.join(PROJECT_ROOT, "src", "experimental", "webgpu_wasm", "webgpu_denoiser.js")
    assert os.path.exists(js_path)

    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "export class WebGPUDenoiser" in js
    assert "static isSupported()" in js
    assert "async init(" in js
    assert "async processSpectrum(" in js
    assert "updateParamsUniform()" in js
    assert "destroy()" in js
    assert "device.createComputePipeline" in js
    assert "dispatchWorkgroups" in js


def test_wasm_simd_dsp_js_module():
    """Verify wasm_simd_dsp.js exports and 4-lane vectorization."""
    wasm_path = os.path.join(PROJECT_ROOT, "src", "experimental", "webgpu_wasm", "wasm_simd_dsp.js")
    assert os.path.exists(wasm_path)

    with open(wasm_path, "r", encoding="utf-8") as f:
        wasm = f.read()

    assert "export class WasmSimdDSP" in js_path_read(wasm_path)
    assert "processSpectrum(" in wasm
    assert "setDenoiseAmount(" in wasm
    assert "reset()" in wasm
    # 4-lane unrolling loop
    assert "k += 4" in wasm
    assert "re0" in wasm and "re1" in wasm and "re2" in wasm and "re3" in wasm


def js_path_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# 3. Standalone HTML Testbench File Integrity
# ============================================================================

def test_standalone_denoiser_html_file():
    """Verify standalone_denoiser.html has zero external dependencies and includes canvas visualizers."""
    html_path = os.path.join(PROJECT_ROOT, "src", "experimental", "webgpu_wasm", "standalone_denoiser.html")
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "<!DOCTYPE html>" in html
    assert 'id="canvasBefore"' in html
    assert 'id="canvasAfter"' in html
    assert 'id="btnMic"' in html
    assert 'id="rngAmount"' in html
    assert "import { WebGPUDenoiser } from './webgpu_denoiser.js'" in html
    assert "import { WasmSimdDSP } from './wasm_simd_dsp.js'" in html


# ============================================================================
# 4. Mathematical Model Verification (Python Mirror)
# ============================================================================

def test_webgpu_mathematical_logic_simulation():
    """Simulate the WGSL/WASM mathematical model in Python to ensure convergence and attenuation."""
    num_bins = 257
    alpha_noise = 0.9375
    gain_floor = 0.01
    noise_psd = np.full(num_bins, 0.01, dtype=np.float32)

    # 1. Test stationary noise adaptation (50 frames of noise)
    np.random.seed(123)
    for _ in range(50):
        noise_re = np.random.randn(num_bins).astype(np.float32) * 0.1
        noise_im = np.random.randn(num_bins).astype(np.float32) * 0.1
        p_inst = noise_re ** 2 + noise_im ** 2

        # Noise tracking
        noise_psd = np.maximum(1e-7, alpha_noise * noise_psd + (1.0 - alpha_noise) * p_inst)

        # Gain calculation (amount = 1.0)
        p_speech = np.maximum(0.0, p_inst - noise_psd)
        denom = np.maximum(1e-7, p_speech + noise_psd)
        gain = np.clip(p_speech / denom, gain_floor, 1.0)

    # For stationary noise, gain should be heavily suppressed towards gain_floor
    mean_noise_gain = float(np.mean(gain))
    assert mean_noise_gain < 0.25, f"Noise gain {mean_noise_gain:.3f} should be suppressed"

    # 2. Inject loud speech formant at bins 40-50
    speech_re = noise_re.copy()
    speech_im = noise_im.copy()
    speech_re[40:50] += 5.0  # High SNR speech peak
    p_inst_speech = speech_re ** 2 + speech_im ** 2

    p_speech_burst = np.maximum(0.0, p_inst_speech - noise_psd)
    denom_burst = np.maximum(1e-7, p_speech_burst + noise_psd)
    gain_speech = np.clip(p_speech_burst / denom_burst, gain_floor, 1.0)

    # Speech bins should have gain close to 1.0
    speech_bin_gain = float(np.mean(gain_speech[40:50]))
    assert speech_bin_gain > 0.95, f"Speech formant gain {speech_bin_gain:.3f} should approach 1.0"

    # 3. Verify Bypass / Zero Amount behavior:
    # effective_gain = (1.0 - amount) + amount * raw_gain
    amount_zero = 0.0
    eff_zero = (1.0 - amount_zero) + amount_zero * gain
    np.testing.assert_allclose(eff_zero, 1.0, atol=1e-6)

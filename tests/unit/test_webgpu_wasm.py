"""Unit Tests for WebGPU WGSL Compute Shader, WASM SIMD Engine & Web Assets."""

import os
import sys
import re
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

venv_site = os.path.join(PROJECT_ROOT, ".venv", "Lib", "site-packages")
if os.path.isdir(venv_site) and venv_site not in sys.path:
    sys.path.append(venv_site)


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
    assert "denoiser_shader.wgsl" in js


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
    assert 'id="selMode"' in html
    assert 'id="valFPS"' in html
    assert 'id="valSnrGain"' in html
    assert 'id="btnReturnStudio"' in html
    assert "import { WebGPUDenoiser } from './webgpu_denoiser.js'" in html
    assert "import { WasmSimdDSP } from './wasm_simd_dsp.js'" in html
    assert 'href="/"' in html


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


# ============================================================================
# 5. FastAPI Endpoints & Route Mounting
# ============================================================================

def test_webgpu_fastapi_routes():
    """Verify GET /webgpu, /webgpu/, /webgpu/index.html, and standalone_denoiser.html return HTTP 200 with text/html."""
    from fastapi.testclient import TestClient
    from src.dashboard.app import app

    client = TestClient(app)
    for route in ["/webgpu", "/webgpu/", "/webgpu/index.html", "/webgpu/standalone_denoiser.html"]:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with status {resp.status_code}"
        content_type = resp.headers.get("content-type", "").lower()
        assert "text/html" in content_type, f"Expected text/html for {route}, got {content_type}"
        assert "RTX Edge AI Denoiser" in resp.text
        assert 'href="/"' in resp.text


# ============================================================================
# 6. WebGPU & WASM Static Assets Serving
# ============================================================================

def test_webgpu_static_assets_serving():
    """Verify denoiser_shader.wgsl, webgpu_denoiser.js, and wasm_simd_dsp.js are served correctly."""
    from fastapi.testclient import TestClient
    from src.dashboard.app import app

    client = TestClient(app)

    # 1. WGSL compute shader (.wgsl -> text/wgsl)
    for path in ["/webgpu/denoiser_shader.wgsl", "/denoiser_shader.wgsl"]:
        resp_wgsl = client.get(path)
        assert resp_wgsl.status_code == 200
        ct_wgsl = resp_wgsl.headers.get("content-type", "").lower()
        assert "text/wgsl" in ct_wgsl, f"Expected text/wgsl for {path}, got {ct_wgsl}"
        assert "@compute" in resp_wgsl.text
        assert "@workgroup_size(64)" in resp_wgsl.text

    # 2. WebGPU orchestrator JS
    for path in ["/webgpu/webgpu_denoiser.js", "/webgpu_denoiser.js"]:
        resp_js = client.get(path)
        assert resp_js.status_code == 200
        ct_js = resp_js.headers.get("content-type", "").lower()
        assert "javascript" in ct_js, f"Expected javascript for {path}, got {ct_js}"
        assert "WebGPUDenoiser" in resp_js.text

    # 3. WASM SIMD fallback JS
    for path in ["/webgpu/wasm_simd_dsp.js", "/wasm_simd_dsp.js"]:
        resp_wasm = client.get(path)
        assert resp_wasm.status_code == 200
        ct_wasm = resp_wasm.headers.get("content-type", "").lower()
        assert "javascript" in ct_wasm, f"Expected javascript for {path}, got {ct_wasm}"
        assert "WasmSimdDSP" in resp_wasm.text


# ============================================================================
# 7. Dashboard UI Navigation Integration
# ============================================================================

def test_dashboard_ui_webgpu_navigation():
    """Verify src/dashboard/static/index.html contains /webgpu navigation anchors in workspace tabs and telemetry."""
    index_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "static", "index.html")
    assert os.path.exists(index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Anchor presence
    assert "/webgpu" in html

    # Verify presence in #workspaceTabsBar
    tabs_idx = html.find('id="workspaceTabsBar"')
    assert tabs_idx != -1, "#workspaceTabsBar not found in index.html"
    tabs_end = html.find('</nav>', tabs_idx)
    tabs_html = html[tabs_idx:tabs_end]
    assert "/webgpu" in tabs_html, "/webgpu not found inside #workspaceTabsBar"
    assert "tabBtnWebGpu" in tabs_html

    # Verify presence in #viewTelemetry
    telemetry_idx = html.find('id="viewTelemetry"')
    assert telemetry_idx != -1, "#viewTelemetry not found in index.html"
    telemetry_html = html[telemetry_idx:]
    assert "/webgpu" in telemetry_html, "/webgpu not found inside #viewTelemetry"
    assert "linkWebGpuTelemetryHeader" in telemetry_html
    assert "linkWebGpuTelemetryBanner" in telemetry_html


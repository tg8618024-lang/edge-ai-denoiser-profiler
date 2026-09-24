"""Unit Tests for VST3 / CLAP Audio Plugin Bridge & Host Simulator."""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.plugins.vst.vst_bridge import (
    AudioCircularFIFO,
    VST3PluginProcessor,
    PARAM_BYPASS,
    PARAM_DENOISE_AMOUNT,
    PARAM_MODEL_SELECT,
    PARAM_CROSSFADE,
    PARAM_PRECISION,
    create_vst_instance,
    destroy_vst_instance,
    vst_set_param,
    vst_get_param,
    vst_get_latency,
    vst_process_buffer,
    vst_reset_instance,
)
from src.plugins.vst.host_simulator import DAWHostSimulator, HostSimulationResult


# ============================================================================
# 1. Circular FIFO Ring Buffer Tests
# ============================================================================

def test_circular_fifo_basic_operations():
    """Verify write, read, peek, count, and clearing operations."""
    fifo = AudioCircularFIFO(capacity=16)
    assert fifo.available_read() == 0
    assert fifo.available_write() == 16

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    written = fifo.write(data)
    assert written == 4
    assert fifo.available_read() == 4
    assert fifo.available_write() == 12

    # Test peek (read without advancing pointer)
    peeked = fifo.peek(4)
    np.testing.assert_array_equal(peeked, data)
    assert fifo.available_read() == 4

    # Test read
    read_data = fifo.read(4)
    np.testing.assert_array_equal(read_data, data)
    assert fifo.available_read() == 0

    # Test clear
    fifo.write(data)
    fifo.clear()
    assert fifo.available_read() == 0
    assert fifo.available_write() == 16


def test_circular_fifo_wrap_around():
    """Verify FIFO wrapping behavior across buffer boundary."""
    fifo = AudioCircularFIFO(capacity=8)
    fifo.write(np.ones(6, dtype=np.float32))
    _ = fifo.read(6)

    # Next write will wrap around index 8 back to index 0
    payload = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    written = fifo.write(payload)
    assert written == 4
    assert fifo.available_read() == 4

    read_back = fifo.read(4)
    np.testing.assert_array_equal(read_back, payload)


# ============================================================================
# 2. VST3 Plugin Processor Parameter Automation & PDC Tests
# ============================================================================

def test_vst_processor_parameters():
    """Verify parameter getting, setting, validation, and metadata."""
    proc = VST3PluginProcessor(sample_rate=16000)

    # Default parameters
    assert proc.get_parameter(PARAM_BYPASS) == 0.0
    assert proc.get_parameter(PARAM_DENOISE_AMOUNT) == 1.0
    assert proc.get_parameter(PARAM_MODEL_SELECT) == 0.0
    assert proc.get_parameter(PARAM_CROSSFADE) == 0.5
    assert proc.get_parameter(PARAM_PRECISION) == 0.0

    # Updates
    proc.set_parameter(PARAM_BYPASS, 1.0)
    assert proc.get_parameter(PARAM_BYPASS) == 1.0

    proc.set_parameter(PARAM_DENOISE_AMOUNT, 0.75)
    assert abs(proc.get_parameter(PARAM_DENOISE_AMOUNT) - 0.75) < 1e-4

    proc.set_parameter(PARAM_MODEL_SELECT, 2.0)
    assert proc.get_parameter(PARAM_MODEL_SELECT) == 2.0

    proc.set_parameter(PARAM_CROSSFADE, 0.3)
    assert abs(proc.get_parameter(PARAM_CROSSFADE) - 0.3) < 1e-4

    proc.set_parameter(PARAM_PRECISION, 2.0)
    assert proc.get_parameter(PARAM_PRECISION) == 2.0

    # Invalid parameter
    with pytest.raises(KeyError):
        proc.set_parameter("invalidParam", 1.0)

    # Definitions
    defs = proc.get_parameter_definitions()
    assert PARAM_BYPASS in defs
    assert PARAM_DENOISE_AMOUNT in defs
    assert PARAM_MODEL_SELECT in defs


def test_vst_processor_pdc_latency():
    """Verify sample-accurate Plugin Delay Compensation latency reporting."""
    proc = VST3PluginProcessor(sample_rate=16000, hop_length=256)
    assert proc.get_latency_samples() == 256
    assert abs(proc.get_latency_ms() - 16.0) < 1e-3


# ============================================================================
# 3. Variable Block Size Streaming Tests (32, 64, 128, 256, 512, 1024)
# ============================================================================

@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512, 1024])
def test_vst_processor_variable_block_sizes(block_size):
    """Verify continuous audio streaming across all standard DAW buffer sizes."""
    proc = VST3PluginProcessor(sample_rate=16000)
    simulator = DAWHostSimulator(sample_rate=16000)

    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out_audio, result = simulator.simulate_streaming(tone, block_size=block_size, processor=proc)

    assert len(out_audio) == len(tone)
    assert not np.isnan(out_audio).any()
    assert not np.isinf(out_audio).any()
    assert result.num_blocks > 0
    assert result.throughput_blocks_per_sec > 100.0


def test_vst_processor_dynamic_block_size_schedule():
    """Verify host dynamically switching block sizes between calls."""
    proc = VST3PluginProcessor(sample_rate=16000)
    simulator = DAWHostSimulator(sample_rate=16000)

    audio_len = 4096
    tone = (0.3 * np.random.randn(audio_len)).astype(np.float32)
    schedule = [32, 64, 128, 256, 512, 64, 32, 1024]

    out_audio, result = simulator.simulate_variable_block_sizes(tone, block_schedule=schedule, processor=proc)

    assert len(out_audio) == audio_len
    assert not np.isnan(out_audio).any()
    assert result.clicks_detected == 0


# ============================================================================
# 4. Bypass Mode & Sample-Exact PDC Alignment Tests
# ============================================================================

def test_vst_processor_bypass_alignment():
    """Verify that bypass mode exactly reproduces the input delayed by PDC latency."""
    simulator = DAWHostSimulator(sample_rate=16000)

    # 1-second sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 300 * t)

    is_aligned = simulator.verify_bypass_alignment(tone, block_size=128)
    assert is_aligned, "Bypassed output should match input delayed by exactly 256 samples"


# ============================================================================
# 5. Parameter Automation Smoothing Tests (Anti-Pop/Click)
# ============================================================================

def test_vst_processor_parameter_automation_ramp():
    """Verify parameter ramp simulation without clicks/pops."""
    simulator = DAWHostSimulator(sample_rate=16000)

    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

    # Ramp denoise amount from 0.0 (dry) to 1.0 (wet)
    out_audio, result = simulator.simulate_parameter_ramp(
        tone,
        param_id=PARAM_DENOISE_AMOUNT,
        start_val=0.0,
        end_val=1.0,
        block_size=64,
    )
    assert len(out_audio) == len(tone)
    assert result.clicks_detected == 0

    # Ramp crossfade from 0.0 to 1.0
    out_audio2, result2 = simulator.simulate_parameter_ramp(
        tone,
        param_id=PARAM_CROSSFADE,
        start_val=0.0,
        end_val=1.0,
        block_size=64,
    )
    assert len(out_audio2) == len(tone)
    assert result2.clicks_detected == 0


# ============================================================================
# 6. C ABI Interface Helper Tests
# ============================================================================

def test_c_abi_interface():
    """Verify C ABI export hooks for JUCE / nih-plug wrappers."""
    handle = create_vst_instance(sample_rate=16000)
    assert handle > 0

    # Latency
    latency = vst_get_latency(handle)
    assert latency == 256

    # Parameter access
    res = vst_set_param(handle, PARAM_DENOISE_AMOUNT, 0.65)
    assert res == 0
    val = vst_get_param(handle, PARAM_DENOISE_AMOUNT)
    assert abs(val - 0.65) < 1e-4

    # Audio block processing
    in_chunk = np.random.randn(128).astype(np.float32) * 0.1
    out_chunk = vst_process_buffer(handle, in_chunk)
    assert out_chunk is not None
    assert len(out_chunk) == 128
    assert not np.isnan(out_chunk).any()

    # Reset
    assert vst_reset_instance(handle) == 0

    # Destroy
    assert destroy_vst_instance(handle) == 0
    assert destroy_vst_instance(handle) == -1  # Already destroyed
    assert vst_get_latency(handle) == -1

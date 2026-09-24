"""Unit Tests for FPGA Verilog RTL Co-Processor & Cycle-Accurate Simulator."""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware.fpga_rtl.verify_rtl_sim import (
    CycleAccurateRTLSimulator,
    verify_numerical_parity,
)


# ============================================================================
# 1. Verilog RTL Synthesizable Source Code Integrity Tests
# ============================================================================

def test_verilog_rtl_module_file_exists_and_syntax_structure():
    """Verify hardware/fpga_rtl/wiener_filter_q15.v structure, ports, and pipeline."""
    v_path = os.path.join(PROJECT_ROOT, "hardware", "fpga_rtl", "wiener_filter_q15.v")
    assert os.path.exists(v_path), f"RTL file missing: {v_path}"

    with open(v_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Module declaration and end
    assert "module wiener_filter_q15" in src
    assert "endmodule" in src

    # AXI4-Stream Slave Interface ports
    assert "input  wire [DATA_WIDTH-1:0]  s_axis_tdata" in src
    assert "input  wire                   s_axis_tvalid" in src
    assert "output wire                   s_axis_tready" in src
    assert "input  wire                   s_axis_tlast" in src
    assert "input  wire [BIN_WIDTH-1:0]   s_axis_tuser" in src

    # AXI4-Stream Master Interface ports
    assert "output wire [DATA_WIDTH-1:0]  m_axis_tdata" in src
    assert "output wire                   m_axis_tvalid" in src
    assert "input  wire                   m_axis_tready" in src
    assert "output wire                   m_axis_tlast" in src
    assert "output wire [BIN_WIDTH-1:0]   m_axis_tuser" in src

    # Verify 5 Pipeline Stages
    assert "Pipeline Stage 1" in src
    assert "Pipeline Stage 2" in src
    assert "Pipeline Stage 3" in src
    assert "Pipeline Stage 4" in src
    assert "Pipeline Stage 5" in src

    # Dual-port Noise RAM
    assert "noise_ram" in src


def test_verilog_testbench_file_exists_and_structure():
    """Verify hardware/fpga_rtl/tb_wiener_filter_q15.v structure and test stimuli."""
    tb_path = os.path.join(PROJECT_ROOT, "hardware", "fpga_rtl", "tb_wiener_filter_q15.v")
    assert os.path.exists(tb_path), f"Testbench file missing: {tb_path}"

    with open(tb_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "module tb_wiener_filter_q15" in src
    assert "wiener_filter_q15" in src  # Instantiation of UUT
    assert "$finish" in src
    assert "endmodule" in src


# ============================================================================
# 2. Cycle-Accurate Simulator Pipeline Latency & Backpressure Tests
# ============================================================================

def test_rtl_pipeline_5_cycle_latency():
    """Verify that a single input sample emerges at master interface exactly 5 cycles later."""
    sim = CycleAccurateRTLSimulator(num_bins=257)
    sim.reset()

    # Apply 1 valid sample at cycle 0
    s_rdy, m_data, m_valid, m_last, m_user = sim.step(
        s_axis_tdata=5000,
        s_axis_tvalid=True,
        s_axis_tlast=False,
        s_axis_tuser=0,
        m_axis_tready=True,
    )
    assert s_rdy is True
    assert m_valid is False  # Cycle 1: not reached Stage 5 yet

    # Advance cycles 1, 2, 3, 4 with no more inputs
    for cycle in range(1, 4):
        s_rdy, m_data, m_valid, m_last, m_user = sim.step(
            s_axis_tdata=0,
            s_axis_tvalid=False,
            s_axis_tlast=False,
            s_axis_tuser=0,
            m_axis_tready=True,
        )
        assert m_valid is False, f"Output unexpectedly valid at cycle {cycle}"

    # Cycle 5: Sample must now emerge from Stage 5
    s_rdy, m_data, m_valid, m_last, m_user = sim.step(
        s_axis_tdata=0,
        s_axis_tvalid=False,
        s_axis_tlast=False,
        s_axis_tuser=0,
        m_axis_tready=True,
    )
    assert m_valid is True, "Output must be valid at cycle 5"
    assert m_user == 0, "Output bin index must match input bin index 0"


def test_rtl_backpressure_stall_handling():
    """Verify that master backpressure (m_axis_tready=False) holds pipeline state."""
    sim = CycleAccurateRTLSimulator(num_bins=257)
    sim.reset()

    # Fill pipeline with 5 samples
    for i in range(5):
        sim.step(
            s_axis_tdata=1000 * (i + 1),
            s_axis_tvalid=True,
            s_axis_tlast=False,
            s_axis_tuser=i,
            m_axis_tready=True,
        )

    # Now stage 5 has a valid sample. If downstream deasserts m_axis_tready,
    # the pipeline should stall
    s_rdy, m_data, m_valid, m_last, m_user = sim.step(
        s_axis_tdata=9999,
        s_axis_tvalid=True,
        s_axis_tlast=False,
        s_axis_tuser=5,
        m_axis_tready=False,  # Stall!
    )
    # Pipeline is holding valid data at stage 5 and cannot advance
    assert m_valid is True
    assert s_rdy is False, "Slave ready should deassert on backpressure stall"


# ============================================================================
# 3. Numerical Parity vs FixedPointWienerDSP Benchmark (SQNR > 60 dB)
# ============================================================================

def test_rtl_numerical_parity_against_fixed_point_dsp():
    """Verify cycle-accurate RTL model matches FixedPointWienerDSP with SQNR > 60 dB."""
    res = verify_numerical_parity(num_frames=5, num_bins=257, verbose=False)
    assert res["passed"] is True, f"Parity test failed: {res}"
    assert res["overall_sqnr_db"] >= 60.0, f"SQNR {res['overall_sqnr_db']} dB < 60 dB threshold"
    assert res["min_frame_sqnr_db"] >= 60.0


def test_verify_rtl_sim_cli(monkeypatch):
    """Verify verify_rtl_sim CLI entrypoint executes and returns exit code 0."""
    from hardware.fpga_rtl.verify_rtl_sim import main
    monkeypatch.setattr(sys, "argv", ["verify_rtl_sim.py", "--frames", "3", "--quiet"])
    exit_code = main()
    assert exit_code == 0


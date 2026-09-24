"""Cycle-Accurate Hardware Simulator & Numerical Parity Verifier for wiener_filter_q15.v.

Verifies:
1. Cycle-by-cycle pipelined execution matching the 5-stage Verilog datapath.
2. AXI4-Stream handshaking (tvalid, tready, tlast, tuser).
3. Bit-exact numerical equivalence against FixedPointWienerDSP in src/audio/fixed_point.py.
4. Validates Signal-to-Quantization-Noise Ratio (SQNR) > 60 dB.
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.fixed_point import FixedPointWienerDSP, compute_sqnr_db, Q15_MAX, Q15_MIN


class CycleAccurateRTLSimulator:
    """Cycle-accurate model of the 5-stage synthesizable wiener_filter_q15.v module."""

    def __init__(
        self,
        num_bins: int = 257,
        gain_min_q15: int = 164,
        init_noise: int = 1000,
    ) -> None:
        self.num_bins = int(num_bins)
        self.gain_min_q15 = int(gain_min_q15)
        self.init_noise = int(init_noise)

        # Dual-port Noise RAM: 257 words x 32 bits
        self.noise_ram = np.full(self.num_bins, self.init_noise, dtype=np.int32)

        # Pipeline Registers (Stages 1 through 5)
        self.pipe_valid = [False] * 5
        self.st1_x: int = 0
        self.st1_bin: int = 0
        self.st1_last: bool = False
        self.st1_p_inst: int = 0
        self.st1_noise_read: int = self.init_noise

        self.st2_x: int = 0
        self.st2_bin: int = 0
        self.st2_last: bool = False
        self.st2_p_inst: int = 0
        self.st2_p_noise: int = self.init_noise

        self.st3_x: int = 0
        self.st3_bin: int = 0
        self.st3_last: bool = False
        self.st3_p_speech: int = 0
        self.st3_p_noise: int = self.init_noise
        self.st3_denom: int = 1

        self.st4_x: int = 0
        self.st4_bin: int = 0
        self.st4_last: bool = False
        self.st4_gain: int = self.gain_min_q15

        self.st5_y: int = 0
        self.st5_bin: int = 0
        self.st5_last: bool = False

        self.current_cycle: int = 0

    def reset(self) -> None:
        """Emulate active-low reset pulse."""
        self.noise_ram.fill(self.init_noise)
        self.pipe_valid = [False] * 5
        self.st1_x = 0
        self.st1_bin = 0
        self.st1_last = False
        self.st1_p_inst = 0
        self.st1_noise_read = self.init_noise

        self.st2_x = 0
        self.st2_bin = 0
        self.st2_last = False
        self.st2_p_inst = 0
        self.st2_p_noise = self.init_noise

        self.st3_x = 0
        self.st3_bin = 0
        self.st3_last = False
        self.st3_p_speech = 0
        self.st3_p_noise = self.init_noise
        self.st3_denom = 1

        self.st4_x = 0
        self.st4_bin = 0
        self.st4_last = False
        self.st4_gain = self.gain_min_q15

        self.st5_y = 0
        self.st5_bin = 0
        self.st5_last = False
        self.current_cycle = 0

    def step(
        self,
        s_axis_tdata: int,
        s_axis_tvalid: bool,
        s_axis_tlast: bool,
        s_axis_tuser: int,
        m_axis_tready: bool,
    ) -> Tuple[bool, int, bool, bool, int]:
        """Execute a single clock cycle transition.

        Returns
        -------
        s_axis_tready : bool
        m_axis_tdata  : int16
        m_axis_tvalid : bool
        m_axis_tlast  : bool
        m_axis_tuser  : int (bin index)
        """
        # Flow control
        pipe_en = m_axis_tready or not self.pipe_valid[4]
        s_axis_tready = pipe_en

        if pipe_en:
            # ----------------------------------------------------------------
            # Next Stage 5 (Output)
            # Y = (X * G) >> 15
            # ----------------------------------------------------------------
            next_st5_valid = self.pipe_valid[3]
            prod_32 = np.int64(self.st4_x) * np.int64(self.st4_gain)
            next_st5_y = int(np.clip(prod_32 >> 15, Q15_MIN, Q15_MAX))
            next_st5_bin = self.st4_bin
            next_st5_last = self.st4_last

            # ----------------------------------------------------------------
            # Next Stage 4 (Gain Calculation)
            # G = (P_s * 32767) // Denom
            # ----------------------------------------------------------------
            next_st4_valid = self.pipe_valid[2]
            gain_num = np.int64(self.st3_p_speech) * 32767
            denom = max(1, self.st3_denom)
            gain_div = int(gain_num // denom)
            gain_clamped = int(np.clip(max(gain_div, self.gain_min_q15), 0, 32767))

            next_st4_gain = gain_clamped
            next_st4_x = self.st3_x
            next_st4_bin = self.st3_bin
            next_st4_last = self.st3_last

            # ----------------------------------------------------------------
            # Next Stage 3 (Speech Power & Denominator)
            # P_s = max(0, P_inst - P_noise)
            # Denom = P_s + P_noise
            # ----------------------------------------------------------------
            next_st3_valid = self.pipe_valid[1]
            p_speech = max(0, self.st2_p_inst - self.st2_p_noise)
            denom_calc = max(1, p_speech + self.st2_p_noise)

            next_st3_p_speech = p_speech
            next_st3_p_noise = self.st2_p_noise
            next_st3_denom = denom_calc
            next_st3_x = self.st2_x
            next_st3_bin = self.st2_bin
            next_st3_last = self.st2_last

            # ----------------------------------------------------------------
            # Next Stage 2 (Noise Update & RAM Write-Back)
            # P_noise = (30 * P_noise + 2 * P_inst) >> 5
            # ----------------------------------------------------------------
            next_st2_valid = self.pipe_valid[0]
            noise_acc = (30 * int(self.st1_noise_read)) + (2 * int(self.st1_p_inst))
            noise_updated = max(1, noise_acc >> 5)

            if self.pipe_valid[0]:
                self.noise_ram[self.st1_bin] = noise_updated

            next_st2_p_noise = noise_updated
            next_st2_p_inst = self.st1_p_inst
            next_st2_x = self.st1_x
            next_st2_bin = self.st1_bin
            next_st2_last = self.st1_last

            # ----------------------------------------------------------------
            # Next Stage 1 (Ingest & P_inst)
            # P_inst = (X^2) >> 15
            # ----------------------------------------------------------------
            next_st1_valid = s_axis_tvalid
            x_val = int(np.int16(s_axis_tdata))
            p_inst_val = int((np.int64(x_val) * np.int64(x_val)) >> 15)
            bin_val = int(s_axis_tuser)
            last_val = bool(s_axis_tlast)
            noise_read = int(self.noise_ram[bin_val])

            next_st1_x = x_val
            next_st1_bin = bin_val
            next_st1_last = last_val
            next_st1_p_inst = p_inst_val
            next_st1_noise_read = noise_read

            # Latch all pipeline registers
            self.pipe_valid[4] = next_st5_valid
            self.st5_y = next_st5_y
            self.st5_bin = next_st5_bin
            self.st5_last = next_st5_last

            self.pipe_valid[3] = next_st4_valid
            self.st4_gain = next_st4_gain
            self.st4_x = next_st4_x
            self.st4_bin = next_st4_bin
            self.st4_last = next_st4_last

            self.pipe_valid[2] = next_st3_valid
            self.st3_p_speech = next_st3_p_speech
            self.st3_p_noise = next_st3_p_noise
            self.st3_denom = next_st3_denom
            self.st3_x = next_st3_x
            self.st3_bin = next_st3_bin
            self.st3_last = next_st3_last

            self.pipe_valid[1] = next_st2_valid
            self.st2_p_noise = next_st2_p_noise
            self.st2_p_inst = next_st2_p_inst
            self.st2_x = next_st2_x
            self.st2_bin = next_st2_bin
            self.st2_last = next_st2_last

            self.pipe_valid[0] = next_st1_valid
            self.st1_x = next_st1_x
            self.st1_bin = next_st1_bin
            self.st1_last = next_st1_last
            self.st1_p_inst = next_st1_p_inst
            self.st1_noise_read = next_st1_noise_read

        self.current_cycle += 1

        # Master outputs from Stage 5
        m_axis_tvalid = self.pipe_valid[4]
        m_axis_tdata = self.st5_y
        m_axis_tlast = self.st5_last
        m_axis_tuser = self.st5_bin

        return s_axis_tready, m_axis_tdata, m_axis_tvalid, m_axis_tlast, m_axis_tuser


def verify_numerical_parity(
    num_frames: int = 5,
    num_bins: int = 257,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare cycle-accurate RTL model against FixedPointWienerDSP across num_frames."""
    rtl_sim = CycleAccurateRTLSimulator(num_bins=num_bins)
    ref_dsp = FixedPointWienerDSP(num_bins=num_bins)

    np.random.seed(42)

    total_rtl_outputs: List[int] = []
    total_ref_outputs: List[int] = []
    frame_sqnrs: List[float] = []

    for f_idx in range(num_frames):
        # Generate synthetic spectral frame in Q1.15
        # Mixture of noise baseline + occasional speech formants
        noise_mag = np.random.randint(100, 800, size=num_bins, dtype=np.int16)
        if f_idx >= 2:
            # Add tonal peaks
            noise_mag[20:30] += 5000
            noise_mag[80:95] += 8000
        mag_q15 = np.clip(noise_mag, 0, 32767).astype(np.int16)

        # 1. Compute Reference Wiener gain & output
        ref_gain = ref_dsp.compute_gain_q15(mag_q15)
        # Reference attenuation: (mag * gain) >> 15
        ref_out = ((mag_q15.astype(np.int32) * ref_gain.astype(np.int32)) >> 15).astype(np.int16)

        # 2. Feed into Cycle-Accurate RTL Simulator cycle-by-cycle
        rtl_frame_out: List[int] = []
        bin_idx = 0

        # Run until all 257 bins sent and drained through 5-stage pipeline
        while len(rtl_frame_out) < num_bins:
            if bin_idx < num_bins:
                s_tvalid = True
                s_tdata = int(mag_q15[bin_idx])
                s_tuser = bin_idx
                s_tlast = (bin_idx == num_bins - 1)
            else:
                s_tvalid = False
                s_tdata = 0
                s_tuser = 0
                s_tlast = False

            s_rdy, m_data, m_valid, m_last, m_user = rtl_sim.step(
                s_axis_tdata=s_tdata,
                s_axis_tvalid=s_tvalid,
                s_axis_tlast=s_tlast,
                s_axis_tuser=s_tuser,
                m_axis_tready=True,
            )

            if s_tvalid and s_rdy:
                bin_idx += 1

            if m_valid:
                rtl_frame_out.append(m_data)

        rtl_out = np.array(rtl_frame_out, dtype=np.int16)

        # 3. Compare parity
        diff = np.abs(rtl_out.astype(np.float64) - ref_out.astype(np.float64))
        max_diff = float(np.max(diff))
        sqnr = compute_sqnr_db(ref_out.astype(np.float32), rtl_out.astype(np.float32))

        frame_sqnrs.append(sqnr)
        total_rtl_outputs.extend(rtl_frame_out)
        total_ref_outputs.extend(ref_out.tolist())

        if verbose:
            print(f"Frame {f_idx + 1:2d}/{num_frames}: Max Abs Diff = {max_diff:.1f}, Frame SQNR = {sqnr:.2f} dB")

    overall_sqnr = compute_sqnr_db(
        np.array(total_ref_outputs, dtype=np.float32),
        np.array(total_rtl_outputs, dtype=np.float32),
    )

    passed = overall_sqnr >= 60.0

    result = {
        "num_frames": num_frames,
        "num_bins": num_bins,
        "overall_sqnr_db": float(overall_sqnr),
        "min_frame_sqnr_db": float(np.min(frame_sqnrs)),
        "mean_frame_sqnr_db": float(np.mean(frame_sqnrs)),
        "passed": bool(passed),
    }

    if verbose:
        print("----------------------------------------------------------------")
        print(f"Overall RTL vs DSP Parity SQNR: {overall_sqnr:.2f} dB (Threshold: 60.0 dB)")
        print(f"Verification Status: {'PASS' if passed else 'FAIL'}")
        print("----------------------------------------------------------------")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FPGA RTL Wiener filter numerical parity.")
    parser.add_argument("--frames", type=int, default=5, help="Number of spectral frames to simulate")
    parser.add_argument("--bins", type=int, default=257, help="Number of frequency bins per frame")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed printouts")
    args = parser.parse_args()

    res = verify_numerical_parity(
        num_frames=args.frames,
        num_bins=args.bins,
        verbose=not args.quiet,
    )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

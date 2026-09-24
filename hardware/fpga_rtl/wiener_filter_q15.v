// ============================================================================
// Module: wiener_filter_q15
// Description: Synthesizable 5-Stage Pipelined Q1.15 Fixed-Point Wiener Filter
//              AXI4-Stream Co-Processor for Ultra-Low-Power Edge Audio DSP.
// Target: Xilinx Vivado (Artix-7, Kintex, Zynq) / Intel Quartus (Cyclone V, Agilex)
// Standard: Verilog-2001
// ============================================================================

`timescale 1ns / 1ps

module wiener_filter_q15 #(
    parameter integer NUM_BINS          = 257,
    parameter integer BIN_WIDTH         = 9,      // ceil(log2(NUM_BINS))
    parameter integer DATA_WIDTH        = 16,     // Q1.15 audio / magnitude
    parameter integer POWER_WIDTH       = 32,     // 32-bit power accumulator
    parameter [15:0]  GAIN_MIN_Q15      = 16'd164, // ~0.005 gain floor
    parameter [31:0]  INIT_NOISE_POWER  = 32'd1000 // Initial noise PSD
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // AXI4-Stream Slave Interface (Input Magnitude Spectrum X[k])
    input  wire [DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                   s_axis_tvalid,
    output wire                   s_axis_tready,
    input  wire                   s_axis_tlast,
    input  wire [BIN_WIDTH-1:0]   s_axis_tuser,    // Frequency Bin Index k (0..256)

    // AXI4-Stream Master Interface (Filtered Output Spectrum Y[k])
    output wire [DATA_WIDTH-1:0]  m_axis_tdata,
    output wire                   m_axis_tvalid,
    input  wire                   m_axis_tready,
    output wire                   m_axis_tlast,
    output wire [BIN_WIDTH-1:0]   m_axis_tuser     // Frequency Bin Index k
);

    // ------------------------------------------------------------------------
    // Pipeline Stall & Flow Control
    // ------------------------------------------------------------------------
    // Pipeline moves forward when master is ready or when output stage is not holding valid data
    wire pipe_en;
    reg  pipe_valid [0:4];

    assign s_axis_tready = pipe_en;
    assign pipe_en       = m_axis_tready || !pipe_valid[4];

    // ------------------------------------------------------------------------
    // Dual-Port Noise Floor RAM: 257 words x 32 bits
    // ------------------------------------------------------------------------
    reg [POWER_WIDTH-1:0] noise_ram [0:NUM_BINS-1];

    integer init_idx;
    initial begin
        for (init_idx = 0; init_idx < NUM_BINS; init_idx = init_idx + 1) begin
            noise_ram[init_idx] = INIT_NOISE_POWER;
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 1: Ingest & Instantaneous Power Estimation
    // P_inst = (X^2) >> 15
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] st1_x;
    reg [BIN_WIDTH-1:0]         st1_bin;
    reg                         st1_last;
    reg [POWER_WIDTH-1:0]       st1_p_inst;
    reg [POWER_WIDTH-1:0]       st1_noise_read;

    wire signed [31:0] x_mult = {{16{s_axis_tdata[15]}}, s_axis_tdata} * {{16{s_axis_tdata[15]}}, s_axis_tdata};
    wire [POWER_WIDTH-1:0] p_inst_calc = x_mult[30:15]; // Shift right by 15 bits

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid[0]  <= 1'b0;
            st1_x          <= {DATA_WIDTH{1'b0}};
            st1_bin        <= {BIN_WIDTH{1'b0}};
            st1_last       <= 1'b0;
            st1_p_inst     <= {POWER_WIDTH{1'b0}};
            st1_noise_read <= INIT_NOISE_POWER;
        end else if (pipe_en) begin
            pipe_valid[0]  <= s_axis_tvalid;
            st1_x          <= s_axis_tdata;
            st1_bin        <= s_axis_tuser;
            st1_last       <= s_axis_tlast;
            st1_p_inst     <= p_inst_calc;
            st1_noise_read <= noise_ram[s_axis_tuser];
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 2: Noise Floor Update & RAM Write-Back
    // P_noise[k] = (30 * P_noise[k] + 2 * P_inst) >> 5
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] st2_x;
    reg [BIN_WIDTH-1:0]         st2_bin;
    reg                         st2_last;
    reg [POWER_WIDTH-1:0]       st2_p_inst;
    reg [POWER_WIDTH-1:0]       st2_p_noise;

    // Fixed-point leaky integrator: 30 * P_noise + 2 * P_inst
    wire [36:0] noise_acc = (37'd30 * st1_noise_read) + (37'd2 * st1_p_inst);
    wire [POWER_WIDTH-1:0] noise_updated = (noise_acc[36:5] == 0) ? 32'd1 : noise_acc[36:5];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid[1] <= 1'b0;
            st2_x         <= {DATA_WIDTH{1'b0}};
            st2_bin       <= {BIN_WIDTH{1'b0}};
            st2_last      <= 1'b0;
            st2_p_inst    <= {POWER_WIDTH{1'b0}};
            st2_p_noise   <= INIT_NOISE_POWER;
        end else if (pipe_en) begin
            pipe_valid[1] <= pipe_valid[0];
            st2_x         <= st1_x;
            st2_bin       <= st1_bin;
            st2_last      <= st1_last;
            st2_p_inst    <= st1_p_inst;
            st2_p_noise   <= noise_updated;

            // Write updated noise floor back to dual-port RAM
            if (pipe_valid[0]) begin
                noise_ram[st1_bin] <= noise_updated;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 3: Speech Power Estimation
    // P_s = max(0, P_inst - P_noise)
    // Denom = max(1, P_s + P_noise)
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] st3_x;
    reg [BIN_WIDTH-1:0]         st3_bin;
    reg                         st3_last;
    reg [POWER_WIDTH-1:0]       st3_p_speech;
    reg [POWER_WIDTH-1:0]       st3_p_noise;
    reg [POWER_WIDTH-1:0]       st3_denom;

    wire [POWER_WIDTH-1:0] p_speech_sub = (st2_p_inst > st2_p_noise) ? (st2_p_inst - st2_p_noise) : 32'd0;
    wire [POWER_WIDTH-1:0] denom_calc   = (p_speech_sub + st2_p_noise == 0) ? 32'd1 : (p_speech_sub + st2_p_noise);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid[2] <= 1'b0;
            st3_x         <= {DATA_WIDTH{1'b0}};
            st3_bin       <= {BIN_WIDTH{1'b0}};
            st3_last      <= 1'b0;
            st3_p_speech  <= {POWER_WIDTH{1'b0}};
            st3_p_noise   <= INIT_NOISE_POWER;
            st3_denom     <= 32'd1;
        end else if (pipe_en) begin
            pipe_valid[2] <= pipe_valid[1];
            st3_x         <= st2_x;
            st3_bin       <= st2_bin;
            st3_last      <= st2_last;
            st3_p_speech  <= p_speech_sub;
            st3_p_noise   <= st2_p_noise;
            st3_denom     <= denom_calc;
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 4: Integer Gain Approximation
    // G = (P_s * 32767) / Denom with Gain Floor Clamp [GAIN_MIN_Q15, 32767]
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] st4_x;
    reg [BIN_WIDTH-1:0]         st4_bin;
    reg                         st4_last;
    reg [15:0]                  st4_gain;

    wire [47:0] gain_num = st3_p_speech * 48'd32767;
    wire [31:0] gain_div = gain_num / st3_denom;
    wire [15:0] gain_clamped = (gain_div < GAIN_MIN_Q15) ? GAIN_MIN_Q15 :
                               (gain_div > 32'd32767)    ? 16'd32767     : gain_div[15:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid[3] <= 1'b0;
            st4_x         <= {DATA_WIDTH{1'b0}};
            st4_bin       <= {BIN_WIDTH{1'b0}};
            st4_last      <= 1'b0;
            st4_gain      <= GAIN_MIN_Q15;
        end else if (pipe_en) begin
            pipe_valid[3] <= pipe_valid[2];
            st4_x         <= st3_x;
            st4_bin       <= st3_bin;
            st4_last      <= st3_last;
            st4_gain      <= gain_clamped;
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 5: Spectral Attenuation & Saturation Output
    // Y[k] = (X[k] * G[k]) >> 15
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] st5_y;
    reg [BIN_WIDTH-1:0]         st5_bin;
    reg                         st5_last;

    wire signed [31:0] prod_32 = st4_x * $signed({1'b0, st4_gain});
    wire signed [15:0] y_atten = prod_32[30:15];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid[4] <= 1'b0;
            st5_y         <= {DATA_WIDTH{1'b0}};
            st5_bin       <= {BIN_WIDTH{1'b0}};
            st5_last      <= 1'b0;
        end else if (pipe_en) begin
            pipe_valid[4] <= pipe_valid[3];
            st5_y         <= y_atten;
            st5_bin       <= st4_bin;
            st5_last      <= st4_last;
        end
    end

    // ------------------------------------------------------------------------
    // Master Interface Output Assignments
    // ------------------------------------------------------------------------
    assign m_axis_tdata  = st5_y;
    assign m_axis_tvalid = pipe_valid[4];
    assign m_axis_tlast  = st5_last;
    assign m_axis_tuser  = st5_bin;

endmodule

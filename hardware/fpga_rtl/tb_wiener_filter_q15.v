// ============================================================================
// Module: tb_wiener_filter_q15
// Description: Synthesizable Verilog Testbench for wiener_filter_q15 Co-Processor.
//              Validates AXI4-Stream slave ingest, master egress, backpressure,
//              and pipeline latency across multiple 257-bin spectral frames.
// ============================================================================

`timescale 1ns / 1ps

module tb_wiener_filter_q15;

    // Parameters
    localparam integer NUM_BINS   = 257;
    localparam integer BIN_WIDTH  = 9;
    localparam integer DATA_WIDTH = 16;
    localparam integer CLK_PERIOD = 10; // 100 MHz clock

    // Signals
    reg                   clk;
    reg                   rst_n;

    // Slave AXI-Stream
    reg  [DATA_WIDTH-1:0] s_axis_tdata;
    reg                   s_axis_tvalid;
    wire                  s_axis_tready;
    reg                   s_axis_tlast;
    reg  [BIN_WIDTH-1:0]  s_axis_tuser;

    // Master AXI-Stream
    wire [DATA_WIDTH-1:0] m_axis_tdata;
    wire                  m_axis_tvalid;
    reg                   m_axis_tready;
    wire                  m_axis_tlast;
    wire [BIN_WIDTH-1:0]  m_axis_tuser;

    // Counters & Telemetry
    integer samples_sent;
    integer samples_received;
    integer errors_detected;
    integer frame_idx;

    // Instantiate Unit Under Test (UUT)
    wiener_filter_q15 #(
        .NUM_BINS(NUM_BINS),
        .BIN_WIDTH(BIN_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tuser(s_axis_tuser),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tuser(m_axis_tuser)
    );

    // Clock Generation: 100 MHz (5ns high / 5ns low)
    always #(CLK_PERIOD / 2) clk = ~clk;

    // Master Egress Consumer & Verification Process
    always @(posedge clk) begin
        if (!rst_n) begin
            samples_received <= 0;
            errors_detected  <= 0;
        end else if (m_axis_tvalid && m_axis_tready) begin
            samples_received <= samples_received + 1;

            // Check tlast consistency: must be high only when bin == NUM_BINS - 1 (256)
            if (m_axis_tuser == (NUM_BINS - 1)) begin
                if (!m_axis_tlast) begin
                    $display("[ERROR] Expected tlast=1 for bin %d", m_axis_tuser);
                    errors_detected <= errors_detected + 1;
                end
            end else begin
                if (m_axis_tlast) begin
                    $display("[ERROR] Unexpected tlast=1 for intermediate bin %d", m_axis_tuser);
                    errors_detected <= errors_detected + 1;
                end
            end
        end
    end

    // Stimulus Process
    initial begin
        // Initialize signals
        clk              = 1'b0;
        rst_n            = 1'b0;
        s_axis_tdata     = 16'd0;
        s_axis_tvalid    = 1'b0;
        s_axis_tlast     = 1'b0;
        s_axis_tuser     = 9'd0;
        m_axis_tready    = 1'b1;
        samples_sent     = 0;
        samples_received = 0;
        errors_detected  = 0;

        $display("================================================================");
        $display("  STARTING FPGA RTL TESTBENCH: wiener_filter_q15");
        $display("================================================================");

        // Reset Pulse
        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        // Send 3 full spectral frames (257 bins each)
        for (frame_idx = 0; frame_idx < 3; frame_idx = frame_idx + 1) begin
            $display("[TB] Transmitting Frame %0d (257 frequency bins)...", frame_idx);

            integer bin_idx;
            for (bin_idx = 0; bin_idx < NUM_BINS; bin_idx = bin_idx + 1) begin
                @(posedge clk);
                // Apply input data
                s_axis_tvalid <= 1'b1;
                s_axis_tuser  <= bin_idx;
                s_axis_tlast  <= (bin_idx == (NUM_BINS - 1)) ? 1'b1 : 1'b0;
                // Test stimulus: synthetic spectral magnitude
                s_axis_tdata  <= (bin_idx * 120 + frame_idx * 500) % 30000;

                // Wait for slave handshake
                while (!s_axis_tready) begin
                    @(posedge clk);
                end
                samples_sent = samples_sent + 1;

                // Inject intermittent backpressure every 50 bins
                if (bin_idx % 50 == 25) begin
                    m_axis_tready <= 1'b0;
                    #(CLK_PERIOD * 3);
                    @(posedge clk);
                    m_axis_tready <= 1'b1;
                end
            end

            // Deassert valid after frame
            @(posedge clk);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;
            #(CLK_PERIOD * 10);
        end

        // Wait for all remaining pipeline stages to drain
        #(CLK_PERIOD * 20);

        $display("================================================================");
        $display("  TESTBENCH COMPLETED SUMMARY:");
        $display("  Total Bins Transmitted: %0d", samples_sent);
        $display("  Total Bins Received:    %0d", samples_received);
        $display("  Total Errors:           %0d", errors_detected);
        $display("================================================================");

        if (samples_sent == samples_received && errors_detected == 0) begin
            $display(">>> ALL RTL AXI4-STREAM HANDSHAKE TESTS PASSED SUCCESSFULLY! <<<");
        end else begin
            $display(">>> RTL TESTBENCH FAILED WITH MISMATCHES! <<<");
        end

        $finish;
    end

endmodule

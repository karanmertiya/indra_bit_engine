`timescale 1ns / 1ps
`include "npu_params.vh"

module npu_tile_top (
    input wire clk,
    input wire rst_n,
    input wire start,
    
    output reg done,
    output reg [31:0] total_cycles,
    output reg [31:0] bytes_read,
    output reg [31:0] bytes_written
);

    // --- Memory Models (SRAMs) ---
    wire [127:0] act_dout;
    reg [5:0] act_addr;
    sram_model #(128, 6, 64, "sram_act.hex") u_act_sram (
        .clk(clk), .we(1'b0), .waddr(6'd0), .raddr(act_addr), .din(128'd0), .dout(act_dout)
    );

    wire [127:0] wgt_dout;
    reg [5:0] wgt_addr;
    sram_model #(128, 6, 64, "sram_wgt.hex") u_wgt_sram (
        .clk(clk), .we(1'b0), .waddr(6'd0), .raddr(wgt_addr), .din(128'd0), .dout(wgt_dout)
    );

    reg buf_we;
    reg [5:0] buf_raddr;
    reg [5:0] buf_waddr;

    // --- Compute Engines ---
    reg array_drain_en;
    // --- FSM Controller Declarations ---
    localparam S_IDLE = 0, S_MATMUL_Q_FILL = 1, S_MATMUL_Q_FLUSH = 2, S_MATMUL_Q_DRAIN = 3;
    localparam S_MATMUL_K_FILL = 4, S_MATMUL_K_FLUSH = 5, S_MATMUL_K_DRAIN = 6;
    localparam S_ROPE_READ = 7, S_ROPE_WAIT = 8;
    localparam S_DONE = 9;

    reg [3:0] state;

    wire [511:0] buf_dout;
    wire [511:0] buf_din_comb;


    sram_model #(512, 6, 64, "") u_buf_sram (
        .clk(clk), .we(buf_we), .waddr(buf_waddr), .raddr(buf_raddr), .din(buf_din_comb), .dout(buf_dout)
    );

    wire [127:0] array_act_in;
    wire [127:0] array_wgt_in;
    wire [511:0] array_psum_out;

    // Mux SRAM output or 0 for flush
    assign array_act_in = (state == S_MATMUL_Q_FILL || state == S_MATMUL_K_FILL) ? act_dout : 128'd0;
    assign array_wgt_in = (state == S_MATMUL_Q_FILL || state == S_MATMUL_K_FILL) ? wgt_dout : 128'd0;
    
    systolic_array_16x16 u_array (
        .clk(clk),
        .rst(~rst_n),
        .drain_en(array_drain_en),
        .act_in_vec(array_act_in),
        .weight_in_vec(array_wgt_in),
        .psum_out_vec(array_psum_out)
    );

    reg rope_in_valid;
    wire rope_out_valid;
    wire [511:0] rope_vec_out;
    reg [7:0] rope_pos;
    
    rope_vector_unit u_rope_vec (
        .clk(clk),
        .rst_n(rst_n),
        .vec_in(buf_dout),
        .pos(rope_pos),
        .in_valid(rope_in_valid),
        .vec_out(rope_vec_out),
        .out_valid(rope_out_valid)
    );

    assign buf_din_comb = (state == S_MATMUL_Q_DRAIN || state == S_MATMUL_K_DRAIN) ? array_psum_out : rope_vec_out;

    // --- FSM Controller ---
    reg [4:0] counter;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            total_cycles <= 0;
            bytes_read <= 0;
            bytes_written <= 0;
            buf_we <= 0;
            buf_raddr <= 0;
            buf_waddr <= 0;
            rope_in_valid <= 0;
            rope_pos <= 0;
            array_drain_en <= 0;
            act_addr <= 0;
            wgt_addr <= 0;
        end else begin
            total_cycles <= total_cycles + 1;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state <= S_MATMUL_Q_FILL;
                        counter <= 0;
                        act_addr <= 1; // Prime for Cycle 1
                        wgt_addr <= 1;
                        done <= 0;
                        bytes_read <= 0;
                        bytes_written <= 0;
                    end
                end
                
                // --- MATMUL Q ---
                S_MATMUL_Q_FILL: begin
                    
                    if (counter < 30) begin
                        counter <= counter + 1;
                        act_addr <= act_addr + 1;
                        wgt_addr <= wgt_addr + 1;
                    end else begin
                        state <= S_MATMUL_Q_FLUSH;
                        counter <= 0;
                    end
                end
                
                S_MATMUL_Q_FLUSH: begin
                    if (counter < 15) begin
                        counter <= counter + 1;
                    end else begin
                        state <= S_MATMUL_Q_DRAIN;
                        counter <= 0;
                        array_drain_en <= 1;
                        buf_waddr <= 15;
                        buf_we <= 1;
                    end
                end
                
                S_MATMUL_Q_DRAIN: begin
                    if (counter < 15) begin
                        buf_we <= 1;
                        counter <= counter + 1;
                        buf_waddr <= buf_waddr - 1;
                        if (counter == 14) begin
                            act_addr <= 32;
                            wgt_addr <= 32;
                        end
                    end else begin
                        state <= S_MATMUL_K_FILL;
                        counter <= 0;
                        array_drain_en <= 0;
                        buf_we <= 0; // Stop writing after Cycle 15
                        act_addr <= 33; // Prime for Cycle 1
                        wgt_addr <= 33;
                    end
                end

                // --- MATMUL K ---
                S_MATMUL_K_FILL: begin
                    if (counter < 30) begin
                        counter <= counter + 1;
                        act_addr <= act_addr + 1;
                        wgt_addr <= wgt_addr + 1;
                    end else begin
                        state <= S_MATMUL_K_FLUSH;
                        counter <= 0;
                    end
                end
                
                S_MATMUL_K_FLUSH: begin
                    if (counter < 15) begin
                        counter <= counter + 1;
                        if (counter == 14) begin
                            buf_raddr <= 15; // Prime ROPE_READ with Q Row 0
                        end
                    end else begin
                        state <= S_MATMUL_K_DRAIN;
                        counter <= 0;
                        array_drain_en <= 1;
                        buf_waddr <= 31;
                        buf_we <= 1;
                    end
                end
                
                S_MATMUL_K_DRAIN: begin
                    if (counter < 15) begin
                        buf_we <= 1;
                        counter <= counter + 1;
                        buf_waddr <= buf_waddr - 1;
                    end else begin
                        state <= S_ROPE_READ;
                        counter <= 0;
                        array_drain_en <= 0;
                        buf_we <= 0; // Don't write yet
                        buf_waddr <= 31; // Prime
                        buf_raddr <= 0; // Prime for Cycle 0
                        rope_pos <= 0;
                        rope_in_valid <= 0; // Wait for Cycle 1
                    end
                end

                // --- RoPE (Only applied to Q) ---
                S_ROPE_READ: begin
                    // Issue read requests (reading forwards from 0 up to 15)
                    if (counter < 15) begin
                        buf_raddr <= buf_raddr + 1;
                    end
                    
                    // RoPE input valid (valid at Cycle 1 to 16, because SRAM output is delayed by 1)
                    if (counter >= 0 && counter < 16) begin
                        rope_in_valid <= 1;
                    end else begin
                        rope_in_valid <= 0;
                    end
                    
                    // Pos logic (must match the input to RoPE at Cycle 1)
                    if (counter >= 0 && counter < 16) begin
                        rope_pos <= counter;
                    end
                    
                    // Write logic (RoPE output arrives 2 cycles after input)
                    // Input at Cycle 1 -> Output at Cycle 3
                    if (counter >= 2 && counter < 18) begin
                        buf_we <= 1;
                        buf_waddr <= 32 + (counter - 2);
                    end else begin
                        buf_we <= 0;
                    end
                    
                    if (counter == 18) begin
                        state <= S_DONE;
                        counter <= 0;
                    end else begin
                        counter <= counter + 1;
                    end
                end
                
                S_ROPE_WAIT: begin
                    state <= S_DONE;
                end
                
                S_DONE: begin
                    done <= 1;
                    buf_we <= 0;
                    act_addr <= 0; // Pre-prime for next run
                    wgt_addr <= 0;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end
endmodule

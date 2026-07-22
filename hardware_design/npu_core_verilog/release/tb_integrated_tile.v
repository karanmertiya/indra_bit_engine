`timescale 1ns / 1ps
`include "npu_params.vh"

module tb_integrated_tile;
    reg clk;
    reg rst_n;
    reg start;
    
    wire done;
    wire [31:0] total_cycles;
    wire [31:0] bytes_read;
    wire [31:0] bytes_written;

    npu_tile_top u_tile (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .total_cycles(total_cycles),
        .bytes_read(bytes_read),
        .bytes_written(bytes_written)
    );

    // --- Verification ---
    reg [511:0] golden_q_rope [0:15];
    reg [511:0] golden_q      [0:15];
    reg [511:0] golden_k      [0:15];
    integer i, errors;
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    initial begin
        $dumpfile("integrated_tile.vcd");
        $dumpvars;
        $readmemh("sram_golden_q_rope.hex", golden_q_rope);
        $readmemh("sram_golden_q.hex", golden_q);
        $readmemh("sram_golden_k.hex", golden_k);

        rst_n = 0;
        start = 0;
        #20 rst_n = 1;
        
        // --- Test 1: Normal Start ---
        #10 start = 1;
        #10 start = 0;
        
        wait(done);
        #20;
        
        $display("----------------------------------------");
        $display("Test 1 Completed.");
        $display("Hardware Inference Latency: %0d clock cycles", total_cycles);
        
        errors = 0;
        for (i = 0; i < 16; i = i + 1) begin
            if (u_tile.u_buf_sram.mem[i] !== golden_q[i]) begin
                $display("FAIL Test 1 (Raw Q) at Row %0d: Expected %h, Got %h", i, golden_q[i], u_tile.u_buf_sram.mem[i]);
                errors = errors + 1;
            end
            if (u_tile.u_buf_sram.mem[16+i] !== golden_k[i]) begin
                $display("FAIL Test 1 (K) at Row %0d: Expected %h, Got %h", i, golden_k[i], u_tile.u_buf_sram.mem[16+i]);
                errors = errors + 1;
            end
            if (u_tile.u_buf_sram.mem[32+i] !== golden_q_rope[i]) begin
                $display("FAIL Test 1 (Q_RoPE) at Row %0d: Expected %h, Got %h", i, golden_q_rope[i], u_tile.u_buf_sram.mem[32+i]);
                errors = errors + 1;
            end
        end

        // --- Test 2: Double Start & Start Held High ---
        $display("Starting Test 2 (Start Held High)...");
        #10 start = 1;
        wait(!done); // wait until the FSM acknowledges start and clears done
        #100 start = 0; // Hold start high for several cycles
        
        wait(done);
        #20;
        $display("Test 2 Completed.");
        
        for (i = 0; i < 16; i = i + 1) begin
            if (u_tile.u_buf_sram.mem[i] !== golden_q[i]) begin
                $display("FAIL Test 2 (Raw Q) at Row %0d: Expected %h, Got %h", i, golden_q[i], u_tile.u_buf_sram.mem[i]);
                errors = errors + 1;
            end
            if (u_tile.u_buf_sram.mem[16+i] !== golden_k[i]) begin
                $display("FAIL Test 2 (K) at Row %0d: Expected %h, Got %h", i, golden_k[i], u_tile.u_buf_sram.mem[16+i]);
                errors = errors + 1;
            end
            if (u_tile.u_buf_sram.mem[32+i] !== golden_q_rope[i]) begin
                $display("FAIL Test 2 (Q_RoPE) at Row %0d: Expected %h, Got %h", i, golden_q_rope[i], u_tile.u_buf_sram.mem[32+i]);
                errors = errors + 1;
            end
        end
        
        $display("----------------------------------------");
        if (errors == 0) begin
            $display("SUCCESS! 100%% Bit-Accurate Match for all tests (Q, K, and Q_RoPE).");
        end else begin
            $display("Test Failed with %0d errors.", errors);
        end
        $finish;
    end
    
    initial begin
        #150000;
        $display("TIMEOUT ERROR: Simulation hung!");
        $display("FSM State: %0d, Counter: %0d, Done: %0d", u_tile.state, u_tile.counter, done);
        $finish;
    end
endmodule

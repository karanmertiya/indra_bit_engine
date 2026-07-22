`timescale 1ns / 1ps

module sram_model #(
    parameter DATA_WIDTH = 128,
    parameter ADDR_WIDTH = 8,
    parameter DEPTH = 256,
    parameter INIT_FILE = ""
)(
    input wire clk,
    input wire we,
    input wire [ADDR_WIDTH-1:0] waddr,
    input wire [ADDR_WIDTH-1:0] raddr,
    input wire [DATA_WIDTH-1:0] din,
    output reg [DATA_WIDTH-1:0] dout
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always @(posedge clk) begin
        if (we) begin
            mem[waddr] <= din;
        end
        dout <= mem[raddr];
    end

endmodule

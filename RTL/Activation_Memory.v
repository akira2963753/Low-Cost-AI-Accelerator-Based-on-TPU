// Activation_Memory.v
// This module manages the storage and retrieval of activation values for the systolic array.
// It handles the activation memory, allowing for pre-loading and updating of activation values
// based on the input signals. The module is designed to work with a clock and reset signal
// and interfaces with the pre-load unit to ensure that the activation values are ready for computation
// in the systolic array. It also manages the compensation rows for the activation values.

module Activation_Memory #(
    parameter SIZE = 8,
    parameter MEM_SIZE = SIZE*SIZE, // Size of the activation memory
    parameter WRITE_ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter READ_ADDR_WIDTH = $clog2(SIZE), // Address width for the memory
    parameter ACTUVATION_OUT_WIDTH = SIZE*7 // Width of the activation output
)(
    input clk,
    input [6:0] Activation,
    input [WRITE_ADDR_WIDTH-1:0] Wr_Addr,
    input Wr_en,
    input Rd_en,
    input [READ_ADDR_WIDTH-1:0] Rd_Addr,
    output [ACTUVATION_OUT_WIDTH-1:0] Activation_out
); 

    genvar i;
    integer j;
    wire [6:0] Mem_out[0:SIZE-1];

    reg [SIZE-1:0] Wr_en_mask;
    wire [READ_ADDR_WIDTH-1:0] Wr_Addr_mask;
    wire [READ_ADDR_WIDTH-1:0] Wr_Addr_in;

    assign Wr_Addr_mask = Wr_Addr / SIZE;
    assign Wr_Addr_in = Wr_Addr % SIZE;

    always @(*) begin
        for(j = 0; j < SIZE; j = j + 1) begin
            if(Wr_en) begin
                if(Wr_Addr_mask == j) Wr_en_mask[j] = 1;
                else Wr_en_mask[j] = 0;
            end
            else Wr_en_mask[j] = 0;
        end
    end

    // Write into Activation Memory
    generate
        for(i = 0; i < SIZE; i = i + 1) begin : ACTIVATION_RAM_GEN
            RAM_7x8 #(
            .SIZE(SIZE),
            .ADDR_WIDTH(READ_ADDR_WIDTH)
            )ACTIVATION_RAM(clk,Activation,Wr_Addr_in,Wr_en_mask[i],Rd_en,Rd_Addr,Mem_out[i]);
            // Memory out
            assign Activation_out[i*7+6 : i*7] = Mem_out[7-i];
        end
    endgenerate
endmodule


module RAM_7x8 #(
    parameter SIZE = 8,
    parameter ADDR_WIDTH = $clog2(SIZE)
)(
    input clk,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Wr_Addr,
    input Wr_en,
    input Rd_en,
    input [ADDR_WIDTH-1:0] Rd_Addr,
    output reg [6:0] Mem_out
);

    reg [6:0] Activation_Mem[0:SIZE-1]; 
    always @(posedge clk) begin
        if(Wr_en) Activation_Mem[Wr_Addr] <= Activation;
        else if(Rd_en) Mem_out <= Activation_Mem[Rd_Addr];
    end
endmodule
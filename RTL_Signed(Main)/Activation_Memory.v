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
    output reg [ACTUVATION_OUT_WIDTH-1:0] Activation_out
); 

    // Activation Memory
    reg [6:0] Activation_Mem[0:MEM_SIZE-1]; 
    always @(posedge clk) begin
        if(Wr_en) Activation_Mem[Wr_Addr] <= Activation;
        else if(Rd_en) begin
            Activation_out[6:0] <= Activation_Mem[56+Rd_Addr];
            Activation_out[13:7] <=  Activation_Mem[48+Rd_Addr];
            Activation_out[20:14] <= Activation_Mem[40+Rd_Addr];
            Activation_out[27:21] <= Activation_Mem[32+Rd_Addr];
            Activation_out[34:28] <= Activation_Mem[24+Rd_Addr];
            Activation_out[41:35] <= Activation_Mem[16+Rd_Addr];
            Activation_out[48:42] <= Activation_Mem[8+Rd_Addr];
            Activation_out[55:49] <= Activation_Mem[0+Rd_Addr];
        end
    end

    /*always @(negedge clk) begin
        if(Rd_en) begin
            Activation_out[6:0] <= Activation_Mem[56+Rd_Addr];
            Activation_out[13:7] <=  Activation_Mem[48+Rd_Addr];
            Activation_out[20:14] <= Activation_Mem[40+Rd_Addr];
            Activation_out[27:21] <= Activation_Mem[32+Rd_Addr];
            Activation_out[34:28] <= Activation_Mem[24+Rd_Addr];
            Activation_out[41:35] <= Activation_Mem[16+Rd_Addr];
            Activation_out[48:42] <= Activation_Mem[8+Rd_Addr];
            Activation_out[55:49] <= Activation_Mem[0+Rd_Addr];
        end
    end*/


endmodule

// Compensation_Memory.v
// This module handles the storage and retrieval of compensation weights.
// It manages the compensation weights for the systolic array, allowing for pre-loading
// and updating of weights based on the input signals. The module is designed to work
// with a clock and reset signal, and it interfaces with the pre-load unit to ensure
// that the compensation weights are ready for computation in the systolic array.
// The module includes logic for changing columns and managing the valid state of the compensation weights.

module Compensation_Memory #(
    parameter SIZE = 8,
    parameter CMEM_SIZE = SIZE * 3, // Compensation Memory Size
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE), // Address width for the compensation memory
    parameter COMPENSATION_WEIGHT_OUT_WIDTH = SIZE * 4 // Width of the compensation output
)(
    input clk,
    input rst,
    input [3:0] Compensation_Weight,
    input [CMEM_ADDR_WIDTH-1:0] Wr_Addr,
    input Wr_en,
    input [1:0] Rd_Addr,
    input Rd_en,
    output reg [COMPENSATION_WEIGHT_OUT_WIDTH-1:0] Compensation_Weight_out
);
    
    // Declare the Register, Net and Integer
    reg [3:0] Compensation_Mem[0:CMEM_SIZE-1];
    integer i;

    // Assignment
    always@(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            for(i=0;i<CMEM_SIZE;i=i+1) Compensation_Mem[i] <= 0;
        end
        else begin
            if(Wr_en) begin
                Compensation_Mem[Wr_Addr] <= Compensation_Weight;
            end
            else if(Rd_en) begin
                Compensation_Weight_out <= {Compensation_Mem[Rd_Addr+21],Compensation_Mem[Rd_Addr+18],Compensation_Mem[Rd_Addr+15],
                                            Compensation_Mem[Rd_Addr+12],Compensation_Mem[Rd_Addr+9],Compensation_Mem[Rd_Addr+6],
                                            Compensation_Mem[Rd_Addr+3],Compensation_Mem[Rd_Addr]};
            end
            else;
        end
    end
    
endmodule
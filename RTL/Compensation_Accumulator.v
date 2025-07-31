// Compensation_Accumulator.v
// This module accumulates compensation values for the systolic array.

module Compensation_Accumulator #(
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 4 + 1
)(
    input clk,
    input rst,
    input CACC_Write_enable,
    input [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Sum_in,
    output [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Sum_out
);
    reg [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Sum_Reg [0:7];
    integer i;
    always @(posedge clk) begin
        if(CACC_Write_enable) begin
            Compensation_Sum_Reg[0] <= Compensation_Sum_in;
            for(i=1;i<8;i=i+1) Compensation_Sum_Reg[i] <= Compensation_Sum_Reg[i-1];
        end
        else;
    end
    assign Compensation_Sum_out = Compensation_Sum_Reg[0];


endmodule
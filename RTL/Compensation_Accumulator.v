// Compensation_Accumulator.v
// This module accumulates compensation values for the systolic array.

module Compensation_Accumulator(
    input clk,
    input rst,
    input Cal,
    input [13:0] Compensation_Sum_in,
    output reg [13:0] Compensation_Sum_out
);

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Compensation_Sum_out <= 33'd0;
        end
        else begin
            Compensation_Sum_out <= (Cal)? Compensation_Sum_in + Compensation_Sum_out : Compensation_Sum_out;
        end
    end

endmodule
module Accumulator #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8 * 4) + 4) + SIZE + 1 // Size of the partial sum
)(
    input clk,
    input rst,
    input Cal,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output reg [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Partial_Sum_out <= 0;
        end
        else begin
            Partial_Sum_out <= (Cal)? Partial_Sum_in + Partial_Sum_out : Partial_Sum_out;
        end
    end

endmodule
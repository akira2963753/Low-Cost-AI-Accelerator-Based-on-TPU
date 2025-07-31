module Accumulator #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8 * 4) + 4) + SIZE + 1 // Size of the partial sum
)(
    input clk,
    input rst,
    input ACC_Write_enable,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
   
    reg [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_Reg [0:7];
    integer i;
    always @(posedge clk) begin
        if(ACC_Write_enable) begin
            Partial_Sum_Reg[0] <= Partial_Sum_in;
            for(i=1;i<8;i=i+1) Partial_Sum_Reg[i] <= Partial_Sum_Reg[i-1];
        end
        else;
    end
    assign Partial_Sum_out = Partial_Sum_Reg[0];

    /*always @(posedge clk or posedge rst) begin
        if(rst) begin
            Partial_Sum_out <= 0;
        end
        else begin
            Partial_Sum_out <= (ACC_Write_enable)? Partial_Sum_in + Partial_Sum_out : Partial_Sum_out;
        end
    end*/


endmodule
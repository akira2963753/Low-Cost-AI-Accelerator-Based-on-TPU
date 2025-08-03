module Accumulator #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8 * 4) + 4) + SIZE + 1, // Size of the partial sum
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 4 + 1
)(
    input clk,
    input [2:0] Acc_Wr_Addr,
    input ACC_Wr_en,
    input [2:0] CAcc_Wr_Addr,
    input CACC_Wr_en,
    input [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum_in,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
   
    reg [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_Mem [0:7];

    always @(posedge clk) begin // Two port write mem
        if(ACC_Wr_en) begin
            Partial_Sum_Mem[Acc_Wr_Addr] <= Partial_Sum_in + Partial_Sum_Mem[Acc_Wr_Addr];
        end
        else;
        if(CACC_Wr_en) begin
            Partial_Sum_Mem[CAcc_Wr_Addr] <= Compensation_Partial_Sum_in;
        end
        else;
    end
    
    assign Partial_Sum_out = Partial_Sum_Mem[0];

   


endmodule
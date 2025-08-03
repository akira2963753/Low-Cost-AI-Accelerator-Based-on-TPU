module Accumulator #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8 * 4) + 4) + SIZE + 1, // Size of the partial sum
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 5 + 1
)(
    input [2:0] Col, // 觀察用
    input clk,
    input [2:0] Acc_Wr_Addr,
    input ACC_Wr_en,
    input [2:0] CAcc_Wr_Addr,
    input CACC_Wr_en,
    input signed [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum_in,
    input signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
   
    reg signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_Mem [0:7];
    integer i;

    always @(posedge clk) begin // Two port write mem
        if(ACC_Wr_en) begin
            Partial_Sum_Mem[Acc_Wr_Addr] <= Partial_Sum_in + Partial_Sum_Mem[Acc_Wr_Addr];
            $display("--------------------------------- Col = [%01d] --------------------------------------",Col);
            $display("==================================================================================");
            $display("Write Addr [%01d] = %05d (Partial Sum) + %04d (Compensation Value) = %d",Acc_Wr_Addr,Partial_Sum_in,Partial_Sum_Mem[Acc_Wr_Addr],Partial_Sum_in + Partial_Sum_Mem[Acc_Wr_Addr]); 
            $display("==================================================================================");
        end
        else;
        if(CACC_Wr_en) begin
            Partial_Sum_Mem[CAcc_Wr_Addr] <= Compensation_Partial_Sum_in;
        end
        else;
    end
    
    assign Partial_Sum_out = Partial_Sum_Mem[0];

   


endmodule
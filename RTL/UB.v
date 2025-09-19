module UB #(
    parameter SIZE = 8,
    parameter ACTIVATION_WIDTH = 7
)(
    input clk,
    input Wr_en,
    input [5:0] Wr_Addr,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_0,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_1,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_2,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_3,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_4,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_5,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_6,
    input [ACTIVATION_WIDTH-1:0] Wr_Data_7,
    input [5:0] Rd_Addr,
    output [ACTIVATION_WIDTH-1:0] Rd_Data
);

    reg signed [ACTIVATION_WIDTH-1:0] Unified_Buffer [0:63];

    always @(posedge clk) begin
        if(Wr_en) begin
            Unified_Buffer[Wr_Addr] <= Wr_Data_0;
            Unified_Buffer[Wr_Addr+1] <= Wr_Data_1;
            Unified_Buffer[Wr_Addr+2] <= Wr_Data_2;
            Unified_Buffer[Wr_Addr+3] <= Wr_Data_3;
            Unified_Buffer[Wr_Addr+4] <= Wr_Data_4;
            Unified_Buffer[Wr_Addr+5] <= Wr_Data_5;
            Unified_Buffer[Wr_Addr+6] <= Wr_Data_6;
            Unified_Buffer[Wr_Addr+7] <= Wr_Data_7;
        end
        else;
    end
    
    assign Rd_Data = Unified_Buffer[Rd_Addr];


endmodule
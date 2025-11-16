module Compensation_Memory #(
    parameter SIZE = 8,
    parameter CMEM_SIZE = SIZE * 3,
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE),
    parameter COMPENSATION_WEIGHT_OUT_WIDTH = SIZE * 4
)(
    input clk,
    input rst,
    input [3:0] Compensation_Weight,
    input [CMEM_ADDR_WIDTH-1:0] Wr_Addr,
    input Wr_en,
    input [1:0] Rd_Addr,
    input Rd_en,
    output [COMPENSATION_WEIGHT_OUT_WIDTH-1:0] Compensation_Weight_out
);
    genvar i;
    integer j;
    wire [3:0] Mem_out[0:SIZE-1];

    reg [SIZE-1:0] Wr_en_mask;
    wire [SIZE-1:0] Wr_Addr_mask;
    wire [1:0] Wr_Addr_in;

    assign Wr_Addr_mask = Wr_Addr / 3;
    assign Wr_Addr_in = Wr_Addr % 3;
    
    always @(*) begin
        for(j = 0; j < CMEM_SIZE; j = j + 1) begin
            if(Wr_en) begin
                if(Wr_Addr_mask == j) Wr_en_mask[j] = 1;
                else Wr_en_mask[j] = 0;
            end
            else Wr_en_mask[j] = 0;
        end
    end

    // Write into Weight Memory
    generate
        for(i = 0; i < SIZE; i = i + 1) begin : COMPENSATION_RAM_GEN
            RAM_4x3 COMPENSATION_RAM(clk,rst,Compensation_Weight,Wr_Addr_in,Wr_en_mask[i],Rd_Addr,Rd_en,Mem_out[i]);
            // Memory out
            assign Compensation_Weight_out[i*4+3 : i*4] = Mem_out[i];
        end
    endgenerate
    
endmodule

module RAM_4x3 (
    input clk,
    input rst,
    input [3:0] Compensation_Weight,
    input [1:0] Wr_Addr,
    input Wr_en,
    input [1:0] Rd_Addr,
    input Rd_en,
    output reg [3:0] Mem_out
);
    integer k;
    reg [3:0] Compensation_Mem[0:2];
   
    always@(posedge clk or posedge rst) begin
        if(rst) for(k=0;k<3;k=k+1) Compensation_Mem[k] <= 0;
        else begin
            if(Wr_en) Compensation_Mem[Wr_Addr] <= Compensation_Weight;
            else if(Rd_en) Mem_out <= Compensation_Mem[Rd_Addr];
        end
    end

endmodule
module Weight_Memory #(
    parameter SIZE = 8,
    parameter MEM_SIZE = SIZE * SIZE,
    parameter WRITE_ADDR_WIDTH = $clog2(MEM_SIZE),
    parameter READ_ADDR_WIDTH = $clog2(SIZE),
    parameter WEIGHT_OUT_WIDTH = SIZE * 5
)(
    input clk,
    input [WRITE_ADDR_WIDTH-1:0] Wr_Addr,
    input [4:0] Weight_Data,
    input Wr_en,
    input Rd_en,
    input [READ_ADDR_WIDTH-1:0] Rd_Addr,
    output [WEIGHT_OUT_WIDTH-1:0] Weight_out
);
    genvar i;
    integer j;

    wire [4:0] Mem_out[0:SIZE-1];

    reg [SIZE-1:0] Wr_en_mask;
    wire [READ_ADDR_WIDTH-1:0] Wr_Addr_mask;
    wire [READ_ADDR_WIDTH-1:0] Wr_Addr_in;

    assign Wr_Addr_mask = Wr_Addr / SIZE;
    assign Wr_Addr_in = Wr_Addr % SIZE;
    
    always @(*) begin
        for(j = 0; j < SIZE; j = j + 1) begin
            if(Wr_en) begin
                if(Wr_Addr_mask == j) Wr_en_mask[j] = 1;
                else Wr_en_mask[j] = 0;
            end
            else Wr_en_mask[j] = 0;
        end
    end

    // Write into Weight Memory
    generate
        for(i = 0; i < SIZE; i = i + 1) begin : WEIGHT_RAM_GEN
            RAM_5x8 #(
            .SIZE(SIZE),
            .ADDR_WIDTH(READ_ADDR_WIDTH)
            )WEIGHT_RAM(clk,Wr_Addr_in,Weight_Data,Wr_en_mask[i],Rd_en,Rd_Addr,Mem_out[i]);
            // Memory out
            assign Weight_out[i*5+4 : i*5] = Mem_out[i];
        end
    endgenerate
endmodule

module RAM_5x8 #(
    parameter SIZE = 8,
    parameter ADDR_WIDTH = $clog2(SIZE)
)(
    input clk,
    input [ADDR_WIDTH-1:0] Wr_Addr,
    input [4:0] Weight_Data,
    input Wr_en,
    input Rd_en,
    input [ADDR_WIDTH-1:0] Rd_Addr,
    output reg [4:0] Mem_out
);
    reg [4:0] Weight_Mem[0:SIZE-1];

    always @(posedge clk) begin
        if(Wr_en) Weight_Mem[Wr_Addr] <= Weight_Data;
        else if(Rd_en) Mem_out <= Weight_Mem[Rd_Addr];
    end

endmodule

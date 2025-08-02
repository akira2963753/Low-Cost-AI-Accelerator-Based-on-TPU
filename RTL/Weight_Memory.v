module Weight_Memory #(
    parameter SIZE = 8, // Size of the weight memory
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter WRITE_ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter READ_ADDR_WIDTH = $clog2(SIZE), // Address width for the memory
    parameter WEIGHT_OUT_WIDTH = SIZE * 5 // Width of the weight output
)(
    input clk,
    input [WRITE_ADDR_WIDTH-1:0] Wr_Addr,
    input [4:0] Weight_Data,
    input Wr_en,
    input Rd_en,
    input [READ_ADDR_WIDTH-1:0] Rd_Addr,
    output reg [WEIGHT_OUT_WIDTH-1:0] Weight_out
);
    // Weight Memory
    reg [4:0] Weight_Mem[0:MEM_SIZE-1];

    // Write into Weight Memory
    always @(posedge clk) begin
        if(Wr_en) Weight_Mem[Wr_Addr] <= Weight_Data;
        else if(Rd_en) begin
            Weight_out[4:0] <= Weight_Mem[0+Rd_Addr];
            Weight_out[9:5] <= Weight_Mem[8+Rd_Addr];
            Weight_out[14:10] <= Weight_Mem[16+Rd_Addr];
            Weight_out[19:15] <= Weight_Mem[24+Rd_Addr];
            Weight_out[24:20] <= Weight_Mem[32+Rd_Addr];
            Weight_out[29:25] <= Weight_Mem[40+Rd_Addr];
            Weight_out[34:30] <= Weight_Mem[48+Rd_Addr];
            Weight_out[39:35] <= Weight_Mem[56+Rd_Addr];
        end
    end

endmodule

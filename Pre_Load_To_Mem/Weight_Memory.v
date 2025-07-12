// Weight Memory for 8x8 Systolic Array 
// 2025.07.12

module Weight_Memory(
    input clk,
    input rst,
    input [5:0] Weight_Mem_Address_in,
    input [4:0] Weight_Data,
    input done   
);
    localparam Size = 8;
    localparam Mem_Size = Size*Size;

    reg [4:0] Weight_Mem[Mem_Size-1:0];

    always @(posedge clk or posedge rst) begin
        if(rst);
        else begin
            if(done==1'b0) begin
                Weight_Mem[Weight_Mem_Address_in] <= Weight_Data;
            end
            else;
        end
    end
endmodule

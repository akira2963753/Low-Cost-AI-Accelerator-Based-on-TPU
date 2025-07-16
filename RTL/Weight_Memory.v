module Weight_Memory #(
    parameter SIZE = 8, // Size of the weight memory
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE) // Address width for the memory
)(
    input clk,
    input rst,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [4:0] Weight_Data,
    input load_mem_done   
);
    reg [4:0] Weight_Mem[0:MEM_SIZE-1]; // Weight Memory
    integer i;

    always @(posedge clk or posedge rst) begin
        if(rst) for(i=0;i<MEM_SIZE;i=i+1) Weight_Mem[i] <= 5'd0;
        else begin
            if(!load_mem_done) Weight_Mem[Weight_Mem_Address_in] <= Weight_Data;
            else;
        end
    end

endmodule

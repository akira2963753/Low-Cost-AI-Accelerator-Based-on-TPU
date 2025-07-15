module Weight_Memory(
    input clk,
    input rst,
    input [5:0] Weight_Mem_Address_in,
    input [4:0] Weight_Data,
    input load_mem_done   
);
    reg [4:0] Weight_Mem[0:63];
    integer i;

    always @(posedge clk or posedge rst) begin
        if(rst) for(i=0;i<64;i=i+1) Weight_Mem[i] <= 5'd0;
        else begin
            if(load_mem_done==1'b0) Weight_Mem[Weight_Mem_Address_in] <= Weight_Data;
            else;
        end
    end

endmodule

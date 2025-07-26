module Weight_Memory #(
    parameter SIZE = 8, // Size of the weight memory
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter INDEX_WIDTH = ADDR_WIDTH, // Index width for the weight memory
    parameter BIAS_WIDTH = ADDR_WIDTH, // Bias width for the activation memory
    parameter WEIGHT_OUT_WIDTH = SIZE * 5, // Width of the weight output
    parameter SHIFT = $clog2(SIZE) // Shift value for the size of the weight memory
)(
    input clk,
    input rst,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [4:0] Weight_Data,
    input load_mem_done,
    input PreLoadWeight,
    output [WEIGHT_OUT_WIDTH-1:0] Weight_out,
    output Weight_out_valid
);
    reg [4:0] Weight_Mem[0:MEM_SIZE-1]; // Weight Memory
    reg [INDEX_WIDTH-1:0] Index;
    integer i;
    wire [BIAS_WIDTH-1:0] bias;

    assign bias = Index << SHIFT;
    assign Weight_out_valid = (load_mem_done&&PreLoadWeight)? 1'b1 : 1'b0;
    
    // Assign Weight output to the weight memory
    assign Weight_out[4:0] = Weight_Mem[0+Index];
    assign Weight_out[9:5] = Weight_Mem[8+Index];
    assign Weight_out[14:10] = Weight_Mem[16+Index];
    assign Weight_out[19:15] = Weight_Mem[24+Index];
    assign Weight_out[24:20] = Weight_Mem[32+Index];
    assign Weight_out[29:25] = Weight_Mem[40+Index];
    assign Weight_out[34:30] = Weight_Mem[48+Index];
    assign Weight_out[39:35] = Weight_Mem[56+Index];


    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<MEM_SIZE;i=i+1) Weight_Mem[i] <= 5'd0;
            Index <= 0;
        end
        else begin
            if(!load_mem_done) Weight_Mem[Weight_Mem_Address_in] <= Weight_Data;
            else if(PreLoadWeight) begin
                Index <= Index + 1;
            end
            else Index <= 0;
        end
    end

endmodule

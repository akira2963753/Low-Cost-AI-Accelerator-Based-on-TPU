// Compensation Memory for 8x8 Systolic Array 
// 2025.07.12
module Compensation_Memory(
    input clk,
    input rst,
    input [2:0] Compensation_Weight,
    input out_Compensation_valid,
    input change_col,
    input done
);
    // Declare the local parameter
    localparam Size = 8;
    localparam Compensation_Size = Size * 3;
    localparam Index_Width = $clog2(Compensation_Size);

    reg [2:0] Compensation_Mem[Compensation_Size-1:0];
    reg [Index_Width-1:0] Index;
    
    integer i;

    always@(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<24;i=i+1) Compensation_Mem[i] <= 3'd0;
            Index <= 5'd0;
        end
        else begin
            if(done==1'b0) begin
                if(out_Compensation_valid) begin
                    Compensation_Mem[Index] <= Compensation_Weight;
                    // Change Column
                    if(change_col) Index <= (Index==5'd23)? 5'd0 : Index + (5'd3 - (Index%5'd3)); 
                    else Index <= (Index==5'd23)? 5'd0 : Index + 5'd1;
                end
                else if(change_col) Index <= (Index==5'd23)? 5'd0 : Index + (5'd3 - (Index%5'd3));
                else;
            end
            else;
        end
    end
    
                 

endmodule
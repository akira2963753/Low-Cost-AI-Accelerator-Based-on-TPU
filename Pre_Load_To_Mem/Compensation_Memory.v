module Compensation_Memory(
    input clk,
    input rst,
    input [2:0] Compensation_Weight,
    input Compensation_out_valid,
    input change_col,
    input load_weight_done
);
    
    // Declare the Register, Net and Integer
    reg [2:0] Compensation_Mem[23:0];
    reg [4:0] Index; // log2(24) = 5
    wire [4:0] Change_Col_Index;
    integer i;

    // Assignment 
    assign Change_Col_Index = Index + (5'd3 - (Index%5'd3));

    always@(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            for(i=0;i<24;i=i+1) Compensation_Mem[i] <= 3'd0;
            Index <= 5'd0;
        end
        else begin
            if(load_weight_done==1'b0) begin
                if(Compensation_out_valid) begin
                    // Load Compensation Weight into Mem
                    Compensation_Mem[Index] <= Compensation_Weight;
                    // Change Column
                    if(change_col) Index <= (Index==5'd23)? 5'd0 : Change_Col_Index; 
                    else Index <= (Index==5'd23)? 5'd0 : Index + 5'd1;
                end
                else if(change_col) Index <= (Index==5'd23)? 5'd0 : Change_Col_Index;
                else;
            end
            else Index = 5'd0;
        end
    end
    
endmodule
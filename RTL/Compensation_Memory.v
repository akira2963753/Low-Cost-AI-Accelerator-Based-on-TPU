// Compensation_Memory.v
// This module handles the storage and retrieval of compensation weights.
// It manages the compensation weights for the systolic array, allowing for pre-loading
// and updating of weights based on the input signals. The module is designed to work
// with a clock and reset signal, and it interfaces with the pre-load unit to ensure
// that the compensation weights are ready for computation in the systolic array.
// The module includes logic for changing columns and managing the valid state of the compensation weights.

module Compensation_Memory(
    input clk,
    input rst,
    input [2:0] Compensation_Weight,
    input Compensation_out_valid,
    input change_col,
    input load_mem_done,
    input PreLoad_CWeight,
    output reg [23:0] Compensation_Weight_out,
    output Compensation_Weight_out_valid
);
    
    // Declare the Register, Net and Integer
    reg [2:0] Compensation_Mem[23:0];
    reg [4:0] Index; // log2(24) = 5
    wire [4:0] Change_Col_Index;
    integer i;

    // Assignment 
    assign Change_Col_Index = Index + (5'd3 - (Index%5'd3));
    assign Compensation_Weight_out_valid = (load_mem_done&&PreLoad_CWeight&&Index!=5'd3) ? 1'b1 : 1'b0;

    always@(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            for(i=0;i<24;i=i+1) Compensation_Mem[i] <= 3'd0;
            Index <= 5'd0;
        end
        else begin
            if(load_mem_done==1'b0) begin
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
            else if(PreLoad_CWeight) begin
                if(Index!=5'd3) begin
                    // Preload Compensation Weight
                    Compensation_Weight_out <= {Compensation_Mem[Index+21],Compensation_Mem[Index+18],Compensation_Mem[Index+15],
                                                    Compensation_Mem[Index+12],Compensation_Mem[Index+9],Compensation_Mem[Index+6],
                                                    Compensation_Mem[Index+3],Compensation_Mem[Index]};
                    Index <= Index + 5'd1;
                end
                else;
            end
            else Index <= 5'd0;
        end
    end
    
endmodule
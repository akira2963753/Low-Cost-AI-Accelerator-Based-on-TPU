module Activation_Memory (
    input clk,
    input rst,
    input [2:0] Compensation_Row, // The Weight Compensation_row
    input out_Compensation_valid,
    input change_col, // The Weight Loading is done.
    input done,
    output out_valid); 

    // Define the local parameter
    localparam Invalid_Value = 8;

    // Declare the Register, Net and Integer
    reg [7:0] Activation_Mem[0:63];
    reg [3:0] Compensation_Row_Reg[0:23]; // Compensation Row Reg Add a bit for recording Non-Compensation-value
    reg [4:0] Index; // log2(24) = 5
    integer i;

    // Assingment 
    assign out_valid = (done==1);

    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<24;i=i+1) Compensation_Row_Reg[i] <= Invalid_Value;
            Index <= 5'd0;
        end
        else begin
            if(done==0) begin
                if(out_Compensation_valid) begin
                    Compensation_Row_Reg[Index] <=  Compensation_Row;
                    Index <= Index + 1; // Index ++
                end
                else if(change_col) begin // Change column
                    Index <= Index + (5'd3 - (Index%5'd3));
                end
                else; 
            end
            else Index <= 5'd0;
        end
    end 


endmodule

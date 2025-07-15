// Activation_Memory.v
// This module manages the storage and retrieval of activation values for the systolic array.
// It handles the activation memory, allowing for pre-loading and updating of activation values
// based on the input signals. The module is designed to work with a clock and reset signal
// and interfaces with the pre-load unit to ensure that the activation values are ready for computation
// in the systolic array. It also manages the compensation rows for the activation values.

module Activation_Memory (
    input clk,
    input rst,
    input [6:0] Activation,
    input [5:0] Activation_Mem_Address_in,
    input [2:0] Compensation_Row, // The Weight Compensation_row
    input Compensation_out_valid,
    input change_col, // The Weight Loading is done.
    input load_mem_done,
    input Cal,
    output [55:0] Activation_out,
    output [167:0] Activation_cout,
    output Activation_cout_valid
); 
    // Define the local parameter
    localparam Invalid_Value = 8;

    // Declare the Register, Net and Integer
    reg [6:0] Activation_Mem[0:63];
    reg [3:0] Compensation_Row_Reg[0:23]; // Compensation Row Reg Add a bit for recording Non-Compensation-value
    reg [4:0] Index; // log2(24) = 5
    integer i;
    wire [5:0] bias;

    // Assingment Activation_out for Comepensation
    assign bias = Index << 3;
    assign Activation_cout_valid = (Cal&&Index!=5'd8)? 1'b1 : 1'b0;
    assign Activation_cout = (Activation_cout_valid)? {Activation_Mem[Compensation_Row_Reg[21]+bias],Activation_Mem[Compensation_Row_Reg[22]+bias],Activation_Mem[Compensation_Row_Reg[23]+bias],
                             Activation_Mem[Compensation_Row_Reg[18]+bias],Activation_Mem[Compensation_Row_Reg[19]+bias],
                             Activation_Mem[Compensation_Row_Reg[20]+bias],Activation_Mem[Compensation_Row_Reg[15]+bias],
                             Activation_Mem[Compensation_Row_Reg[16]+bias],Activation_Mem[Compensation_Row_Reg[17]+bias],
                             Activation_Mem[Compensation_Row_Reg[12]+bias],Activation_Mem[Compensation_Row_Reg[13]+bias],
                             Activation_Mem[Compensation_Row_Reg[14]+bias],Activation_Mem[Compensation_Row_Reg[9]+bias],
                             Activation_Mem[Compensation_Row_Reg[10]+bias],Activation_Mem[Compensation_Row_Reg[11]+bias],
                             Activation_Mem[Compensation_Row_Reg[6]+bias],Activation_Mem[Compensation_Row_Reg[7]+bias],
                             Activation_Mem[Compensation_Row_Reg[8]+bias],Activation_Mem[Compensation_Row_Reg[5]+bias],
                             Activation_Mem[Compensation_Row_Reg[4]+bias],Activation_Mem[Compensation_Row_Reg[3]+bias],
                             Activation_Mem[Compensation_Row_Reg[2]+bias],Activation_Mem[Compensation_Row_Reg[1]+bias],
                             Activation_Mem[Compensation_Row_Reg[0]+bias]} : 168'd0;

    // Assigment for Activation output to systolic array
    assign Activation_out = {Activation_Mem[7+bias], Activation_Mem[6+bias], Activation_Mem[5+bias],
                             Activation_Mem[4+bias], Activation_Mem[3+bias], Activation_Mem[2+bias],
                             Activation_Mem[1+bias], Activation_Mem[0+bias]};
    

    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<24;i=i+1) Compensation_Row_Reg[i] <= Invalid_Value;
            Index <= 5'd0;
        end
        else begin
            if(load_mem_done==0) begin
                // Load Activation Value into Memory
                Activation_Mem[Activation_Mem_Address_in] <= Activation;

                if(Compensation_out_valid) begin
                    Compensation_Row_Reg[Index] <=  Compensation_Row;
                    Index <= Index + 5'd1; // Index ++
                end
                else if(change_col) begin // Change column
                    Index <= Index + (5'd3 - (Index%5'd3));
                end
                else; 
            end
            else if(Cal) begin
                Index <= (Index==5'd8)? 5'd8 : Index + 5'd1;
            end
            else Index <= 5'd0;
        end
    end 


endmodule

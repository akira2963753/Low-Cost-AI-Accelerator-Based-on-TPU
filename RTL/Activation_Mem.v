// Activation_Memory.v
// This module manages the storage and retrieval of activation values for the systolic array.
// It handles the activation memory, allowing for pre-loading and updating of activation values
// based on the input signals. The module is designed to work with a clock and reset signal
// and interfaces with the pre-load unit to ensure that the activation values are ready for computation
// in the systolic array. It also manages the compensation rows for the activation values.

module Activation_Memory #(
    parameter SIZE = 8,
    parameter SHIFT = $clog2(SIZE), // Shift value for the size of the activation memory
    parameter CROW_WIDTH = $clog2(SIZE), // Compensation row size 
    parameter MEM_SIZE = SIZE*SIZE, // Size of the activation memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter COMPENSATIOPN_ROW_SIZE = SIZE * 3, // Compensation row size
    parameter COMPENSATIOPN_ROW_ADDR_WIDTH = $clog2(COMPENSATIOPN_ROW_SIZE), // Compensation row address width
    parameter INVALID_VALUE = SIZE, // Invalid value for compensation row
    parameter BIAS_WIDTH = ADDR_WIDTH, // Bias width for the activation memory
    parameter ACTUVATION_OUT_WIDTH = SIZE*7 // Width of the activation output
)(
    input clk,
    input rst,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input load_mem_done,
    input Cal,
    output [ACTUVATION_OUT_WIDTH-1:0] Activation_out,
    output [7:0] Activation_out_valid
); 

    // Declare the Register, Net and Integer
    reg [6:0] Activation_Mem[0:MEM_SIZE-1]; // Activation Memory
    reg [COMPENSATIOPN_ROW_ADDR_WIDTH-1:0] Index; // Index for Compensation Row Reg
    wire [BIAS_WIDTH-1:0] bias;
    wire [BIAS_WIDTH-1:0] bias_1;
    wire [BIAS_WIDTH-1:0] bias_2;
    wire [BIAS_WIDTH-1:0] bias_3;
    wire [BIAS_WIDTH-1:0] bias_4;
    wire [BIAS_WIDTH-1:0] bias_5;
    wire [BIAS_WIDTH-1:0] bias_6;
    wire [BIAS_WIDTH-1:0] bias_7;
    integer i;

    // Assingment Activation_out for Comepensation
    assign bias = Index << SHIFT; // Bias for the activation memory address
    assign bias_1 = (Index<1)? 0 : (Index - 1) << SHIFT;
    assign bias_2 = (Index<2)? 0 : (Index - 2) << SHIFT;
    assign bias_3 = (Index<3)? 0 : (Index - 3) << SHIFT;
    assign bias_4 = (Index<4)? 0 : (Index - 4) << SHIFT;
    assign bias_5 = (Index<5)? 0 : (Index - 5) << SHIFT;
    assign bias_6 = (Index<6)? 0 : (Index - 6) << SHIFT;
    assign bias_7 = (Index<7)? 0 : (Index - 7) << SHIFT;
    
    assign Activation_out_valid[0] = (Cal&&Index<SIZE)? 1'b1 : 1'b0;
    assign Activation_out_valid[1] = (Cal&&Index<SIZE+1&&Index>0)? 1'b1 : 1'b0;
    assign Activation_out_valid[2] = (Cal&&Index<SIZE+2&&Index>1)? 1'b1 : 1'b0;
    assign Activation_out_valid[3] = (Cal&&Index<SIZE+3&&Index>2)? 1'b1 : 1'b0;
    assign Activation_out_valid[4] = (Cal&&Index<SIZE+4&&Index>3)? 1'b1 : 1'b0;
    assign Activation_out_valid[5] = (Cal&&Index<SIZE+5&&Index>4)? 1'b1 : 1'b0;
    assign Activation_out_valid[6] = (Cal&&Index<SIZE+6&&Index>5)? 1'b1 : 1'b0;
    assign Activation_out_valid[7] = (Cal&&Index<SIZE+7&&Index>6)? 1'b1 : 1'b0;

    // Assigment for Activation output to systolic array
    assign Activation_out[6:0] = (Index>7)? 0 : Activation_Mem[7+bias];
    assign Activation_out[13:7] = (Index<1||Index>8)? 0 : Activation_Mem[6+bias_1];
    assign Activation_out[20:14] = (Index<2||Index>9)? 0 : Activation_Mem[5+bias_2];
    assign Activation_out[27:21] = (Index<3||Index>10)? 0 : Activation_Mem[4+bias_3];
    assign Activation_out[34:28] = (Index<4||Index>11)? 0 : Activation_Mem[3+bias_4];
    assign Activation_out[41:35] = (Index<5||Index>12)? 0 : Activation_Mem[2+bias_5];
    assign Activation_out[48:42] = (Index<6||Index>13)? 0 : Activation_Mem[1+bias_6];
    assign Activation_out[55:49] = (Index<7||Index>14)? 0 : Activation_Mem[0+bias_7];

    
    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Index <= 0;
        end
        else begin
            if(!load_mem_done) begin
                // Load Activation Value into Memory
                Activation_Mem[Activation_Mem_Address_in] <= Activation;
            end
            else if(Cal) begin
                Index <= Index + 1;
            end
            else Index <= 0;
        end
    end 

endmodule

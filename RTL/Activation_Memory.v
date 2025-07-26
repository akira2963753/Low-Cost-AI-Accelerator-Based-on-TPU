// Activation_Memory.v
// This module manages the storage and retrieval of activation values for the systolic array.
// It handles the activation memory, allowing for pre-loading and updating of activation values
// based on the input signals. The module is designed to work with a clock and reset signal
// and interfaces with the pre-load unit to ensure that the activation values are ready for computation
// in the systolic array. It also manages the compensation rows for the activation values.

module Activation_Memory #(
    parameter SIZE = 8,
    parameter SHIFT = $clog2(SIZE), // Shift value for the size of the activation memory
    parameter MEM_SIZE = SIZE*SIZE, // Size of the activation memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
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
    output Activation_out_valid
); 

    // Declare the Register, Net and Integer
    reg [6:0] Activation_Mem[0:MEM_SIZE-1]; // Activation Memory
    reg [ADDR_WIDTH-1:0] Index;
    wire [BIAS_WIDTH-1:0] bias;

    // Assingment Activation_out for Comepensation
    assign bias = Index << SHIFT; // Bias for the activation memory address
    assign Activation_out_valid = (Cal&&Index<SIZE)? 1'b1 : 1'b0;
    // Assigment for Activation output to systolic array
    assign Activation_out[6:0] = (Cal)? Activation_Mem[7+bias] : 0;
    assign Activation_out[13:7] = (Cal)? Activation_Mem[6+bias] : 0;
    assign Activation_out[20:14] = (Cal)? Activation_Mem[5+bias] : 0;
    assign Activation_out[27:21] = (Cal)? Activation_Mem[4+bias] : 0;
    assign Activation_out[34:28] = (Cal)? Activation_Mem[3+bias] : 0;
    assign Activation_out[41:35] = (Cal)? Activation_Mem[2+bias] : 0;
    assign Activation_out[48:42] = (Cal)? Activation_Mem[1+bias] : 0;
    assign Activation_out[55:49] = (Cal)? Activation_Mem[0+bias] : 0;
    
    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Index <= 0;
        end
        else begin
            if(!load_mem_done) begin
                Activation_Mem[Activation_Mem_Address_in] <= Activation;
            end
            else if(Cal) begin
                Index <= Index + 1;
            end
            else Index <= 0;
        end
    end 

endmodule

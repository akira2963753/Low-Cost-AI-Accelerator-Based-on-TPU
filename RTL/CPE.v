// CPE.v
// CPE (Compensation Processing Element) module for the systolic array
// This module processes compensation weights and activations, performing the necessary
// calculations to generate the compensation output. It interfaces with the Compensation_Memory
// and handles the pre-loading of compensation weights. The module is designed to work
// with a clock and reset signal, and it manages the valid state of the compensation weights.

module CPE (
    input clk,
    input rst,
    input [2:0] Compensation_Weight,
    input [6:0] Activation_cout,
    input [32:0] Compensation_Partial_Sum,
    input Activation_cout_valid,
    input Compensation_Weight_out_valid,
    output reg [2:0] Compensation_Weight_Pass,
    output Compensation_Weight_Pass_valid,
    output reg [32:0] Compensation_out
);
    
    // Valid state for Compensation Weight Pass
    assign Compensation_Weight_Pass_valid = (Compensation_Weight_out_valid)? 1'b1 : 1'b0;

    // Always block to handle the processing of compensation weights and activations
    always @(posedge clk or posedge rst) begin
        if(rst) begin // Reset the outputs to zero
            Compensation_Weight_Pass <= 3'd0;
            Compensation_out <= 33'd0;
        end 
        else begin
            // If Compensation Weight is valid, pass it through
            if(Compensation_Weight_out_valid) Compensation_Weight_Pass <= Compensation_Weight;
            // If Activation output is valid, perform the multiplication and addition
            else if(Activation_cout_valid) Compensation_out <= ({Compensation_Weight,1'b1} * {Activation_cout,1'b1}) + Compensation_Partial_Sum;
            // If neither is valid, retain the previous output
            else Compensation_out <= Compensation_Partial_Sum; 
        end
    end
endmodule
// CPE.v
// CPE (Compensation Processing Element) module for the systolic array
// This module processes compensation weights and activations, performing the necessary
// calculations to generate the compensation output. It interfaces with the Compensation_Memory
// and handles the pre-loading of compensation weights. The module is designed to work
// with a clock and reset signal, and it manages the valid state of the compensation weights.

module CPE(
    input clk,
    input rst,
    input [2:0] Compensation_Weight,
    input [6:0] Activation,
    input [32:0] Compensation_Partial_Sum,
    input PreLoad_CWeight,
    input Cal,
    input Activation_cout_valid,
    input Compensation_Weight_out_valid,
    output reg [2:0] Compensation_Weight_Pass,
    output Compensation_Weight_Pass_valid,
    output reg [32:0] Compensation_out
);

    assign Compensation_Weight_Pass_valid = (PreLoad_CWeight&&Compensation_Weight_out_valid)? 1'b1 : 1'b0;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Compensation_Weight_Pass <= 3'd0;
            Compensation_out <= 33'd0;
        end
        else begin
            if(PreLoad_CWeight&&Compensation_Weight_out_valid) Compensation_Weight_Pass <= Compensation_Weight;
            else if(Cal&&Activation_cout_valid) Compensation_out <= ({Compensation_Weight,1'b1} * {Activation,1'b1}) + Compensation_Partial_Sum;
            else if(Cal) Compensation_out <= Compensation_Partial_Sum; // Maintain the previous value if not in calculation mode
            else Compensation_out <= 33'd0; // Reset Compensation_out if not in calculation mode
        end
    end
endmodule
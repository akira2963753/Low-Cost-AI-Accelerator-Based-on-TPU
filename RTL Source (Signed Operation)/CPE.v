// CPE.v
// CPE (Compensation Processing Element) module for the systolic array
// This module processes compensation weights and activations, performing the necessary
// calculations to generate the compensation output. It interfaces with the Compensation_Memory
// and handles the pre-loading of compensation weights. The module is designed to work
// with a clock and reset signal, and it manages the valid state of the compensation weights.

module CPE #(
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 4 + 1
)(
    input clk,
    input [3:0] Compensation_Weight,
    input [6:0] Activation_cin,
    input signed [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum,
    input Activation_cout_valid,
    input Compensation_Weight_out_valid,
    output reg [3:0] Compensation_Weight_Pass,
    output Compensation_Weight_Pass_valid,
    output reg signed[COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_out
);
    
    wire signed [7:0] Expected_Activation_cin;
    wire signed [4:0] Expected_Weight_cin;
    wire signed [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] MAC_OUT;
    assign Expected_Activation_cin = {Activation_cin,1'b1};
    assign Expected_Weight_cin = {Compensation_Weight_Pass,1'b1};

    MAC #(
        .COMPENSATION_PARTIAL_SUM_WIDTH(COMPENSATION_PARTIAL_SUM_WIDTH)
    )CPE_MAC_UNIT(
        .Activation(Expected_Activation_cin),
        .Weight(Expected_Weight_cin),
        .Partial_Sum_in(Compensation_Partial_Sum),
        .Partial_Sum_out(MAC_OUT)
    );

    assign Compensation_Weight_Pass_valid = Compensation_Weight_out_valid;

    // Always block to handle the processing of compensation weights and activations
    always @(posedge clk) begin
        if(Compensation_Weight_out_valid) Compensation_Weight_Pass <= Compensation_Weight;
        else if(Activation_cout_valid) Compensation_out <= MAC_OUT;
        else Compensation_out <= Compensation_Partial_Sum;
    end

endmodule

module MAC #(
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 5 + 1    
)(
    input signed [7:0] Activation,
    input signed [4:0] Weight,
    input signed [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output signed [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire signed [COMPENSATION_PARTIAL_SUM_WIDTH-2:0] Mul_Result;
    assign Mul_Result = Activation * Weight;
    assign Partial_Sum_out = Mul_Result + Partial_Sum_in;

endmodule
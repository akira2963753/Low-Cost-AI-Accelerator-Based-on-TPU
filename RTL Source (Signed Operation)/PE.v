module PE#(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8+4) + 4) + $clog2(SIZE), // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8 // Width of the extended activation
)(
    input clk,
    input [7:0] Weight_in,
    input [7:0] Activation_in,
    input signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    input Weight_in_valid,
    output reg [7:0] Weight_Pass,
    output Weight_Pass_valid,
    output reg [7:0] Activation_Pass,
    output reg signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire signed [PARTIAL_SUM_WIDTH-1:0] MAC_OUT;

    MAC_Unit #(
        .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH)
    ) MAC_u1(
        .Activation(Activation_in),
        .Weight(Weight_Pass),
        .Partial_Sum_in(Partial_Sum_in),
        .Partial_Sum_out(MAC_OUT)
    );

    assign Weight_Pass_valid = Weight_in_valid;

    always @(posedge clk) begin
        // Weight pass downward
        if(Weight_in_valid) begin
            Weight_Pass <= Weight_in;
        end
        else begin
            Partial_Sum_out <= MAC_OUT;
            Activation_Pass <= Activation_in;
        end
    end

endmodule

module MAC_Unit #(
    parameter PARTIAL_SUM_WIDTH = 20
)(
    input signed [7:0] Activation,
    input signed [7:0] Weight,
    input signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire signed [15:0] Mul_Result;
    assign Mul_Result = Activation * Weight;
    assign Partial_Sum_out = Mul_Result + Partial_Sum_in;

endmodule
module RPE#(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE), // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8 // Width of the extended activation
)(
    input clk,
    input [4:0] Weight_in,
    input [6:0] Activation_in,
    input signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    input Weight_in_valid,
    output reg [4:0] Weight_Pass,
    output Weight_Pass_valid,
    output reg [6:0] Activation_Pass,
    output reg signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire signed [7:0] Expected_Activation_in;
    assign Expected_Activation_in = {Activation_in,1'b1};
    wire signed [PARTIAL_SUM_WIDTH-1:0] MAC_OUT;

    MAC_Unit #(
        .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH)
    ) MAC_u1(
        .Activation(Expected_Activation_in),
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
    input signed [4:0] Weight,
    input signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output signed [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire [3:0] Weight_in;
    wire [7:0] Activation_in;
    wire [11:0] Mul_Result;
    wire [11:0] Sign_Result;
    wire signed [12:0] Shift_Result;
    wire signed [15:0] Non_MSR4_Result;
    wire signed [13:0] MSR4_Result;
    wire signed [15:0] Result;

    assign Weight_in = (Weight[3])? ((Weight[4])? ~(Weight[3:0]) : ~(Weight[3:0]) + 1) : Weight[3:0];
    assign Activation_in = (Activation[7])? ~(Activation) + 1 : Activation;
    assign Mul_Result = Activation_in * Weight_in;
    assign Sign_Result = (Activation[7]^Weight[3])? ~ Mul_Result + 1 : Mul_Result;
    assign Shift_Result = {Sign_Result,1'b0};
    assign MSR4_Result = Shift_Result + Activation;
    assign Non_MSR4_Result = {Shift_Result,3'b000};
    assign Result = (Weight[4])? Non_MSR4_Result : {{2{MSR4_Result[13]}},MSR4_Result};   
    assign Partial_Sum_out = Result + Partial_Sum_in;

endmodule
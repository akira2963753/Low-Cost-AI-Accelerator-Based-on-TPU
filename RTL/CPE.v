// CPE.v
// CPE (Compensation Processing Element) module for the systolic array
// This module processes compensation weights and activations, performing the necessary
// calculations to generate the compensation output. It interfaces with the Compensation_Memory
// and handles the pre-loading of compensation weights. The module is designed to work
// with a clock and reset signal, and it manages the valid state of the compensation weights.

module CPE #(
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 5 + 1
)(
    input clk,
    input [3:0] Compensation_Weight,
    input [6:0] Activation_cin,
    input [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum,
    input Activation_cout_valid,
    input Compensation_Weight_out_valid,
    output reg [3:0] Compensation_Weight_Pass,
    output Compensation_Weight_Pass_valid,
    output reg [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_out
);
    
    wire [7:0] Expected_Activation_cin;
    wire [4:0] Expected_Weight_cin;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] MAC_OUT;
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
    input [7:0] Activation,
    input [4:0] Weight,
    input [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);

    wire [7:0] Actiation_in;
    wire [4:0] Weight_in;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-2:0] Mul_Result;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-2:0] Sign_Result;
    

    assign Weight_in = (Weight[4])? ~(Weight) + 1 : Weight;
    assign Actiation_in = (Activation[7])? ~(Activation) + 1 : Activation;
    multiplier_8x5 m1(Actiation_in,Weight_in,Mul_Result);
    assign Sign_Result = (Activation[7]^Weight[4])? ~(Mul_Result) + 1 : Mul_Result;
    assign Partial_Sum_out = {Sign_Result[COMPENSATION_PARTIAL_SUM_WIDTH-2],Sign_Result} + Partial_Sum_in;

endmodule

module multiplier_8x5(
    input [7:0] a,
    input [4:0] b,
    output [12:0] p
);
    
    wire [7:0] pp0, pp1, pp2, pp3, pp4;
    
    assign pp0 = {8{b[0]}} & a;
    assign pp1 = {8{b[1]}} & a;
    assign pp2 = {8{b[2]}} & a;
    assign pp3 = {8{b[3]}} & a;
    assign pp4 = {8{b[4]}} & a;
    
    wire [7:0] c1, c2, c3, c4;
    wire [7:0] s1, s2, s3, s4;
    
    assign p[0] = pp0[0];
    
    half_adder ha1_0(.a(pp0[1]), .b(pp1[0]), .sum(p[1]), .carry(c1[0]));
    
    full_adder fa1_1(.a(pp0[2]), .b(pp1[1]), .cin(c1[0]), .sum(s1[1]), .cout(c1[1]));
    full_adder fa1_2(.a(pp0[3]), .b(pp1[2]), .cin(c1[1]), .sum(s1[2]), .cout(c1[2]));
    full_adder fa1_3(.a(pp0[4]), .b(pp1[3]), .cin(c1[2]), .sum(s1[3]), .cout(c1[3]));
    full_adder fa1_4(.a(pp0[5]), .b(pp1[4]), .cin(c1[3]), .sum(s1[4]), .cout(c1[4]));
    full_adder fa1_5(.a(pp0[6]), .b(pp1[5]), .cin(c1[4]), .sum(s1[5]), .cout(c1[5]));
    full_adder fa1_6(.a(pp0[7]), .b(pp1[6]), .cin(c1[5]), .sum(s1[6]), .cout(c1[6]));
    
    half_adder ha1_7(.a(pp1[7]), .b(c1[6]), .sum(s1[7]), .carry(c1[7]));
    
    half_adder ha2_0(.a(s1[1]), .b(pp2[0]), .sum(p[2]), .carry(c2[0]));
    
    full_adder fa2_1(.a(s1[2]), .b(pp2[1]), .cin(c2[0]), .sum(s2[1]), .cout(c2[1]));
    full_adder fa2_2(.a(s1[3]), .b(pp2[2]), .cin(c2[1]), .sum(s2[2]), .cout(c2[2]));
    full_adder fa2_3(.a(s1[4]), .b(pp2[3]), .cin(c2[2]), .sum(s2[3]), .cout(c2[3]));
    full_adder fa2_4(.a(s1[5]), .b(pp2[4]), .cin(c2[3]), .sum(s2[4]), .cout(c2[4]));
    full_adder fa2_5(.a(s1[6]), .b(pp2[5]), .cin(c2[4]), .sum(s2[5]), .cout(c2[5]));
    full_adder fa2_6(.a(s1[7]), .b(pp2[6]), .cin(c2[5]), .sum(s2[6]), .cout(c2[6]));
    full_adder fa2_7(.a(c1[7]), .b(pp2[7]), .cin(c2[6]), .sum(s2[7]), .cout(c2[7]));
    
    half_adder ha3_0(.a(s2[1]), .b(pp3[0]), .sum(p[3]), .carry(c3[0]));
    
    full_adder fa3_1(.a(s2[2]), .b(pp3[1]), .cin(c3[0]), .sum(s3[1]), .cout(c3[1]));
    full_adder fa3_2(.a(s2[3]), .b(pp3[2]), .cin(c3[1]), .sum(s3[2]), .cout(c3[2]));
    full_adder fa3_3(.a(s2[4]), .b(pp3[3]), .cin(c3[2]), .sum(s3[3]), .cout(c3[3]));
    full_adder fa3_4(.a(s2[5]), .b(pp3[4]), .cin(c3[3]), .sum(s3[4]), .cout(c3[4]));
    full_adder fa3_5(.a(s2[6]), .b(pp3[5]), .cin(c3[4]), .sum(s3[5]), .cout(c3[5]));
    full_adder fa3_6(.a(s2[7]), .b(pp3[6]), .cin(c3[5]), .sum(s3[6]), .cout(c3[6]));
    full_adder fa3_7(.a(c2[7]), .b(pp3[7]), .cin(c3[6]), .sum(s3[7]), .cout(c3[7]));
    
    half_adder ha4_0(.a(s3[1]), .b(pp4[0]), .sum(p[4]), .carry(c4[0]));
    
    full_adder fa4_1(.a(s3[2]), .b(pp4[1]), .cin(c4[0]), .sum(p[5]), .cout(c4[1]));
    full_adder fa4_2(.a(s3[3]), .b(pp4[2]), .cin(c4[1]), .sum(p[6]), .cout(c4[2]));
    full_adder fa4_3(.a(s3[4]), .b(pp4[3]), .cin(c4[2]), .sum(p[7]), .cout(c4[3]));
    full_adder fa4_4(.a(s3[5]), .b(pp4[4]), .cin(c4[3]), .sum(p[8]), .cout(c4[4]));
    full_adder fa4_5(.a(s3[6]), .b(pp4[5]), .cin(c4[4]), .sum(p[9]), .cout(c4[5]));
    full_adder fa4_6(.a(s3[7]), .b(pp4[6]), .cin(c4[5]), .sum(p[10]), .cout(c4[6]));
    full_adder fa4_7(.a(c3[7]), .b(pp4[7]), .cin(c4[6]), .sum(p[11]), .cout(p[12]));
    
endmodule
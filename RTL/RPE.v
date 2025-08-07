module RPE#(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE), // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8 // Width of the extended activation
)(
    input clk,
    input [4:0] Weight_in,
    input [6:0] Activation_in,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    input Weight_in_valid,
    output reg [4:0] Weight_Pass,
    output Weight_Pass_valid,
    output reg [6:0] Activation_Pass,
    output reg [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire [7:0] Expected_Activation_in;
    assign Expected_Activation_in = {Activation_in,1'b1};
    wire [PARTIAL_SUM_WIDTH-1:0] MAC_OUT;

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
    input [7:0] Activation,
    input [4:0] Weight,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    localparam RESULT_EXTENSION = PARTIAL_SUM_WIDTH - 16;
    wire [3:0] Weight_in;
    wire [7:0] Activation_in;
    wire [11:0] Mul_Result;
    wire [11:0] Sign_Result;
    wire [12:0] Shift_Result;
    wire [12:0] MSR4_Result;
    wire [15:0] Non_MSR4_Result;
    wire [15:0] Result;

    assign Weight_in = (Weight[3])? ((Weight[4])? ~(Weight[3:0]) : ~(Weight[3:0]) + 1) : Weight[3:0];
    assign Activation_in = (Activation[7])? ~(Activation) + 1 : Activation;

    multiplier_8x4 m0(Activation_in,Weight_in,Mul_Result);

    assign Sign_Result = (Activation[7]^Weight[3])? ~(Mul_Result) + 1 : Mul_Result;
    assign Shift_Result = {Sign_Result,1'b0};
    assign MSR4_Result = Shift_Result + {{5{Activation[7]}},Activation};
    assign Non_MSR4_Result = {Shift_Result,3'b000};
    assign Result = (Weight[4])? Non_MSR4_Result : {{3{MSR4_Result[11]}},MSR4_Result};   
    assign Partial_Sum_out = {{RESULT_EXTENSION{Result[14]}},Result} + Partial_Sum_in;

endmodule

module half_adder(
    input a,
    input b,
    output sum,
    output carry
);
    assign sum = a ^ b;
    assign carry = a & b;
endmodule

module full_adder(
    input a,
    input b,
    input cin,
    output sum,
    output cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (a & cin) | (b & cin);
endmodule

module multiplier_8x4(
    input [7:0] a, 
    input [3:0] b,    
    output [11:0] p    
);
    wire [7:0] pp0, pp1, pp2, pp3;
    
    assign pp0 = {8{b[0]}} & a;  // a × b[0]
    assign pp1 = {8{b[1]}} & a;  // a × b[1]  
    assign pp2 = {8{b[2]}} & a;  // a × b[2]
    assign pp3 = {8{b[3]}} & a;  // a × b[3]
    
    wire [7:0] c1, c2, c3;
    wire [7:0] s1, s2, s3; 
    
    assign p[0] = pp0[0];

    half_adder uha1_0(.a(pp0[1]), .b(pp1[0]), .sum(p[1]), .carry(c1[0]));
    
    full_adder ufa1_1(.a(pp0[2]), .b(pp1[1]), .cin(c1[0]), .sum(s1[1]), .cout(c1[1]));
    full_adder ufa1_2(.a(pp0[3]), .b(pp1[2]), .cin(c1[1]), .sum(s1[2]), .cout(c1[2]));
    full_adder ufa1_3(.a(pp0[4]), .b(pp1[3]), .cin(c1[2]), .sum(s1[3]), .cout(c1[3]));
    full_adder ufa1_4(.a(pp0[5]), .b(pp1[4]), .cin(c1[3]), .sum(s1[4]), .cout(c1[4]));
    full_adder ufa1_5(.a(pp0[6]), .b(pp1[5]), .cin(c1[4]), .sum(s1[5]), .cout(c1[5]));
    full_adder ufa1_6(.a(pp0[7]), .b(pp1[6]), .cin(c1[5]), .sum(s1[6]), .cout(c1[6]));
    
    half_adder uha1_7(.a(pp1[7]), .b(c1[6]), .sum(s1[7]), .carry(c1[7]));
    
    half_adder uha2_0(.a(s1[1]), .b(pp2[0]), .sum(p[2]), .carry(c2[0]));
    full_adder ufa2_1(.a(s1[2]), .b(pp2[1]), .cin(c2[0]), .sum(s2[1]), .cout(c2[1]));
    full_adder ufa2_2(.a(s1[3]), .b(pp2[2]), .cin(c2[1]), .sum(s2[2]), .cout(c2[2]));
    full_adder ufa2_3(.a(s1[4]), .b(pp2[3]), .cin(c2[2]), .sum(s2[3]), .cout(c2[3]));
    full_adder ufa2_4(.a(s1[5]), .b(pp2[4]), .cin(c2[3]), .sum(s2[4]), .cout(c2[4]));
    full_adder ufa2_5(.a(s1[6]), .b(pp2[5]), .cin(c2[4]), .sum(s2[5]), .cout(c2[5]));
    full_adder ufa2_6(.a(s1[7]), .b(pp2[6]), .cin(c2[5]), .sum(s2[6]), .cout(c2[6]));
    full_adder ufa2_7(.a(c1[7]), .b(pp2[7]), .cin(c2[6]), .sum(s2[7]), .cout(c2[7]));
    
    half_adder uha3_0(.a(s2[1]), .b(pp3[0]), .sum(p[3]), .carry(c3[0]));
    full_adder ufa3_1(.a(s2[2]), .b(pp3[1]), .cin(c3[0]), .sum(p[4]), .cout(c3[1]));
    full_adder ufa3_2(.a(s2[3]), .b(pp3[2]), .cin(c3[1]), .sum(p[5]), .cout(c3[2]));
    full_adder ufa3_3(.a(s2[4]), .b(pp3[3]), .cin(c3[2]), .sum(p[6]), .cout(c3[3]));
    full_adder ufa3_4(.a(s2[5]), .b(pp3[4]), .cin(c3[3]), .sum(p[7]), .cout(c3[4]));
    full_adder ufa3_5(.a(s2[6]), .b(pp3[5]), .cin(c3[4]), .sum(p[8]), .cout(c3[5]));
    full_adder ufa3_6(.a(s2[7]), .b(pp3[6]), .cin(c3[5]), .sum(p[9]), .cout(c3[6]));
    full_adder ufa3_7(.a(c2[7]), .b(pp3[7]), .cin(c3[6]), .sum(p[10]), .cout(p[11]));
    
endmodule

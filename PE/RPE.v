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
    wire Weight_add;
    wire [11:0] Mul_Result;
    wire [12:0] Shift_Result;
    wire [12:0] MSR4_Result;
    wire [15:0] Non_MSR4_Result;
    wire [15:0] Result;

    assign Weight_add = Weight[3]&&(!Weight[4]);
    Multiplier_RPE m0(Activation,Weight[3:0],Weight_add,Mul_Result);

    assign Shift_Result = {Mul_Result,1'b0};
    assign MSR4_Result = Shift_Result + {{5{Activation[7]}},Activation};
    assign Non_MSR4_Result = {Shift_Result,3'b000};
    assign Result = (Weight[4])? Non_MSR4_Result : {{3{MSR4_Result[11]}},MSR4_Result};   
    assign Partial_Sum_out = {{RESULT_EXTENSION{Result[14]}},Result} + Partial_Sum_in;

endmodule

module Multiplier_RPE(input [7:0]activation, input[3:0]weight, input Weight_add, output [11:0]sum);
	wire [52:0]C;
	wire [16:0]T_sum;
	wire [7:0]activation_c;
	wire [3:0]weight_c;
	wire [11:0]sum_c;

assign	sum_c[0]=activation_c[0]&&weight_c[0];

//2's complement for activation (8-bit)
Half_adder F200(activation[0]^activation[7],activation[7],activation_c[0],C[31]);
Half_adder F201(activation[1]^activation[7],C[31],activation_c[1],C[32]);
Half_adder F202(activation[2]^activation[7],C[32],activation_c[2],C[33]);
Half_adder F203(activation[3]^activation[7],C[33],activation_c[3],C[34]);
Half_adder F204(activation[4]^activation[7],C[34],activation_c[4],C[35]);
Half_adder F205(activation[5]^activation[7],C[35],activation_c[5],C[36]);
Half_adder F206(activation[6]^activation[7],C[36],activation_c[6],C[37]);
Half_adder F207(activation[7]^activation[7],C[37],activation_c[7],);

//2's complement for weight (4-bit)
Half_adder F208(weight[0]^weight[3],Weight_add,weight_c[0],C[38]);
Half_adder F209(weight[1]^weight[3],C[38],weight_c[1],C[39]);
Half_adder F210(weight[2]^weight[3],C[39],weight_c[2],C[40]);
Half_adder F211(weight[3]^weight[3],C[40],weight_c[3],);

// First partial product addition (bit 1)
Half_adder F213(activation_c[1]&&weight_c[0],activation_c[0]&&weight_c[1],sum_c[1],C[0]);

// Second partial product addition (bit 2-7)
Full_adder F214(activation_c[2]&&weight_c[0],activation_c[1]&&weight_c[1],C[0],T_sum[0],C[1]);
Full_adder F215(activation_c[3]&&weight_c[0],activation_c[2]&&weight_c[1],C[1],T_sum[1],C[2]);
Full_adder F216(activation_c[4]&&weight_c[0],activation_c[3]&&weight_c[1],C[2],T_sum[2],C[3]);
Full_adder F217(activation_c[5]&&weight_c[0],activation_c[4]&&weight_c[1],C[3],T_sum[3],C[4]);
Full_adder F218(activation_c[6]&&weight_c[0],activation_c[5]&&weight_c[1],C[4],T_sum[4],C[5]);
Full_adder F219(activation_c[7]&&weight_c[0],activation_c[6]&&weight_c[1],C[5],T_sum[5],C[6]);
Half_adder F220(activation_c[7]&&weight_c[1],C[6],T_sum[6],C[7]);

// Third partial product addition (bit 2-8)
Half_adder F221(T_sum[0],activation_c[0]&&weight_c[2],sum_c[2],C[8]);
Full_adder F222(T_sum[1],activation_c[1]&&weight_c[2],C[8],T_sum[7],C[9]);
Full_adder F223(T_sum[2],activation_c[2]&&weight_c[2],C[9],T_sum[8],C[10]);
Full_adder F224(T_sum[3],activation_c[3]&&weight_c[2],C[10],T_sum[9],C[11]);
Full_adder F225(T_sum[4],activation_c[4]&&weight_c[2],C[11],T_sum[10],C[12]);
Full_adder F226(T_sum[5],activation_c[5]&&weight_c[2],C[12],T_sum[11],C[13]);
Full_adder F227(T_sum[6],activation_c[6]&&weight_c[2],C[13],T_sum[12],C[14]);
Full_adder F228(C[7],activation_c[7]&&weight_c[2],C[14],T_sum[13],C[15]);

// Fourth partial product addition (bit 3-10)
Half_adder F229(T_sum[7],activation_c[0]&&weight_c[3],sum_c[3],C[16]);
Full_adder F230(T_sum[8],activation_c[1]&&weight_c[3],C[16],sum_c[4],C[17]);
Full_adder F231(T_sum[9],activation_c[2]&&weight_c[3],C[17],sum_c[5],C[18]);
Full_adder F232(T_sum[10],activation_c[3]&&weight_c[3],C[18],sum_c[6],C[19]);
Full_adder F233(T_sum[11],activation_c[4]&&weight_c[3],C[19],sum_c[7],C[20]);
Full_adder F234(T_sum[12],activation_c[5]&&weight_c[3],C[20],sum_c[8],C[21]);
Full_adder F235(T_sum[13],activation_c[6]&&weight_c[3],C[21],sum_c[9],C[22]);
Full_adder F236(C[15],activation_c[7]&&weight_c[3],C[22],sum_c[10],sum_c[11]);

// Final 2's complement conversion
Half_adder F245(sum_c[0]^(activation[7]^weight[3]),(activation[7]^weight[3]),sum[0],C[42]);
Half_adder F246(sum_c[1]^(activation[7]^weight[3]),C[42],sum[1],C[43]);
Half_adder F247(sum_c[2]^(activation[7]^weight[3]),C[43],sum[2],C[44]);
Half_adder F248(sum_c[3]^(activation[7]^weight[3]),C[44],sum[3],C[45]);
Half_adder F249(sum_c[4]^(activation[7]^weight[3]),C[45],sum[4],C[46]);
Half_adder F250(sum_c[5]^(activation[7]^weight[3]),C[46],sum[5],C[47]);
Half_adder F251(sum_c[6]^(activation[7]^weight[3]),C[47],sum[6],C[48]);
Half_adder F252(sum_c[7]^(activation[7]^weight[3]),C[48],sum[7],C[49]);
Half_adder F253(sum_c[8]^(activation[7]^weight[3]),C[49],sum[8],C[50]);
Half_adder F254(sum_c[9]^(activation[7]^weight[3]),C[50],sum[9],C[51]);
Half_adder F255(sum_c[10]^(activation[7]^weight[3]),C[51],sum[10],C[52]);
Half_adder F256(sum_c[11]^(activation[7]^weight[3]),C[52],sum[11],);

endmodule

module Half_adder(input A,B,output S,C);

assign	S=A^B;
assign C=A&&B;

endmodule

module Full_adder(input A,B,C_in,output S,C);

assign	S=A^B^C_in;
assign 	C=((A^B)&&C_in)+(A&&B);

endmodule
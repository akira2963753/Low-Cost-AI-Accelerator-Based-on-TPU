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

    wire [COMPENSATION_PARTIAL_SUM_WIDTH-2:0] Mul_Result;
    
    Multiplier_CPE m1(Activation,Weight,Mul_Result);
    assign Partial_Sum_out = {Mul_Result[COMPENSATION_PARTIAL_SUM_WIDTH-2],Mul_Result} + Partial_Sum_in;

endmodule

module Multiplier_CPE(input [7:0]activation,input[4:0]weight,output [12:0]sum);
	wire [53:0]C;
	wire [20:0]T_sum;
	wire [7:0]activation_c;
	wire [4:0]weight_c;
	wire [12:0]sum_c;

assign	sum_c[0]=activation_c[0]&&weight_c[0];

//2's complement
Half_adder CF200(activation[0]^activation[7],activation[7],activation_c[0],C[31]);
Half_adder CF201(activation[1]^activation[7],C[31],activation_c[1],C[32]);
Half_adder CF202(activation[2]^activation[7],C[32],activation_c[2],C[33]);
Half_adder CF203(activation[3]^activation[7],C[33],activation_c[3],C[34]);
Half_adder CF204(activation[4]^activation[7],C[34],activation_c[4],C[35]);
Half_adder CF205(activation[5]^activation[7],C[35],activation_c[5],C[36]);
Half_adder CF206(activation[6]^activation[7],C[36],activation_c[6],C[37]);
Half_adder CF207(activation[7]^activation[7],C[37],activation_c[7],);

Half_adder CF208(weight[0]^weight[4],weight[4],weight_c[0],C[38]);
Half_adder CF209(weight[1]^weight[4],C[38],weight_c[1],C[39]);
Half_adder CF210(weight[2]^weight[4],C[39],weight_c[2],C[40]);
Half_adder CF211(weight[3]^weight[4],C[40],weight_c[3],C[41]);
Half_adder CF212(weight[4]^weight[4],C[41],weight_c[4],);

Half_adder CF213(activation_c[1]&&weight_c[0],activation_c[0]&&weight_c[1],sum_c[1],C[0]);
Full_adder CF214(activation_c[2]&&weight_c[0],activation_c[1]&&weight_c[1],C[0],T_sum[0],C[1]);
Full_adder CF215(activation_c[3]&&weight_c[0],activation_c[2]&&weight_c[1],C[1],T_sum[1],C[2]);
Full_adder CF216(activation_c[4]&&weight_c[0],activation_c[3]&&weight_c[1],C[2],T_sum[2],C[3]);
Full_adder CF217(activation_c[5]&&weight_c[0],activation_c[4]&&weight_c[1],C[3],T_sum[3],C[4]);
Full_adder CF218(activation_c[6]&&weight_c[0],activation_c[5]&&weight_c[1],C[4],T_sum[4],C[5]);
Full_adder CF219(activation_c[7]&&weight_c[0],activation_c[6]&&weight_c[1],C[5],T_sum[5],C[6]);
Half_adder CF220(activation_c[7]&&weight_c[1],C[6],T_sum[6],C[7]);

Half_adder CF221(T_sum[0],activation_c[0]&&weight_c[2],sum_c[2],C[8]);
Full_adder CF222(T_sum[1],activation_c[1]&&weight_c[2],C[8],T_sum[7],C[9]);
Full_adder CF223(T_sum[2],activation_c[2]&&weight_c[2],C[9],T_sum[8],C[10]);
Full_adder CF224(T_sum[3],activation_c[3]&&weight_c[2],C[10],T_sum[9],C[11]);
Full_adder CF225(T_sum[4],activation_c[4]&&weight_c[2],C[11],T_sum[10],C[12]);
Full_adder CF226(T_sum[5],activation_c[5]&&weight_c[2],C[12],T_sum[11],C[13]);
Full_adder CF227(T_sum[6],activation_c[6]&&weight_c[2],C[13],T_sum[12],C[14]);
Full_adder CF228(C[7],activation_c[7]&&weight_c[2],C[14],T_sum[13],C[15]);

Half_adder CF229(T_sum[7],activation_c[0]&&weight_c[3],sum_c[3],C[16]);
Full_adder CF230(T_sum[8],activation_c[1]&&weight_c[3],C[16],T_sum[14],C[17]);
Full_adder CF231(T_sum[9],activation_c[2]&&weight_c[3],C[17],T_sum[15],C[18]);
Full_adder CF232(T_sum[10],activation_c[3]&&weight_c[3],C[18],T_sum[16],C[19]);
Full_adder CF233(T_sum[11],activation_c[4]&&weight_c[3],C[19],T_sum[17],C[20]);
Full_adder CF234(T_sum[12],activation_c[5]&&weight_c[3],C[20],T_sum[18],C[21]);
Full_adder CF235(T_sum[13],activation_c[6]&&weight_c[3],C[21],T_sum[19],C[22]);
Full_adder CF236(C[15],activation_c[7]&&weight_c[3],C[22],T_sum[20],C[23]);

Half_adder CF237(T_sum[14],activation_c[0]&&weight_c[4],sum_c[4],C[24]);
Full_adder CF238(T_sum[15],activation_c[1]&&weight_c[4],C[24],sum_c[5],C[25]);
Full_adder CF239(T_sum[16],activation_c[2]&&weight_c[4],C[25],sum_c[6],C[26]);
Full_adder CF240(T_sum[17],activation_c[3]&&weight_c[4],C[26],sum_c[7],C[27]);
Full_adder CF241(T_sum[18],activation_c[4]&&weight_c[4],C[27],sum_c[8],C[28]);
Full_adder CF242(T_sum[19],activation_c[5]&&weight_c[4],C[28],sum_c[9],C[29]);
Full_adder CF243(T_sum[20],activation_c[6]&&weight_c[4],C[29],sum_c[10],C[30]);
Full_adder CF244(C[23],activation_c[7]&&weight_c[4],C[30],sum_c[11],sum_c[12]);

Half_adder CF245(sum_c[0]^(activation[7]^weight[4]),(activation[7]^weight[4]),sum[0],C[42]);
Half_adder CF246(sum_c[1]^(activation[7]^weight[4]),C[42],sum[1],C[43]);
Half_adder CF247(sum_c[2]^(activation[7]^weight[4]),C[43],sum[2],C[44]);
Half_adder CF248(sum_c[3]^(activation[7]^weight[4]),C[44],sum[3],C[45]);
Half_adder CF249(sum_c[4]^(activation[7]^weight[4]),C[45],sum[4],C[46]);
Half_adder CF250(sum_c[5]^(activation[7]^weight[4]),C[46],sum[5],C[47]);
Half_adder CF251(sum_c[6]^(activation[7]^weight[4]),C[47],sum[6],C[48]);
Half_adder CF252(sum_c[7]^(activation[7]^weight[4]),C[48],sum[7],C[49]);
Half_adder CF253(sum_c[8]^(activation[7]^weight[4]),C[49],sum[8],C[50]);
Half_adder CF254(sum_c[9]^(activation[7]^weight[4]),C[50],sum[9],C[51]);
Half_adder CF255(sum_c[10]^(activation[7]^weight[4]),C[51],sum[10],C[52]);
Half_adder CF256(sum_c[11]^(activation[7]^weight[4]),C[52],sum[11],C[53]);
Half_adder CF257(sum_c[12]^(activation[7]^weight[4]),C[53],sum[12],);

endmodule

module Half_adder(input A,B,output S,C);

assign	S=A^B;
assign C=A&&B;

endmodule

module Full_adder(input A,B,C_in,output S,C);

assign	S=A^B^C_in;
assign 	C=((A^B)&&C_in)+(A&&B);

endmodule

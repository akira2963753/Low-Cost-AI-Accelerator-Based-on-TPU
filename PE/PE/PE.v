module PE#(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE), // Size of the partial sum
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
    input [7:0] Activation,
    input [7:0] Weight,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    wire [15:0] Mul_Result;

    Multiplier_PE m3(Activation,Weight,Mul_Result);
    assign Partial_Sum_out = {{5{Mul_Result[15]}},Mul_Result} + Partial_Sum_in;

endmodule

module Multiplier_PE(input [7:0]activation, input[7:0]weight, output [15:0]sum);
	wire [119:0]C;
	wire [55:0]T_sum;
	wire [7:0]activation_c;
	wire [7:0]weight_c;
	wire [15:0]sum_c;

assign	sum_c[0]=activation_c[0]&&weight_c[0];

//2's complement for activation (8-bit)
Half_adder F200(activation[0]^activation[7],activation[7],activation_c[0],C[80]);
Half_adder F201(activation[1]^activation[7],C[80],activation_c[1],C[81]);
Half_adder F202(activation[2]^activation[7],C[81],activation_c[2],C[82]);
Half_adder F203(activation[3]^activation[7],C[82],activation_c[3],C[83]);
Half_adder F204(activation[4]^activation[7],C[83],activation_c[4],C[84]);
Half_adder F205(activation[5]^activation[7],C[84],activation_c[5],C[85]);
Half_adder F206(activation[6]^activation[7],C[85],activation_c[6],C[86]);
Half_adder F207(activation[7]^activation[7],C[86],activation_c[7],);

//2's complement for weight (8-bit)
Half_adder F208(weight[0]^weight[7],weight[7],weight_c[0],C[87]);
Half_adder F209(weight[1]^weight[7],C[87],weight_c[1],C[88]);
Half_adder F210(weight[2]^weight[7],C[88],weight_c[2],C[89]);
Half_adder F211(weight[3]^weight[7],C[89],weight_c[3],C[90]);
Half_adder F212(weight[4]^weight[7],C[90],weight_c[4],C[91]);
Half_adder F213(weight[5]^weight[7],C[91],weight_c[5],C[92]);
Half_adder F214(weight[6]^weight[7],C[92],weight_c[6],C[93]);
Half_adder F215(weight[7]^weight[7],C[93],weight_c[7],);

// First partial product addition (bit 1)
Half_adder F216(activation_c[1]&&weight_c[0],activation_c[0]&&weight_c[1],sum_c[1],C[0]);

// Second partial product addition (bit 2-8)
Full_adder F217(activation_c[2]&&weight_c[0],activation_c[1]&&weight_c[1],C[0],T_sum[0],C[1]);
Full_adder F218(activation_c[3]&&weight_c[0],activation_c[2]&&weight_c[1],C[1],T_sum[1],C[2]);
Full_adder F219(activation_c[4]&&weight_c[0],activation_c[3]&&weight_c[1],C[2],T_sum[2],C[3]);
Full_adder F220(activation_c[5]&&weight_c[0],activation_c[4]&&weight_c[1],C[3],T_sum[3],C[4]);
Full_adder F221(activation_c[6]&&weight_c[0],activation_c[5]&&weight_c[1],C[4],T_sum[4],C[5]);
Full_adder F222(activation_c[7]&&weight_c[0],activation_c[6]&&weight_c[1],C[5],T_sum[5],C[6]);
Half_adder F223(activation_c[7]&&weight_c[1],C[6],T_sum[6],C[7]);

// Third partial product addition (bit 2-9)
Half_adder F224(T_sum[0],activation_c[0]&&weight_c[2],sum_c[2],C[8]);
Full_adder F225(T_sum[1],activation_c[1]&&weight_c[2],C[8],T_sum[7],C[9]);
Full_adder F226(T_sum[2],activation_c[2]&&weight_c[2],C[9],T_sum[8],C[10]);
Full_adder F227(T_sum[3],activation_c[3]&&weight_c[2],C[10],T_sum[9],C[11]);
Full_adder F228(T_sum[4],activation_c[4]&&weight_c[2],C[11],T_sum[10],C[12]);
Full_adder F229(T_sum[5],activation_c[5]&&weight_c[2],C[12],T_sum[11],C[13]);
Full_adder F230(T_sum[6],activation_c[6]&&weight_c[2],C[13],T_sum[12],C[14]);
Full_adder F231(C[7],activation_c[7]&&weight_c[2],C[14],T_sum[13],C[15]);

// Fourth partial product addition (bit 3-10)
Half_adder F232(T_sum[7],activation_c[0]&&weight_c[3],sum_c[3],C[16]);
Full_adder F233(T_sum[8],activation_c[1]&&weight_c[3],C[16],T_sum[14],C[17]);
Full_adder F234(T_sum[9],activation_c[2]&&weight_c[3],C[17],T_sum[15],C[18]);
Full_adder F235(T_sum[10],activation_c[3]&&weight_c[3],C[18],T_sum[16],C[19]);
Full_adder F236(T_sum[11],activation_c[4]&&weight_c[3],C[19],T_sum[17],C[20]);
Full_adder F237(T_sum[12],activation_c[5]&&weight_c[3],C[20],T_sum[18],C[21]);
Full_adder F238(T_sum[13],activation_c[6]&&weight_c[3],C[21],T_sum[19],C[22]);
Full_adder F239(C[15],activation_c[7]&&weight_c[3],C[22],T_sum[20],C[23]);

// Fifth partial product addition (bit 4-11)
Half_adder F240(T_sum[14],activation_c[0]&&weight_c[4],sum_c[4],C[24]);
Full_adder F241(T_sum[15],activation_c[1]&&weight_c[4],C[24],T_sum[21],C[25]);
Full_adder F242(T_sum[16],activation_c[2]&&weight_c[4],C[25],T_sum[22],C[26]);
Full_adder F243(T_sum[17],activation_c[3]&&weight_c[4],C[26],T_sum[23],C[27]);
Full_adder F244(T_sum[18],activation_c[4]&&weight_c[4],C[27],T_sum[24],C[28]);
Full_adder F245(T_sum[19],activation_c[5]&&weight_c[4],C[28],T_sum[25],C[29]);
Full_adder F246(T_sum[20],activation_c[6]&&weight_c[4],C[29],T_sum[26],C[30]);
Full_adder F247(C[23],activation_c[7]&&weight_c[4],C[30],T_sum[27],C[31]);

// Sixth partial product addition (bit 5-12)
Half_adder F248(T_sum[21],activation_c[0]&&weight_c[5],sum_c[5],C[32]);
Full_adder F249(T_sum[22],activation_c[1]&&weight_c[5],C[32],T_sum[28],C[33]);
Full_adder F250(T_sum[23],activation_c[2]&&weight_c[5],C[33],T_sum[29],C[34]);
Full_adder F251(T_sum[24],activation_c[3]&&weight_c[5],C[34],T_sum[30],C[35]);
Full_adder F252(T_sum[25],activation_c[4]&&weight_c[5],C[35],T_sum[31],C[36]);
Full_adder F253(T_sum[26],activation_c[5]&&weight_c[5],C[36],T_sum[32],C[37]);
Full_adder F254(T_sum[27],activation_c[6]&&weight_c[5],C[37],T_sum[33],C[38]);
Full_adder F255(C[31],activation_c[7]&&weight_c[5],C[38],T_sum[34],C[39]);

// Seventh partial product addition (bit 6-13)
Half_adder F256(T_sum[28],activation_c[0]&&weight_c[6],sum_c[6],C[40]);
Full_adder F257(T_sum[29],activation_c[1]&&weight_c[6],C[40],T_sum[35],C[41]);
Full_adder F258(T_sum[30],activation_c[2]&&weight_c[6],C[41],T_sum[36],C[42]);
Full_adder F259(T_sum[31],activation_c[3]&&weight_c[6],C[42],T_sum[37],C[43]);
Full_adder F260(T_sum[32],activation_c[4]&&weight_c[6],C[43],T_sum[38],C[44]);
Full_adder F261(T_sum[33],activation_c[5]&&weight_c[6],C[44],T_sum[39],C[45]);
Full_adder F262(T_sum[34],activation_c[6]&&weight_c[6],C[45],T_sum[40],C[46]);
Full_adder F263(C[39],activation_c[7]&&weight_c[6],C[46],T_sum[41],C[47]);

// Eighth partial product addition (bit 7-14)
Half_adder F264(T_sum[35],activation_c[0]&&weight_c[7],sum_c[7],C[48]);
Full_adder F265(T_sum[36],activation_c[1]&&weight_c[7],C[48],sum_c[8],C[49]);
Full_adder F266(T_sum[37],activation_c[2]&&weight_c[7],C[49],sum_c[9],C[50]);
Full_adder F267(T_sum[38],activation_c[3]&&weight_c[7],C[50],sum_c[10],C[51]);
Full_adder F268(T_sum[39],activation_c[4]&&weight_c[7],C[51],sum_c[11],C[52]);
Full_adder F269(T_sum[40],activation_c[5]&&weight_c[7],C[52],sum_c[12],C[53]);
Full_adder F270(T_sum[41],activation_c[6]&&weight_c[7],C[53],sum_c[13],C[54]);
Full_adder F271(C[47],activation_c[7]&&weight_c[7],C[54],sum_c[14],sum_c[15]);

// Final 2's complement conversion
Half_adder F272(sum_c[0]^(activation[7]^weight[7]),(activation[7]^weight[7]),sum[0],C[94]);
Half_adder F273(sum_c[1]^(activation[7]^weight[7]),C[94],sum[1],C[95]);
Half_adder F274(sum_c[2]^(activation[7]^weight[7]),C[95],sum[2],C[96]);
Half_adder F275(sum_c[3]^(activation[7]^weight[7]),C[96],sum[3],C[97]);
Half_adder F276(sum_c[4]^(activation[7]^weight[7]),C[97],sum[4],C[98]);
Half_adder F277(sum_c[5]^(activation[7]^weight[7]),C[98],sum[5],C[99]);
Half_adder F278(sum_c[6]^(activation[7]^weight[7]),C[99],sum[6],C[100]);
Half_adder F279(sum_c[7]^(activation[7]^weight[7]),C[100],sum[7],C[101]);
Half_adder F280(sum_c[8]^(activation[7]^weight[7]),C[101],sum[8],C[102]);
Half_adder F281(sum_c[9]^(activation[7]^weight[7]),C[102],sum[9],C[103]);
Half_adder F282(sum_c[10]^(activation[7]^weight[7]),C[103],sum[10],C[104]);
Half_adder F283(sum_c[11]^(activation[7]^weight[7]),C[104],sum[11],C[105]);
Half_adder F284(sum_c[12]^(activation[7]^weight[7]),C[105],sum[12],C[106]);
Half_adder F285(sum_c[13]^(activation[7]^weight[7]),C[106],sum[13],C[107]);
Half_adder F286(sum_c[14]^(activation[7]^weight[7]),C[107],sum[14],C[108]);
Half_adder F287(sum_c[15]^(activation[7]^weight[7]),C[108],sum[15],);

endmodule

module Half_adder(input A,B,output S,C);

assign	S=A^B;
assign C=A&&B;

endmodule

module Full_adder(input A,B,C_in,output S,C);

assign	S=A^B^C_in;
assign 	C=((A^B)&&C_in)+(A&&B);

endmodule
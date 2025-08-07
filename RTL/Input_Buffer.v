module Input_Buffer #(
    parameter SIZE = 8,
    parameter CROW_WIDTH = $clog2(SIZE), // Compensation row size 
    parameter COMPENSATIOPN_ROW_SIZE = SIZE * 3, // Compensation row size
    parameter CMEM_SIZE = SIZE * 3, // Compensation Memory Size
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE), // Address width for the compensation memory
    parameter INVALID_VALUE = SIZE, // Invalid value for compensation row
    parameter ACTUVATION_OUT_WIDTH = SIZE*7, // Width of the activation output
    parameter COMPENSATION_ACTIVATION_OUT_WIDTH = SIZE*3*7 // Width of the compensation output
)(
    input clk,
    input rst,
    input [ACTUVATION_OUT_WIDTH-1:0] Activation,
    input [CROW_WIDTH-1:0] Compensation_Row, // The Weight Compensation_row
    input Compensation_out_valid,
    input [CMEM_ADDR_WIDTH-1:0] Compensation_Row_Reg_Addr, 
    input Cal,
    output [ACTUVATION_OUT_WIDTH-1:0] Activation_out,
    output [COMPENSATION_ACTIVATION_OUT_WIDTH-1:0] Activation_cout,
    output [CMEM_SIZE-1:0] Activation_cout_valid
);
    // Compensation Row Reg Add a bit for recording Non-Compensation-value
    reg [CROW_WIDTH:0] Compensation_Row_Reg[0:COMPENSATIOPN_ROW_SIZE-1];

    // Row0 Temp
    reg [6:0] Row0_Temp[0:2];
    // Row 1 Temp 
    reg [6:0] Row1_Temp[0:2];
    // Row 2 Temp
    reg [6:0] Row2_Temp[0:2];
    // Row 3 Temp
    reg [6:0] Row3_Temp[0:3];
    // Row 4 Temp
    reg [6:0] Row4_Temp[0:4];
    // Row 5 Temp
    reg [6:0] Row5_Temp[0:5];
    // Row 6 Temp
    reg [6:0] Row6_Temp[0:6];
    // Row 7 Temp
    reg [6:0] Row7_Temp[0:7];

    // Assigment for Activation output to systolic array
    assign Activation_out[6:0] = Row0_Temp[0];
    assign Activation_out[13:7] = Row1_Temp[1];
    assign Activation_out[20:14] = Row2_Temp[2];
    assign Activation_out[27:21] = Row3_Temp[3];
    assign Activation_out[34:28] = Row4_Temp[4];
    assign Activation_out[41:35] = Row5_Temp[5];
    assign Activation_out[48:42] = Row6_Temp[6];
    assign Activation_out[55:49] = Row7_Temp[7];
    
    assign Activation_cout_valid[0] = (Compensation_Row_Reg[2]!=8);
    assign Activation_cout_valid[1] = (Compensation_Row_Reg[1]!=8);
    assign Activation_cout_valid[2] = (Compensation_Row_Reg[0]!=8);
    assign Activation_cout_valid[3] = (Compensation_Row_Reg[5]!=8);
    assign Activation_cout_valid[4] = (Compensation_Row_Reg[4]!=8);
    assign Activation_cout_valid[5] = (Compensation_Row_Reg[3]!=8);
    assign Activation_cout_valid[6] = (Compensation_Row_Reg[8]!=8);
    assign Activation_cout_valid[7] = (Compensation_Row_Reg[7]!=8);
    assign Activation_cout_valid[8] = (Compensation_Row_Reg[6]!=8);
    assign Activation_cout_valid[9] = (Compensation_Row_Reg[11]!=8);
    assign Activation_cout_valid[10] = (Compensation_Row_Reg[10]!=8);
    assign Activation_cout_valid[11] = (Compensation_Row_Reg[9]!=8);
    assign Activation_cout_valid[12] = (Compensation_Row_Reg[14]!=8);
    assign Activation_cout_valid[13] = (Compensation_Row_Reg[13]!=8);
    assign Activation_cout_valid[14] = (Compensation_Row_Reg[12]!=8);
    assign Activation_cout_valid[15] = (Compensation_Row_Reg[17]!=8);
    assign Activation_cout_valid[16] = (Compensation_Row_Reg[16]!=8);
    assign Activation_cout_valid[17] = (Compensation_Row_Reg[15]!=8);
    assign Activation_cout_valid[18] = (Compensation_Row_Reg[20]!=8);
    assign Activation_cout_valid[19] = (Compensation_Row_Reg[19]!=8);
    assign Activation_cout_valid[20] = (Compensation_Row_Reg[18]!=8);
    assign Activation_cout_valid[21] = (Compensation_Row_Reg[23]!=8);
    assign Activation_cout_valid[22] = (Compensation_Row_Reg[22]!=8);
    assign Activation_cout_valid[23] = (Compensation_Row_Reg[21]!=8);

    assign Activation_cout[6:0] = (Compensation_Row_Reg[2] == 0) ? Row7_Temp[0] :
                                  (Compensation_Row_Reg[2] == 1) ? Row6_Temp[0] :
                                  (Compensation_Row_Reg[2] == 2) ? Row5_Temp[0] :
                                  (Compensation_Row_Reg[2] == 3) ? Row4_Temp[0] :
                                  (Compensation_Row_Reg[2] == 4) ? Row3_Temp[0] :
                                  (Compensation_Row_Reg[2] == 5) ? Row2_Temp[0] :
                                  (Compensation_Row_Reg[2] == 6) ? Row1_Temp[0] :
                                  (Compensation_Row_Reg[2] == 7) ? Row0_Temp[0] : 7'b0;


    assign Activation_cout[13:7] = (Compensation_Row_Reg[1] == 0) ? Row7_Temp[1] :
                                   (Compensation_Row_Reg[1] == 1) ? Row6_Temp[1] :
                                   (Compensation_Row_Reg[1] == 2) ? Row5_Temp[1] :
                                   (Compensation_Row_Reg[1] == 3) ? Row4_Temp[1] :
                                   (Compensation_Row_Reg[1] == 4) ? Row3_Temp[1] :
                                   (Compensation_Row_Reg[1] == 5) ? Row2_Temp[1] :
                                   (Compensation_Row_Reg[1] == 6) ? Row1_Temp[1] :
                                   (Compensation_Row_Reg[1] == 7) ? Row0_Temp[1] : 7'b0;


    assign Activation_cout[20:14] = (Compensation_Row_Reg[0] == 0) ? Row7_Temp[2] :
                                    (Compensation_Row_Reg[0] == 1) ? Row6_Temp[2] :
                                    (Compensation_Row_Reg[0] == 2) ? Row5_Temp[2] :
                                    (Compensation_Row_Reg[0] == 3) ? Row4_Temp[2] :
                                    (Compensation_Row_Reg[0] == 4) ? Row3_Temp[2] :
                                    (Compensation_Row_Reg[0] == 5) ? Row2_Temp[2] :
                                    (Compensation_Row_Reg[0] == 6) ? Row1_Temp[2] :
                                    (Compensation_Row_Reg[0] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[27:21] = (Compensation_Row_Reg[5] == 0) ? Row7_Temp[0] :
                                    (Compensation_Row_Reg[5] == 1) ? Row6_Temp[0] :
                                    (Compensation_Row_Reg[5] == 2) ? Row5_Temp[0] :
                                    (Compensation_Row_Reg[5] == 3) ? Row4_Temp[0] :
                                    (Compensation_Row_Reg[5] == 4) ? Row3_Temp[0] :
                                    (Compensation_Row_Reg[5] == 5) ? Row2_Temp[0] :
                                    (Compensation_Row_Reg[5] == 6) ? Row1_Temp[0] :
                                    (Compensation_Row_Reg[5] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[34:28] = (Compensation_Row_Reg[4] == 0) ? Row7_Temp[1] :
                                    (Compensation_Row_Reg[4] == 1) ? Row6_Temp[1] :
                                    (Compensation_Row_Reg[4] == 2) ? Row5_Temp[1] :
                                    (Compensation_Row_Reg[4] == 3) ? Row4_Temp[1] :
                                    (Compensation_Row_Reg[4] == 4) ? Row3_Temp[1] :
                                    (Compensation_Row_Reg[4] == 5) ? Row2_Temp[1] :
                                    (Compensation_Row_Reg[4] == 6) ? Row1_Temp[1] :
                                    (Compensation_Row_Reg[4] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[41:35] = (Compensation_Row_Reg[3] == 0) ? Row7_Temp[2] :
                                    (Compensation_Row_Reg[3] == 1) ? Row6_Temp[2] :
                                    (Compensation_Row_Reg[3] == 2) ? Row5_Temp[2] :
                                    (Compensation_Row_Reg[3] == 3) ? Row4_Temp[2] :
                                    (Compensation_Row_Reg[3] == 4) ? Row3_Temp[2] :
                                    (Compensation_Row_Reg[3] == 5) ? Row2_Temp[2] :
                                    (Compensation_Row_Reg[3] == 6) ? Row1_Temp[2] :
                                    (Compensation_Row_Reg[3] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[48:42] = (Compensation_Row_Reg[8] == 0) ? Row7_Temp[0] :
                                    (Compensation_Row_Reg[8] == 1) ? Row6_Temp[0] :
                                    (Compensation_Row_Reg[8] == 2) ? Row5_Temp[0] :
                                    (Compensation_Row_Reg[8] == 3) ? Row4_Temp[0] :
                                    (Compensation_Row_Reg[8] == 4) ? Row3_Temp[0] :
                                    (Compensation_Row_Reg[8] == 5) ? Row2_Temp[0] :
                                    (Compensation_Row_Reg[8] == 6) ? Row1_Temp[0] :
                                    (Compensation_Row_Reg[8] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[55:49] = (Compensation_Row_Reg[7] == 0) ? Row7_Temp[1] :
                                    (Compensation_Row_Reg[7] == 1) ? Row6_Temp[1] :
                                    (Compensation_Row_Reg[7] == 2) ? Row5_Temp[1] :
                                    (Compensation_Row_Reg[7] == 3) ? Row4_Temp[1] :
                                    (Compensation_Row_Reg[7] == 4) ? Row3_Temp[1] :
                                    (Compensation_Row_Reg[7] == 5) ? Row2_Temp[1] :
                                    (Compensation_Row_Reg[7] == 6) ? Row1_Temp[1] :
                                    (Compensation_Row_Reg[7] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[62:56] = (Compensation_Row_Reg[6] == 0) ? Row7_Temp[2] :
                                    (Compensation_Row_Reg[6] == 1) ? Row6_Temp[2] :
                                    (Compensation_Row_Reg[6] == 2) ? Row5_Temp[2] :
                                    (Compensation_Row_Reg[6] == 3) ? Row4_Temp[2] :
                                    (Compensation_Row_Reg[6] == 4) ? Row3_Temp[2] :
                                    (Compensation_Row_Reg[6] == 5) ? Row2_Temp[2] :
                                    (Compensation_Row_Reg[6] == 6) ? Row1_Temp[2] :
                                    (Compensation_Row_Reg[6] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[69:63] = (Compensation_Row_Reg[11] == 0) ? Row7_Temp[0] :
                                    (Compensation_Row_Reg[11] == 1) ? Row6_Temp[0] :
                                    (Compensation_Row_Reg[11] == 2) ? Row5_Temp[0] :
                                    (Compensation_Row_Reg[11] == 3) ? Row4_Temp[0] :
                                    (Compensation_Row_Reg[11] == 4) ? Row3_Temp[0] :
                                    (Compensation_Row_Reg[11] == 5) ? Row2_Temp[0] :
                                    (Compensation_Row_Reg[11] == 6) ? Row1_Temp[0] :
                                    (Compensation_Row_Reg[11] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[76:70] = (Compensation_Row_Reg[10] == 0) ? Row7_Temp[1] :
                                    (Compensation_Row_Reg[10] == 1) ? Row6_Temp[1] :
                                    (Compensation_Row_Reg[10] == 2) ? Row5_Temp[1] :
                                    (Compensation_Row_Reg[10] == 3) ? Row4_Temp[1] :
                                    (Compensation_Row_Reg[10] == 4) ? Row3_Temp[1] :
                                    (Compensation_Row_Reg[10] == 5) ? Row2_Temp[1] :
                                    (Compensation_Row_Reg[10] == 6) ? Row1_Temp[1] :
                                    (Compensation_Row_Reg[10] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[83:77] = (Compensation_Row_Reg[9] == 0) ? Row7_Temp[2] :
                                    (Compensation_Row_Reg[9] == 1) ? Row6_Temp[2] :
                                    (Compensation_Row_Reg[9] == 2) ? Row5_Temp[2] :
                                    (Compensation_Row_Reg[9] == 3) ? Row4_Temp[2] :
                                    (Compensation_Row_Reg[9] == 4) ? Row3_Temp[2] :
                                    (Compensation_Row_Reg[9] == 5) ? Row2_Temp[2] :
                                    (Compensation_Row_Reg[9] == 6) ? Row1_Temp[2] :
                                    (Compensation_Row_Reg[9] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[90:84] = (Compensation_Row_Reg[14] == 0) ? Row7_Temp[0] :
                                    (Compensation_Row_Reg[14] == 1) ? Row6_Temp[0] :
                                    (Compensation_Row_Reg[14] == 2) ? Row5_Temp[0] :
                                    (Compensation_Row_Reg[14] == 3) ? Row4_Temp[0] :
                                    (Compensation_Row_Reg[14] == 4) ? Row3_Temp[0] :
                                    (Compensation_Row_Reg[14] == 5) ? Row2_Temp[0] :
                                    (Compensation_Row_Reg[14] == 6) ? Row1_Temp[0] :
                                    (Compensation_Row_Reg[14] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[97:91] = (Compensation_Row_Reg[13] == 0) ? Row7_Temp[1] :
                                    (Compensation_Row_Reg[13] == 1) ? Row6_Temp[1] :
                                    (Compensation_Row_Reg[13] == 2) ? Row5_Temp[1] :
                                    (Compensation_Row_Reg[13] == 3) ? Row4_Temp[1] :
                                    (Compensation_Row_Reg[13] == 4) ? Row3_Temp[1] :
                                    (Compensation_Row_Reg[13] == 5) ? Row2_Temp[1] :
                                    (Compensation_Row_Reg[13] == 6) ? Row1_Temp[1] :
                                    (Compensation_Row_Reg[13] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[104:98] = (Compensation_Row_Reg[12] == 0) ? Row7_Temp[2] :
                                     (Compensation_Row_Reg[12] == 1) ? Row6_Temp[2] :
                                     (Compensation_Row_Reg[12] == 2) ? Row5_Temp[2] :
                                     (Compensation_Row_Reg[12] == 3) ? Row4_Temp[2] :
                                     (Compensation_Row_Reg[12] == 4) ? Row3_Temp[2] :
                                     (Compensation_Row_Reg[12] == 5) ? Row2_Temp[2] :
                                     (Compensation_Row_Reg[12] == 6) ? Row1_Temp[2] :
                                     (Compensation_Row_Reg[12] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[111:105] = (Compensation_Row_Reg[17] == 0) ? Row7_Temp[0] :
                                      (Compensation_Row_Reg[17] == 1) ? Row6_Temp[0] :
                                      (Compensation_Row_Reg[17] == 2) ? Row5_Temp[0] :
                                      (Compensation_Row_Reg[17] == 3) ? Row4_Temp[0] :
                                      (Compensation_Row_Reg[17] == 4) ? Row3_Temp[0] :
                                      (Compensation_Row_Reg[17] == 5) ? Row2_Temp[0] :
                                      (Compensation_Row_Reg[17] == 6) ? Row1_Temp[0] :
                                      (Compensation_Row_Reg[17] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[118:112] = (Compensation_Row_Reg[16] == 0) ? Row7_Temp[1] :
                                      (Compensation_Row_Reg[16] == 1) ? Row6_Temp[1] :
                                      (Compensation_Row_Reg[16] == 2) ? Row5_Temp[1] :
                                      (Compensation_Row_Reg[16] == 3) ? Row4_Temp[1] :
                                      (Compensation_Row_Reg[16] == 4) ? Row3_Temp[1] :
                                      (Compensation_Row_Reg[16] == 5) ? Row2_Temp[1] :
                                      (Compensation_Row_Reg[16] == 6) ? Row1_Temp[1] :
                                      (Compensation_Row_Reg[16] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[125:119] = (Compensation_Row_Reg[15] == 0) ? Row7_Temp[2] :
                                      (Compensation_Row_Reg[15] == 1) ? Row6_Temp[2] :
                                      (Compensation_Row_Reg[15] == 2) ? Row5_Temp[2] :
                                      (Compensation_Row_Reg[15] == 3) ? Row4_Temp[2] :
                                      (Compensation_Row_Reg[15] == 4) ? Row3_Temp[2] :
                                      (Compensation_Row_Reg[15] == 5) ? Row2_Temp[2] :
                                      (Compensation_Row_Reg[15] == 6) ? Row1_Temp[2] :
                                      (Compensation_Row_Reg[15] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[132:126] = (Compensation_Row_Reg[20] == 0) ? Row7_Temp[0] :
                                      (Compensation_Row_Reg[20] == 1) ? Row6_Temp[0] :
                                      (Compensation_Row_Reg[20] == 2) ? Row5_Temp[0] :
                                      (Compensation_Row_Reg[20] == 3) ? Row4_Temp[0] :
                                      (Compensation_Row_Reg[20] == 4) ? Row3_Temp[0] :
                                      (Compensation_Row_Reg[20] == 5) ? Row2_Temp[0] :
                                      (Compensation_Row_Reg[20] == 6) ? Row1_Temp[0] :
                                      (Compensation_Row_Reg[20] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[139:133] = (Compensation_Row_Reg[19] == 0) ? Row7_Temp[1] :
                                      (Compensation_Row_Reg[19] == 1) ? Row6_Temp[1] :
                                      (Compensation_Row_Reg[19] == 2) ? Row5_Temp[1] :
                                      (Compensation_Row_Reg[19] == 3) ? Row4_Temp[1] :
                                      (Compensation_Row_Reg[19] == 4) ? Row3_Temp[1] :
                                      (Compensation_Row_Reg[19] == 5) ? Row2_Temp[1] :
                                      (Compensation_Row_Reg[19] == 6) ? Row1_Temp[1] :
                                      (Compensation_Row_Reg[19] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[146:140] = (Compensation_Row_Reg[18] == 0) ? Row7_Temp[2] :
                                      (Compensation_Row_Reg[18] == 1) ? Row6_Temp[2] :
                                      (Compensation_Row_Reg[18] == 2) ? Row5_Temp[2] :
                                      (Compensation_Row_Reg[18] == 3) ? Row4_Temp[2] :
                                      (Compensation_Row_Reg[18] == 4) ? Row3_Temp[2] :
                                      (Compensation_Row_Reg[18] == 5) ? Row2_Temp[2] :
                                      (Compensation_Row_Reg[18] == 6) ? Row1_Temp[2] :
                                      (Compensation_Row_Reg[18] == 7) ? Row0_Temp[2] : 7'b0;

    assign Activation_cout[153:147] = (Compensation_Row_Reg[23] == 0) ? Row7_Temp[0] :
                                      (Compensation_Row_Reg[23] == 1) ? Row6_Temp[0] :
                                      (Compensation_Row_Reg[23] == 2) ? Row5_Temp[0] :
                                      (Compensation_Row_Reg[23] == 3) ? Row4_Temp[0] :
                                      (Compensation_Row_Reg[23] == 4) ? Row3_Temp[0] :
                                      (Compensation_Row_Reg[23] == 5) ? Row2_Temp[0] :
                                      (Compensation_Row_Reg[23] == 6) ? Row1_Temp[0] :
                                      (Compensation_Row_Reg[23] == 7) ? Row0_Temp[0] : 7'b0;

    assign Activation_cout[160:154] = (Compensation_Row_Reg[22] == 0) ? Row7_Temp[1] :
                                      (Compensation_Row_Reg[22] == 1) ? Row6_Temp[1] :
                                      (Compensation_Row_Reg[22] == 2) ? Row5_Temp[1] :
                                      (Compensation_Row_Reg[22] == 3) ? Row4_Temp[1] :
                                      (Compensation_Row_Reg[22] == 4) ? Row3_Temp[1] :
                                      (Compensation_Row_Reg[22] == 5) ? Row2_Temp[1] :
                                      (Compensation_Row_Reg[22] == 6) ? Row1_Temp[1] :
                                      (Compensation_Row_Reg[22] == 7) ? Row0_Temp[1] : 7'b0;

    assign Activation_cout[167:161] = (Compensation_Row_Reg[21] == 0) ? Row7_Temp[2] :
                                      (Compensation_Row_Reg[21] == 1) ? Row6_Temp[2] :
                                      (Compensation_Row_Reg[21] == 2) ? Row5_Temp[2] :
                                      (Compensation_Row_Reg[21] == 3) ? Row4_Temp[2] :
                                      (Compensation_Row_Reg[21] == 4) ? Row3_Temp[2] :
                                      (Compensation_Row_Reg[21] == 5) ? Row2_Temp[2] :
                                      (Compensation_Row_Reg[21] == 6) ? Row1_Temp[2] :
                                      (Compensation_Row_Reg[21] == 7) ? Row0_Temp[2] : 7'b0;

    integer i;
    
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<COMPENSATIOPN_ROW_SIZE;i=i+1) Compensation_Row_Reg[i] <= INVALID_VALUE;
        end
        else begin
            if(Compensation_out_valid) begin
                Compensation_Row_Reg[Compensation_Row_Reg_Addr] <= Compensation_Row; 
            end
            else begin
                if(Cal) begin
                    // Activation Pipeline 
                    // Row 0
                    Row0_Temp[0] <= Activation[6:0];
                    Row0_Temp[1] <= Row0_Temp[0];
                    Row0_Temp[2] <= Row0_Temp[1];

                    // Row 1
                    Row1_Temp[0] <= Activation[13:7];
                    Row1_Temp[1] <= Row1_Temp[0];
                    Row1_Temp[2] <= Row1_Temp[1];
                    
                    // Row 2
                    Row2_Temp[0] <= Activation[20:14];
                    Row2_Temp[1] <= Row2_Temp[0];
                    Row2_Temp[2] <= Row2_Temp[1];

                    // Row 3
                    Row3_Temp[0] <= Activation[27:21];
                    Row3_Temp[1] <= Row3_Temp[0];
                    Row3_Temp[2] <= Row3_Temp[1];
                    Row3_Temp[3] <= Row3_Temp[2];

                    // Row 4
                    Row4_Temp[0] <= Activation[34:28];
                    Row4_Temp[1] <= Row4_Temp[0];
                    Row4_Temp[2] <= Row4_Temp[1];
                    Row4_Temp[3] <= Row4_Temp[2];
                    Row4_Temp[4] <= Row4_Temp[3];

                    // Row 5                    
                    Row5_Temp[0] <= Activation[41:35];
                    Row5_Temp[1] <= Row5_Temp[0];
                    Row5_Temp[2] <= Row5_Temp[1];
                    Row5_Temp[3] <= Row5_Temp[2];
                    Row5_Temp[4] <= Row5_Temp[3];
                    Row5_Temp[5] <= Row5_Temp[4];

                    // Row 6
                    Row6_Temp[0] <= Activation[48:42];
                    Row6_Temp[1] <= Row6_Temp[0];
                    Row6_Temp[2] <= Row6_Temp[1];
                    Row6_Temp[3] <= Row6_Temp[2];
                    Row6_Temp[4] <= Row6_Temp[3];
                    Row6_Temp[5] <= Row6_Temp[4];
                    Row6_Temp[6] <= Row6_Temp[5];

                    // Row 7
                    Row7_Temp[0] <= Activation[55:49];
                    Row7_Temp[1] <= Row7_Temp[0];
                    Row7_Temp[2] <= Row7_Temp[1];
                    Row7_Temp[3] <= Row7_Temp[2];
                    Row7_Temp[4] <= Row7_Temp[3];
                    Row7_Temp[5] <= Row7_Temp[4];
                    Row7_Temp[6] <= Row7_Temp[5];
                    Row7_Temp[7] <= Row7_Temp[6];
                end
                else;
            end
        end
    end


endmodule
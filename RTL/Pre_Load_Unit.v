// Pre-load unit for weight and activation data
// This module handles the pre-loading of weights and activations,
// manages the compensation weights, and prepares the data for the systolic array.
// It includes weight processing, activation memory management, and compensation calculations.
// The module is designed to work with a clock and reset signal, and it interfaces with
// various memory components to ensure that the data is ready for computation.  

module Pre_Load_Unit(
    input clk,
    input rst,
    input [7:0] Weight,
    input [5:0] Weight_Mem_Address_in,
    input [6:0] Activation,
    input [5:0] Activation_Mem_Address_in,
    input load_mem_done,
    input PreLoad_CWeight,
    input Cal
);

    wire change_col;
    wire [2:0] Compensation_Row;
    wire [4:0] Reduced_Weight;
    wire [2:0] Compensation_Weight;
    wire [5:0] Weight_Mem_Address_out;
    wire Compensation_out_valid;
    wire [55:0] Activation_out;
    wire [167:0] Activation_cout;
  

    wire [23:0] Compensation_Weight_out;
    wire Compensation_Weight_out_valid;
    wire [2:0] Compensation_Weight_Pass[0:23];
    wire [32:0] Compensation_out[0:23];
    wire Compensation_Weight_Pass_valid[0:23];
    wire [32:0] Compensation_Acc_Sum_out[0:7];
    wire Activation_cout_valid;
    
    WPU u0(clk,rst,Weight,Weight_Mem_Address_in,load_mem_done,Reduced_Weight,Compensation_Weight,
    Compensation_Row,Compensation_out_valid,Weight_Mem_Address_out,change_col);

    Activation_Memory u1(clk,rst,Activation,Activation_Mem_Address_in,Compensation_Row,Compensation_out_valid,
    change_col,load_mem_done,Cal,Activation_out,Activation_cout,Activation_cout_valid);

    Weight_Memory u2(clk,rst,Weight_Mem_Address_out,Reduced_Weight,load_mem_done);

    Compensation_Memory u3(clk,rst,Compensation_Weight,Compensation_out_valid,change_col,load_mem_done,
    PreLoad_CWeight,Compensation_Weight_out,Compensation_Weight_out_valid);

    // 8 x 3 Compensation Array for 8x8 Systolic Array
    genvar i;
    generate
        for (i=0; i<8; i=i+1) begin: Compensation_Array
            // Instantiate CPE for each column (3 CPEs per column)
            CPE u4(clk,rst,Compensation_Weight_out[i*3+2:i*3],Activation_cout[21*i+6:21*i],
            33'd0,PreLoad_CWeight,Cal,Activation_cout_valid,Compensation_Weight_out_valid,Compensation_Weight_Pass[3*i],
            Compensation_Weight_Pass_valid[i*3],Compensation_out[3*i]);

            CPE u5(clk,rst,Compensation_Weight_Pass[3*i],Activation_cout[21*i+13:21*i+7],
            Compensation_out[3*i],PreLoad_CWeight,Cal,Activation_cout_valid,Compensation_Weight_Pass_valid[3*i],
            Compensation_Weight_Pass[3*i+1],Compensation_Weight_Pass_valid[3*i+1],Compensation_out[3*i+1]);

            CPE u6(clk,rst,Compensation_Weight_Pass[3*i+1],Activation_cout[21*i+20:21*i+14],
            Compensation_out[3*i+1],PreLoad_CWeight,Cal,Activation_cout_valid,Compensation_Weight_Pass_valid[3*i+1],
            Compensation_Weight_Pass[3*i+2],Compensation_Weight_Pass_valid[3*i+2],Compensation_out[3*i+2]);

            // Instantiate Compensation Accumulator for each column
            Compensation_Accumulator u7(clk,rst,Cal,Compensation_out[3*i+2],Compensation_Acc_Sum_out[i]);
        end
    endgenerate

endmodule
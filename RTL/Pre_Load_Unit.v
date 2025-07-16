// Pre-load unit for weight and activation data
// This module handles the pre-loading of weights and activations,
// manages the compensation weights, and prepares the data for the systolic array.
// It includes weight processing, activation memory management, and compensation calculations.
// The module is designed to work with a clock and reset signal, and it interfaces with
// various memory components to ensure that the data is ready for computation.  

module Pre_Load_Unit#(
    parameter SIZE = 8, // Size of the systolic array
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter COMPENSATIOPN_ROW_SIZE = SIZE * 3, // Compensation row size
    parameter COMPENSATIOPN_ROW_ADDR_WIDTH = $clog2(COMPENSATIOPN_ROW_SIZE), // Address width for the compensation row
    parameter INVALID_VALUE = SIZE, // Invalid value for compensation row
    parameter BIAS_WIDTH = ADDR_WIDTH, // Bias width for the activation memory
    parameter ACTUVATION_OUT_WIDTH = SIZE * 7, // Width of the activation output
    parameter COMPENSATION_OUT_WIDTH = SIZE * 3 * 7, // Width of the compensation output
    parameter SHIFT = $clog2(SIZE), // Shift value for the size of the activation memory
    parameter CROW_WIDTH = $clog2(SIZE), // Compensation row size
    parameter CMEM_SIZE = SIZE * 3, // Compensation Memory Size
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE) // Address width for the compensation memory
)(
    input clk,
    input rst,
    input [7:0] Weight,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input load_mem_done,
    input PreLoad_CWeight,
    input Cal
);
    
    wire change_col;
    wire [CROW_WIDTH-1:0] Compensation_Row;
    wire [4:0] Reduced_Weight;
    wire [2:0] Compensation_Weight;
    wire [ADDR_WIDTH-1:0] Weight_Mem_Address_out;
    wire Compensation_out_valid;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_out;
    wire [COMPENSATION_OUT_WIDTH-1:0] Activation_cout;

    wire [CMEM_SIZE-1:0] Compensation_Weight_out;
    wire Compensation_Weight_out_valid;
    wire [2:0] Compensation_Weight_Pass[0:CMEM_SIZE-1];
    wire [32:0] Compensation_out[0:CMEM_SIZE-1];
    wire Compensation_Weight_Pass_valid[0:CMEM_SIZE-1];
    wire [32:0] Compensation_Acc_Sum_out[0:SIZE-1];
    wire Activation_cout_valid;
    
    WPU #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .CROW_WIDTH(CROW_WIDTH)    
    )u0(clk,rst,Weight,Weight_Mem_Address_in,load_mem_done,Reduced_Weight,Compensation_Weight,
    Compensation_Row,Compensation_out_valid,Weight_Mem_Address_out,change_col);

    Activation_Memory #(
    .SIZE(SIZE),
    .SHIFT($clog2(SIZE)),
    .CROW_WIDTH($clog2(SIZE)),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .COMPENSATIOPN_ROW_SIZE(COMPENSATIOPN_ROW_SIZE),
    .COMPENSATIOPN_ROW_ADDR_WIDTH(COMPENSATIOPN_ROW_ADDR_WIDTH),
    .INVALID_VALUE(INVALID_VALUE),
    .BIAS_WIDTH(BIAS_WIDTH),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH),
    .COMPENSATION_OUT_WIDTH(COMPENSATION_OUT_WIDTH)
    )u1(clk,rst,Activation,Activation_Mem_Address_in,Compensation_Row,Compensation_out_valid,
    change_col,load_mem_done,Cal,Activation_out,Activation_cout,Activation_cout_valid);

    Weight_Memory #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH)
    )u2(clk,rst,Weight_Mem_Address_out,Reduced_Weight,load_mem_done);

    Compensation_Memory #(
    .SIZE(SIZE),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH)
    )u3(clk,rst,Compensation_Weight,Compensation_out_valid,change_col,load_mem_done,
    PreLoad_CWeight,Compensation_Weight_out,Compensation_Weight_out_valid);

    // 8 x 3 Compensation Array for 8x8 Systolic Array
    genvar i;
    generate 
        for (i=0; i<SIZE; i=i+1) begin: Compensation_Array
            // Instantiate CPE for each column (3 CPEs per column)
            CPE u4(clk,rst,Compensation_Weight_out[i*3+2:i*3],Activation_cout[21*i+20:21*i+14],
            33'd0,Activation_cout_valid,Compensation_Weight_out_valid,Compensation_Weight_Pass[3*i],
            Compensation_Weight_Pass_valid[i*3],Compensation_out[3*i]);

            CPE u5(clk,rst,Compensation_Weight_Pass[3*i],Activation_cout[21*i+13:21*i+7],
            Compensation_out[3*i],Activation_cout_valid,Compensation_Weight_Pass_valid[3*i],
            Compensation_Weight_Pass[3*i+1],Compensation_Weight_Pass_valid[3*i+1],Compensation_out[3*i+1]);

            CPE u6(clk,rst,Compensation_Weight_Pass[3*i+1],Activation_cout[21*i+6:21*i],
            Compensation_out[3*i+1],Activation_cout_valid,Compensation_Weight_Pass_valid[3*i+1],
            Compensation_Weight_Pass[3*i+2],Compensation_Weight_Pass_valid[3*i+2],Compensation_out[3*i+2]);

            // Instantiate Compensation Accumulator for each column
            Compensation_Accumulator u7(clk,rst,Cal,Compensation_out[3*i+2],Compensation_Acc_Sum_out[i]);
        end
    endgenerate

endmodule
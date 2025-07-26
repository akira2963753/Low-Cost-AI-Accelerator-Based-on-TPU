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
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE), // Address width for the compensation memory
    parameter WEIGHT_OUT_WIDTH = SIZE * 5, // Width of the weight output
    parameter INDEX_WIDTH = ADDR_WIDTH, // Index width for the weight memory
    parameter PARTIAL_SUM_WIDTH = ((8+4) + 4) + $clog2(SIZE) + 1, // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8 // Width of the extended activation
)(
    input clk,
    input rst,
    input [7:0] Weight,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input load_mem_done
);
    // ==============================================================================================
    // --------------------------------- TPU System Controller --------------------------------------
    // ==============================================================================================

    // ================================== Total Cycles Analysis ===================================== 
    // || Load Weight Mem & Activation Mem = 64                                                    ||
    // || Pre-Load Weight into PE = 8                                                              ||
    // || TPU Cal = 7 + 15 + 1(Output) = 23                                                        ||
    // || Total Cal Cycles = 8 + 23 = 31 Cycles                                                    ||
    // ==============================================================================================
    
    localparam LOAD_MEM = 2'd0, PRE_LOAD_WEIGHT = 2'd1, CAL = 2'd2, OUT = 2'd3;
    reg [1:0] state;
    reg PreLoad_CWeight;
    reg PreLoad_Weight;
    reg Cal;
    reg [4:0] CNT;
    
    always @(negedge clk or posedge rst) begin
        if(rst) begin
            state <= LOAD_MEM;
            PreLoad_CWeight <= 0;
            PreLoad_Weight <= 0;
            Cal <= 0;
            CNT <= 0;
        end
        else begin
            case(state)
                LOAD_MEM: state <= (load_mem_done)? PRE_LOAD_WEIGHT : LOAD_MEM;
                PRE_LOAD_WEIGHT: begin
                    PreLoad_CWeight <= (CNT<3);
                    PreLoad_Weight <= 1'b1;
                    state <= (CNT==7)? CAL : PRE_LOAD_WEIGHT;
                    CNT <= CNT + 1;
                end
                CAL: begin
                    PreLoad_Weight <= 1'b0;
                    Cal <= 1'b1;
                    state <= (CNT==31)? OUT : CAL;
                    CNT <= CNT + 1;
                end
                OUT: begin
                    CNT <= 0;
                    Cal <= 1'b0;
                    state <= OUT;
                end
            endcase
        end
    end

    // ==============================================================================================
    // -------------------------------- Activation Memory Net ---------------------------------------
    // ==============================================================================================

    wire change_col;
    wire [CROW_WIDTH-1:0] Compensation_Row;
    wire [4:0] Reduced_Weight;
    wire [2:0] Compensation_Weight;
    wire [ADDR_WIDTH-1:0] Weight_Mem_Address_out;
    wire Compensation_out_valid;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_out;
    wire [COMPENSATION_OUT_WIDTH-1:0] Activation_cout;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_Bff_out;
    wire [COMPENSATION_OUT_WIDTH-1:0] Activation_Bff_cout;

    // ==============================================================================================
    // -------------------------------- Compensation Memory Net -------------------------------------
    // ==============================================================================================

    wire [CMEM_SIZE-1:0] Compensation_Weight_out;
    wire Compensation_Weight_out_valid;
    wire [2:0] Compensation_Weight_Pass[0:CMEM_SIZE-1];
    wire [13:0] Compensation_out[0:CMEM_SIZE-1];
    wire Compensation_Weight_Pass_valid[0:CMEM_SIZE-1];
    wire [21:0] Compensation_Acc_Sum_out[0:SIZE-1];
    wire Activation_out_valid_in;
    wire [7:0] Activation_out_valid;

    // ==============================================================================================
    // ------------------------------------ Weight Memory Net ---------------------------------------
    // ==============================================================================================

    wire [WEIGHT_OUT_WIDTH-1:0] Weight_out;
    wire Weight_out_valid;

    // ==============================================================================================
    // --------------------------------- 8 x 8 Systolic Array Net -----------------------------------
    // ==============================================================================================
    
    wire [4:0] Weight_Wire [0:8][0:7];
    wire [6:0] Activation_Wire [0:7][0:8];
    wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_Wire [0:8][0:7];

    wire [4:0] Weight_in[0:7];
    wire [6:0] Activation_in[0:7];
    //wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in[0:7];

    wire Weight_Pass_valid [0:7][0:7];
    wire Activation_Pass_valid [0:7][0:7];
    wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out [0:7];
    wire [PARTIAL_SUM_WIDTH-1:0] Final_Partial_Sum[0:7];

    // ==============================================================================================
    // ------------------------------- Weight Pre-Processing Unit -----------------------------------
    // ==============================================================================================

    WPU #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .CROW_WIDTH(CROW_WIDTH)    
    )u0(clk,rst,Weight,Weight_Mem_Address_in,load_mem_done,Reduced_Weight,Compensation_Weight,
    Compensation_Row,Compensation_out_valid,Weight_Mem_Address_out,change_col);

    // ==============================================================================================
    // ---------------------------------- Activation Memory Unit ------------------------------------
    // ==============================================================================================

    Activation_Memory #(
    .SIZE(SIZE),
    .SHIFT($clog2(SIZE)),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .BIAS_WIDTH(BIAS_WIDTH),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH)
    )u1(clk,rst,Activation,Activation_Mem_Address_in,load_mem_done,Cal,Activation_out,Activation_out_valid_in);

    // ==============================================================================================
    // ------------------------------------------ Buffer --------------------------------------------
    // ==============================================================================================    


    Buffer #(
    .SIZE(SIZE),
    .SHIFT(SHIFT),
    .CROW_WIDTH(CROW_WIDTH),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .COMPENSATIOPN_ROW_SIZE(COMPENSATIOPN_ROW_SIZE),
    .COMPENSATIOPN_ROW_ADDR_WIDTH(COMPENSATIOPN_ROW_ADDR_WIDTH),
    .INVALID_VALUE(INVALID_VALUE),
    .BIAS_WIDTH(BIAS_WIDTH),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH),
    .COMPENSATION_OUT_WIDTH(COMPENSATION_OUT_WIDTH)
    )B0(clk,rst,Activation_out,Compensation_Row,Compensation_out_valid,change_col,Cal,Activation_out_valid_in,
    Activation_Bff_out,Activation_Bff_cout,Activation_out_valid);
    


    // ==============================================================================================
    // ---------------------------------- Weight Memory Unit ----------------------------------------
    // ==============================================================================================

    Weight_Memory #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .INDEX_WIDTH(ADDR_WIDTH),    
    .BIAS_WIDTH(BIAS_WIDTH),
    .WEIGHT_OUT_WIDTH(WEIGHT_OUT_WIDTH),
    .SHIFT(SHIFT)
    )u2(clk,rst,Weight_Mem_Address_out,Reduced_Weight,load_mem_done,PreLoad_Weight,Weight_out,Weight_out_valid);


    // ==============================================================================================
    // ------------------------------ Compensation Memory Unit --------------------------------------
    // ==============================================================================================

    Compensation_Memory #(
    .SIZE(SIZE),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH)
    )u3(clk,rst,Compensation_Weight,Compensation_out_valid,change_col,load_mem_done,
    PreLoad_CWeight,Compensation_Weight_out,Compensation_Weight_out_valid);

    // ==============================================================================================
    // --------------------- 8 x 3 Compensation Array for 8x8 Systolic Array ------------------------
    // ==============================================================================================

    genvar i,j;
    generate 
        for (i=0; i<SIZE; i=i+1) begin: Compensation_Array
            // Instantiate CPE for each column (3 CPEs per column)
            CPE u4(clk,rst,Compensation_Weight_out[i*3+2:i*3],Activation_Bff_cout[21*i+6:21*i],
            14'd0,Activation_out_valid[0],Compensation_Weight_out_valid,Compensation_Weight_Pass[3*i],
            Compensation_Weight_Pass_valid[i*3],Compensation_out[3*i]);

            CPE u5(clk,rst,Compensation_Weight_Pass[3*i],Activation_Bff_cout[21*i+13:21*i+7],
            Compensation_out[3*i],Activation_out_valid[1],Compensation_Weight_Pass_valid[3*i],
            Compensation_Weight_Pass[3*i+1],Compensation_Weight_Pass_valid[3*i+1],Compensation_out[3*i+1]);

            CPE u6(clk,rst,Compensation_Weight_Pass[3*i+1],Activation_Bff_cout[21*i+20:21*i+14],
            Compensation_out[3*i+1],Activation_out_valid[2],Compensation_Weight_Pass_valid[3*i+1],
            Compensation_Weight_Pass[3*i+2],Compensation_Weight_Pass_valid[3*i+2],Compensation_out[3*i+2]);

            // Instantiate Compensation Accumulator for each column
            Compensation_Accumulator u7(clk,rst,Cal,Compensation_out[3*i+2],Compensation_Acc_Sum_out[i]);
        end
    endgenerate

    // ==============================================================================================
    // ----------------  Systolic Array for 8x8 RPE (Reduced Processing Element) --------------------
    // ==============================================================================================

    generate  // Unpack the weight and activation inputs for the systolic array
        for (i = 0; i < 8; i = i + 1) begin : unpack_input
            assign Weight_in[i] = Weight_out[5*i+4:5*i];
            assign Activation_in[i] = Activation_Bff_out[7*i+6:7*i];
            //assign Partial_Sum_in[i] = Partial_Sum_out[29*i + 28 : 29*i];
        end
    endgenerate
    
    generate  // Initialize the wires for the systolic array
        for(i=0;i<8;i=i+1) begin
            assign Weight_Wire[0][i] = Weight_in[i]; // weight input
            assign Activation_Wire[i][0] = Activation_in[i];
            assign Partial_Sum_Wire[0][i] = 0; // Initialize partial sum
            assign Activation_Pass_valid[i][0] = Activation_out_valid[i];
            assign Weight_Pass_valid[0][i] = Weight_out_valid;
        end
    endgenerate

    generate // Instantiate the RPE for the systolic array
        for (i = 0; i < 8; i = i + 1) begin : Row_gen
            for (j = 0; j < 8; j = j + 1) begin : Col_gen
                RPE #(
                .SIZE(SIZE),
                .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH),
                .ACTIVATION_EXTEND_WIDTH(ACTIVATION_EXTEND_WIDTH)
                )u8(clk,rst,Weight_Wire[i][j],Activation_Wire[i][j],
                Partial_Sum_Wire[i][j],Weight_Pass_valid[i][j],Activation_Pass_valid[i][j],
                Weight_Wire[i+1][j],Weight_Pass_valid[i+1][j],
                Activation_Wire[i][j+1],Activation_Pass_valid[i][j+1],
                Partial_Sum_Wire[i+1][j]);
            end
        end
    endgenerate

    // ==============================================================================================
    // ----------------  The Accumulator of 8x8 Systolic Array with Compensation --------------------
    // ==============================================================================================

    generate  // Assign the final partial sum outputs
        for (i = 0; i < 8; i = i + 1) begin : Output_gen
            Accumulator #(
            .SIZE(SIZE),
            .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH)
            )u9(clk,rst,Cal,Partial_Sum_Wire[8][i],Partial_Sum_out[i]);
            assign Final_Partial_Sum[i] = Partial_Sum_out[i] + Compensation_Acc_Sum_out[i];
        end
    endgenerate

endmodule
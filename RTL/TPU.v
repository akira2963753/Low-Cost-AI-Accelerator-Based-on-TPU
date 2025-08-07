module TPU#(
    parameter SIZE = 8, // Size of the systolic array
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter WRITE_ADDR_WIDTH = $clog2(MEM_SIZE), // Write Address width for the memory
    parameter READ_ADDR_WIDTH = $clog2(SIZE), // Read Address width for the memory
    parameter COMPENSATIOPN_ROW_SIZE = SIZE * 3, // Compensation row size
    parameter INVALID_VALUE = SIZE, // Invalid value for compensation row
    parameter ACTUVATION_OUT_WIDTH = SIZE * 7, // Width of the activation output
    parameter COMPENSATION_ACTIVATION_OUT_WIDTH = SIZE * 3 * 7, // Width of the compensation output
    parameter CROW_WIDTH = $clog2(SIZE), // Compensation row size
    parameter CMEM_SIZE = SIZE * 3, // Compensation Memory Size
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE), // Address width for the compensation memory
    parameter COMPENSATION_WEIGHT_OUT_WIDTH = SIZE * 4, // Width of the compensation output
    parameter WEIGHT_OUT_WIDTH = SIZE * 5, // Width of the weight output
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE), // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8, // Width of the extended activation
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 5 + 1 
)(
    input clk,
    input rst,
    input [7:0] Weight,
    input [WRITE_ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [6:0] Activation,
    input [WRITE_ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input Mem_Write,
    input [WRITE_ADDR_WIDTH-1:0] UB_Rd_Address_in,
    output [6:0] UB_Data,
    output Done
);
    // ==============================================================================================
    // ----------------------------- Declare the Integer and Genvar ---------------------------------
    // ==============================================================================================  
    
    genvar i,j;
    integer k;

    // ==============================================================================================
    // ------------------------- Weight Pre-Processing Unit Net and Register ------------------------
    // ==============================================================================================

    wire [4:0] Reduced_Weight;
    wire [3:0] Compensation_Weight;
    wire [CROW_WIDTH-1:0] Compensation_Row;
    wire Compensation_out_valid;
    wire [WRITE_ADDR_WIDTH-1:0] Weight_Mem_Address_out;
    wire [CMEM_ADDR_WIDTH-1:0] Compensation_Mem_Wr_Addr;

    // ==============================================================================================
    // ------------------------------- Weight Memory Net and Register -------------------------------
    // ==============================================================================================   
    
    wire Weight_Mem_Wr_en;
    wire Weight_Mem_Rd_en;
    wire [READ_ADDR_WIDTH-1:0] Weight_Mem_Rd_Addr;
    wire [WEIGHT_OUT_WIDTH-1:0] Weight_out;

    // ==============================================================================================
    // ----------------------------- Activation Memory Net and Register -----------------------------
    // ==============================================================================================  

    wire Activation_Mem_Wr_en;
    wire Activation_Mem_Rd_en;
    wire [READ_ADDR_WIDTH-1:0] Activation_Mem_Rd_Addr;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_out;

    // ==============================================================================================
    // ---------------------------- Compensation Memory Net and Register ----------------------------
    // ==============================================================================================  

    wire [COMPENSATION_WEIGHT_OUT_WIDTH-1:0] Compensation_Weight_out;
    wire Compensation_Mem_Wr_en;
    wire Compensation_Mem_Rd_en;
    wire [1:0] Compensation_Mem_Rd_Addr;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_out[0:CMEM_SIZE-1];

    // ==============================================================================================
    // -------------------------------- Input Buffer Net and Register -------------------------------
    // ==============================================================================================  

    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_Buf_out;
    wire [COMPENSATION_ACTIVATION_OUT_WIDTH-1:0] Activation_Buf_cout;
    wire [CMEM_SIZE-1:0] Activation_cout_valid;

    // ==============================================================================================
    // -------------------------- Systolic Array Net and Register (8 x 8) ---------------------------
    // ==============================================================================================
    // --------------------------- Compensation Processing Element (CPE) ----------------------------
    // ==============================================================================================
    
    wire Compensation_Weight_out_valid;
    wire [3:0] Compensation_Weight_Pass[0:CMEM_SIZE-1];
    wire Compensation_Weight_Pass_valid[0:CMEM_SIZE-1];

    // ==============================================================================================
    // ------------------------- Reduced-Precision Processing Element (RPE) ------------------------- 
    // ==============================================================================================

    wire Weight_out_valid;
    wire [4:0] Weight_Wire [0:8][0:7];
    wire [6:0] Activation_Wire [0:7][0:8];
    wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_Wire [0:8][0:7];
    wire [4:0] Weight_in[0:7];
    wire [6:0] Activation_in[0:7];
    wire Weight_Pass_valid [0:8][0:7];

    // ==============================================================================================
    // -------------------------------- Accumulator Net and Register -------------------------------- 
    // ==============================================================================================   

    wire CACC_Wr_en;
    wire [7:0] ACC_Wr_en;
    wire Acc_Rd_en;
    reg [2:0] Acc_Wr_Addr[0:7];
    reg [2:0] CAcc_Wr_Addr;
    reg [2:0] Acc_Rd_Addr;
    wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out [0:7];
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum_in[0:7];

    // ==============================================================================================
    // ----------------- Actiation Function / Unified Buffer Net and Register ----------------------- 
    // ==============================================================================================   
    
    wire [PARTIAL_SUM_WIDTH-1:0] Activation_Function_out [0:7];
    wire UB_Wr_en;
    wire [5:0] UB_Wr_Addr;
    assign UB_Wr_Addr = (UB_Wr_en)? (Acc_Rd_Addr-1) << 3 : 0  ;

    // ==============================================================================================
    // ------------------------- TPU System Controller Net and Reigster -----------------------------
    // ==============================================================================================  
    
    wire PreLoad_CWeight,PreLoad_Weight,Cal;
    
    // ==============================================================================================
    // --------------------------------- TPU System Controller --------------------------------------
    // ==============================================================================================

    TSC #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .READ_ADDR_WIDTH(READ_ADDR_WIDTH)
    )TPU_System_Controller(.*);

    
    assign Compensation_Partial_Sum_in[0] = Compensation_out[2];
    assign Compensation_Partial_Sum_in[1] = Compensation_out[5];
    assign Compensation_Partial_Sum_in[2] = Compensation_out[8];
    assign Compensation_Partial_Sum_in[3] = Compensation_out[11];
    assign Compensation_Partial_Sum_in[4] = Compensation_out[14];
    assign Compensation_Partial_Sum_in[5] = Compensation_out[17];
    assign Compensation_Partial_Sum_in[6] = Compensation_out[20];
    assign Compensation_Partial_Sum_in[7] = Compensation_out[23];

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(k=0;k<8;k=k+1) Acc_Wr_Addr[k] <= 0;
            CAcc_Wr_Addr <= 0;
            Acc_Rd_Addr <= 0;
        end
        else begin
            CAcc_Wr_Addr <= (CACC_Wr_en)? CAcc_Wr_Addr + 1 : 0;
            for(k=0;k<8;k=k+1) Acc_Wr_Addr[k] <= (ACC_Wr_en[k])? Acc_Wr_Addr[k] + 1 : 0;
            Acc_Rd_Addr <= (Acc_Rd_en)? Acc_Rd_Addr + 1 : 0;
        end
    end

    // ==============================================================================================
    // ------------------------------- Weight Pre-Processing Unit -----------------------------------
    // ==============================================================================================

    WPU #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(WRITE_ADDR_WIDTH),
    .CROW_WIDTH(CROW_WIDTH),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH)  
    )Weight_Pre_Processing_Unit(.*);

    // ==============================================================================================
    // ---------------------------------- Activation Memory Unit ------------------------------------
    // ==============================================================================================

    Activation_Memory #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .WRITE_ADDR_WIDTH(WRITE_ADDR_WIDTH),
    .READ_ADDR_WIDTH(READ_ADDR_WIDTH),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH)
    )Activation_Memory_Unit(
    .clk(clk),
    .Activation(Activation),
    .Wr_Addr(Activation_Mem_Address_in),
    .Wr_en(Activation_Mem_Wr_en),
    .Rd_en(Activation_Mem_Rd_en),
    .Rd_Addr(Activation_Mem_Rd_Addr),
    .Activation_out(Activation_out));

    // ==============================================================================================
    // ------------------------------------------ Buffer --------------------------------------------
    // ==============================================================================================    


    Input_Buffer #(
    .SIZE(SIZE),
    .CROW_WIDTH(CROW_WIDTH),
    .COMPENSATIOPN_ROW_SIZE(COMPENSATIOPN_ROW_SIZE),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH),
    .INVALID_VALUE(INVALID_VALUE),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH),
    .COMPENSATION_ACTIVATION_OUT_WIDTH(COMPENSATION_ACTIVATION_OUT_WIDTH)
    )Input_Buffer_Unit(
    .clk(clk),
    .rst(rst),
    .Activation(Activation_out),
    .Compensation_Row(Compensation_Row),
    .Compensation_out_valid(Compensation_out_valid),
    .Compensation_Row_Reg_Addr(Compensation_Mem_Wr_Addr),
    .Cal(Cal),
    .Activation_out(Activation_Buf_out),
    .Activation_cout(Activation_Buf_cout),
    .Activation_cout_valid(Activation_cout_valid));
    
    // ==============================================================================================
    // ---------------------------------- Weight Memory Unit ----------------------------------------
    // ==============================================================================================

    Weight_Memory #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .WRITE_ADDR_WIDTH(WRITE_ADDR_WIDTH),  
    .READ_ADDR_WIDTH(READ_ADDR_WIDTH), 
    .WEIGHT_OUT_WIDTH(WEIGHT_OUT_WIDTH)
    )Weight_Memory_Unit(
    .clk(clk),
    .Wr_Addr(Weight_Mem_Address_out),
    .Weight_Data(Reduced_Weight),
    .Wr_en(Weight_Mem_Wr_en),
    .Rd_en(Weight_Mem_Rd_en),
    .Rd_Addr(Weight_Mem_Rd_Addr),
    .Weight_out(Weight_out));

    // ==============================================================================================
    // ------------------------------ Compensation Memory Unit --------------------------------------
    // ==============================================================================================

    Compensation_Memory #(
    .SIZE(SIZE),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH),
    .COMPENSATION_WEIGHT_OUT_WIDTH(COMPENSATION_WEIGHT_OUT_WIDTH)
    )Compensation_Memory_Unit(
    .clk(clk),
    .rst(rst),
    .Compensation_Weight(Compensation_Weight),
    .Wr_Addr(Compensation_Mem_Wr_Addr),
    .Wr_en(Compensation_Mem_Wr_en),
    .Rd_Addr(Compensation_Mem_Rd_Addr),
    .Rd_en(Compensation_Mem_Rd_en),
    .Compensation_Weight_out(Compensation_Weight_out));

    // ==============================================================================================
    // --------------------- 8 x 3 Compensation Array for 8x8 Systolic Array ------------------------
    // ==============================================================================================

    generate 
        for (i=0; i<SIZE; i=i+1) begin: Compensation_Array
            // Instantiate CPE for each column (3 CPEs per column)
            CPE #(
            .COMPENSATION_PARTIAL_SUM_WIDTH(COMPENSATION_PARTIAL_SUM_WIDTH)
            )Compensation_Processing_Element_u0(
            .clk(clk),
            .Compensation_Weight(Compensation_Weight_out[i*4+3:i*4]),
            .Activation_cin(Activation_Buf_cout[21*i+6:21*i]),
            .Compensation_Partial_Sum({COMPENSATION_PARTIAL_SUM_WIDTH{1'b0}}),
            .Activation_cout_valid(Activation_cout_valid[i*3]),
            .Compensation_Weight_out_valid(Compensation_Weight_out_valid),
            .Compensation_Weight_Pass(Compensation_Weight_Pass[3*i]),
            .Compensation_Weight_Pass_valid(Compensation_Weight_Pass_valid[i*3]),
            .Compensation_out(Compensation_out[3*i]));

            CPE #(
            .COMPENSATION_PARTIAL_SUM_WIDTH(COMPENSATION_PARTIAL_SUM_WIDTH)
            )Compensation_Processing_Element_u1(
            .clk(clk),
            .Compensation_Weight(Compensation_Weight_Pass[3*i]),
            .Activation_cin(Activation_Buf_cout[21*i+13:21*i+7]),
            .Compensation_Partial_Sum(Compensation_out[3*i]),
            .Activation_cout_valid(Activation_cout_valid[i*3+1]),
            .Compensation_Weight_out_valid(Compensation_Weight_Pass_valid[3*i]),
            .Compensation_Weight_Pass(Compensation_Weight_Pass[3*i+1]),
            .Compensation_Weight_Pass_valid(Compensation_Weight_Pass_valid[3*i+1]),
            .Compensation_out(Compensation_out[3*i+1]));

            CPE #(
            .COMPENSATION_PARTIAL_SUM_WIDTH(COMPENSATION_PARTIAL_SUM_WIDTH)
            )Compensation_Processing_Element_u2(
            .clk(clk),
            .Compensation_Weight(Compensation_Weight_Pass[3*i+1]),
            .Activation_cin(Activation_Buf_cout[21*i+20:21*i+14]),
            .Compensation_Partial_Sum(Compensation_out[3*i+1]),
            .Activation_cout_valid(Activation_cout_valid[i*3+2]),
            .Compensation_Weight_out_valid(Compensation_Weight_Pass_valid[3*i+1]),
            .Compensation_Weight_Pass(Compensation_Weight_Pass[3*i+2]),
            .Compensation_Weight_Pass_valid(Compensation_Weight_Pass_valid[3*i+2]),
            .Compensation_out(Compensation_out[3*i+2]));
        end
    endgenerate

    // ==============================================================================================
    // ----------------  Systolic Array for 8x8 RPE (Reduced Processing Element) --------------------
    // ==============================================================================================

    generate  // Unpack the weight and activation inputs for the systolic array
        for (i = 0; i < 8; i = i + 1) begin : unpack_input
            assign Weight_in[i] = Weight_out[5*i+4:5*i];
            assign Activation_in[i] = Activation_Buf_out[7*i+6:7*i];
        end
    endgenerate
    
    generate  // Initialize the wires for the systolic array
        for(i=0;i<8;i=i+1) begin
            assign Weight_Wire[0][i] = Weight_in[i]; // weight input
            assign Activation_Wire[i][0] = Activation_in[i];
            assign Partial_Sum_Wire[0][i] = 0; // Initialize partial sum
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
                )Reduced_Processing_Element(
                .clk(clk),
                .Weight_in(Weight_Wire[i][j]),
                .Activation_in(Activation_Wire[i][j]),
                .Partial_Sum_in(Partial_Sum_Wire[i][j]),
                .Weight_in_valid(Weight_Pass_valid[i][j]),
                .Weight_Pass(Weight_Wire[i+1][j]),
                .Weight_Pass_valid(Weight_Pass_valid[i+1][j]),
                .Activation_Pass(Activation_Wire[i][j+1]),
                .Partial_Sum_out(Partial_Sum_Wire[i+1][j]));
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
            )Accumulator_Unit(
            .clk(clk),
            .Acc_Wr_Addr(Acc_Wr_Addr[i]),
            .ACC_Wr_en(ACC_Wr_en[i]),
            .CAcc_Wr_Addr(CAcc_Wr_Addr),
            .CACC_Wr_en(CACC_Wr_en),
            .Acc_Rd_en(Acc_Rd_en),
            .Acc_Rd_Addr(Acc_Rd_Addr),
            .Compensation_Partial_Sum_in(Compensation_Partial_Sum_in[i]),
            .Partial_Sum_in(Partial_Sum_Wire[8][i]),
            .Partial_Sum_out(Partial_Sum_out[i]));
        
            Activation_Function #(
            .SIZE(SIZE),
            .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH)
            )Activation_Function_Unit(
            .Partial_Sum_in(Partial_Sum_out[i]),
            .Partial_Sum_out(Activation_Function_out[i]));
        end 
    endgenerate

    // ==============================================================================================
    // ----------------------------------- Unified Buffer -------------------------------------------
    // ==============================================================================================

    UB #(
    .SIZE(SIZE),
    .PARTIAL_SUM_WIDTH(PARTIAL_SUM_WIDTH)
    )Unified_Buffer(
    .clk(clk),
    .Wr_en(UB_Wr_en),
    .Wr_Addr(UB_Wr_Addr),
    .Wr_Data_0(Activation_Function_out[0]),
    .Wr_Data_1(Activation_Function_out[1]),
    .Wr_Data_2(Activation_Function_out[2]),
    .Wr_Data_3(Activation_Function_out[3]),
    .Wr_Data_4(Activation_Function_out[4]),
    .Wr_Data_5(Activation_Function_out[5]),
    .Wr_Data_6(Activation_Function_out[6]),
    .Wr_Data_7(Activation_Function_out[7]),
    .Rd_Addr(UB_Rd_Address_in),
    .Rd_Data(UB_Data));

endmodule
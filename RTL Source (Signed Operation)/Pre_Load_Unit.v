// Pre-load unit for weight and activation data
// This module handles the pre-loading of weights and activations,
// manages the compensation weights, and prepares the data for the systolic array.
// It includes weight processing, activation memory management, and compensation calculations.
// The module is designed to work with a clock and reset signal, and it interfaces with
// various memory components to ensure that the data is ready for computation.  

module Pre_Load_Unit#(
    parameter SIZE = 8, // Size of the systolic array
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter WRITE_ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter READ_ADDR_WIDTH = $clog2(SIZE), // Address width for the memory
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
    parameter COMPENSATION_WEIGHT_OUT_WIDTH = SIZE * 4, // Width of the compensation output
    parameter WEIGHT_OUT_WIDTH = SIZE * 5, // Width of the weight output
    parameter INDEX_WIDTH = ADDR_WIDTH, // Index width for the weight memory
    parameter PARTIAL_SUM_WIDTH = ((8+4) + 4) + $clog2(SIZE), // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8, // Width of the extended activation
    parameter COMPENSATION_PARTIAL_SUM_WIDTH = 8 + 5 + 1 
)(
    input clk,
    input rst,
    input [7:0] Weight,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input Mem_Write
);
    // ==============================================================================================
    // ----------------------------- Declare the Integer and Genvar ---------------------------------
    // ==============================================================================================  
    
    genvar i,j;
    integer k;

    // ==============================================================================================
    // ------------------------- TPU System Controller Net and Reigster -----------------------------
    // ==============================================================================================  
    
    localparam LOAD_MEM = 2'd0, PRE_LOAD_WEIGHT = 2'd1, CAL = 2'd2, OUT = 2'd3;
    reg [1:0] state;
    reg [5:0] Cycle_CNT; // True Cycle Count = Cycle_CNT - 1
    reg PreLoad_CWeight,PreLoad_Weight,Cal;

    // ==============================================================================================
    // ------------------------- Weight Pre-Processing Unit Net and Register ------------------------
    // ==============================================================================================

    wire [4:0] Reduced_Weight;
    wire [3:0] Compensation_Weight;
    wire [CROW_WIDTH-1:0] Compensation_Row;
    wire Compensation_out_valid;
    wire [ADDR_WIDTH-1:0] Weight_Mem_Address_out;
    wire [CMEM_ADDR_WIDTH-1:0] Compensation_Mem_Wr_Addr;

    // ==============================================================================================
    // ------------------------------- Weight Memory Net and Register -------------------------------
    // ==============================================================================================   
    
    wire Weight_Mem_Wr_en;
    wire Weight_Mem_Rd_en;
    reg [READ_ADDR_WIDTH-1:0] Weight_Mem_Rd_Addr;
    wire [WEIGHT_OUT_WIDTH-1:0] Weight_out;

    // ==============================================================================================
    // ----------------------------- Activation Memory Net and Register -----------------------------
    // ==============================================================================================  

    wire Activation_Mem_Wr_en;
    wire Activation_Mem_Rd_en;
    reg [READ_ADDR_WIDTH-1:0] Activation_Mem_Rd_Addr;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_out;

    // ==============================================================================================
    // ---------------------------- Compensation Memory Net and Register ----------------------------
    // ==============================================================================================  

    wire [COMPENSATION_WEIGHT_OUT_WIDTH-1:0] Compensation_Weight_out;
    wire Compensation_Mem_Wr_en;
    wire Compensation_Mem_Rd_en;
    reg [1:0] Compensation_Mem_Rd_Addr;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_out[0:CMEM_SIZE-1];

    // ==============================================================================================
    // -------------------------------- Input Buffer Net and Register -------------------------------
    // ==============================================================================================  

    wire Activation_out_valid_in;
    wire [ACTUVATION_OUT_WIDTH-1:0] Activation_Buf_out;
    wire [COMPENSATION_OUT_WIDTH-1:0] Activation_Buf_cout;
    wire [7:0] Activation_out_valid;
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
    wire Weight_Pass_valid [0:7][0:7];
    wire Activation_Pass_valid [0:7][0:7];

    // ==============================================================================================
    // ----------------------------- Compensation Acc and Original Acc ------------------------------ 
    // ==============================================================================================   

    reg CACC_Wr_en;
    reg [7:0] ACC_Wr_en;
    wire [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out [0:7];
    reg [2:0] Acc_Wr_Addr[0:7];
    reg [2:0] CAcc_Wr_Addr;
    wire [COMPENSATION_PARTIAL_SUM_WIDTH-1:0] Compensation_Partial_Sum_in[0:7];
    
    // ==============================================================================================
    // --------------------------------- TPU System Controller --------------------------------------
    // ==============================================================================================
    // || ----------------- Total Cycle (PreLoadWeight + Calculate 1 Pattern) -------------------- ||
    // ||                                                                                          ||
    // ||    Pre-load Weight need 7 cycles and Buffer need 1 cycles Delay = 8 cycles [n]           ||
    // ||    Activation boardcast needed 7 (Input Delay) + 8 (Matrix Size) + 7 (Boardcast Delay)   ||
    // ||    So Activation totally need 22 cycles [(n-1) + n + (n-1)]                              ||  
    // ||    Ouput to Register of Accumulater need 1 cycle                                         ||
    // ||    Total Cycle = 8 + 22 + 1 = 31 cycles [4n-1]                                           ||
    // ||                                                                                          ||   
    // ==============================================================================================

    assign Weight_Mem_Wr_en = Mem_Write;
    assign Weight_Mem_Rd_en = (state==PRE_LOAD_WEIGHT);
    assign Activation_Mem_Wr_en = Mem_Write;
    assign Activation_Mem_Rd_en = (state==CAL);
    assign Compensation_Mem_Wr_en = (Mem_Write)&&Compensation_out_valid;
    assign Compensation_Mem_Rd_en = (state==PRE_LOAD_WEIGHT&&Cycle_CNT<3);

    assign Weight_out_valid = PreLoad_Weight;
    assign Activation_out_valid_in = (Cal&&Activation_Mem_Rd_Addr!=7)? 1'b1 : 1'b0;
    assign Compensation_Weight_out_valid = PreLoad_CWeight;

    assign Compensation_Partial_Sum_in[0] = Compensation_out[2];
    assign Compensation_Partial_Sum_in[1] = Compensation_out[5];
    assign Compensation_Partial_Sum_in[2] = Compensation_out[8];
    assign Compensation_Partial_Sum_in[3] = Compensation_out[11];
    assign Compensation_Partial_Sum_in[4] = Compensation_out[14];
    assign Compensation_Partial_Sum_in[5] = Compensation_out[17];
    assign Compensation_Partial_Sum_in[6] = Compensation_out[20];
    assign Compensation_Partial_Sum_in[7] = Compensation_out[23];

    always @(negedge clk or posedge rst) begin
        if(rst) begin
            state <= LOAD_MEM;
            PreLoad_CWeight <= 0;
            PreLoad_Weight <= 0;
            Cal <= 0;
            Cycle_CNT <= 0;
            CACC_Wr_en <= 0;
            ACC_Wr_en <= 0;
            Weight_Mem_Rd_Addr <= 0;
            Activation_Mem_Rd_Addr <= 0;
            Compensation_Mem_Rd_Addr <= 0;
        end
        else begin
            case(state)
                LOAD_MEM: state <= (!Mem_Write)? PRE_LOAD_WEIGHT : LOAD_MEM;
                PRE_LOAD_WEIGHT: begin
                    // Pre-load Compensation Weight just need 3 Cycles
                    PreLoad_CWeight <= (Cycle_CNT<3);
                    // Compensation_Mem_Rd_Addr ++ 
                    Compensation_Mem_Rd_Addr <= (Compensation_Mem_Rd_Addr==2)? 2 : Compensation_Mem_Rd_Addr + 1;
                    // Pre-load Weight Signal Until the PRE_LOAD_WEIGHT State done
                    PreLoad_Weight <= 1'b1;
                    // Weight_Mem_Rd_Addr ++
                    Weight_Mem_Rd_Addr <= Weight_Mem_Rd_Addr + 1;
                    // Cycle_CNT ++
                    Cycle_CNT <= Cycle_CNT + 1;
                    // 8 Cycles turn to CAL
                    state <= (Cycle_CNT==(SIZE-1))? CAL : PRE_LOAD_WEIGHT;
                end
                CAL: begin
                    // Compensation Acc Wr_enable
                    CACC_Wr_en <= (Cycle_CNT>(SIZE+3)&&Cycle_CNT<(2*SIZE+3+1))? 1'b1 : 1'b0;
                    // Acc Wr_enable
                    ACC_Wr_en[0] <= (Cycle_CNT>(2*SIZE)&&Cycle_CNT<(3*SIZE+1))? 1'b1 : 1'b0;
                    // Use 45 angles operation on the Acc Wr_enable
                    for(k=1;k<8;k=k+1) ACC_Wr_en[k] <= ACC_Wr_en[k-1];
                    // Pre-load Weight Signal turn to 0
                    PreLoad_Weight <= 1'b0;
                    // Cal Signal turn to 1
                    Cal <= 1'b1;    
                    // Activation_Mem_Rd_Addr ++ 
                    Activation_Mem_Rd_Addr <= (Activation_Mem_Rd_Addr==7)? 7 : Activation_Mem_Rd_Addr + 1;
                    // Cycle_CNT ++
                    Cycle_CNT <= Cycle_CNT + 1;
                    // 25 (9+25=32) Cycles turn to CAL
                    state <= (Cycle_CNT==32)? OUT : CAL;
                end
                OUT: begin
                    Cal <= 1'b0;
                    state <= OUT;
                end
            endcase
        end
    end

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(k=0;k<8;k=k+1) Acc_Wr_Addr[k] <= 0;
            CAcc_Wr_Addr <= 0;
        end
        else begin
            CAcc_Wr_Addr <= (CACC_Wr_en)? CAcc_Wr_Addr + 1 : 0;
            for(k=0;k<8;k=k+1) Acc_Wr_Addr[k] <= (ACC_Wr_en[k])? Acc_Wr_Addr[k] + 1 : 0;
        end
    end

    // ==============================================================================================
    // ------------------------------- Weight Pre-Processing Unit -----------------------------------
    // ==============================================================================================

    WPU #(
    .SIZE(SIZE),
    .MEM_SIZE(MEM_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .CROW_WIDTH(CROW_WIDTH),
    .CMEM_SIZE(CMEM_SIZE),
    .CMEM_ADDR_WIDTH(CMEM_ADDR_WIDTH)  
    )Weight_Pre_Processing_Unit(
    .clk(clk),
    .rst(rst),
    .Weight(Weight),
    .Weight_Mem_Address_in(Weight_Mem_Address_in),
    .Mem_Write(Mem_Write),
    .Reduced_Weight(Reduced_Weight),
    .Compensation_Weight(Compensation_Weight),
    .Compensation_Row(Compensation_Row),
    .Compensation_out_valid(Compensation_out_valid),
    .Weight_Mem_Address_out(Weight_Mem_Address_out),
    .Compensation_Mem_Wr_Addr(Compensation_Mem_Wr_Addr));

    // ==============================================================================================
    // ---------------------------------- Activation Memory Unit ------------------------------------
    // ==============================================================================================

    Activation_Memory #(
    .SIZE(SIZE),
    .SHIFT($clog2(SIZE)),
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
    .COMPENSATIOPN_ROW_ADDR_WIDTH(COMPENSATIOPN_ROW_ADDR_WIDTH),
    .INVALID_VALUE(INVALID_VALUE),
    .ACTUVATION_OUT_WIDTH(ACTUVATION_OUT_WIDTH),
    .COMPENSATION_OUT_WIDTH(COMPENSATION_OUT_WIDTH)
    )Input_Buffer_Unit(
    .clk(clk),
    .rst(rst),
    .Activation(Activation_out),
    .Compensation_Row(Compensation_Row),
    .Compensation_out_valid(Compensation_out_valid),
    .Compensation_Row_Reg_Addr(Compensation_Mem_Wr_Addr),
    .Cal(Cal),
    .Activation_out_valid_in(Activation_out_valid_in),
    .Activation_out(Activation_Buf_out),
    .Activation_cout(Activation_Buf_cout),
    .Activation_out_valid(Activation_out_valid),
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
            .Compensation_Partial_Sum_in(Compensation_Partial_Sum_in[i]),
            .Partial_Sum_in(Partial_Sum_Wire[8][i]),
            .Partial_Sum_out(Partial_Sum_out[i]));
        end
    endgenerate

endmodule
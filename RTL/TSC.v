// Tensor Processing Unit (TPU) System Controller
module TSC #(
    parameter SIZE = 8,
    parameter MEM_SIZE = SIZE * SIZE,
    parameter READ_ADDR_WIDTH = $clog2(SIZE)
)(
    input clk,
    input rst,
    input Mem_Write,
    input Compensation_out_valid,
    output Weight_Mem_Wr_en,
    output Weight_Mem_Rd_en,
    output Activation_Mem_Wr_en,
    output Activation_Mem_Rd_en,
    output Compensation_Mem_Wr_en,
    output Compensation_Mem_Rd_en,
    output Weight_out_valid,
    output Compensation_Weight_out_valid,
    output Acc_Rd_en,
    output reg [READ_ADDR_WIDTH-1:0] Weight_Mem_Rd_Addr,
    output reg [1:0] Compensation_Mem_Rd_Addr,
    output reg [READ_ADDR_WIDTH-1:0] Activation_Mem_Rd_Addr,
    output reg PreLoad_CWeight,
    output reg PreLoad_Weight,
    output reg Cal,
    output reg CACC_Wr_en,
    output reg [7:0] ACC_Wr_en,
    output reg UB_Wr_en,
    output reg Done
);
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

    localparam LOAD_MEM = 3'd0, PRE_LOAD_WEIGHT = 3'd1, CAL = 3'd2, OUT = 3'd3, DONE = 3'd4;
    integer k;
    reg [2:0] state;
    reg [5:0] Cycle_CNT; // True Cycle Count = Cycle_CNT - 1

    assign Weight_Mem_Wr_en = Mem_Write;
    assign Weight_Mem_Rd_en = (state==PRE_LOAD_WEIGHT);
    assign Activation_Mem_Wr_en = Mem_Write;
    assign Activation_Mem_Rd_en = (state==CAL);
    assign Compensation_Mem_Wr_en = (Mem_Write)&&Compensation_out_valid;
    assign Compensation_Mem_Rd_en = (state==PRE_LOAD_WEIGHT&&Cycle_CNT<3);

    assign Weight_out_valid = PreLoad_Weight;
    assign Compensation_Weight_out_valid = PreLoad_CWeight;
    assign Acc_Rd_en = (state==OUT);

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
            UB_Wr_en <= 0;
            Done <= 0;
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
                    Cycle_CNT <= Cycle_CNT + 1;
                    Cal <= 0;
                    UB_Wr_en <= 1;
                    state <= (Cycle_CNT==40)? DONE : OUT;
                end
                DONE: begin
                    UB_Wr_en <= 0;
                    Cycle_CNT <= 0;
                    Done <= 1;
                end
            endcase
        end
    end





endmodule
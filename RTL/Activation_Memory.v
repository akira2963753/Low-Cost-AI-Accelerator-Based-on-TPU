// Activation_Memory.v
// This module manages the storage and retrieval of activation values for the systolic array.
// It handles the activation memory, allowing for pre-loading and updating of activation values
// based on the input signals. The module is designed to work with a clock and reset signal
// and interfaces with the pre-load unit to ensure that the activation values are ready for computation
// in the systolic array. It also manages the compensation rows for the activation values.

module Activation_Memory #(
    parameter SIZE = 8,
    parameter SHIFT = $clog2(SIZE), // Shift value for the size of the activation memory
    parameter CROW_WIDTH = $clog2(SIZE), // Compensation row size 
    parameter MEM_SIZE = SIZE*SIZE, // Size of the activation memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter COMPENSATIOPN_ROW_SIZE = SIZE * 3, // Compensation row size
    parameter COMPENSATIOPN_ROW_ADDR_WIDTH = $clog2(COMPENSATIOPN_ROW_SIZE), // Compensation row address width
    parameter INVALID_VALUE = SIZE, // Invalid value for compensation row
    parameter BIAS_WIDTH = ADDR_WIDTH, // Bias width for the activation memory
    parameter ACTUVATION_OUT_WIDTH = SIZE*7, // Width of the activation output
    parameter COMPENSATION_OUT_WIDTH = SIZE*3*7 // Width of the compensation output
)(
    input clk,
    input rst,
    input [6:0] Activation,
    input [ADDR_WIDTH-1:0] Activation_Mem_Address_in,
    input [CROW_WIDTH-1:0] Compensation_Row, // The Weight Compensation_row
    input Compensation_out_valid,
    input change_col, // The Weight Loading is done.
    input load_mem_done,
    input Cal,
    output [ACTUVATION_OUT_WIDTH-1:0] Activation_out,
    output [COMPENSATION_OUT_WIDTH-1:0] Activation_cout,
    output Activation_cout_valid
); 

    // Declare the Register, Net and Integer
    reg [6:0] Activation_Mem[0:MEM_SIZE-1]; // Activation Memory
    reg [CROW_WIDTH:0] Compensation_Row_Reg[0:COMPENSATIOPN_ROW_SIZE-1]; // Compensation Row Reg Add a bit for recording Non-Compensation-value
    reg [COMPENSATIOPN_ROW_ADDR_WIDTH-1:0] Index; // Index for Compensation Row Reg
    wire [BIAS_WIDTH-1:0] bias;
    integer i;

    // Assingment Activation_out for Comepensation
    assign bias = Index << SHIFT; // Bias for the activation memory address
    assign Activation_cout_valid = (Cal&&Index<SIZE)? 1'b1 : 1'b0;

    // Assign Activation output to the compensation shadow array
    assign Activation_cout = {Activation_Mem[Compensation_Row_Reg[21]+bias],Activation_Mem[Compensation_Row_Reg[22]+bias],
                            Activation_Mem[Compensation_Row_Reg[23]+bias],Activation_Mem[Compensation_Row_Reg[18]+bias],
                            Activation_Mem[Compensation_Row_Reg[19]+bias],Activation_Mem[Compensation_Row_Reg[20]+bias],
                            Activation_Mem[Compensation_Row_Reg[15]+bias],Activation_Mem[Compensation_Row_Reg[16]+bias],
                            Activation_Mem[Compensation_Row_Reg[17]+bias],Activation_Mem[Compensation_Row_Reg[12]+bias],
                            Activation_Mem[Compensation_Row_Reg[13]+bias],Activation_Mem[Compensation_Row_Reg[14]+bias],
                            Activation_Mem[Compensation_Row_Reg[9]+bias],Activation_Mem[Compensation_Row_Reg[10]+bias],
                            Activation_Mem[Compensation_Row_Reg[11]+bias],Activation_Mem[Compensation_Row_Reg[6]+bias],
                            Activation_Mem[Compensation_Row_Reg[7]+bias],Activation_Mem[Compensation_Row_Reg[8]+bias],
                            Activation_Mem[Compensation_Row_Reg[5]+bias],Activation_Mem[Compensation_Row_Reg[4]+bias],
                            Activation_Mem[Compensation_Row_Reg[3]+bias],Activation_Mem[Compensation_Row_Reg[2]+bias],
                            Activation_Mem[Compensation_Row_Reg[1]+bias],Activation_Mem[Compensation_Row_Reg[0]+bias]};

    // Assigment for Activation output to systolic array
    assign Activation_out = {Activation_Mem[7+bias], Activation_Mem[6+bias], Activation_Mem[5+bias],
                             Activation_Mem[4+bias], Activation_Mem[3+bias], Activation_Mem[2+bias],
                             Activation_Mem[1+bias], Activation_Mem[0+bias]};
    

    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<COMPENSATIOPN_ROW_SIZE;i=i+1) Compensation_Row_Reg[i] <= INVALID_VALUE;
            Index <= 0;
        end
        else begin
            if(!load_mem_done) begin
                // Load Activation Value into Memory
                Activation_Mem[Activation_Mem_Address_in] <= Activation;

                if(Compensation_out_valid) begin
                    Compensation_Row_Reg[Index] <=  Compensation_Row;
                    Index <= Index + 1; // Index ++
                end
                else if(change_col) begin // Change column
                    Index <= Index + (3 - (Index%3));
                end
                else; 
            end
            else if(Cal) begin
                Index <= (Index==SIZE)? SIZE : Index + 1;
            end
            else Index <= 0;
        end
    end 

endmodule

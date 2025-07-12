// Activation Memory for 8x8 Systolic Array 
// 2025.07.12

module  Activation_Memory (
    input clk,
    input rst,
    // The Weight Compensation_row
    input [2:0] Compensation_Row,
    input out_Compensation_valid,
    input change_col,
    // The Weight Loading is done.
    input done,
    output out_valid); 

    // Declare the local parameter
    localparam Size = 8;
    localparam Mem_Size = Size*Size;
    localparam Compensation_Size = Size * 3;
    localparam Width = Size / 2;
    localparam Index_Width = $clog2(Compensation_Size);
    localparam Invalid_Value = Size;

    // Declare the Register and Net
    reg [7:0] Activation_Mem[0:Mem_Size-1];
    reg [Width:0] Compensation_Reg[Compensation_Size-1:0];  // Compensation Reg Add a bit for recording Non-Compensation-value
    reg [Index_Width-1:0] Index;
    integer i;
    // Assingment index & out_valid
    assign out_valid = (done==1);

    // Set Compensation Register to save compensation row
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i=0;i<Compensation_Size;i=i+1) Compensation_Reg[i] <= Invalid_Value;
            Index <= 0;
        end
        else begin
            if(done==0) begin
                if(out_Compensation_valid) begin
                    Compensation_Reg[Index] <=  Compensation_Row;
                    Index <= Index + 1; // Index ++
                end
                else if(change_col) begin // Change column
                    Index <= Index + (3 - (Index%3));
                end
                else;
            end
            else;
        end
    end 


endmodule

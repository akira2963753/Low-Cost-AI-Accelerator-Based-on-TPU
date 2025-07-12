/// Pre-Load Unit for 8x8 Systolic Array 
// 2025.07.12
module Pre_Load_Unit(
    input clk,
    input rst,
    input [7:0] Weight,
    input [5:0] Weight_Mem_Address_in,
    input done
);

    wire change_col;
    wire [2:0] Compensation_Row;
    wire [4:0] Reduced_Weight;
    wire [2:0] Compensation_Weight;
    wire [5:0] Weight_Mem_Address_out;
    wire out_valid;
    wire out_Compensation_valid;

    
    WPU u0(clk,rst,Weight,Weight_Mem_Address_in,Reduced_Weight,Compensation_Weight,
    Compensation_Row,out_valid,out_Compensation_valid,Weight_Mem_Address_out,change_col);

    Activation_Memory u1(clk,rst,Compensation_Row,out_Compensation_valid,change_col,done,out_valid);

    Weight_Memory u2(clk,rst,Weight_Mem_Address_out,Reduced_Weight,done);

    Compensation_Memory u3(clk,rst,Compensation_Weight,out_Compensation_valid,change_col,done);

    


endmodule
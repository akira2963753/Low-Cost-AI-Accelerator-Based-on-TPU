module Activation_Function #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE) // Size of the partial sum
)(
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    // Relu Function 
    assign Partial_Sum_out = (Partial_Sum_in[PARTIAL_SUM_WIDTH-1])? 0 : Partial_Sum_in;
    
endmodule
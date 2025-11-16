module Activation_Function #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 2*SIZE + $clog2(SIZE)
)(
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    output [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    // Relu Function 
    assign Partial_Sum_out = (Partial_Sum_in[PARTIAL_SUM_WIDTH-1])? 0 : Partial_Sum_in;
    
endmodule
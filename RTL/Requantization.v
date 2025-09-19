module Requantization #(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = 8 + 4 + 4 + $clog2(SIZE),
    parameter ACTIVATION_WIDTH = 7
)(
    input [PARTIAL_SUM_WIDTH-1:0] Requant_in,
    output [ACTIVATION_WIDTH-1:0] Requant_out
);

    assign Requant_out = Requant_in[PARTIAL_SUM_WIDTH - 1 : PARTIAL_SUM_WIDTH - ACTIVATION_WIDTH];

endmodule
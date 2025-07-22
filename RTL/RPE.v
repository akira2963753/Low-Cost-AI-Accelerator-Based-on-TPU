module RPE#(
    parameter SIZE = 8,
    parameter PARTIAL_SUM_WIDTH = ((8+4) + 4) + $clog2(SIZE) + 1, // Size of the partial sum
    parameter ACTIVATION_EXTEND_WIDTH = PARTIAL_SUM_WIDTH - 8 // Width of the extended activation
)(
    input clk,
    input rst,
    input [4:0] Weight_out,
    input [6:0] Activation_out,
    input [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_in,
    input Weight_out_valid,
    input Activation_out_valid,
    output reg [4:0] Weight_Pass,
    output Weight_Pass_valid,
    output reg [6:0] Activation_Pass,
    output reg Activation_Pass_valid,
    output reg [PARTIAL_SUM_WIDTH-1:0] Partial_Sum_out
);
    reg [4:0] Weight_Reg;

    assign Weight_Pass_valid = Weight_out_valid;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            // Reset all registers
            Weight_Reg <= 0;
            Weight_Pass <= 0;
            Partial_Sum_out <= 0;
            Activation_Pass <= 0;
            Activation_Pass_valid <= 0;
        end
        else begin
            Activation_Pass_valid <= Activation_out_valid;
            // Weight pass downward
            if(Weight_out_valid) begin
                Weight_Pass <= Weight_out;
                Weight_Reg <= Weight_out;
            end
            else if(Activation_out_valid) begin
                case(Weight_Reg[4])
                    0: Partial_Sum_out <= Partial_Sum_in + {{{ACTIVATION_EXTEND_WIDTH{1'b0}},Activation_out,1'b1}*Weight_Reg[3:0],1'b0} + {Activation_out,1'b1};
                    1: Partial_Sum_out <= Partial_Sum_in + {{{ACTIVATION_EXTEND_WIDTH{1'b0}},Activation_out,1'b1}*Weight_Reg[3:0],4'b0000};
                endcase
                Activation_Pass <= Activation_out;
            end
            else begin
                Partial_Sum_out <= 0; // If no computation is needed, pass directly downward
            end
        end
    end






endmodule
// Weight Preprocessing Unit for 8x8 Systolic Array 
// 2025.07.12

module WPU (
    input clk,
    input rst,
    input [7:0] Weight,
    input [5:0] Weight_Mem_Address_in,
    output reg [4:0] Reduced_Weight,
    output reg [2:0] Compensation_Weight,
    output reg [2:0] Compensation_Row,
    output reg out_valid,
    output reg out_Compensation_valid,
    output reg [5:0] Weight_Mem_Address_out,
    output change_col
);
    wire Non_MSR_4;
    assign Non_MSR_4 = (Weight[7]&Weight[6]&Weight[5]&Weight[4])^(Weight[7]|Weight[6]|Weight[5]|Weight[4]);
    assign change_col = (Weight_Mem_Address_out[2:0] == 3'b111);
    
    always @(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            Weight_Mem_Address_out <= 6'd0;
            out_valid <= 1'b0;
            Reduced_Weight <= 5'd0;
            Compensation_Weight <= 3'd0;
            Compensation_Row <= 3'd0;
            out_Compensation_valid <= 1'b0;
        end
        else begin
            // Pass Weight Address
            Weight_Mem_Address_out <= Weight_Mem_Address_in;
            if(Non_MSR_4) begin
                Reduced_Weight <= {1'b1,Weight[7:4]};
                Compensation_Row <= Weight_Mem_Address_in[2:0]; // Weight_Mem_Address_in % 8
                Compensation_Weight <= Weight[3:1];
                out_Compensation_valid <= 1'b1;
                out_valid <= 1'b1;
            end
            else begin
                Reduced_Weight <= {1'b0,Weight[4:1]};
                out_Compensation_valid <= 1'b0;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
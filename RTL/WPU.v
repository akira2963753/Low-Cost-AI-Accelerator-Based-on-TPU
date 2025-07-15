module WPU (
    input clk,
    input rst,
    input [7:0] Weight,
    input [5:0] Weight_Mem_Address_in,
    input load_mem_done,
    output reg [4:0] Reduced_Weight,
    output reg [2:0] Compensation_Weight,
    output reg [2:0] Compensation_Row,
    output reg Compensation_out_valid,
    output reg [5:0] Weight_Mem_Address_out,
    output change_col
);
    // Declare the Register, Net and Integer
    wire Non_MSR_4;
    reg [1:0] Boundary_limit;

    // Assigment 
    assign Non_MSR_4 = (Weight[7]&Weight[6]&Weight[5]&Weight[4])^(Weight[7]|Weight[6]|Weight[5]|Weight[4]);
    assign change_col = (Weight_Mem_Address_out[2:0] == 3'b111); 
    
    always @(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            Weight_Mem_Address_out <= 6'd0;
            Reduced_Weight <= 5'd0;
            Compensation_Weight <= 3'd0;
            Compensation_Row <= 3'd0;
            Compensation_out_valid <= 1'b0;
            Boundary_limit <= 2'd0;
        end
        else begin
            if(load_mem_done==1'b0) begin
                Weight_Mem_Address_out <= Weight_Mem_Address_in; // Pass Weight Address
                if(Non_MSR_4) begin
                    Reduced_Weight <= {1'b1,Weight[7:4]};
                    if(Boundary_limit==2'd3) begin
                        Compensation_out_valid <= 1'b0;
                        Boundary_limit <= 2'd0;
                    end
                    else begin
                        Compensation_Row <= Weight_Mem_Address_in[2:0]; // Weight_Mem_Address_in % 8
                        Compensation_Weight <= Weight[3:1];
                        Compensation_out_valid <= 1'b1;
                        Boundary_limit <= Boundary_limit + 2'd1;
                    end
                end
                else begin
                    Reduced_Weight <= {1'b0,Weight[4:1]};
                    Compensation_out_valid <= 1'b0;
                end
            end
            else;
        end
    end
endmodule
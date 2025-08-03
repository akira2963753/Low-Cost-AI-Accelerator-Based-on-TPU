module WPU #(
    parameter SIZE = 8, // Size of the systolic array
    parameter MEM_SIZE = SIZE * SIZE, // Total size of the weight memory
    parameter ADDR_WIDTH = $clog2(MEM_SIZE), // Address width for the memory
    parameter CROW_WIDTH = $clog2(SIZE),
    parameter CMEM_SIZE = SIZE * 3, // Compensation Memory Size
    parameter CMEM_ADDR_WIDTH = $clog2(CMEM_SIZE) // Address width for the compensation memory
)(
    input clk,
    input rst,
    input [7:0] Weight,
    input [ADDR_WIDTH-1:0] Weight_Mem_Address_in,
    input Mem_Write,
    output reg [4:0] Reduced_Weight,
    output reg [3:0] Compensation_Weight,
    output reg [CROW_WIDTH-1:0] Compensation_Row,
    output reg Compensation_out_valid,
    output reg [ADDR_WIDTH-1:0] Weight_Mem_Address_out,
    output reg [CMEM_ADDR_WIDTH-1:0] Compensation_Mem_Wr_Addr
);
    // Declare the Register, Net and Integer
    wire Non_MSR_4;
    reg [1:0] Boundary_limit;
    wire [1:0] Judge;

    // Assigment 
    assign Judge = Compensation_Mem_Wr_Addr % 3;
    assign Non_MSR_4 = (Weight[7]&Weight[6]&Weight[5]&Weight[4])^(Weight[7]|Weight[6]|Weight[5]|Weight[4]);
    assign change_col = (Weight_Mem_Address_out[2:0] == 3'b111&&Mem_Write); 
    
    always @(posedge clk or posedge rst) begin
        if(rst) begin // Reset to zero
            Weight_Mem_Address_out <= 0;
            Reduced_Weight <= 0;
            Compensation_Weight <= 0;
            Compensation_Row <= 0;
            Compensation_out_valid <= 0;
            Boundary_limit <= 0;
        end
        else begin
            if(Mem_Write) begin
                Weight_Mem_Address_out <= Weight_Mem_Address_in; // Pass Weight Address
                if(Non_MSR_4) begin
                    Reduced_Weight <= {1'b1,Weight[7:4]};
                    if(Boundary_limit==2'd3) begin
                        Compensation_out_valid <= 0;
                        Boundary_limit <= 0;
                    end
                    else begin
                        Compensation_Row <= Weight_Mem_Address_in[2:0]; // Weight_Mem_Address_in % 8
                        Compensation_Weight <= (Weight[7])? {1'b1,Weight[3:1]} : {1'b0,Weight[3:1]};
                        Compensation_out_valid <= 1;
                        Boundary_limit <= (change_col)? 0 : Boundary_limit + 1;
                    end
                end
                else begin
                    if(change_col) begin
                        Boundary_limit <= 0;
                    end
                    else;
                    Reduced_Weight <= {1'b0,Weight[4:1]};
                    Compensation_out_valid <= 0;
                end
            end
            else begin
                Compensation_out_valid <= 0;
            end
        end
    end

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            Compensation_Mem_Wr_Addr <= 0;
        end
        else begin
            if(Compensation_out_valid) begin
                Compensation_Mem_Wr_Addr <= (Judge==2)? Compensation_Mem_Wr_Addr : Compensation_Mem_Wr_Addr + 1;
            end
            else if(change_col) begin
                Compensation_Mem_Wr_Addr <= Compensation_Mem_Wr_Addr + (3 - Judge);
            end
            else;
        end

    end
endmodule
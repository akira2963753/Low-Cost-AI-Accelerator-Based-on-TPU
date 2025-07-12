module tb_Pre_Load_Unit();
    reg clk;
    reg rst;
    reg [7:0] Weight;
    reg [5:0] Weight_Mem_Address_in;
    reg done;
    integer i;
    integer weight_file,comp_file,comp2_file;

    Pre_Load_Unit  P0(clk,rst,Weight,Weight_Mem_Address_in,done);

    initial begin
        clk = 0;
        rst = 0;
        Weight = 0;
        Weight_Mem_Address_in = 0;
        done = 0;
    end

    always #5 clk = ~clk;

    always @(posedge clk or posedge rst) begin
        if(rst) Weight_Mem_Address_in <= 0;
        else if(Weight_Mem_Address_in!=6'd63) Weight_Mem_Address_in <= Weight_Mem_Address_in + 1;
        else;
    end

    initial begin
        #1 rst = 1;
        #4 begin 
            rst = 0;
            Weight = 8'b00110011; // Col0 - 0 *
        end
        #10 Weight = 8'b00001111; // Col0 - 1
        #10 Weight = 8'b00001111; // Col0 - 2
        #10 Weight = 8'b11110000; // Col0 - 3
        #10 Weight = 8'b00001111; // Col0 - 4
        #10 Weight = 8'b00110011; // Col0 - 5 *
        #10 Weight = 8'b00001111; // Col0 - 6
        #10 Weight = 8'b11110000; // Col0 - 7
        #10 Weight = 8'b00001111; // Col1 - 0
        #10 Weight = 8'b00110110; // Col1 - 1 *
        #10 Weight = 8'b00001111; // Col1 - 2
        #10 Weight = 8'b11110000; // Col1 - 3
        #10 Weight = 8'b00001111; // Col1 - 4
        #10 Weight = 8'b11110000; // Col1 - 5
        #10 Weight = 8'b00001111; // Col1 - 6
        #10 Weight = 8'b11110000; // Col1 - 7
        #10 Weight = 8'b00110011; // Col2 - 0 *
        #10 Weight = 8'b11110000; // Col2 - 1
        #10 Weight = 8'b11001001; // Col2 - 2 *
        #10 Weight = 8'b11110000; // Col2 - 3
        #10 Weight = 8'b00001111; // Col2 - 4
        #10 Weight = 8'b11110000; // Col2 - 5
        #10 Weight = 8'b00001111; // Col2 - 6
        #10 Weight = 8'b11110000; // Col2 - 7
        #10 Weight = 8'b00001111; // Col3 - 0
        #10 Weight = 8'b11110000; // Col3 - 1
        #10 Weight = 8'b00001111; // Col3 - 2
        #10 Weight = 8'b00110110; // Col3 - 3 *
        #10 Weight = 8'b00001111; // Col3 - 4
        #10 Weight = 8'b11110000; // Col3 - 5
        #10 Weight = 8'b00001111; // Col3 - 6
        #10 Weight = 8'b11110000; // Col3 - 7
        #10 Weight = 8'b00001111; // Col4 - 0
        #10 Weight = 8'b11110000; // Col4 - 1
        #10 Weight = 8'b00001111; // Col4 - 2
        #10 Weight = 8'b11110000; // Col4 - 3
        #10 Weight = 8'b11000011; // Col4 - 4 *
        #10 Weight = 8'b11110000; // Col4 - 5
        #10 Weight = 8'b00001111; // Col4 - 6
        #10 Weight = 8'b11110000; // Col4 - 7
        #10 Weight = 8'b00001111; // Col5 - 0
        #10 Weight = 8'b11110000; // Col5 - 1
        #10 Weight = 8'b00001111; // Col5 - 2
        #10 Weight = 8'b11110000; // Col5 - 3
        #10 Weight = 8'b00001111; // Col5 - 4
        #10 Weight = 8'b00110110; // Col5 - 5 *
        #10 Weight = 8'b00001111; // Col5 - 6
        #10 Weight = 8'b11110000; // Col5 - 7
        #10 Weight = 8'b00001111; // Col6 - 0
        #10 Weight = 8'b11110000; // Col6 - 1
        #10 Weight = 8'b00001111; // Col6 - 2
        #10 Weight = 8'b11110000; // Col6 - 3
        #10 Weight = 8'b00001111; // Col6 - 4
        #10 Weight = 8'b11110000; // Col6 - 5
        #10 Weight = 8'b11001111; // Col6 - 6 *
        #10 Weight = 8'b11110000; // Col6 - 7
        #10 Weight = 8'b00001111; // Col7 - 0
        #10 Weight = 8'b11110000; // Col7 - 1
        #10 Weight = 8'b00001111; // Col7 - 2
        #10 Weight = 8'b11110000; // Col7 - 3
        #10 Weight = 8'b00001111; // Col7 - 4
        #10 Weight = 8'b11110000; // Col7 - 5
        #10 Weight = 8'b00001111; // Col7 - 6
        #10 Weight = 8'b00110110; // Col7 - 7 *
        #15 done = 1;
        #15 begin
            // Weight_Mem.out
            weight_file = $fopen("Weight_Mem.out", "w"); 
            if (weight_file) begin
                $fdisplay(weight_file, "// Weight Memory Contents with Index");
                $fdisplay(weight_file, "// Format: [Index] Data");
                for (i = 0; i < 64; i = i + 1) begin
                    $fdisplay(weight_file, "%05b  //[%02d]", P0.u2.Weight_Mem[i], i);
                end
                $fclose(weight_file);  
                $display("Weight Memory written to Weight_Mem.out");
            end
            else begin
                $display("Error: Cannot open Weight_Mem.out for writing");
            end

            comp_file = $fopen("Compensation_Mem.out", "w");
            if (comp_file) begin
                $fdisplay(comp_file, "// Compensation Memory Contents with Index");
                $fdisplay(comp_file, "// Format: [Index] Data");
                for (i = 0; i < 24; i = i + 1) begin
                    $fdisplay(comp_file, "%03b  //[%02d]", P0.u3.Compensation_Mem[i], i);
                end
                $fclose(comp_file);
                $display("Compensation Memory written to Compensation_Mem.out");
            end
            else begin
                $display("Error: Cannot open Compensation_Mem.out for writing");
            end
            
            comp2_file = $fopen("Compensation_Row.out", "w");
            if (comp2_file) begin
                $fdisplay(comp2_file, "// Compensation Row Contents with Index");
                $fdisplay(comp2_file, "// Format: [Index] Data");
                for (i = 0; i < 24; i = i + 1) begin
                    $fdisplay(comp2_file, "%d  //[%02d]", P0.u1.Compensation_Reg[i], i);
                end
                $fclose(comp2_file);
                $display("Compensation Row written to Compensation_Mem.out");
            end
            else begin
                $display("Error: Cannot open Compensation_Row.out for writing");
            end

            $finish;           
        end
    end

endmodule
// tb_Pre_Load_Unit.v
// Testbench for Pre_Load_Unit module
// This testbench simulates the Pre_Load_Unit module, providing inputs for weights and activations,
// and checking the outputs. It includes a clock generator, reset signal, and stimulus for the module.
// The testbench also writes the memory contents to files for verification.


module tb_Pre_Load_Unit();
    reg clk;
    reg rst;
    reg [7:0] Weight;
    reg [5:0] Weight_Mem_Address_in;
    reg [6:0] Activation;
    reg [5:0] Activation_Mem_Address_in;
    reg load_mem_done;
    //reg PreLoad_CWeight;
    //reg Cal;
    //reg PreLoad_Weight;

    reg [7:0] Weight_Data [0:63];
    reg [6:0] Activation_Data [0:63];
    integer i;
    integer weight_file,activation_file,comp_file,comp2_file;

    Pre_Load_Unit  P0(clk,rst,Weight,Weight_Mem_Address_in,Activation,
    Activation_Mem_Address_in,load_mem_done);//,PreLoad_CWeight,PreLoad_Weight,Cal);

    initial begin
        clk = 0;
        rst = 0;
        Weight = 0;
        Weight_Mem_Address_in = 0;
        load_mem_done = 0;
        //PreLoad_CWeight = 0;
        //Cal = 0;
        //PreLoad_Weight = 0;
        Activation = 0;
        Activation_Mem_Address_in = 0;
        $readmemb("Weight_Data.dat", Weight_Data);
        $readmemb("Activation_Data.dat", Activation_Data);
    end

    always #5 clk = ~clk;

    always @(posedge clk or posedge rst) begin
        if(rst) begin 
            Weight_Mem_Address_in <= 0;
            Activation_Mem_Address_in <= 0;
        end
        else if(Weight_Mem_Address_in!=6'd63||Activation_Mem_Address_in!=6'd63) begin
            Weight_Mem_Address_in <= Weight_Mem_Address_in + 1;
            Activation_Mem_Address_in <= Activation_Mem_Address_in + 1;
        end
        else;
    end

    initial begin
        #1 rst = 1;
        #4 begin 
            rst = 0;
            Weight = Weight_Data[0];
            Activation = Activation_Data[0];
        end
        #10 begin
            Weight = Weight_Data[1];
            Activation = Activation_Data[1];
        end
        #10 begin
            Weight = Weight_Data[2];
            Activation = Activation_Data[2];
        end
        #10 begin
            Weight = Weight_Data[3];
            Activation = Activation_Data[3];
        end
        #10 begin
            Weight = Weight_Data[4];
            Activation = Activation_Data[4];
        end
        #10 begin
            Weight = Weight_Data[5];
            Activation = Activation_Data[5];
        end
        #10 begin
            Weight = Weight_Data[6];
            Activation = Activation_Data[6];
        end
        #10 begin
            Weight = Weight_Data[7];
            Activation = Activation_Data[7];
        end
        #10 begin
            Weight = Weight_Data[8];
            Activation = Activation_Data[8];
        end
        #10 begin
            Weight = Weight_Data[9];
            Activation = Activation_Data[9];
        end
        #10 begin
            Weight = Weight_Data[10];
            Activation = Activation_Data[10];
        end
        #10 begin
            Weight = Weight_Data[11];
            Activation = Activation_Data[11];
        end
        #10 begin
            Weight = Weight_Data[12];
            Activation = Activation_Data[12];
        end
        #10 begin
            Weight = Weight_Data[13];
            Activation = Activation_Data[13];
        end
        #10 begin
            Weight = Weight_Data[14];
            Activation = Activation_Data[14];
        end
        #10 begin
            Weight = Weight_Data[15];
            Activation = Activation_Data[15];
        end
        #10 begin
            Weight = Weight_Data[16];
            Activation = Activation_Data[16];
        end
        #10 begin
            Weight = Weight_Data[17];
            Activation = Activation_Data[17];
        end
        #10 begin
            Weight = Weight_Data[18];
            Activation = Activation_Data[18];
        end
        #10 begin
            Weight = Weight_Data[19];
            Activation = Activation_Data[19];
        end
        #10 begin
            Weight = Weight_Data[20];
            Activation = Activation_Data[20];
        end
        #10 begin
            Weight = Weight_Data[21];
            Activation = Activation_Data[21];
        end
        #10 begin
            Weight = Weight_Data[22];
            Activation = Activation_Data[22];
        end
        #10 begin
            Weight = Weight_Data[23];
            Activation = Activation_Data[23];
        end
        #10 begin
            Weight = Weight_Data[24];
            Activation = Activation_Data[24];
        end
        #10 begin
            Weight = Weight_Data[25];
            Activation = Activation_Data[25];
        end
        #10 begin
            Weight = Weight_Data[26];
            Activation = Activation_Data[26];
        end
        #10 begin
            Weight = Weight_Data[27];
            Activation = Activation_Data[27];
        end
        #10 begin
            Weight = Weight_Data[28];
            Activation = Activation_Data[28];
        end
        #10 begin
            Weight = Weight_Data[29];
            Activation = Activation_Data[29];
        end
        #10 begin
            Weight = Weight_Data[30];
            Activation = Activation_Data[30];
        end
        #10 begin
            Weight = Weight_Data[31];
            Activation = Activation_Data[31];
        end
        #10 begin
            Weight = Weight_Data[32];
            Activation = Activation_Data[32];
        end
        #10 begin
            Weight = Weight_Data[33];
            Activation = Activation_Data[33];
        end
        #10 begin
            Weight = Weight_Data[34];
            Activation = Activation_Data[34];
        end
        #10 begin
            Weight = Weight_Data[35];
            Activation = Activation_Data[35];
        end
        #10 begin
            Weight = Weight_Data[36];
            Activation = Activation_Data[36];
        end
        #10 begin
            Weight = Weight_Data[37];
            Activation = Activation_Data[37];
        end
        #10 begin
            Weight = Weight_Data[38];
            Activation = Activation_Data[38];
        end
        #10 begin
            Weight = Weight_Data[39];
            Activation = Activation_Data[39];
        end
        #10 begin
            Weight = Weight_Data[40];
            Activation = Activation_Data[40];
        end
        #10 begin
            Weight = Weight_Data[41];
            Activation = Activation_Data[41];
        end
        #10 begin
            Weight = Weight_Data[42];
            Activation = Activation_Data[42];
        end
        #10 begin
            Weight = Weight_Data[43];
            Activation = Activation_Data[43];
        end
        #10 begin
            Weight = Weight_Data[44];
            Activation = Activation_Data[44];
        end
        #10 begin
            Weight = Weight_Data[45];
            Activation = Activation_Data[45];
        end
        #10 begin
            Weight = Weight_Data[46];
            Activation = Activation_Data[46];
        end
        #10 begin
            Weight = Weight_Data[47];
            Activation = Activation_Data[47];
        end
        #10 begin
            Weight = Weight_Data[48];
            Activation = Activation_Data[48];
        end
        #10 begin
            Weight = Weight_Data[49];
            Activation = Activation_Data[49];
        end
        #10 begin
            Weight = Weight_Data[50];
            Activation = Activation_Data[50];
        end
        #10 begin
            Weight = Weight_Data[51];
            Activation = Activation_Data[51];
        end
        #10 begin
            Weight = Weight_Data[52];
            Activation = Activation_Data[52];
        end
        #10 begin
            Weight = Weight_Data[53];
            Activation = Activation_Data[53];
        end
        #10 begin
            Weight = Weight_Data[54];
            Activation = Activation_Data[54];
        end
        #10 begin
            Weight = Weight_Data[55];
            Activation = Activation_Data[55];
        end
        #10 begin
            Weight = Weight_Data[56];
            Activation = Activation_Data[56];
        end
        #10 begin
            Weight = Weight_Data[57];
            Activation = Activation_Data[57];
        end
        #10 begin
            Weight = Weight_Data[58];
            Activation = Activation_Data[58];
        end
        #10 begin
            Weight = Weight_Data[59];
            Activation = Activation_Data[59];
        end
        #10 begin
            Weight = Weight_Data[60];
            Activation = Activation_Data[60];
        end
        #10 begin
            Weight = Weight_Data[61];
            Activation = Activation_Data[61];
        end
        #10 begin
            Weight = Weight_Data[62];
            Activation = Activation_Data[62];
        end
        #10 begin
            Weight = Weight_Data[63];
            Activation = Activation_Data[63];
        end
        #15 begin 
            load_mem_done = 1; 
        end
        #330 begin
            $display("==================================================================================");
            $display("Compensation Accumulator Output for Col 0 : %d", P0.Compensation_Acc_Sum_out[0]);
            $display("Compensation Accumulator Output for Col 1 : %d", P0.Compensation_Acc_Sum_out[1]);
            $display("Compensation Accumulator Output for Col 2 : %d", P0.Compensation_Acc_Sum_out[2]);
            $display("Compensation Accumulator Output for Col 3 : %d", P0.Compensation_Acc_Sum_out[3]);
            $display("Compensation Accumulator Output for Col 4 : %d", P0.Compensation_Acc_Sum_out[4]);
            $display("Compensation Accumulator Output for Col 5 : %d", P0.Compensation_Acc_Sum_out[5]);
            $display("Compensation Accumulator Output for Col 6 : %d", P0.Compensation_Acc_Sum_out[6]);
            $display("Compensation Accumulator Output for Col 7 : %d", P0.Compensation_Acc_Sum_out[7]);
            $display("==================================================================================");
            $display("Systolic Array Output for Col 0 : %d", P0.Partial_Sum_out[0]);
            $display("Systolic Array Output for Col 1 : %d", P0.Partial_Sum_out[1]);
            $display("Systolic Array Output for Col 2 : %d", P0.Partial_Sum_out[2]);
            $display("Systolic Array Output for Col 3 : %d", P0.Partial_Sum_out[3]);
            $display("Systolic Array Output for Col 4 : %d", P0.Partial_Sum_out[4]);
            $display("Systolic Array Output for Col 5 : %d", P0.Partial_Sum_out[5]);
            $display("Systolic Array Output for Col 6 : %d", P0.Partial_Sum_out[6]);
            $display("Systolic Array Output for Col 7 : %d", P0.Partial_Sum_out[7]);
            $display("==================================================================================");
            $display("Final Partial Sum for Col 0 : %d", P0.Final_Partial_Sum[0]);
            $display("Final Partial Sum for Col 1 : %d", P0.Final_Partial_Sum[1]);
            $display("Final Partial Sum for Col 2 : %d", P0.Final_Partial_Sum[2]);
            $display("Final Partial Sum for Col 3 : %d", P0.Final_Partial_Sum[3]);
            $display("Final Partial Sum for Col 4 : %d", P0.Final_Partial_Sum[4]);
            $display("Final Partial Sum for Col 5 : %d", P0.Final_Partial_Sum[5]);
            $display("Final Partial Sum for Col 6 : %d", P0.Final_Partial_Sum[6]);
            $display("Final Partial Sum for Col 7 : %d", P0.Final_Partial_Sum[7]);
        end
        #20
        // Write memory contents to files
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

            // Activation_Mem.out
            activation_file = $fopen("Activation_Mem.out", "w");
            if (activation_file) begin
                $fdisplay(activation_file, "// Activation Memory Contents with Index");
                $fdisplay(activation_file, "// Format: [Index] Data");
                for (i = 0; i < 64; i = i + 1) begin
                    $fdisplay(activation_file, "%07b  //[%02d]", P0.u1.Activation_Mem[i], i);
                end
                $fclose(activation_file);
                $display("Activation Memory written to Activation_Mem.out");
            end
            else begin
                $display("Error: Cannot open Activation_Mem.out for writing");
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
                    $fdisplay(comp2_file, "%d  //[%02d]", P0.u1.Compensation_Row_Reg[i], i);
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
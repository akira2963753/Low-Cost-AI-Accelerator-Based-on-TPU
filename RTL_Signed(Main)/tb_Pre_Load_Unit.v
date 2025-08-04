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
    reg Mem_Write;


    reg [7:0] Weight_Data [0:63];
    reg [6:0] Activation_Data [0:63];
    integer i,j;
    integer weight_file,activation_file,comp_file,comp2_file,output_file;

    Pre_Load_Unit  P0(clk,rst,Weight,Weight_Mem_Address_in,Activation,
    Activation_Mem_Address_in,Mem_Write);

    initial begin
        clk = 0;
        rst = 0;
        Weight = 0;
        Weight_Mem_Address_in = 0;
        Mem_Write = 1;
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
        #10 begin
            Weight = 0;
        end
        #5 begin 
            Mem_Write = 0; 
        end
        #420 begin
            $display("==================================================================================");
            $display("Accumulator Output for Col 0 [0] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 0 [1] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 0 [2] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 0 [3] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 0 [4] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 0 [5] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 0 [6] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 0 [7] : %d", P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 1 [0] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 1 [1] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 1 [2] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 1 [3] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 1 [4] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 1 [5] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 1 [6] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 1 [7] : %d", P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 2 [0] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 2 [1] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 2 [2] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 2 [3] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 2 [4] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 2 [5] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 2 [6] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 2 [7] : %d", P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 3 [0] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 3 [1] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 3 [2] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 3 [3] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 3 [4] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 3 [5] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 3 [6] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 3 [7] : %d", P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 4 [0] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 4 [1] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 4 [2] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 4 [3] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 4 [4] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 4 [5] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 4 [6] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 4 [7] : %d", P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 5 [0] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 5 [1] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 5 [2] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 5 [3] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 5 [4] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 5 [5] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 5 [6] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 5 [7] : %d", P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[7]);
            $display("==================================================================================");
            $display("Accumulator Output for Col 6 [0] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 6 [1] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 6 [2] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 6 [3] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 6 [4] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 6 [5] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 6 [6] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 6 [7] : %d", P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[7]);            
            $display("==================================================================================");
            $display("Accumulator Output for Col 7 [0] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[0]);
            $display("Accumulator Output for Col 7 [1] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[1]);
            $display("Accumulator Output for Col 7 [2] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[2]);
            $display("Accumulator Output for Col 7 [3] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[3]);
            $display("Accumulator Output for Col 7 [4] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[4]);
            $display("Accumulator Output for Col 7 [5] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[5]);
            $display("Accumulator Output for Col 7 [6] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[6]);
            $display("Accumulator Output for Col 7 [7] : %d", P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[7]);     
            $display("==================================================================================");
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
                    $fdisplay(weight_file,"%d", P0.Weight_Memory_Unit.Weight_Mem[i]);
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
                    $fdisplay(activation_file, "%d  // [%02d]", P0.Activation_Memory_Unit.Activation_Mem[i],i);
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
                    $fdisplay(comp_file, "%d  // [%02d]", P0.Compensation_Memory_Unit.Compensation_Mem[i],i);
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
                    $fdisplay(comp2_file, "%d   // [%02d]", P0.Input_Buffer_Unit.Compensation_Row_Reg[i],i);
                end
                $fclose(comp2_file);
                $display("Compensation Row written to Compensation_Mem.out");
            end
            else begin
                $display("Error: Cannot open Compensation_Row.out for writing");
            end
            // output.out
            output_file = $fopen("Output.out", "w"); 
            if (output_file) begin
                $fdisplay(output_file, "// Output Contents with Index");
                $fdisplay(output_file, "// Format: [Index] Data");
                $fdisplay(output_file,"===================================Col[0]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[0].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[1]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[1].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[2]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[2].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[3]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[3].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[4]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[4].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[5]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[5].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[6]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[6].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"===================================Col[7]=========================================");
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[0],0);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[1],1);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[2],2);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[3],3);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[4],4);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[5],5);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[6],6);
                $fdisplay(output_file,"%d  //  [%01d] ",P0.Output_gen[7].Accumulator_Unit.Partial_Sum_Mem[7],7);

                $fdisplay(output_file,"=========================== After Relu Function =================================");
                $fdisplay(output_file,"========================= Save into Unified Buffer ==============================");
                for(i=0;i<64;i=i+1) begin
                    $fdisplay(output_file,"%d  //  [%02d]", P0.Unified_Buffer[i],i);
                end
                $fclose(output_file);  
                $display("Output written to Output.out");
            end
            else begin
                $display("Error: Cannot open Output.out for writing");
            end


            $finish;           
        end
    end

endmodule
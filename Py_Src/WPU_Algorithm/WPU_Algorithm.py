class WeightProcessor:
    def __init__(self): 
        self.weight_mem = [[0]*8 for _ in range(8)] 
        self.compensation_mem = [0] * 24
        self.compensation_row = [8] * 24
        
        self.weight_count = 0
        self.current_group = 0 
    
    def is_consecutive_msb4(self, weight_8bit):
        msb4 = (weight_8bit >> 4) & 0xF
        return msb4 == 0x0 or msb4 == 0xF
    
    def msr4_compress(self, weight_8bit):
        if self.is_consecutive_msb4(weight_8bit):
            # MSR-4
            low_4bits = weight_8bit & 0xF 
            sign_bit = (weight_8bit >> 7) & 0x1 
            reduced_weight_3bits = (low_4bits >> 1) & 0x7  
            reduced_weight = (sign_bit << 3) | reduced_weight_3bits 
            output = (0 << 4) | reduced_weight 
            return output, None, None
        else:
            # Non-MSR-4
            msb4 = (weight_8bit >> 4) & 0xF  
            reduced_weight = (1 << 4) | msb4 
            
            # Compensation
            sign_bit = (weight_8bit >> 7) & 0x1 
            compensation_3bits = (msb4 >> 1) & 0x7 
            compensation = (sign_bit << 3) | compensation_3bits
            
            return reduced_weight, compensation, self.weight_count % 8
    
    def process_weight(self, weight_8bit):
        if self.weight_count > 0 and self.weight_count % 8 == 0:
            self.current_group += 1
        
        row = self.weight_count // 8
        col = self.weight_count % 8
        
        reduced_weight, compensation, comp_row = self.msr4_compress(weight_8bit)

        self.weight_mem[row][col] = reduced_weight
        
        if compensation is not None:
            group_offset = self.current_group * 3
            
            write_position = 0
            for j in range(3):
                if group_offset + j < 24 and self.compensation_mem[group_offset + j] == 0:
                    write_position = j
                    break
            
            final_write_addr = group_offset + write_position
            
            if final_write_addr < 24:
                self.compensation_mem[final_write_addr] = compensation
                self.compensation_row[final_write_addr] = comp_row
        
        self.weight_count += 1
        
# ====================================================== read and save file ======================================================
    def read_weight_file(self, filename):
        weights = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith('//'):
                        if len(line) == 8 and all(c in '01' for c in line):
                            weights.append(int(line, 2))
        except FileNotFoundError:
            print(f"file {filename} is not exist")
            return []
        return weights
    
    def process_weight_file(self, filename):
        weights = self.read_weight_file(filename)
        for i in range(len(weights)):
            self.process_weight(weights[i])
    
    def save_to_files(self, weight_file="Weight_Mem.out", comp_mem_file="Compensation_Mem.out", comp_row_file="Compensation_Row.out"):
        with open(weight_file, 'w') as f:
            f.write("// Weight Memory Output - 8x8x5bit\n")
            f.write("// Format: 5-bit binary values\n")
            for i in range(8):
                for j in range(8):
                    f.write(f"{self.weight_mem[i][j]:05b}\n")
        
        with open(comp_mem_file, 'w') as f:
            f.write("// Compensation Memory Output - 24x4bit\n")
            f.write("// Format: 4-bit binary values\n")
            for i in range(24):
                f.write(f"{self.compensation_mem[i]:04b}\n")
        
        with open(comp_row_file, 'w') as f:
            f.write("// Compensation Row Output - 24 values\n")
            f.write("// Format: decimal values (8 = no compensation)\n")
            for i in range(24):
                f.write(f"{self.compensation_row[i]}\n")
        
        print(f"\n save the file:")
        print(f"  - {weight_file}")
        print(f"  - {comp_mem_file}")
        print(f"  - {comp_row_file}")

def test_with_file():
    processor = WeightProcessor()
    processor.process_weight_file('Weight_Data.dat')
    processor.save_to_files()

if __name__ == "__main__":
    test_with_file()
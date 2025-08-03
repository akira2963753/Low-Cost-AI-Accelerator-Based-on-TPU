import numpy as np

def read_weight_data(filename):
    """讀取權重資料檔案"""
    weights = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):  # 跳過註釋和空行
                weights.append(line)
    return weights

def read_activation_data(filename):
    """讀取激活值資料檔案"""
    activations = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):  # 跳過註釋和空行
                activations.append(line)
    return activations

def preprocess_weights_unsigned(weight_list):
    """預處理權重：LSB設為1，轉換為無號數"""
    processed_weights = []
    for weight_str in weight_list:
        # LSB設為1
        weight_bin = weight_str[:-1] + '1'  # 前7位保持不變，最後1位設為1
        
        # 轉換為無號數 (8位元)
        weight_int = int(weight_bin, 2)  # 直接轉換，範圍 0-255
        
        processed_weights.append(weight_int)
    
    return processed_weights

def preprocess_activations_unsigned(activation_list):
    """預處理激活值：LSB補1位元，轉換為無號數"""
    processed_activations = []
    for activation_str in activation_list:
        # LSB補1位元 (補1)
        activation_bin = activation_str + '1'  # 7位原始值 + 1位補1
        
        # 轉換為無號數 (8位元)
        activation_int = int(activation_bin, 2)  # 直接轉換，範圍 0-255
        
        processed_activations.append(activation_int)
    
    return processed_activations

def systolic_array_computation_unsigned(weights, activations):
    """執行8x8脈動陣列計算 (無號數版本)"""
    results = []
    
    # 8個輸出欄位
    for col in range(8):
        col_results = []
        print(f"\n{'='*60}")
        print(f"計算 Col-{col} (使用權重 W{col*8}-W{col*8+7}) - 無號數計算")
        print(f"{'='*60}")
        
        # 每個欄位8個輸出
        for output_idx in range(8):
            total = 0
            print(f"\nCol-{col} Output-{output_idx} 計算:")
            
            # 計算該輸出的加權和
            for row in range(8):
                weight_idx = col * 8 + row  # 權重索引: Col-0用W0-W7, Col-1用W8-W15...
                activation_idx = row * 8 + output_idx  # 激活值索引: 跳8個取值模式
                
                product = weights[weight_idx] * activations[activation_idx]
                total += product
                
                print(f"  W{weight_idx:2d}({weights[weight_idx]:3d}) × A{activation_idx:2d}({activations[activation_idx]:3d}) = {product:6d}")
            
            col_results.append(total)
            print(f"  → Col-{col} Output-{output_idx} 總和: {total}")
            print("-" * 50)
        
        results.append(col_results)
    
    return results

def main():
    print("8x8 Systolic Array Simulator - 無號數版本")
    print("=" * 70)
    
    # 讀取資料檔案
    print("讀取資料檔案...")
    try:
        weight_raw = read_weight_data('Weight_Data.dat')
        activation_raw = read_activation_data('Activation_Data.dat')
        
        print(f"讀取到 {len(weight_raw)} 個權重值")
        print(f"讀取到 {len(activation_raw)} 個激活值")
        print()
        
        # 預處理資料
        print("預處理資料 (無號數版本)...")
        weights = preprocess_weights_unsigned(weight_raw)
        activations = preprocess_activations_unsigned(activation_raw)
        
        print("權重預處理結果 (全部64個) - 無號數:")
        for i in range(len(weight_raw)):
            print(f"W{i:2d}: {weight_raw[i]} -> LSB設1 -> {weights[i]:3d} (無號數, 範圍0-255)")
        
        print("\n激活值預處理結果 (全部64個) - 無號數:")
        for i in range(len(activation_raw)):
            print(f"A{i:2d}: {activation_raw[i]} -> LSB補1 -> {activations[i]:3d} (無號數, 範圍0-255)")
        
        print("\n" + "=" * 70)
        print("開始脈動陣列計算 (無號數運算)...")
        print("=" * 70)
        
        # 執行脈動陣列計算
        results = systolic_array_computation_unsigned(weights, activations)
        
        # 顯示最終結果
        print("\n" + "=" * 80)
        print("完整計算結果 (無號數版本):")
        print("=" * 80)
        
        for col in range(8):
            print(f"\nCol-{col} 所有輸出 (無號數結果):")
            for output_idx in range(8):
                print(f"  Col-{col} Output-{output_idx}: {results[col][output_idx]:7d}")
        
        # 建立結果矩陣顯示
        print("\n" + "=" * 80)
        print("結果矩陣 (8行x8列) - 無號數版本:")
        print("行=輸出索引, 列=Col索引")
        print("=" * 80)
        print("      ", end="")
        for col in range(8):
            print(f"Col-{col:1d}".rjust(10), end="")
        print()
        print("      " + "-" * 80)
        
        for row in range(8):
            print(f"Out-{row}: ", end="")
            for col in range(8):
                print(f"{results[col][row]:9d}", end=" ")
            print()
        
        # 顯示數值範圍資訊
        print("\n" + "=" * 80)
        print("數值範圍資訊:")
        print("=" * 80)
        print(f"權重範圍: {min(weights)} - {max(weights)} (無號數 0-255)")
        print(f"激活值範圍: {min(activations)} - {max(activations)} (無號數 0-255)")
        
        all_results = [result for col_results in results for result in col_results]
        print(f"輸出結果範圍: {min(all_results)} - {max(all_results)}")
        print(f"最大可能乘積: {max(weights)} × {max(activations)} = {max(weights) * max(activations)}")
        print(f"最大可能輸出: 8 × {max(weights) * max(activations)} = {8 * max(weights) * max(activations)}")
        
    except FileNotFoundError as e:
        print(f"檔案不存在: {e}")
        print("請確保 Weight_Data.dat 和 Activation_Data.dat 在同一目錄下")
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
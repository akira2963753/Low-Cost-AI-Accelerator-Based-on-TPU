#!/usr/bin/env python3
"""
Top-Level Processing Element Area Calculator
計算 Vivado Area Report 中 RPE 和 CPE 頂層模組的面積總和
新增 Total cell area 占比計算和 Gate count 計算
"""
import re
import sys

def calculate_top_level_areas(filename):
    """
    從 area report 文件中計算頂層 RPE 和 CPE 的面積總和，並找出 Total cell area
    
    Args:
        filename: area report 文件路徑
        
    Returns:
        dict: 包含 RPE、CPE 的計算結果和 Total cell area
    """
    # RPE 數據
    rpe_areas = []
    rpe_modules = []
    rpe_total = 0.0
    
    # CPE 數據
    cpe_areas = []
    cpe_modules = []
    cpe_total = 0.0
    
    # Total cell area
    total_cell_area = 0.0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # 檢查 Total cell area
                if 'Total cell area:' in line_stripped:
                    match = re.search(r'Total cell area:\s*(\d+\.\d+)', line_stripped)
                    if match:
                        total_cell_area = float(match[1])
                        print(f"找到 Total cell area: {total_cell_area}")
                
                # 檢查 RPE 模組
                if (line_stripped.startswith('Row_gen[') and 
                    'Reduced_Processing_Element' in line_stripped and 
                    '/' not in line_stripped):
                    
                    module_name = line_stripped
                    rpe_modules.append(module_name)
                    
                    # 檢查下一行數值
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        match = re.search(r'(\d+\.\d+)', next_line)
                        if match:
                            value = float(match[1])
                            rpe_areas.append(value)
                            rpe_total += value
                            print(f"RPE 找到: {module_name} = {value}")
                
                # 檢查 CPE 模組 (u0, u1, u2)
                elif (line_stripped.startswith('Compensation_Array[') and 
                      'Compensation_Processing_Element_u' in line_stripped and 
                      ('_u0' in line_stripped or '_u1' in line_stripped or '_u2' in line_stripped) and
                      '/' not in line_stripped):
                    
                    module_name = line_stripped
                    cpe_modules.append(module_name)
                    
                    # 檢查下一行數值
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        match = re.search(r'(\d+\.\d+)', next_line)
                        if match:
                            value = float(match[1])
                            cpe_areas.append(value)
                            cpe_total += value
                            print(f"CPE 找到: {module_name} = {value}")
    
    except FileNotFoundError:
        print(f"錯誤: 找不到文件 '{filename}'")
        return None
    except Exception as e:
        print(f"錯誤: 讀取文件時發生問題 - {e}")
        return None
    
    return {
        'rpe': {
            'areas': rpe_areas,
            'modules': rpe_modules,
            'total': rpe_total,
            'count': len(rpe_areas)
        },
        'cpe': {
            'areas': cpe_areas,
            'modules': cpe_modules,
            'total': cpe_total,
            'count': len(cpe_areas)
        },
        'total_cell_area': total_cell_area
    }

def calculate_gate_count(area, gate_area=2.8):
    """
    計算 Gate count
    Gate count = AREA / 2.8
    
    Args:
        area: 面積數值
        gate_area: 每個 gate 的面積，預設 2.8
        
    Returns:
        int: Gate count
    """
    return int(area / gate_area)

def print_results(data, module_type, full_name, total_cell_area=None):
    """印出計算結果，包含 Gate count 和正確的占比"""
    if data['count'] > 0:
        print(f"\n=== {full_name} ({module_type}) 計算結果 ===")
        print(f"找到的 {module_type} 模組數量: {data['count']}")
        print(f"{module_type} 面積總和: {data['total']:.4f}")
        print(f"平均面積: {data['total']/data['count']:.4f}")
        
        # 計算 Gate count
        gate_count = calculate_gate_count(data['total'])
        print(f"{module_type} Gate count: {gate_count:,} gates (面積 {data['total']:.4f} / 2.8)")
        
        # 如果有 Total cell area，計算正確的占比
        if total_cell_area and total_cell_area > 0:
            percentage = (data['total'] / total_cell_area) * 100
            print(f"{module_type} 占 Total cell area 比例: {percentage:.2f}% ({data['total']:.4f} / {total_cell_area:.4f})")
        
        print(f"\n=== {module_type} 統計信息 ===")
        print(f"最大 {module_type} 面積: {max(data['areas']):.4f}")
        print(f"最小 {module_type} 面積: {min(data['areas']):.4f}")
        
        # 如果是 CPE，按照 u0, u1, u2 分組顯示
        if module_type == 'CPE':
            print(f"\n=== {module_type} 按子模組分組 ===")
            u0_modules = [(name, value) for name, value in zip(data['modules'], data['areas']) if '_u0' in name]
            u1_modules = [(name, value) for name, value in zip(data['modules'], data['areas']) if '_u1' in name]
            u2_modules = [(name, value) for name, value in zip(data['modules'], data['areas']) if '_u2' in name]
            
            for suffix, modules in [('u0', u0_modules), ('u1', u1_modules), ('u2', u2_modules)]:
                if modules:
                    total_area = sum(value for _, value in modules)
                    gate_count = calculate_gate_count(total_area)
                    print(f"\n{suffix} 子模組 ({len(modules)} 個):")
                    print(f"  總面積: {total_area:.4f}, Gate count: {gate_count:,}")
                    for i, (name, value) in enumerate(modules):
                        individual_gate_count = calculate_gate_count(value)
                        print(f"  {i+1}. {name}: {value:.4f} ({individual_gate_count:,} gates)")
        
        print(f"\n=== 所有 {module_type} 面積數值 ===")
        for i, (name, value) in enumerate(zip(data['modules'], data['areas'])):
            individual_gate_count = calculate_gate_count(value)
            print(f"{i+1:2d}. {name}: {value:.4f} ({individual_gate_count:,} gates)")
        
        print(f"\n=== {module_type} 加法算式 ===")
        formula = " + ".join(f"{v:.4f}" for v in data['areas'])
        print(f"{formula} = {data['total']:.4f}")
    else:
        print(f"\n=== {full_name} ({module_type}) 計算結果 ===")
        print(f"未找到任何 {module_type} 模組")

def main():
    """主函數"""
    # 檢查命令行參數
    if len(sys.argv) != 2:
        print("使用方法: python rpe_cpe_area_calculator.py <area_report_file>")
        print("範例: python rpe_cpe_area_calculator.py area.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"正在處理文件: {filename}")
    print("尋找頂層 RPE 和 CPE 模組以及 Total cell area...")
    print("CPE 模組包括: u0, u1, u2 (每個 n 從 0~7)")
    print("=" * 80)
    
    # 計算頂層面積
    results = calculate_top_level_areas(filename)
    
    if results is not None:
        print("=" * 80)
        
        total_cell_area = results['total_cell_area']
        
        # 顯示 RPE 結果
        print_results(results['rpe'], 'RPE', 'Reduced Processing Element', total_cell_area)
        
        # 顯示 CPE 結果
        print_results(results['cpe'], 'CPE', 'Compensation Processing Element', total_cell_area)
        
        # 顯示總結
        total_modules = results['rpe']['count'] + results['cpe']['count']
        total_area = results['rpe']['total'] + results['cpe']['total']
        total_gate_count = calculate_gate_count(total_area)
        
        print(f"\n{'='*80}")
        print(f"=== 總結 ===")
        
        if total_cell_area > 0:
            print(f"Total cell area: {total_cell_area:.4f}")
            total_cell_gate_count = calculate_gate_count(total_cell_area)
            print(f"Total cell area Gate count: {total_cell_gate_count:,} gates")
            print()
        
        print(f"RPE 模組數量: {results['rpe']['count']}, 總面積: {results['rpe']['total']:.4f}")
        if results['rpe']['count'] > 0:
            rpe_gate_count = calculate_gate_count(results['rpe']['total'])
            print(f"RPE Gate count: {rpe_gate_count:,} gates")
        
        print(f"CPE 模組數量: {results['cpe']['count']}, 總面積: {results['cpe']['total']:.4f}")
        if results['cpe']['count'] > 0:
            cpe_gate_count = calculate_gate_count(results['cpe']['total'])
            print(f"CPE Gate count: {cpe_gate_count:,} gates")
        
        print(f"總模組數量: {total_modules}")
        print(f"總面積 (RPE + CPE): {total_area:.4f}")
        print(f"總 Gate count (RPE + CPE): {total_gate_count:,} gates")
        
        # 計算相對於 Total cell area 的正確占比
        if total_cell_area > 0:
            print(f"\n=== 相對於 Total cell area 的占比 ===")
            if results['rpe']['count'] > 0:
                rpe_percentage = (results['rpe']['total'] / total_cell_area) * 100
                print(f"RPE 占 Total cell area: {rpe_percentage:.2f}%")
            if results['cpe']['count'] > 0:
                cpe_percentage = (results['cpe']['total'] / total_cell_area) * 100
                print(f"CPE 占 Total cell area: {cpe_percentage:.2f}%")
            
            combined_percentage = (total_area / total_cell_area) * 100
            print(f"RPE + CPE 合計占 Total cell area: {combined_percentage:.2f}%")
        else:
            print(f"\n警告: 未找到 Total cell area 數據，無法計算正確占比")
            if results['rpe']['count'] > 0 and results['cpe']['count'] > 0:
                print(f"RPE 相對占比 (僅 RPE+CPE): {results['rpe']['total']/total_area*100:.2f}%")
                print(f"CPE 相對占比 (僅 RPE+CPE): {results['cpe']['total']/total_area*100:.2f}%")
        
        print(f"\n計算完成！")
    else:
        print("計算失敗！")
        sys.exit(1)

if __name__ == "__main__":
    main()
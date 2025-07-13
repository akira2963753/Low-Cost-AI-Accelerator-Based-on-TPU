import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

np.random.seed(42)                  
torch.manual_seed(42) 

# Define the same LeNet model architecture as original
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_size = 16 * 5 * 5
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 改进版量化 LeNet - 更保守的量化策略
class ImprovedQuantizedLeNet(nn.Module):
    def __init__(self, num_classes=10, quantize_activations=True):
        super(ImprovedQuantizedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_size = 16 * 5 * 5
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.quantize_activations = quantize_activations
        
    def forward(self, x):
        # 卷积层：只在池化后量化，减少量化次数
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        if self.quantize_activations:
            x = improved_quantize_activation(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        if self.quantize_activations:
            x = improved_quantize_activation(x)
        
        x = x.view(x.size(0), -1)
        
        # 全连接层：更谨慎的量化
        x = F.relu(self.fc1(x))
        if self.quantize_activations:
            x = improved_quantize_activation(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if self.quantize_activations:
            x = improved_quantize_activation(x)
        x = self.dropout(x)
        
        # 最后一层不量化
        x = self.fc3(x)
        return x

# ==================== 改进的量化函数 ====================

def analyze_tensor_range(tensor, tensor_name):
    """
    分析张量的数值范围，用于确定最佳量化参数
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()
    stats = {
        'name': tensor_name,
        'min': np.min(tensor_np),
        'max': np.max(tensor_np),
        'mean': np.mean(tensor_np),
        'std': np.std(tensor_np),
        'q1': np.percentile(tensor_np, 25),
        'median': np.percentile(tensor_np, 50),
        'q3': np.percentile(tensor_np, 75),
        'q95': np.percentile(tensor_np, 95),
        'q99': np.percentile(tensor_np, 99)
    }
    return stats

def determine_optimal_scale(tensor, target_bits=8):
    """
    根据张量分布确定最优缩放因子
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()
    
    # 使用 95% 分位数来确定范围，避免极值影响
    abs_max = max(abs(np.percentile(tensor_np, 2.5)), abs(np.percentile(tensor_np, 97.5)))
    
    # 为8位有符号整数计算缩放因子
    max_int = 2**(target_bits-1) - 1  # 127 for 8-bit
    
    if abs_max > 0:
        scale = max_int / abs_max
    else:
        scale = 128.0  # 默认 Q1.7 缩放
    
    return scale

def improved_float_to_fixed_point(weight, scale=128.0, bits=8):
    """
    改进的浮点到定点转换，使用动态缩放
    """
    # 使用提供的缩放因子
    scaled_weight = weight * scale
    
    # 舍入到最近整数
    rounded_weight = math.floor(scaled_weight)
    
    # 限制到指定位数的有符号范围
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    clamped_weight = max(min_val, min(max_val, rounded_weight))
    
    # 转换为二进制字符串
    if clamped_weight >= 0:
        binary_str = format(clamped_weight, f'0{bits}b')
    else:
        unsigned_val = (1 << bits) + clamped_weight
        binary_str = format(unsigned_val, f'0{bits}b')
    
    return binary_str

def improved_binary_to_float(binary_str, scale=128.0, bits=8):
    """
    改进的二进制到浮点转换
    """
    int_val = int(binary_str, 2)
    
    # 处理负数（二进制补码）
    if int_val >= 2**(bits-1):
        int_val = int_val - 2**bits
    
    # 转换回浮点数
    float_val = int_val / scale
    return float_val

def conservative_msr4_compensation(binary_str):
    """
    保守的 MSR-4 补偿策略
    """
    if has_msr4(binary_str):
        # MSR-4: 只设置最后一位为1
        compensated = binary_str[:-1] + '1'
    else:
        # Non-MSR-4: 保持原样或轻微调整
        #compensated = binary_str
        compensated = binary_str[:-1] + '1'
        #compensated = binary_str[:-4] + '1000'
    return compensated

def improved_quantize_activation(activation_tensor):
    """
    改进的激活值量化，使用动态缩放和更保守的补偿
    """
    # 分析激活值范围
    scale = determine_optimal_scale(activation_tensor)
    
    # 转换为numpy处理
    activations_np = activation_tensor.detach().cpu().numpy()
    original_shape = activations_np.shape
    activations_flat = activations_np.flatten()
    
    quantized_activations = []
    
    for activation in activations_flat:
        # 使用动态缩放进行量化
        binary_str = improved_float_to_fixed_point(activation, scale)
        
        # 保守的补偿策略
        compensated_binary = conservative_msr4_compensation(binary_str[:-4] + '1000')
        
        # 转换回浮点
        quantized_float = improved_binary_to_float(compensated_binary, scale)
        quantized_activations.append(quantized_float)
    
    # 转换回张量
    quantized_tensor = torch.tensor(quantized_activations, dtype=torch.float32).reshape(original_shape)
    return quantized_tensor.to(activation_tensor.device)

def improved_quantize_weight_tensor(weight_tensor, layer_name):
    """
    改进的权重量化，使用动态缩放
    """
    # 分析权重范围
    stats = analyze_tensor_range(weight_tensor, layer_name)
    print(f"  {layer_name} range: [{stats['min']:.6f}, {stats['max']:.6f}], "
          f"mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
    
    # 确定最优缩放因子
    scale = determine_optimal_scale(weight_tensor)
    print(f"  Using scale factor: {scale:.2f}")
    
    original_shape = weight_tensor.shape
    weights_flat = weight_tensor.flatten()
    
    quantized_weights = []
    msr4_count = 0
    non_msr4_count = 0
    
    for i, weight in enumerate(weights_flat):
        # 使用动态缩放进行量化
        binary_str = improved_float_to_fixed_point(weight.item(), scale)
        
        # 检查 MSR-4
        if has_msr4(binary_str):
            msr4_count += 1
        else:
            non_msr4_count += 1
        
        # 保守的补偿策略
        compensated_binary = conservative_msr4_compensation(binary_str)
        
        # 转换回浮点
        quantized_float = improved_binary_to_float(compensated_binary, scale)
        quantized_weights.append(quantized_float)
    
    # 转换回张量
    quantized_tensor = torch.tensor(quantized_weights, dtype=torch.float32).reshape(original_shape)
    
    stats_dict = {
        'layer_name': layer_name,
        'total_weights': len(weights_flat),
        'msr4_count': msr4_count,
        'non_msr4_count': non_msr4_count,
        'msr4_percentage': (msr4_count / len(weights_flat)) * 100,
        'non_msr4_percentage': (non_msr4_count / len(weights_flat)) * 100,
        'scale_factor': scale,
        'original_range': f"[{stats['min']:.6f}, {stats['max']:.6f}]"
    }
    
    return quantized_tensor, stats_dict

# 复用原有的辅助函数
def has_msr4(binary_str):
    """检查是否有 MSR-4"""
    if len(binary_str) != 8:
        return False
    first_four = binary_str[:4]
    return first_four == '0000' or first_four == '1111'

# ==================== 多种量化策略 ====================

def create_quantized_model_strategy(original_model, strategy="conservative"):
    """
    创建不同策略的量化模型
    """
    strategies = {
        "weights_only": {"quantize_weights": True, "quantize_activations": False},
        "conservative": {"quantize_weights": True, "quantize_activations": True},
        "aggressive": {"quantize_weights": True, "quantize_activations": True}
    }
    
    if strategy not in strategies:
        strategy = "conservative"
    
    config = strategies[strategy]
    print(f"Creating quantized model with strategy: {strategy}")
    print(f"  - Quantize weights: {config['quantize_weights']}")
    print(f"  - Quantize activations: {config['quantize_activations']}")
    
    # 创建量化模型
    quantized_model = ImprovedQuantizedLeNet(quantize_activations=config['quantize_activations']).to(device)
    
    # 量化权重
    quantization_stats = []
    original_state_dict = original_model.state_dict()
    quantized_state_dict = {}
    
    for name, param in original_state_dict.items():
        if 'weight' in name and config['quantize_weights']:
            # 量化权重
            quantized_tensor, stats = improved_quantize_weight_tensor(param, name)
            quantized_state_dict[name] = quantized_tensor
            quantization_stats.append(stats)
            print(f"  ✓ {name}: MSR-4: {stats['msr4_count']:,} ({stats['msr4_percentage']:.2f}%), "
                  f"Scale: {stats['scale_factor']:.2f}")
        else:
            # 保持原始参数
            quantized_state_dict[name] = param
            param_type = "bias" if 'bias' in name else "weight (not quantized)"
            print(f"  ✓ {name}: Kept original ({param_type})")
    
    # 加载量化权重
    quantized_model.load_state_dict(quantized_state_dict)
    
    return quantized_model, quantization_stats

# ==================== 评估函数 ====================

def evaluate_model(model, test_loader, model_name="Model"):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    print(f"Evaluating {model_name}...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100 * correct / total
                print(f"  Batch {batch_idx + 1}: Current accuracy: {current_acc:.2f}%")
    
    accuracy = 100 * correct / total
    print(f"  Final {model_name} Accuracy: {accuracy:.2f}%")
    return accuracy

def get_data_loaders(batch_size=64):
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# ==================== 结果分析 ====================

def display_improved_results(original_acc, strategies_results):
    """显示改进的结果对比"""
    print("\n" + "="*90)
    print("IMPROVED LENET Q1.7 QUANTIZATION RESULTS")
    print("="*90)
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Drop':>10} {'Retention':>12} {'Description'}")
    print("-" * 80)
    
    print(f"{'Original':<20} {original_acc:>9.2f}% {0.0:>9.2f}% {100.0:>11.2f}% {'Float32 baseline'}")
    
    for strategy, (accuracy, stats) in strategies_results.items():
        drop = original_acc - accuracy
        retention = (accuracy / original_acc) * 100
        
        descriptions = {
            "weights_only": "Weights quantized only",
            "conservative": "Weights + careful activations",
            "aggressive": "Full quantization"
        }
        
        print(f"{strategy:<20} {accuracy:>9.2f}% {drop:>9.2f}% {retention:>11.2f}% {descriptions.get(strategy, '')}")
    
    # 找出最佳策略
    best_strategy = max(strategies_results.keys(), key=lambda x: strategies_results[x][0])
    best_acc = strategies_results[best_strategy][0]
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy} (Accuracy: {best_acc:.2f}%)")
    
    # 显示量化统计
    print(f"\n📊 QUANTIZATION STATISTICS:")
    for strategy, (accuracy, stats) in strategies_results.items():
        if stats:
            total_weights = sum(s['total_weights'] for s in stats)
            total_msr4 = sum(s['msr4_count'] for s in stats)
            msr4_percentage = (total_msr4 / total_weights) * 100 if total_weights > 0 else 0
            
            print(f"{strategy}: {total_weights:,} weights, {msr4_percentage:.2f}% MSR-4")
    
    print("="*90)

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("Improved LeNet Q1.7 Quantization Analysis")
    print("="*50)
    
    # 加载原始模型
    print(f"\n📁 Loading original trained LeNet model...")
    original_model = LeNet().to(device)
    
    try:
        state_dict = torch.load('mnist_lenet_model.pth', map_location=device)
        original_model.load_state_dict(state_dict)
        print("✓ Original LeNet model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'mnist_lenet_model.pth' not found!")
        print("Please train the model first using LeNet.py")
        return
    
    # 获取测试数据
    print(f"\n📊 Loading test dataset...")
    test_loader = get_data_loaders(batch_size=1000)
    print("✓ Test dataset loaded")
    
    # 评估原始模型
    print(f"\n🔍 Evaluating original Float32 LeNet model...")
    original_accuracy = evaluate_model(original_model, test_loader, "Original Float32 LeNet")
    
    # 测试多种量化策略
    strategies = ["weights_only", "conservative", "aggressive"]
    strategies_results = {}
    
    for strategy in strategies:
        print(f"\n⚙️  Testing strategy: {strategy}")
        quantized_model, stats = create_quantized_model_strategy(original_model, strategy)
        
        print(f"\n🔍 Evaluating {strategy} quantized model...")
        accuracy = evaluate_model(quantized_model, test_loader, f"LeNet ({strategy})")
        
        strategies_results[strategy] = (accuracy, stats)
        
        # 保存模型
        torch.save(quantized_model.state_dict(), f'mnist_lenet_q17_{strategy}.pth')
        print(f"✓ Model saved as 'mnist_lenet_q17_{strategy}.pth'")
    
    # 显示结果
    display_improved_results(original_accuracy, strategies_results)
    
    print(f"\n🎉 Improved LeNet Q1.7 quantization analysis completed!")
    
    return original_model, strategies_results

if __name__ == "__main__":
    main()
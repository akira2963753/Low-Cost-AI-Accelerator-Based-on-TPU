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

def quantize_and_compensate_activation(tensor):
    """
    向量化版本：將 activation 量化為 Q1.7 並強制 LSB=1（補償），模擬 8-bit 定點行為。
    輸入: float32 tensor
    輸出: float32 tensor (模擬量化後效果)
    """
    # 轉 numpy array
    tensor_np = tensor.detach().cpu().numpy()

    # Step1: Q1.7 量化：乘以 128，再用 floor 模擬 truncation
    scaled = np.floor(tensor_np * 128).astype(np.int32)

    # Step2: 限制在 8-bit 範圍 [-128, 127]
    clamped = np.clip(scaled, -128, 127)

    # Step3: 強制 LSB=1（進行 OR 運算）
    #compensated = clamped | 0b00000001
    compensated = (clamped | 0b00001000) & 0b11111000
    # Step4: 轉回 float（模擬定點乘法輸入）
    float_output = compensated.astype(np.float32) / 128.0

    # 回傳為 Torch tensor 並保留裝置位置
    return torch.tensor(float_output, dtype=torch.float32).to(tensor.device)



# Define the Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Initial preprocessing convolutional layer (not counted in 17 layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers (16 conv layers + 1 FC = 17 layers total)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 4 conv layers
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 4 conv layers
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 4 conv layers
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4 conv layers
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 1 FC layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, quant_act=True):
        out = F.relu(self.bn1(self.conv1(x)))

        # ✅ 只在 conv1 後做 activation 補償
        if quant_act:
            out = quantize_and_compensate_activation(out)

        # 正常傳遞：不額外補償每層
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        # ✅ 再補償一次（模擬進入 FC 前）
        if quant_act:
            out = quantize_and_compensate_activation(out)

        out = self.dropout(out)
        out = self.fc(out)

        return out


# Function to create ResNet-18
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Quantization Functions
def float_to_q1_7_fixed_point(weight):
    """
    Convert a floating-point weight to Q1.7 fixed-point format (8-bit)
    Returns the 8-bit binary representation as a string
    """
    # Scale by 128 (2^7) for Q1.7 format
    scaled_weight = weight * 128
    
    # Round to nearest integer
    rounded_weight = math.floor(scaled_weight)
    
    # Clamp to 8-bit signed range [-128, 127]
    clamped_weight = max(-128, min(127, rounded_weight))
    
    # Convert to 8-bit two's complement representation
    if clamped_weight >= 0:
        # Positive number: direct binary conversion
        binary_str = format(clamped_weight, '08b')
    else:
        # Negative number: two's complement
        # Convert to unsigned 8-bit representation
        unsigned_val = (1 << 8) + clamped_weight  # Add 256 to get positive equivalent
        binary_str = format(unsigned_val, '08b')
    
    return binary_str

def has_msr4(binary_str):
    """
    Check if an 8-bit binary string has MSR-4 (4 consecutive 0s or 1s from MSB)
    Returns True if it has MSR-4, False otherwise
    """
    if len(binary_str) != 8:
        return False
    
    # Check first 4 bits from MSB
    first_four = binary_str[:4]
    
    # MSR-4 means first 4 bits are all 0s or all 1s
    return first_four == '0000' or first_four == '1111'

def apply_no_compensation(binary_str):
    """
    Apply no compensation - just return the original quantized value
    """
    return binary_str

def apply_uniform_lsb_compensation(binary_str):
    """
    Apply uniform LSB=1 compensation to all weights
    """
    return binary_str[:-1] + '1'

def apply_msr4_compensation(binary_str):
    """
    Apply MSR-4 compensation rules:
    - For MSR-4 weights: set last bit to 1
    - For Non-MSR-4 weights: set last 4 bits to 1000
    """
    if has_msr4(binary_str):
        # MSR-4: Set last bit to 1 for expectation compensation
        compensated = binary_str[:-1] + '1'
    else:
        # Non-MSR-4: Set last 4 bits to 1000
        compensated = binary_str[:4] + '1000'
        #compensated = binary_str[:-1] + '1'
    
    return compensated

def binary_to_float(binary_str):
    """
    Convert 8-bit two's complement binary string back to float
    """
    # Convert binary string to integer
    int_val = int(binary_str, 2)
    
    # Handle two's complement for negative numbers
    if int_val >= 128:  # MSB is 1, so it's negative
        int_val = int_val - 256
    
    # Convert back to float by dividing by scale factor (128 for Q1.7)
    float_val = int_val / 128.0
    
    return float_val

def get_layer_info(layer_name):
    """
    Get layer type and category for ResNet-18
    """
    if layer_name == 'conv1.weight':
        return 'Initial', 'Conv1'
    elif 'shortcut' in layer_name:
        return 'Shortcut', f"Shortcut-{layer_name.split('.')[0]}"
    elif 'fc.weight' in layer_name:
        return 'FC', 'Classifier'
    elif any(f'layer{i}' in layer_name for i in range(1, 5)):
        # Extract layer group (layer1, layer2, etc.)
        layer_group = layer_name.split('.')[0]
        return 'Block', layer_group
    else:
        return 'Other', 'Other'

def quantize_weight_tensor(weight_tensor, layer_name, compensation_fn, compensation_name):
    """
    Quantize a weight tensor using Q1.7 format with specified compensation
    Returns quantized tensor and statistics
    """
    original_shape = weight_tensor.shape
    weights_flat = weight_tensor.flatten()
    
    quantized_weights = []
    msr4_count = 0
    non_msr4_count = 0
    
    layer_type, category = get_layer_info(layer_name)
    print(f"  Quantizing {layer_name} ({layer_type}) with {compensation_name}: {len(weights_flat):,} weights")
    
    for i, weight in enumerate(weights_flat):
        # Convert to Q1.7 binary
        binary_str = float_to_q1_7_fixed_point(weight.item())
        
        # Check MSR-4 and count
        if has_msr4(binary_str):
            msr4_count += 1
        else:
            non_msr4_count += 1
        
        # Apply specified compensation
        compensated_binary = compensation_fn(binary_str)
        
        # Convert back to float
        quantized_float = binary_to_float(compensated_binary)
        quantized_weights.append(quantized_float)
        
        # Show progress for large layers
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,} / {len(weights_flat):,} weights")
    
    # Convert back to tensor and reshape
    quantized_tensor = torch.tensor(quantized_weights, dtype=torch.float32).reshape(original_shape)
    
    stats = {
        'layer_name': layer_name,
        'layer_type': layer_type,
        'category': category,
        'compensation_name': compensation_name,
        'total_weights': len(weights_flat),
        'msr4_count': msr4_count,
        'non_msr4_count': non_msr4_count,
        'msr4_percentage': (msr4_count / len(weights_flat)) * 100,
        'non_msr4_percentage': (non_msr4_count / len(weights_flat)) * 100
    }
    
    return quantized_tensor, stats

def quantize_model_weights(model, compensation_fn, compensation_name):
    """
    Quantize all weight tensors in the ResNet-18 model with specified compensation
    """
    print(f"Starting ResNet-18 weight quantization with {compensation_name}...")
    print("Note: This may take a while due to the large number of weights...")
    quantization_stats = []
    
    # Create a new model with quantized weights
    quantized_model = ResNet18().to(device)
    
    # Copy the state dict and quantize weights
    original_state_dict = model.state_dict()
    quantized_state_dict = {}
    
    for name, param in original_state_dict.items():
        if 'weight' in name and 'bn' not in name:  # Skip BatchNorm weights
            # Quantize conv and fc weight tensors
            quantized_tensor, stats = quantize_weight_tensor(param, name, compensation_fn, compensation_name)
            quantized_state_dict[name] = quantized_tensor
            quantization_stats.append(stats)
            print(f"  ✓ {name}: MSR-4: {stats['msr4_count']:,} ({stats['msr4_percentage']:.2f}%), "
                  f"Non-MSR-4: {stats['non_msr4_count']:,} ({stats['non_msr4_percentage']:.2f}%)")
        else:
            # Keep BatchNorm weights, biases and other parameters unchanged
            quantized_state_dict[name] = param
            if 'bn' in name or 'bias' in name:
                print(f"  ✓ {name}: Kept original (BatchNorm/bias)")
    
    # Load quantized weights into new model
    quantized_model.load_state_dict(quantized_state_dict)
    
    return quantized_model, quantization_stats

def evaluate_model(model, test_loader, model_name="Model", quant_act=True):
    model.eval()
    correct = 0
    total = 0

    print(f"Evaluating {model_name}...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data, quant_act=quant_act)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if (batch_idx + 1) % 25 == 0:
                current_acc = 100 * correct / total
                print(f"  Batch {batch_idx + 1}: Current accuracy: {current_acc:.2f}%")

    accuracy = 100 * correct / total
    print(f"  Final {model_name} Accuracy: {accuracy:.2f}%")

    return accuracy

def get_data_loaders(batch_size=64):
    """
    Get MNIST data loaders for evaluation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load test data
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def display_compensation_comparison_summary(original_accuracy, no_comp_accuracy, lsb_comp_accuracy, 
                                          no_comp_stats, lsb_comp_stats):
    """
    Display comprehensive comparison of different compensation strategies for ResNet-18
    """
    print("\n" + "="*110)
    print("RESNET-18 COMPENSATION STRATEGY COMPARISON")
    print("="*110)
    
    # Calculate drops
    no_comp_drop = original_accuracy - no_comp_accuracy
    lsb_comp_drop = original_accuracy - lsb_comp_accuracy
    compensation_impact = lsb_comp_accuracy - no_comp_accuracy
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"{'Model Variant':<35} {'Accuracy':<12} {'Drop from Original':<18} {'Notes'}")
    print("-" * 90)
    print(f"{'Original Float32 ResNet-18':<35} {original_accuracy:<11.2f}% {'-':<17} {'Baseline'}")
    print(f"{'Q1.7 No Compensation':<35} {no_comp_accuracy:<11.2f}% {no_comp_drop:<17.2f}% {'Pure quantization'}")
    print(f"{'Q1.7 LSB=1 Compensation':<35} {lsb_comp_accuracy:<11.2f}% {lsb_comp_drop:<17.2f}% {'Uniform LSB=1'}")
    print("-" * 90)
    
    # Compensation impact analysis
    print(f"\n📊 COMPENSATION IMPACT ANALYSIS:")
    print(f"• Compensation Effect: {compensation_impact:+.2f} percentage points")
    if compensation_impact > 0:
        print(f"• LSB=1 compensation IMPROVES accuracy by {compensation_impact:.2f}%")
        compensation_assessment = "BENEFICIAL"
    elif compensation_impact < 0:
        print(f"• LSB=1 compensation HURTS accuracy by {abs(compensation_impact):.2f}%")
        compensation_assessment = "HARMFUL"
    else:
        print(f"• LSB=1 compensation has NO EFFECT on accuracy")
        compensation_assessment = "NEUTRAL"
    
    print(f"• Overall Assessment: LSB=1 compensation is {compensation_assessment} for ResNet-18")
    
    # Statistical comparison
    total_weights_no_comp = sum(stats['total_weights'] for stats in no_comp_stats)
    total_msr4_no_comp = sum(stats['msr4_count'] for stats in no_comp_stats)
    msr4_percentage = (total_msr4_no_comp / total_weights_no_comp) * 100
    
    print(f"\n📈 QUANTIZATION STATISTICS:")
    print(f"• Total weights: {total_weights_no_comp:,}")
    print(f"• MSR-4 weights: {total_msr4_no_comp:,} ({msr4_percentage:.2f}%)")
    print(f"• Non-MSR-4 weights: {total_weights_no_comp - total_msr4_no_comp:,} ({100 - msr4_percentage:.2f}%)")
    
    # Layer type analysis
    type_stats = {
        'Initial': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'Block': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'Shortcut': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'FC': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0}
    }
    
    # Block group analysis
    block_stats = {
        'layer1': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'layer2': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'layer3': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'layer4': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0}
    }
    
    for stats in no_comp_stats:
        layer_type = stats['layer_type']
        category = stats['category']
        
        # Accumulate type-specific statistics
        if layer_type in type_stats:
            type_stats[layer_type]['total_weights'] += stats['total_weights']
            type_stats[layer_type]['total_msr4'] += stats['msr4_count']
            type_stats[layer_type]['total_non_msr4'] += stats['non_msr4_count']
        
        # Accumulate block-specific statistics
        if any(layer in category for layer in ['layer1', 'layer2', 'layer3', 'layer4']):
            for block in block_stats:
                if block in category:
                    block_stats[block]['total_weights'] += stats['total_weights']
                    block_stats[block]['total_msr4'] += stats['msr4_count']
                    block_stats[block]['total_non_msr4'] += stats['non_msr4_count']
    
    # Layer type breakdown
    print(f"\n🔍 LAYER TYPE BREAKDOWN:")
    print(f"{'Type':<15} {'Weights':<12} {'MSR-4 %':<10} {'Impact of LSB=1'}")
    print("-" * 65)
    
    for layer_type, stats in type_stats.items():
        if stats['total_weights'] > 0:
            msr4_pct = (stats['total_msr4'] / stats['total_weights']) * 100
            print(f"{layer_type:<15} {stats['total_weights']:<12,} {msr4_pct:<9.1f}% {'Applied to all weights'}")
    
    # Block depth analysis
    print(f"\n📋 DEPTH ANALYSIS:")
    print(f"{'Block':<15} {'Weights':<12} {'MSR-4 %':<10} {'Depth Pattern'}")
    print("-" * 65)
    
    depth_msr4_percentages = []
    for block, stats in block_stats.items():
        if stats['total_weights'] > 0:
            msr4_pct = (stats['total_msr4'] / stats['total_weights']) * 100
            depth_msr4_percentages.append(msr4_pct)
            depth_level = block[-1]  # Extract the number from layer1, layer2, etc.
            print(f"{block:<15} {stats['total_weights']:<12,} {msr4_pct:<9.1f}% {'Level ' + depth_level}")
    
    # Depth pattern analysis
    if len(depth_msr4_percentages) >= 2:
        print(f"\n📊 DEPTH PATTERN ANALYSIS:")
        if depth_msr4_percentages[-1] > depth_msr4_percentages[0]:
            print(f"• Deeper layers have higher MSR-4 percentage ({depth_msr4_percentages[-1]:.1f}% vs {depth_msr4_percentages[0]:.1f}%)")
        elif depth_msr4_percentages[0] > depth_msr4_percentages[-1]:
            print(f"• Earlier layers have higher MSR-4 percentage ({depth_msr4_percentages[0]:.1f}% vs {depth_msr4_percentages[-1]:.1f}%)")
        else:
            print(f"• Consistent MSR-4 percentage across depth")
    
    # Hardware implications
    print(f"\n🔧 HARDWARE IMPLICATIONS:")
    print(f"• Memory usage: 8-bit weights (75% reduction from float32)")
    print(f"• MSR-4 optimization potential: {msr4_percentage:.1f}% of weights")
    print(f"• Deep architecture: {len(no_comp_stats)} quantized weight layers")
    print(f"• Residual connections: Preserved in quantized format")
    
    if compensation_assessment == "HARMFUL":
        print(f"• Recommendation: AVOID uniform LSB=1 compensation in ResNet-18")
        print(f"• Reason: Systematic bias accumulates across deep architecture")
        print(f"• Deep networks amplify bias effects through residual connections")
    elif compensation_assessment == "BENEFICIAL":
        print(f"• Recommendation: USE uniform LSB=1 compensation in ResNet-18")
        print(f"• Reason: Improves quantization accuracy in deep networks")
    else:
        print(f"• Recommendation: LSB=1 compensation is optional for ResNet-18")
        print(f"• Reason: No significant impact on accuracy")
    
    # Quality assessment
    print(f"\n📋 QUANTIZATION QUALITY ASSESSMENT:")
    if no_comp_drop < 1.0:
        no_comp_quality = "Excellent"
    elif no_comp_drop < 2.0:
        no_comp_quality = "Good"  
    elif no_comp_drop < 5.0:
        no_comp_quality = "Acceptable"
    else:
        no_comp_quality = "Poor"
    
    if lsb_comp_drop < 1.0:
        lsb_comp_quality = "Excellent"
    elif lsb_comp_drop < 2.0:
        lsb_comp_quality = "Good"
    elif lsb_comp_drop < 5.0:
        lsb_comp_quality = "Acceptable"
    else:
        lsb_comp_quality = "Poor"
    
    print(f"• Q1.7 No Compensation: {no_comp_quality} ({no_comp_drop:.2f}% drop)")
    print(f"• Q1.7 LSB=1 Compensation: {lsb_comp_quality} ({lsb_comp_drop:.2f}% drop)")
    
    # ResNet-specific insights
    print(f"\n🧠 RESNET-18 SPECIFIC INSIGHTS:")
    
    # Compare different layer types
    if type_stats['Block']['total_weights'] > 0 and type_stats['Shortcut']['total_weights'] > 0:
        block_msr4_pct = (type_stats['Block']['total_msr4'] / type_stats['Block']['total_weights']) * 100
        shortcut_msr4_pct = (type_stats['Shortcut']['total_msr4'] / type_stats['Shortcut']['total_weights']) * 100
        print(f"• Regular block convolutions MSR-4: {block_msr4_pct:.2f}%")
        print(f"• Shortcut convolutions MSR-4: {shortcut_msr4_pct:.2f}%")
        
        if compensation_assessment == "HARMFUL":
            print(f"• Bias propagates through both main path and residual shortcuts")
    
    print("="*110)

def demonstrate_compensation_examples():
    """
    Demonstrate the different compensation strategies with examples
    """
    print("\n" + "="*80)
    print("RESNET-18 COMPENSATION STRATEGY EXAMPLES")
    print("="*80)
    
    examples = [
        (0.004096, "MSR-4 example (small positive)"),
        (-0.114851, "MSR-4 example (negative)"),  
        (0.174329, "Non-MSR-4 example (positive)"),
        (-0.134351, "Non-MSR-4 example (negative)")
    ]
    
    print(f"{'Weight':<12} {'Original':<10} {'No Comp':<10} {'LSB=1':<10} {'MSR-4':<10} {'Type'}")
    print("-" * 80)
    
    for weight, description in examples:
        original_binary = float_to_q1_7_fixed_point(weight)
        is_msr4 = has_msr4(original_binary)
        
        no_comp_binary = apply_no_compensation(original_binary)
        lsb_comp_binary = apply_uniform_lsb_compensation(original_binary)
        msr4_comp_binary = apply_msr4_compensation(original_binary)
        
        no_comp_float = binary_to_float(no_comp_binary)
        lsb_comp_float = binary_to_float(lsb_comp_binary)
        msr4_comp_float = binary_to_float(msr4_comp_binary)
        
        weight_type = "MSR-4" if is_msr4 else "Non-MSR-4"
        
        print(f"{weight:<12.6f} {original_binary:<10} {no_comp_binary:<10} {lsb_comp_binary:<10} {msr4_comp_binary:<10} {weight_type}")
        print(f"{'Float:':<12} {weight:<10.6f} {no_comp_float:<10.6f} {lsb_comp_float:<10.6f} {msr4_comp_float:<10.6f}")
        print()

def main():
    """
    Main quantization and evaluation pipeline for ResNet-18 with compensation comparison
    """
    print("ResNet-18 Quantization: Compensation Strategy Comparison")
    print("="*65)
    
    # Demonstrate compensation examples
    demonstrate_compensation_examples()
    
    # Load original trained model
    print(f"\n📁 Loading original trained ResNet-18 model...")
    original_model = ResNet18().to(device)
    
    try:
        state_dict = torch.load('mnist_resnet18_model.pth', map_location=device)
        original_model.load_state_dict(state_dict)
        print("✓ Original ResNet-18 model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'mnist_resnet18_model.pth' not found!")
        print("Please train the ResNet-18 model first using ResNet.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test data loader
    print(f"\n📊 Loading test dataset...")
    test_loader = get_data_loaders(batch_size=500)  # Moderate batch for memory efficiency
    print("✓ Test dataset loaded")
    
    # Evaluate original model
    print(f"\n🔍 Evaluating original ResNet-18 model...")
    original_accuracy = evaluate_model(original_model, test_loader, "Original Float32 ResNet-18")
    
    # Create and evaluate Q1.7 model WITHOUT compensation
    print(f"\n⚙️  Creating Q1.7 ResNet-18 WITHOUT compensation...")
    no_comp_model, no_comp_stats = quantize_model_weights(
        original_model, 
        apply_no_compensation, 
        "No Compensation"
    )
    print("✓ Q1.7 quantization (no compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 ResNet-18 without compensation...")
    no_comp_accuracy = evaluate_model(no_comp_model, test_loader, "Q1.7 No Compensation")
    
    # Create and evaluate Q1.7 model WITH uniform LSB=1 compensation
    print(f"\n⚙️  Creating Q1.7 ResNet-18 WITH uniform LSB=1 compensation...")
    lsb_comp_model, lsb_comp_stats = quantize_model_weights(
        original_model, 
        apply_uniform_lsb_compensation, 
        "Uniform LSB=1"
    )
    print("✓ Q1.7 quantization (LSB=1 compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 ResNet-18 with LSB=1 compensation...")
    lsb_comp_accuracy = evaluate_model(lsb_comp_model, test_loader, "Q1.7 LSB=1 Compensation")
    
    # Display comprehensive comparison
    display_compensation_comparison_summary(
        original_accuracy, no_comp_accuracy, lsb_comp_accuracy, 
        no_comp_stats, lsb_comp_stats
    )
    
    # Save models
    print(f"\n💾 Saving quantized ResNet-18 models...")
    torch.save(no_comp_model.state_dict(), 'mnist_resnet18_q17_no_compensation.pth')
    torch.save(lsb_comp_model.state_dict(), 'mnist_resnet18_q17_lsb_compensation.pth')
    print("✓ Models saved:")
    print("  - mnist_resnet18_q17_no_compensation.pth")
    print("  - mnist_resnet18_q17_lsb_compensation.pth")
    
    print(f"\n🎉 ResNet-18 compensation comparison completed successfully!")
    
    return original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats

if __name__ == "__main__":
    # Run the compensation comparison pipeline
    original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats = main()
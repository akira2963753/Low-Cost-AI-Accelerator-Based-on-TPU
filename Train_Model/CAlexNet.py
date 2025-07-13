import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

np.random.seed(666)                  
torch.manual_seed(666)  

def quantize_and_compensate_activation(tensor):
    """
    向量化版本：將 activation 量化為 Q1.7 並強制 LSB=1，模擬 8-bit 定點補償乘法。
    """
    # 轉成 numpy 陣列
    tensor_np = tensor.detach().cpu().numpy()

    # Q1.7 量化（乘 128 並無條件捨去）
    scaled = np.floor(tensor_np * 128).astype(np.int32)

    # 限制範圍 [-128, 127]
    clamped = np.clip(scaled, -128, 127)

    # LSB = 1 補償（與 1 做 OR）
    compensated = (clamped | 0b00001000) & 0b11111000
    # 轉回 float32（模擬乘法前的行為）
    float_output = compensated.astype(np.float32) / 128.0

    # 回傳 tensor
    return torch.tensor(float_output, dtype=torch.float32).to(tensor.device)


# Define the same AlexNet model architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (5 convolutional layers)
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Second convolutional layer
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Third convolutional layer
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Classifier layers (3 fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # First fully connected layer
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Second fully connected layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # Third fully connected layer (output layer)
            nn.Linear(4096, num_classes),
        )
         
    def forward(self, x, quant_act=True):
        # Feature extraction
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU) and quant_act:
                x = quantize_and_compensate_activation(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU) and quant_act:
                x = quantize_and_compensate_activation(x)
    
        return x

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
        #compensated = binary_str[:4] + '1000'
        compensated = binary_str[:-1] + '1'
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
    Get layer type and category for AlexNet
    """
    if 'features' in layer_name:
        layer_type = 'Conv'
        if 'features.0' in layer_name:
            category = 'Early-Conv'
        elif 'features.3' in layer_name or 'features.6' in layer_name:
            category = 'Mid-Conv'
        elif 'features.8' in layer_name or 'features.10' in layer_name:
            category = 'Late-Conv'
        else:
            category = 'Other-Conv'
    elif 'classifier' in layer_name:
        layer_type = 'FC'
        if 'classifier.1' in layer_name:
            category = 'FC1'
        elif 'classifier.4' in layer_name:
            category = 'FC2'
        elif 'classifier.6' in layer_name:
            category = 'FC3'
        else:
            category = 'Other-FC'
    else:
        layer_type = 'Other'
        category = 'Other'
    
    return layer_type, category

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
    
    start_time = time.time()
    
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
        
        # Show progress for very large layers (AlexNet has massive FC layers)
        if (i + 1) % 1000000 == 0:  # Every 1M weights
            elapsed_time = time.time() - start_time
            progress = ((i + 1) / len(weights_flat)) * 100
            eta = (elapsed_time / progress) * (100 - progress) if progress > 0 else 0
            print(f"    Processed {i + 1:,} / {len(weights_flat):,} weights ({progress:.1f}%) - ETA: {eta:.1f}s")
        elif (i + 1) % 100000 == 0 and len(weights_flat) < 1000000:  # Every 100K for smaller layers
            print(f"    Processed {i + 1:,} / {len(weights_flat):,} weights")
    
    elapsed_time = time.time() - start_time
    print(f"    Completed in {elapsed_time:.1f}s")
    
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
        'non_msr4_percentage': (non_msr4_count / len(weights_flat)) * 100,
        'quantization_time': elapsed_time
    }
    
    return quantized_tensor, stats

def quantize_model_weights(model, compensation_fn, compensation_name):
    """
    Quantize all weight tensors in the AlexNet model with specified compensation
    """
    print(f"Starting AlexNet weight quantization with {compensation_name}...")
    print("⚠️  Note: This will take significant time due to ~28.5M weights!")
    quantization_stats = []
    
    # Create a new model with quantized weights
    quantized_model = AlexNet().to(device)
    
    # Copy the state dict and quantize weights
    original_state_dict = model.state_dict()
    quantized_state_dict = {}
    
    total_start_time = time.time()
    
    for name, param in original_state_dict.items():
        if 'weight' in name:
            # Quantize weight tensors
            print(f"\n🔄 Starting quantization of {name}...")
            quantized_tensor, stats = quantize_weight_tensor(param, name, compensation_fn, compensation_name)
            quantized_state_dict[name] = quantized_tensor
            quantization_stats.append(stats)
            print(f"  ✓ {name}: MSR-4: {stats['msr4_count']:,} ({stats['msr4_percentage']:.2f}%), "
                  f"Non-MSR-4: {stats['non_msr4_count']:,} ({stats['non_msr4_percentage']:.2f}%)")
        else:
            # Keep bias and other parameters unchanged
            quantized_state_dict[name] = param
            print(f"  ✓ {name}: Kept original (bias/other)")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n🔄 Loading quantized weights into model...")
    quantized_model.load_state_dict(quantized_state_dict)
    
    print(f"✓ Total quantization time: {total_elapsed_time:.1f}s ({total_elapsed_time/60:.1f} minutes)")
    
    return quantized_model, quantization_stats

def evaluate_model(model, test_loader, model_name="Model", quant_act=True):
    model.eval()
    correct = 0
    total = 0

    print(f"Evaluating {model_name}...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
            data, target = data.to(device), target.to(device)
            outputs = model(data, quant_act=quant_act)  # 加入 quant_act
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Current accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    print(f"  Final {model_name} Accuracy: {accuracy:.2f}%")

    return accuracy

def get_data_loaders(batch_size=64):
    """
    Get MNIST data loaders for evaluation (with AlexNet's resize)
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
    Display comprehensive comparison of different compensation strategies for AlexNet
    """
    print("\n" + "="*120)
    print("ALEXNET COMPENSATION STRATEGY COMPARISON")
    print("="*120)
    
    # Calculate drops
    no_comp_drop = original_accuracy - no_comp_accuracy
    lsb_comp_drop = original_accuracy - lsb_comp_accuracy
    compensation_impact = lsb_comp_accuracy - no_comp_accuracy
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"{'Model Variant':<40} {'Accuracy':<12} {'Drop from Original':<18} {'Notes'}")
    print("-" * 95)
    print(f"{'Original Float32 AlexNet':<40} {original_accuracy:<11.2f}% {'-':<17} {'Baseline'}")
    print(f"{'Q1.7 No Compensation':<40} {no_comp_accuracy:<11.2f}% {no_comp_drop:<17.2f}% {'Pure quantization'}")
    print(f"{'Q1.7 LSB=1 Compensation':<40} {lsb_comp_accuracy:<11.2f}% {lsb_comp_drop:<17.2f}% {'Uniform LSB=1'}")
    print("-" * 95)
    
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
    
    print(f"• Overall Assessment: LSB=1 compensation is {compensation_assessment} for AlexNet")
    
    # Statistical comparison
    total_weights_no_comp = sum(stats['total_weights'] for stats in no_comp_stats)
    total_msr4_no_comp = sum(stats['msr4_count'] for stats in no_comp_stats)
    msr4_percentage = (total_msr4_no_comp / total_weights_no_comp) * 100
    
    print(f"\n📈 QUANTIZATION STATISTICS:")
    print(f"• Total weights: {total_weights_no_comp:,} (~{total_weights_no_comp/1000000:.1f}M)")
    print(f"• MSR-4 weights: {total_msr4_no_comp:,} ({msr4_percentage:.2f}%)")
    print(f"• Non-MSR-4 weights: {total_weights_no_comp - total_msr4_no_comp:,} ({100 - msr4_percentage:.2f}%)")
    
    # Layer section analysis
    features_stats = {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0}
    classifier_stats = {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0}
    
    # Position analysis
    position_stats = {
        'Early-Conv': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'Mid-Conv': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'Late-Conv': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'FC1': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'FC2': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0},
        'FC3': {'total_weights': 0, 'total_msr4': 0, 'total_non_msr4': 0}
    }
    
    for stats in no_comp_stats:
        layer_type = stats['layer_type']
        category = stats['category']
        
        # Accumulate section-specific statistics
        if layer_type == 'Conv':
            features_stats['total_weights'] += stats['total_weights']
            features_stats['total_msr4'] += stats['msr4_count']
            features_stats['total_non_msr4'] += stats['non_msr4_count']
        elif layer_type == 'FC':
            classifier_stats['total_weights'] += stats['total_weights']
            classifier_stats['total_msr4'] += stats['msr4_count']
            classifier_stats['total_non_msr4'] += stats['non_msr4_count']
        
        # Accumulate position-specific statistics
        if category in position_stats:
            position_stats[category]['total_weights'] += stats['total_weights']
            position_stats[category]['total_msr4'] += stats['msr4_count']
            position_stats[category]['total_non_msr4'] += stats['non_msr4_count']
    
    # Architecture section breakdown
    print(f"\n🔍 ARCHITECTURE SECTION BREAKDOWN:")
    print(f"{'Section':<20} {'Weights':<12} {'Weight %':<10} {'MSR-4 %':<10} {'Impact of LSB=1'}")
    print("-" * 80)
    
    if features_stats['total_weights'] > 0:
        features_msr4_pct = (features_stats['total_msr4'] / features_stats['total_weights']) * 100
        features_weight_pct = (features_stats['total_weights'] / total_weights_no_comp) * 100
        print(f"{'Features (Conv)':<20} {features_stats['total_weights']:<12,} {features_weight_pct:<9.1f}% {features_msr4_pct:<9.1f}% {'Applied to all weights'}")
    
    if classifier_stats['total_weights'] > 0:
        classifier_msr4_pct = (classifier_stats['total_msr4'] / classifier_stats['total_weights']) * 100
        classifier_weight_pct = (classifier_stats['total_weights'] / total_weights_no_comp) * 100
        print(f"{'Classifier (FC)':<20} {classifier_stats['total_weights']:<12,} {classifier_weight_pct:<9.1f}% {classifier_msr4_pct:<9.1f}% {'Applied to all weights'}")
    
    # Position breakdown
    print(f"\n📋 POSITION-WISE BREAKDOWN:")
    print(f"{'Position':<15} {'Weights':<12} {'MSR-4 %':<10} {'Scale Impact'}")
    print("-" * 65)
    
    largest_layer = None
    largest_size = 0
    
    for position, stats in position_stats.items():
        if stats['total_weights'] > 0:
            msr4_pct = (stats['total_msr4'] / stats['total_weights']) * 100
            if stats['total_weights'] > largest_size:
                largest_size = stats['total_weights']
                largest_layer = position
            
            if stats['total_weights'] > 1000000:  # 1M+ weights
                scale_note = "Massive"
            elif stats['total_weights'] > 100000:  # 100K+ weights
                scale_note = "Large"
            else:
                scale_note = "Moderate"
            
            print(f"{position:<15} {stats['total_weights']:<12,} {msr4_pct:<9.1f}% {scale_note}")
    
    # Scale-specific insights
    print(f"\n📊 SCALE-SPECIFIC INSIGHTS:")
    print(f"• Largest layer type: {largest_layer} ({largest_size:,} weights)")
    print(f"• Architecture dominance: {classifier_weight_pct:.1f}% FC weights, {features_weight_pct:.1f}% Conv weights")
    
    if compensation_assessment == "HARMFUL":
        print(f"• Bias accumulation: {total_weights_no_comp:,} weights amplify systematic bias")
        print(f"• Scale impact: Large FC layers ({classifier_stats['total_weights']:,} weights) most affected")
    
    # Processing time analysis
    total_time_no_comp = sum(stats.get('quantization_time', 0) for stats in no_comp_stats)
    total_time_lsb_comp = sum(stats.get('quantization_time', 0) for stats in lsb_comp_stats)
    
    print(f"\n⏱️  PROCESSING TIME ANALYSIS:")
    print(f"• No compensation: {total_time_no_comp:.1f}s ({total_time_no_comp/60:.1f} minutes)")
    print(f"• LSB=1 compensation: {total_time_lsb_comp:.1f}s ({total_time_lsb_comp/60:.1f} minutes)")
    print(f"• Processing rate: ~{total_weights_no_comp/total_time_no_comp:.0f} weights/second")
    
    # Hardware implications
    print(f"\n🔧 HARDWARE IMPLICATIONS:")
    print(f"• Memory usage: 8-bit weights (75% reduction from float32)")
    print(f"• Memory savings: ~{(total_weights_no_comp * 3) / (1024**3):.1f}GB saved (float32 → int8)")
    print(f"• MSR-4 optimization potential: {msr4_percentage:.1f}% of weights")
    print(f"• Production scale: {total_weights_no_comp:,} weights suitable for deployment")
    
    if compensation_assessment == "HARMFUL":
        print(f"• Recommendation: AVOID uniform LSB=1 compensation in AlexNet")
        print(f"• Reason: Systematic bias accumulates across massive weight matrices")
        print(f"• Alternative: Use pure Q1.7 quantization for better accuracy")
    elif compensation_assessment == "BENEFICIAL":
        print(f"• Recommendation: USE uniform LSB=1 compensation in AlexNet")
        print(f"• Reason: Improves quantization accuracy at scale")
    else:
        print(f"• Recommendation: LSB=1 compensation is optional for AlexNet")
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
    
    # AlexNet-specific insights
    print(f"\n🧠 ALEXNET-SPECIFIC INSIGHTS:")
    print(f"• Feature extraction MSR-4: {features_msr4_pct:.2f}%")
    print(f"• Classifier MSR-4: {classifier_msr4_pct:.2f}%")
    
    if features_msr4_pct != classifier_msr4_pct:
        if features_msr4_pct > classifier_msr4_pct:
            print(f"• Feature layers quantize better than classifier layers")
        else:
            print(f"• Classifier layers quantize better than feature layers")
    
    if compensation_assessment == "HARMFUL":
        print(f"• Large FC layers amplify bias effects significantly")
        print(f"• {largest_layer} layer alone contributes substantial bias")
    
    print("="*120)

def demonstrate_compensation_examples():
    """
    Demonstrate the different compensation strategies with examples
    """
    print("\n" + "="*90)
    print("ALEXNET COMPENSATION STRATEGY EXAMPLES")
    print("="*90)
    
    examples = [
        (0.004096, "MSR-4 example (small positive)"),
        (-0.114851, "MSR-4 example (negative)"),  
        (0.174329, "Non-MSR-4 example (positive)"),
        (-0.134351, "Non-MSR-4 example (negative)")
    ]
    
    print(f"{'Weight':<12} {'Original':<10} {'No Comp':<10} {'LSB=1':<10} {'MSR-4':<10} {'Type'}")
    print("-" * 90)
    
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
    Main quantization and evaluation pipeline for AlexNet with compensation comparison
    """
    print("AlexNet Quantization: Compensation Strategy Comparison")
    print("="*70)
    print("⚠️  WARNING: This will process ~28.5M weights and may take 30-60 minutes!")
    
    # Demonstrate compensation examples
    demonstrate_compensation_examples()
    
    # Load original trained model
    print(f"\n📁 Loading original trained AlexNet model...")
    original_model = AlexNet().to(device)
    
    try:
        state_dict = torch.load('mnist_alexnet_model.pth', map_location=device)
        original_model.load_state_dict(state_dict)
        print("✓ Original AlexNet model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'mnist_alexnet_model.pth' not found!")
        print("Please train the AlexNet model first using AlexNet.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test data loader
    print(f"\n📊 Loading test dataset...")
    test_loader = get_data_loaders(batch_size=256)  # Larger batch for efficiency
    print("✓ Test dataset loaded")
    
    # Evaluate original model
    print(f"\n🔍 Evaluating original AlexNet model...")
    original_accuracy = evaluate_model(original_model, test_loader, "Original Float32 AlexNet")
    
    # Create and evaluate Q1.7 model WITHOUT compensation
    print(f"\n⚙️  Creating Q1.7 AlexNet WITHOUT compensation...")
    print("🕐 This will take significant time due to the massive model size...")
    no_comp_model, no_comp_stats = quantize_model_weights(
        original_model, 
        apply_no_compensation, 
        "No Compensation"
    )
    print("✓ Q1.7 quantization (no compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 AlexNet without compensation...")
    no_comp_accuracy = evaluate_model(no_comp_model, test_loader, "Q1.7 No Compensation")
    
    # Create and evaluate Q1.7 model WITH uniform LSB=1 compensation
    print(f"\n⚙️  Creating Q1.7 AlexNet WITH uniform LSB=1 compensation...")
    print("🕐 Processing another ~28.5M weights...")
    lsb_comp_model, lsb_comp_stats = quantize_model_weights(
        original_model, 
        apply_uniform_lsb_compensation, 
        "Uniform LSB=1"
    )
    print("✓ Q1.7 quantization (LSB=1 compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 AlexNet with LSB=1 compensation...")
    lsb_comp_accuracy = evaluate_model(lsb_comp_model, test_loader, "Q1.7 LSB=1 Compensation")
    
    # Display comprehensive comparison
    display_compensation_comparison_summary(
        original_accuracy, no_comp_accuracy, lsb_comp_accuracy, 
        no_comp_stats, lsb_comp_stats
    )
    
    # Save models
    print(f"\n💾 Saving quantized AlexNet models...")
    torch.save(no_comp_model.state_dict(), 'mnist_alexnet_q17_no_compensation.pth')
    torch.save(lsb_comp_model.state_dict(), 'mnist_alexnet_q17_lsb_compensation.pth')
    print("✓ Models saved:")
    print("  - mnist_alexnet_q17_no_compensation.pth")
    print("  - mnist_alexnet_q17_lsb_compensation.pth")
    
    total_weights = sum(s['total_weights'] for s in no_comp_stats)
    print(f"\n🎉 AlexNet compensation comparison completed successfully!")
    print(f"📊 Final stats: {total_weights:,} weights quantized (~{total_weights/1000000:.1f}M)")
    
    return original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats

if __name__ == "__main__":
    # Run the compensation comparison pipeline
    original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats = main()
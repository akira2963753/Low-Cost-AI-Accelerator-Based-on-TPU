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

def quantize_and_compensate_activation(tensor):
    """
    Quantize activations to Q1.7 and apply LSB=1 compensation.
    Return float32 tensor simulating fixed-point behavior.
    """
    tensor_np = tensor.detach().cpu().numpy()
    flat = tensor_np.flatten()
    result = []
    for val in flat:
        binary = float_to_q1_7_fixed_point(val)
        compensated = binary[:-1] + '1'  # LSB = 1
        float_val = binary_to_float(compensated)
        result.append(float_val)
    reshaped = torch.tensor(result, dtype=torch.float32).reshape(tensor.shape)
    return reshaped.to(tensor.device)


# Define the same LeNet model architecture
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

    def forward(self, x, quant_act=False):
        if quant_act:
            x = quantize_and_compensate_activation(x)
        x = self.pool1(F.relu(self.conv1(x)))
        if quant_act:
            x = quantize_and_compensate_activation(x)

        x = self.pool2(F.relu(self.conv2(x)))
        if quant_act:
            x = quantize_and_compensate_activation(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if quant_act:
            x = quantize_and_compensate_activation(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if quant_act:
            x = quantize_and_compensate_activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
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
        compensated = binary_str[:4] + '1000'
    
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
    Get layer type and category for LeNet
    """
    if 'conv' in layer_name:
        layer_type = 'Conv'
        if 'conv1' in layer_name:
            category = 'Conv1'
        elif 'conv2' in layer_name:
            category = 'Conv2'
        else:
            category = 'Other Conv'
    elif 'fc' in layer_name:
        layer_type = 'FC'
        if 'fc1' in layer_name:
            category = 'FC1'
        elif 'fc2' in layer_name:
            category = 'FC2'
        elif 'fc3' in layer_name:
            category = 'FC3'
        else:
            category = 'Other FC'
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
        if (i + 1) % 10000 == 0:
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
    Quantize all weight and bias tensors in the LeNet model with specified compensation.
    """
    print(f"Starting LeNet weight quantization with {compensation_name}...")
    quantization_stats = []

    # Create a new model with quantized weights
    quantized_model = LeNet().to(device)

    # Copy the state dict and quantize weights/biases
    original_state_dict = model.state_dict()
    quantized_state_dict = {}

    for name, param in original_state_dict.items():
        if 'weight' in name:
            # Quantize weight tensors
            quantized_tensor, stats = quantize_weight_tensor(param, name, compensation_fn, compensation_name)
            quantized_state_dict[name] = quantized_tensor
            quantization_stats.append(stats)
            print(f"  ✓ {name}: MSR-4: {stats['msr4_count']:,} ({stats['msr4_percentage']:.2f}%), "
                  f"Non-MSR-4: {stats['non_msr4_count']:,} ({stats['non_msr4_percentage']:.2f}%)")
        
        elif 'bias' in name:
            print(f"  Quantizing bias: {name}")
            bias_flat = param.flatten()
            compensated_bias = []

            for val in bias_flat:
                binary = float_to_q1_7_fixed_point(val.item())
                compensated = apply_uniform_lsb_compensation(binary)
                float_val = binary_to_float(compensated)
                compensated_bias.append(float_val)

            quantized_bias = torch.tensor(compensated_bias, dtype=torch.float32).reshape(param.shape)
            quantized_state_dict[name] = quantized_bias
            print(f"  ✓ {name}: Bias quantized to Q1.7 with LSB=1 compensation")

        else:
            # Keep other parameters unchanged (e.g., running_mean, running_var)
            quantized_state_dict[name] = param
            print(f"  ✓ {name}: Kept original (non-weight/bias parameter)")

    # Load quantized weights and biases into new model
    quantized_model.load_state_dict(quantized_state_dict)

    return quantized_model, quantization_stats


def evaluate_model(model, test_loader, model_name="Model", quant_act=False):
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
            if (batch_idx + 1) % 50 == 0:
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
    Display comprehensive comparison of different compensation strategies
    """
    print("\n" + "="*100)
    print("LENET COMPENSATION STRATEGY COMPARISON")
    print("="*100)
    
    # Calculate drops
    no_comp_drop = original_accuracy - no_comp_accuracy
    lsb_comp_drop = original_accuracy - lsb_comp_accuracy
    compensation_impact = lsb_comp_accuracy - no_comp_accuracy
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"{'Model Variant':<30} {'Accuracy':<12} {'Drop from Original':<18} {'Notes'}")
    print("-" * 85)
    print(f"{'Original Float32':<30} {original_accuracy:<11.2f}% {'-':<17} {'Baseline'}")
    print(f"{'Q1.7 No Compensation':<30} {no_comp_accuracy:<11.2f}% {no_comp_drop:<17.2f}% {'Pure quantization'}")
    print(f"{'Q1.7 LSB=1 Compensation':<30} {lsb_comp_accuracy:<11.2f}% {lsb_comp_drop:<17.2f}% {'Uniform LSB=1'}")
    print("-" * 85)
    
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
    
    print(f"• Overall Assessment: LSB=1 compensation is {compensation_assessment}")
    
    # Statistical comparison
    total_weights_no_comp = sum(stats['total_weights'] for stats in no_comp_stats)
    total_msr4_no_comp = sum(stats['msr4_count'] for stats in no_comp_stats)
    msr4_percentage = (total_msr4_no_comp / total_weights_no_comp) * 100
    
    print(f"\n📈 QUANTIZATION STATISTICS:")
    print(f"• Total weights: {total_weights_no_comp:,}")
    print(f"• MSR-4 weights: {total_msr4_no_comp:,} ({msr4_percentage:.2f}%)")
    print(f"• Non-MSR-4 weights: {total_weights_no_comp - total_msr4_no_comp:,} ({100 - msr4_percentage:.2f}%)")
    
    # Layer-wise comparison
    print(f"\n🔍 LAYER-WISE IMPACT:")
    print(f"{'Layer':<20} {'Type':<6} {'Weights':<10} {'MSR-4 %':<8} {'Impact of LSB=1'}")
    print("-" * 70)
    
    for i, stats in enumerate(no_comp_stats):
        layer_name = stats['layer_name']
        layer_type = stats['layer_type']
        total_weights = stats['total_weights']
        msr4_pct = stats['msr4_percentage']
        
        print(f"{layer_name:<20} {layer_type:<6} {total_weights:<10,} {msr4_pct:<7.1f}% {'Applied to all weights'}")
    
    # Hardware implications
    print(f"\n🔧 HARDWARE IMPLICATIONS:")
    print(f"• Memory usage: 8-bit weights (75% reduction from float32)")
    print(f"• MSR-4 optimization potential: {msr4_percentage:.1f}% of weights")
    
    if compensation_assessment == "HARMFUL":
        print(f"• Recommendation: AVOID uniform LSB=1 compensation")
        print(f"• Reason: Introduces systematic bias that accumulates across layers")
    elif compensation_assessment == "BENEFICIAL":
        print(f"• Recommendation: USE uniform LSB=1 compensation")
        print(f"• Reason: Improves quantization accuracy")
    else:
        print(f"• Recommendation: LSB=1 compensation is optional")
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
    
    print("="*100)

def demonstrate_compensation_examples():
    """
    Demonstrate the different compensation strategies with examples
    """
    print("\n" + "="*70)
    print("COMPENSATION STRATEGY EXAMPLES")
    print("="*70)
    
    examples = [
        (0.004096, "MSR-4 example (small positive)"),
        (-0.114851, "MSR-4 example (negative)"),  
        (0.174329, "Non-MSR-4 example (positive)"),
        (-0.134351, "Non-MSR-4 example (negative)")
    ]
    
    print(f"{'Weight':<12} {'Original':<10} {'No Comp':<10} {'LSB=1':<10} {'MSR-4':<10} {'Type'}")
    print("-" * 70)
    
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
    Main quantization and evaluation pipeline for LeNet with compensation comparison
    """
    print("LeNet Quantization: Compensation Strategy Comparison")
    print("="*60)
    
    # Demonstrate compensation examples
    demonstrate_compensation_examples()
    
    # Load original trained model
    print(f"\n📁 Loading original trained LeNet model...")
    original_model = LeNet().to(device)
    
    try:
        state_dict = torch.load('mnist_lenet_model.pth', map_location=device)
        original_model.load_state_dict(state_dict)
        print("✓ Original LeNet model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'mnist_lenet_model.pth' not found!")
        print("Please train the LeNet model first using LeNet.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test data loader
    print(f"\n📊 Loading test dataset...")
    test_loader = get_data_loaders(batch_size=1000)  # Larger batch for faster evaluation
    print("✓ Test dataset loaded")
    
    # Evaluate original model
    print(f"\n🔍 Evaluating original LeNet model...")
    original_accuracy = evaluate_model(original_model, test_loader, "Original Float32 LeNet")
    
    # Create and evaluate Q1.7 model WITHOUT compensation
    print(f"\n⚙️  Creating Q1.7 LeNet WITHOUT compensation...")
    no_comp_model, no_comp_stats = quantize_model_weights(
        original_model, 
        apply_no_compensation, 
        "No Compensation"
    )
    print("✓ Q1.7 quantization (no compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 LeNet without compensation...")
    no_comp_accuracy = evaluate_model(no_comp_model, test_loader, "Q1.7 No Compensation")
    
    # Create and evaluate Q1.7 model WITH uniform LSB=1 compensation
    print(f"\n⚙️  Creating Q1.7 LeNet WITH uniform LSB=1 compensation...")
    lsb_comp_model, lsb_comp_stats = quantize_model_weights(
        original_model, 
        apply_uniform_lsb_compensation, 
        "Uniform LSB=1"
    )
    print("✓ Q1.7 quantization (LSB=1 compensation) completed")
    
    print(f"\n🔍 Evaluating Q1.7 LeNet with LSB=1 compensation...")
    lsb_comp_accuracy = evaluate_model(lsb_comp_model, test_loader, "Q1.7 LSB=1 Compensation")
    
    # Display comprehensive comparison
    display_compensation_comparison_summary(
        original_accuracy, no_comp_accuracy, lsb_comp_accuracy, 
        no_comp_stats, lsb_comp_stats
    )
    
    # Save models
    print(f"\n💾 Saving quantized LeNet models...")
    torch.save(no_comp_model.state_dict(), 'mnist_lenet_q17_no_compensation.pth')
    torch.save(lsb_comp_model.state_dict(), 'mnist_lenet_q17_lsb_compensation.pth')
    print("✓ Models saved:")
    print("  - mnist_lenet_q17_no_compensation.pth")
    print("  - mnist_lenet_q17_lsb_compensation.pth")
    
    print(f"\n🎉 LeNet compensation comparison completed successfully!")
    
    return original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats

if __name__ == "__main__":
    # Run the compensation comparison pipeline
    original_model, no_comp_model, lsb_comp_model, no_comp_stats, lsb_comp_stats = main()
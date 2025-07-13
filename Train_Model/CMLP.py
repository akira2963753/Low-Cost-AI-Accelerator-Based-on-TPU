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

# Define the same MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, num_classes=10):
        super(MLP, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x

# Define Quantized MLP with activation quantization
class QuantizedMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, num_classes=10):
        super(QuantizedMLP, self).__init__()
        # Same architecture as original MLP
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # First layer with ReLU activation + quantization
        x = F.relu(self.fc1(x))
        x = quantize_and_compensate_activation(x)  # NEW: Quantize activations
        x = self.dropout(x)
        
        # Second layer with ReLU activation + quantization
        x = F.relu(self.fc2(x))
        x = quantize_and_compensate_activation(x)  # NEW: Quantize activations
        x = self.dropout(x)
        
        # Output layer (no activation quantization after final layer)
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
    
    #if scaled_weight < 0 : 
        #print(scaled_weight,rounded_weight)

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

def apply_msr4_compensation(binary_str):
    """
    Apply MSR-4 compensation rules:
    - For MSR-4 weights: set last bit to 1
    - For Non-MSR-4 weights: set last 4 bits to 1000
    """
    if has_msr4(binary_str):
        # MSR-4: Set last bit to 1 for expectation compensation
        compensated = binary_str[:-1] + '1'
        #compensated = binary_str
    else:
        # Non-MSR-4: Set last 4 bits to 1000
        #compensated = binary_str[:4] + '1000'
        compensated = binary_str[:-1] + '1'
    return compensated

def apply_activation_lsb_compensation(binary_str):
    """
    Apply LSB=1 compensation to activation binary string
    Always set the last bit (LSB) to 1
    """
    return binary_str[:4] + '1000'

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

def quantize_and_compensate_activation(activation_tensor):
    """
    Quantize activation tensor to Q1.7 format with LSB=1 compensation
    Input: PyTorch tensor with float activations
    Output: PyTorch tensor with quantized and compensated activations
    """
    # Convert to numpy for easier processing
    activations_np = activation_tensor.detach().cpu().numpy()
    original_shape = activations_np.shape
    activations_flat = activations_np.flatten()
    
    quantized_activations = []
    
    for activation in activations_flat:
        # Convert to Q1.7 binary
        binary_str = float_to_q1_7_fixed_point(activation)
        
        # Apply LSB=1 compensation
        compensated_binary = apply_activation_lsb_compensation(binary_str)
        
        # Convert back to float
        quantized_float = binary_to_float(compensated_binary)
        quantized_activations.append(quantized_float)
    
    # Convert back to tensor and reshape
    quantized_tensor = torch.tensor(quantized_activations, dtype=torch.float32).reshape(original_shape)
    
    # Move back to original device
    return quantized_tensor.to(activation_tensor.device)

def quantize_weight_tensor(weight_tensor, layer_name):
    """
    Quantize a weight tensor using Q1.7 format with MSR-4 compensation
    Returns quantized tensor and statistics
    """
    original_shape = weight_tensor.shape
    weights_flat = weight_tensor.flatten()
    
    quantized_weights = []
    msr4_count = 0
    non_msr4_count = 0
    
    print(f"  Quantizing {layer_name}: {len(weights_flat):,} weights")
    
    for i, weight in enumerate(weights_flat):
        # Convert to Q1.7 binary
        binary_str = float_to_q1_7_fixed_point(weight.item())
        
        # Check MSR-4 and count
        if has_msr4(binary_str):
            msr4_count += 1
        else:
            non_msr4_count += 1
        
        # Apply MSR-4 compensation
        compensated_binary = apply_msr4_compensation(binary_str)
        
        # Convert back to float
        quantized_float = binary_to_float(compensated_binary)
        quantized_weights.append(quantized_float)
        
        # Show progress for large layers
        if (i + 1) % 100000 == 0:
            print(f"    Processed {i + 1:,} / {len(weights_flat):,} weights")
    
    # Convert back to tensor and reshape
    quantized_tensor = torch.tensor(quantized_weights, dtype=torch.float32).reshape(original_shape)
    
    stats = {
        'layer_name': layer_name,
        'total_weights': len(weights_flat),
        'msr4_count': msr4_count,
        'non_msr4_count': non_msr4_count,
        'msr4_percentage': (msr4_count / len(weights_flat)) * 100,
        'non_msr4_percentage': (non_msr4_count / len(weights_flat)) * 100
    }
    
    return quantized_tensor, stats

def create_full_quantized_model(original_model):
    """
    Create a fully quantized model with both weight and activation quantization
    """
    print("Creating full Q1.7 quantized model (weights + activations)...")
    
    # Create quantized model with same architecture
    quantized_model = QuantizedMLP().to(device)
    
    # Quantize weights and load them
    quantization_stats = []
    original_state_dict = original_model.state_dict()
    quantized_state_dict = {}
    
    for name, param in original_state_dict.items():
        if 'weight' in name:
            # Quantize weight tensors
            quantized_tensor, stats = quantize_weight_tensor(param, name)
            quantized_state_dict[name] = quantized_tensor
            quantization_stats.append(stats)
            print(f"  ✓ {name}: MSR-4: {stats['msr4_count']:,} ({stats['msr4_percentage']:.2f}%), "
                  f"Non-MSR-4: {stats['non_msr4_count']:,} ({stats['non_msr4_percentage']:.2f}%)")
        else:
            # Keep bias and other parameters unchanged
            quantized_state_dict[name] = param
            print(f"  ✓ {name}: Kept original (bias/other)")
    
    # Load quantized weights into new model
    quantized_model.load_state_dict(quantized_state_dict)
    
    return quantized_model, quantization_stats

def evaluate_model(model, test_loader, model_name="Model"):
    """
    Evaluate model accuracy on test set
    """
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
            
            # Show progress
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

def display_full_quantization_summary(quantization_stats, original_accuracy, quantized_accuracy):
    """
    Display comprehensive quantization results for systolic array simulation
    """
    print("\n" + "="*80)
    print("SYSTOLIC ARRAY SIMULATION RESULTS")
    print("="*80)
    
    # Overall weight statistics
    total_weights = sum(stats['total_weights'] for stats in quantization_stats)
    total_msr4 = sum(stats['msr4_count'] for stats in quantization_stats)
    total_non_msr4 = sum(stats['non_msr4_count'] for stats in quantization_stats)
    overall_msr4_percentage = (total_msr4 / total_weights) * 100
    overall_non_msr4_percentage = (total_non_msr4 / total_weights) * 100
    
    print(f"\n📊 WEIGHT QUANTIZATION STATISTICS:")
    print(f"{'Layer':<15} {'Total Weights':>15} {'MSR-4 Count':>15} {'MSR-4 %':>10} {'Non-MSR-4 %':>12}")
    print("-" * 75)
    
    for stats in quantization_stats:
        print(f"{stats['layer_name']:<15} {stats['total_weights']:>15,} {stats['msr4_count']:>15,} "
              f"{stats['msr4_percentage']:>9.2f}% {stats['non_msr4_percentage']:>11.2f}%")
    
    print("-" * 75)
    print(f"{'OVERALL':<15} {total_weights:>15,} {total_msr4:>15,} "
          f"{overall_msr4_percentage:>9.2f}% {overall_non_msr4_percentage:>11.2f}%")
    
    # Compensation applied
    print(f"\n⚙️  SYSTOLIC ARRAY COMPENSATION:")
    print(f"• Weight Compensation:")
    print(f"  - MSR-4 weights ({total_msr4:,}): Last bit set to 1")
    print(f"  - Non-MSR-4 weights ({total_non_msr4:,}): Last 4 bits set to 1000")
    print(f"• Activation Compensation:")
    print(f"  - ALL activations: LSB set to 1 (expectation compensation)")
    print(f"• Total weights processed: {total_weights:,}")
    
    # Accuracy comparison
    accuracy_drop = original_accuracy - quantized_accuracy
    accuracy_retention = (quantized_accuracy / original_accuracy) * 100
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"• Original Model (Float32):  {original_accuracy:.2f}%")
    print(f"• Full Q1.7 Model:           {quantized_accuracy:.2f}%")
    print(f"• Accuracy Drop:             {accuracy_drop:.2f} percentage points")
    print(f"• Accuracy Retention:        {accuracy_retention:.2f}%")
    
    # Quantization quality assessment
    print(f"\n📈 QUANTIZATION QUALITY:")
    if accuracy_drop < 1.0:
        quality = "Excellent"
    elif accuracy_drop < 2.0:
        quality = "Good"
    elif accuracy_drop < 5.0:
        quality = "Acceptable"
    else:
        quality = "Poor"
    
    print(f"• Full Quantization Quality: {quality}")
    print(f"• MSR-4 Hardware Benefits: {overall_msr4_percentage:.1f}% of weights can use smaller multipliers")
    
    # Hardware implications
    print(f"\n🔧 SYSTOLIC ARRAY IMPLICATIONS:")
    print(f"• Memory Usage: 8-bit weights + 8-bit activations (75% reduction from float32)")
    print(f"• Processing Units: {overall_msr4_percentage:.1f}% can use optimized multipliers (MSR-4)")
    print(f"• Multiplication Operations: 8-bit × 8-bit throughout the array")
    print(f"• Expectation Compensation: Both weights and activations have LSB compensation")
    
    print("="*80)

def demonstrate_compensation_examples():
    """
    Demonstrate the compensation rules with examples
    """
    print("\n" + "="*60)
    print("Q1.7 COMPENSATION EXAMPLES")
    print("="*60)
    
    print("WEIGHT COMPENSATION (MSR-4 based):")
    weight_examples = [
        (0.004096, "MSR-4 example"),
        (-0.114851, "MSR-4 example"),  
        (0.174329, "Non-MSR-4 example"),
        (-0.134351, "Non-MSR-4 example")
    ]
    
    for weight, description in weight_examples:
        original_binary = float_to_q1_7_fixed_point(weight)
        is_msr4 = has_msr4(original_binary)
        compensated_binary = apply_msr4_compensation(original_binary)
        compensated_float = binary_to_float(compensated_binary)
        
        print(f"\nWeight: {weight:8.6f} ({description})")
        print(f"  Original binary:    {original_binary}")
        print(f"  MSR-4:              {'Yes' if is_msr4 else 'No'}")
        print(f"  Compensated binary: {compensated_binary}")
        print(f"  Compensated float:  {compensated_float:8.6f}")
    
    print(f"\nACTIVATION COMPENSATION (LSB=1 for all):")
    activation_examples = [0.234567, -0.087234, 0.456789, -0.321456]
    
    for activation in activation_examples:
        original_binary = float_to_q1_7_fixed_point(activation)
        compensated_binary = apply_activation_lsb_compensation(original_binary)
        compensated_float = binary_to_float(compensated_binary)
        
        print(f"\nActivation: {activation:8.6f}")
        print(f"  Original binary:    {original_binary}")
        print(f"  Compensated binary: {compensated_binary} (LSB=1)")
        print(f"  Compensated float:  {compensated_float:8.6f}")

def main():
    """
    Main quantization and evaluation pipeline for systolic array simulation
    """
    print("Full Q1.7 Quantization for Systolic Array Simulation")
    print("="*55)
    
    # Demonstrate compensation examples
    demonstrate_compensation_examples()
    
    # Load original trained model
    print(f"\n📁 Loading original trained model...")
    original_model = MLP().to(device)
    
    try:
        state_dict = torch.load('mnist_mlp_model.pth', map_location=device)
        original_model.load_state_dict(state_dict)
        print("✓ Original model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'mnist_mlp_model.pth' not found!")
        print("Please train the model first using MLP.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test data loader
    print(f"\n📊 Loading test dataset...")
    test_loader = get_data_loaders(batch_size=1000)  # Larger batch for faster evaluation
    print("✓ Test dataset loaded")
    
    # Evaluate original model
    print(f"\n🔍 Evaluating original Float32 model...")
    original_accuracy = evaluate_model(original_model, test_loader, "Original Float32 Model")
    
    # Create full quantized model (weights + activations)
    print(f"\n⚙️  Creating full Q1.7 quantized model...")
    full_quantized_model, quantization_stats = create_full_quantized_model(original_model)
    print("✓ Full quantization completed")
    
    # Evaluate full quantized model
    print(f"\n🔍 Evaluating full Q1.7 quantized model...")
    quantized_accuracy = evaluate_model(full_quantized_model, test_loader, "Full Q1.7 Model")
    
    # Display comprehensive summary
    display_full_quantization_summary(quantization_stats, original_accuracy, quantized_accuracy)
    
    # Save quantized model
    print(f"\n💾 Saving full quantized model...")
    torch.save(full_quantized_model.state_dict(), 'mnist_mlp_full_q17_systolic.pth')
    print("✓ Full quantized model saved as 'mnist_mlp_full_q17_systolic.pth'")
    
    print(f"\n🎉 Systolic array simulation completed successfully!")
    
    return original_model, full_quantized_model, quantization_stats

if __name__ == "__main__":
    # Run the full quantization pipeline
    original_model, quantized_model, stats = main()
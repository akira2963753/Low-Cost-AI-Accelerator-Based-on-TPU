import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

np.random.seed(42)                  
torch.manual_seed(42)  

# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # First pooling layer: 2x2 max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # Second pooling layer: 2x2 max pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        # After conv1 + pool1: 28x28 -> 14x14 (with 6 channels)
        # After conv2 + pool2: 10x10 -> 5x5 (with 16 channels)
        self.fc1_input_size = 16 * 5 * 5  # 400
        
        # First fully connected layer
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        # Second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Output layer
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # First fully connected layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second fully connected layer with ReLU and dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x

# Data preprocessing and loading
def get_data_loaders(batch_size=64):
    # Define transforms for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load the test data
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        #with torch.no_grad():
            #model.conv1.weight.clamp_(-0.125,0.1172)
            #model.conv2.weight.clamp_(-0.125,0.1172)
            #model.fc1.weight.data.clamp_(-0.125,0.1172)
            #model.fc2.weight.data.clamp_(-0.125,0.1172)
            #model.fc3.weight.data.clamp_(-0.125,0.1172)
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 200 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc

# Function to visualize model architecture
def print_model_summary(model, input_size=(1, 1, 28, 28)):
    print("LeNet Architecture Summary:")
    print("=" * 50)
    
    # Create a dummy input to trace through the model
    dummy_input = torch.randn(input_size)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Manually trace through each layer
    x = dummy_input
    
    # Conv1 + Pool1
    x = model.pool1(F.relu(model.conv1(x)))
    print(f"After Conv1 + Pool1: {x.shape}")
    
    # Conv2 + Pool2
    x = model.pool2(F.relu(model.conv2(x)))
    print(f"After Conv2 + Pool2: {x.shape}")
    
    # Flatten
    x = x.view(x.size(0), -1)
    print(f"After Flatten: {x.shape}")
    
    # FC layers
    x = F.relu(model.fc1(x))
    print(f"After FC1: {x.shape}")
    
    x = F.relu(model.fc2(x))
    print(f"After FC2: {x.shape}")
    
    x = model.fc3(x)
    print(f"After FC3 (Output): {x.shape}")
    
    print("=" * 50)

# Main training loop
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.000055
    #learning_rate = 0.0001
    num_epochs = 10
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Initialize the model
    model = LeNet().to(device)
    print(f"Model architecture:\n{model}")
    
    # Print model summary
    print_model_summary(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        
        # Store history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    print(f'\nTraining completed!')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Visualize feature maps
    visualize_feature_maps(model, test_loader, device)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_lenet_model.pth')
    print("Model saved as 'mnist_lenet_model.pth'")
    
    return model

# Function to plot training history
def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('LeNet Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('LeNet Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Function to visualize feature maps
def visualize_feature_maps(model, test_loader, device):
    model.eval()
    
    # Get a sample image
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    sample_image = images[0:1].to(device)  # Take first image
    
    # Hook to capture feature maps
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_image)
    
    # Visualize original image
    plt.figure(figsize=(15, 4))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(sample_image.cpu().squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Conv1 feature maps (show first 3 channels)
    conv1_features = feature_maps['conv1'][0]  # Remove batch dimension
    for i in range(min(3, conv1_features.size(0))):
        plt.subplot(1, 4, i + 2)
        plt.imshow(conv1_features[i], cmap='gray')
        plt.title(f'Conv1 Feature {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show more Conv1 feature maps
    plt.figure(figsize=(12, 8))
    for i in range(min(6, conv1_features.size(0))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(conv1_features[i], cmap='gray')
        plt.title(f'Conv1 Channel {i+1}')
        plt.axis('off')
    plt.suptitle('Conv1 Feature Maps (6 channels)')
    plt.tight_layout()
    plt.show()

# Function to test with sample images
def test_sample_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select first num_samples
    sample_images = images[:num_samples]
    sample_labels = labels[:num_samples]
    
    # Make predictions
    with torch.no_grad():
        sample_images_gpu = sample_images.to(device)
        outputs = model(sample_images_gpu)
        _, predicted = torch.max(outputs, 1)
    
    # Plot results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f'True: {sample_labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the model
    trained_model = main()
    
    # Test with sample predictions
    print("\nTesting with sample images...")
    _, test_loader = get_data_loaders(batch_size=64)
    test_sample_predictions(trained_model, test_loader, device)
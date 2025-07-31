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

# Define the AlexNet model adapted for MNIST
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (5 convolutional layers)
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            # Second convolutional layer
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            
            # Third convolutional layer
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3
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
        
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Data preprocessing and loading
def get_data_loaders(batch_size=64):
    # Define transforms for the training and test sets
    # AlexNet typically uses data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to slightly larger size for better feature extraction
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load the test data
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
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

# Main training loop
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 15  # Slightly more epochs for CNN
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Initialize the model
    model = AlexNet().to(device)
    print(f"Model architecture:\n{model}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
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
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print(f'\nTraining completed!')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_alexnet_model.pth')
    print("Model saved as 'mnist_alexnet_model.pth'")
    
    return model

# Function to plot training history
def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
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
        sample_images = sample_images.to(device)
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # Note: images are resized to 32x32, so we need to handle this
        img = sample_images[i].cpu().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {sample_labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')
    
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
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Register hooks for convolutional layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot feature maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, fmap in enumerate(feature_maps):
        if i >= 6:  # Only show first 6 conv layers
            break
        
        # Take first 16 channels of the feature map
        fmap_display = fmap[0, :16].cpu()  # Shape: [16, H, W]
        
        # Create a grid of feature maps
        grid_size = 4
        combined = torch.zeros(grid_size * fmap_display.shape[1], grid_size * fmap_display.shape[2])
        
        for j in range(min(16, fmap_display.shape[0])):
            row = j // grid_size
            col = j % grid_size
            h_start = row * fmap_display.shape[1]
            h_end = h_start + fmap_display.shape[1]
            w_start = col * fmap_display.shape[2]
            w_end = w_start + fmap_display.shape[2]
            combined[h_start:h_end, w_start:w_end] = fmap_display[j]
        
        axes[i].imshow(combined, cmap='viridis')
        axes[i].set_title(f'Conv Layer {i+1} Features')
        axes[i].axis('off')
    
    # Show original image
    if len(feature_maps) < 6:
        axes[5].imshow(sample_image[0, 0].cpu(), cmap='gray')
        axes[5].set_title('Original Image')
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the model
    trained_model = main()
    
    # Test with sample predictions
    print("\nTesting with sample images...")
    _, test_loader = get_data_loaders(batch_size=64)
    test_sample_predictions(trained_model, test_loader, device)
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    visualize_feature_maps(trained_model, test_loader, device)
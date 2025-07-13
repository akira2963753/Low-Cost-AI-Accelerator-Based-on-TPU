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

# Define the MLP model
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
    learning_rate = 0.0001
    num_epochs = 10
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Initialize the model
    model = MLP().to(device)
    print(f"Model architecture:\n{model}")
    
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
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_mlp_model.pth')
    print("Model saved as 'mnist_mlp_model.pth'")
    
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
        plt.imshow(sample_images[i].cpu().squeeze(), cmap='gray')
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
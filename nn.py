import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, learning_rate=0.001):
        super(SimpleNeuralNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Forward pass
        x = x.view(x.size(0), -1)  # Flatten input tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, num_epochs, device):
        self.to(device)  # Move model to device
        self.train()  # Set model to training mode
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self(images)
                
                # Compute loss and backpropagate
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                print(f"Loss: {loss.item()}")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, device):
        self.load_state_dict(torch.load(file_path))
        self.to(device)
        print(f"Model loaded from {file_path}")

    def evaluate_model(self, test_loader, device):
        self.to(device)  # Move model to device
        self.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# -------------------------------
# Example Usage

# Hyperparameters
input_dim = 64 * 64 * 3  # Assuming input image size is 64x64x3
hidden_dim = 512  # Hidden layer size
num_classes = 29  # Number of classes (sign language letters)
learning_rate = 0.001
num_epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation and dataset preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

train_data_path = "archive/asl_alphabet_train"
test_data_path = "archive/asl_alphabet_test"
model_save_path = "training_files/simple_nn_model.pth"

# Check paths
if not os.path.exists(train_data_path):
    raise ValueError(f"Training data path {train_data_path} does not exist!")
if not os.path.exists(test_data_path):
    raise ValueError(f"Test data path {test_data_path} does not exist!")

# Load datasets
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create the model instance
model = SimpleNeuralNetwork(input_dim, hidden_dim, num_classes, learning_rate)

# Train or load model
if os.path.exists(model_save_path):
    print("Model already exists, skipping training.")
    model.load_model(model_save_path, device)
else:
    print("Training the model...")
    model.train_model(train_loader, num_epochs, device)
    model.save_model(model_save_path)

# Evaluate the model
print("Evaluating the model...")
model.evaluate_model(test_loader, device)

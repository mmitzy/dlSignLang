import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes, learning_rate=0.001):
        super(LogisticRegressionModel, self).__init__()
        # Fully connected layer
        self.fc = nn.Linear(input_dim, num_classes)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Flatten the input tensor
        return self.fc(x.view(x.size(0), -1))  # Flatten (batch_size, channels, height, width) -> (batch_size, input_dim)

    def train_model(self, train_loader, num_epochs, device):
        self.to(device)  # Move model to the specified device 
        self.train()  # Set the model to training mode
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self(images)
                
                # Compute the loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                print(f"Loss: {loss.item()}")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    def save_model(self, file_path):
        # Save model weights
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, device):
        # Load model weights from a file
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, weights_only=True))
            self.to(device)  # Move model to the specified device 
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model file not found at {file_path}, initializing a new model.")

    def evaluate_model(self, test_loader, device):
        self.to(device)  # Move model to the specified device 
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():  
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(images)
                
                # Get the predicted class with highest probability
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# -------------------------------
# Example Usage

# Hyperparameters
input_dim = 64 * 64 * 3 
num_classes = 29  
learning_rate = 0.001
num_epochs = 5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformation and Datasets
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor()  
])

# Define paths
train_data_path = "archive/asl_alphabet_train"
test_data_path = "archive/asl_alphabet_test"
model_save_path = "training_files/logistic_regression_model.pth"  

# Check the paths
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
model = LogisticRegressionModel(input_dim, num_classes, learning_rate)

# Train the model if we don't have a saved model
if not os.path.exists(model_save_path):
    print("Training the model...")
    model.train_model(train_loader, num_epochs, device)

    # Save the trained model
    model.save_model(model_save_path)
else:
    print("Model already exists, skipping training.")
    # Load the model if it exists
    model.load_model(model_save_path, device)

# Load the model from file (later)
print("Loading the model...")
model.load_model(model_save_path, device)

# Evaluate the model
print("Evaluating the model...")
model.evaluate_model(test_loader, device)

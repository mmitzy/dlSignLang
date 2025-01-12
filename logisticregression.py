import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        self.to(device)
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for later visualization
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
        
        # Create a figure with subplots
        plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=285)
        plt.title('Confusion Matrix (Results)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        
        # # 2. Per-class Accuracy
        # plt.subplot(2, 2, 2)
        # class_accuracy = cm.diagonal() / cm.sum(axis=1)
        # plt.bar(range(len(class_accuracy)), class_accuracy * 100)
        # plt.title('Per-class Accuracy')
        # plt.xlabel('Class')
        # plt.ylabel('Accuracy (%)')
        
        # # 3. Prediction Confidence Distribution
        # plt.subplot(2, 2, 3)
        # max_probs = np.max(all_probabilities, axis=1)
        # plt.hist(max_probs, bins=50)
        # plt.title('Prediction Confidence Distribution')
        # plt.xlabel('Confidence')
        # plt.ylabel('Count')
        
        # # 4. ROC Curve (for multi-class, using one-vs-rest)
        # plt.subplot(2, 2, 4)
        # all_labels_onehot = np.eye(len(np.unique(all_labels)))[all_labels]
        # fpr = dict()
        # tpr = dict()
        # for i in range(len(np.unique(all_labels))):
        #     fpr[i], tpr[i], _ = roc_curve(all_labels_onehot[:, i], 
        #                                 np.array(all_probabilities)[:, i])
        #     plt.plot(fpr[i], tpr[i], label=f'Class {i}')
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curves (One-vs-Rest)')
        # plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300)
        print("Evaluation plots saved as 'model_evaluation.png'")
        plt.show()
        
        return accuracy

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

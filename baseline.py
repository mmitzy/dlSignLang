import random
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BaselineModel:
    def __init__(self, data_folder):
        self.labels = self._get_labels_from_folder(data_folder)  # Get label folder names
        self.uniform_label = random.choice(self.labels)  # Select a random label from the training set

    def _get_labels_from_folder(self, folder_path):
        # Get all label names from subfolders (class labels) as strings
        return [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    def predict(self, num_samples):
        # Predict the same label for all samples (the uniform label)
        return np.array([self.uniform_label] * num_samples)

    def evaluate(self, true_labels):
        # Compare predicted labels with true labels
        predictions = self.predict(len(true_labels))
        accuracy = np.mean(predictions == true_labels)  # Correct predictions / Total predictions
        return accuracy

    def load_test_data(self, test_data_path):
        # Data transformation and loading test dataset
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.ToTensor()  # Convert image to a tensor
        ])
        
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return test_dataset

if __name__ == "__main__":
    train_data_folder = "archive/asl_alphabet_train"
    test_data_folder = "archive/asl_alphabet_test"
    baseline_model = BaselineModel(train_data_folder)

    # Load the test dataset
    test_dataset = baseline_model.load_test_data(test_data_folder)

    # Extract true labels as strings (class names, not integers)
    true_labels = np.array([test_dataset.classes[label] for _, label in test_dataset])  # Convert integer to class name

    # Ensure that the true labels are correctly loaded
    print(f"True labels: {true_labels[:28]}")  # Check first 10 true labels

    # Evaluate the baseline model
    accuracy = baseline_model.evaluate(true_labels) * 100
    print(f"Baseline accuracy with label '{baseline_model.uniform_label}': {accuracy:.2f}%")

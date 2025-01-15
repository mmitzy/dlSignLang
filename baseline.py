import random
import numpy as np
import os
from torchvision import datasets, transforms

class BaselineModel:
    def __init__(self, data_folder):
        # Initialize labels from the folder structure
        self.labels = self._get_labels_from_folder(data_folder)
        self.uniform_label = random.choice(self.labels)

    def _get_labels_from_folder(self, folder_path):
        # Extract folder names as labels
        return [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    def predict(self, num_samples):
        # Predict a single uniform label for all samples
        return np.array([self.uniform_label] * num_samples)

    def evaluate(self, test_dataset):
        # Extract true labels as strings (class names from the folder structure)
        true_labels = np.array([test_dataset.classes[label] for label in test_dataset.targets])

        # Get baseline predictions
        predictions = self.predict(len(true_labels))
        
        # Compute accuracy
        accuracy = np.mean(predictions == true_labels)
        return accuracy

if __name__ == "__main__":
    # Paths
    test_data_path = "archive/asl_alphabet_test"

    # Data Transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load Test Dataset
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

    baseline_model = BaselineModel(test_data_path)
    accuracy = baseline_model.evaluate(test_dataset) * 100
    print(f"Baseline accuracy with label '{baseline_model.uniform_label}': {accuracy:.2f}%")

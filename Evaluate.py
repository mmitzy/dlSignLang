import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Evaluate():
    def evaluate_model(self, test_loader, device):
            self.to(device)  # Move model to device
            self.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            all_labels = []
            all_predicitions = []
            
            with torch.no_grad():  # Disable gradient computation
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predicitions.extend(predicted.cpu().numpy())
            
            accuracy = 100 * correct / total
            print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

            cm = confusion_matrix(all_labels, all_predicitions)
            plt.figure(figsize=(15,12))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=285)
            plt.xlabel('Predicted label', labelpad=20)
            plt.title("Confusion Matrix(Results)")
            plt.tight_layout()
            plt.savefig("confusion_matrix_nn.png", dpi=300)
            plt.show()
            print("Confusion matrix saved as 'confusion_matrix.png'.")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class CNN(nn.Module):

    def __init__(self):
        num_classes=10

        # varying the kernel size and other variables to see effect on Model
        kern_conv=3
        stride_conv=1
        pad_conv=1

        super(CNN, self).__init__()

        # Define the VGG11 architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=kern_conv, stride=stride_conv, padding=pad_conv),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def train_model(self, train_loader, device, epochs=10, learning_rate=0.01, momentum=0.9):
            
            self.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

            for epoch in range(epochs):
                self.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")



    @staticmethod
    def evaluate_model(model, test_loader, model_name="VGG11"):
        """
        Evaluate the model on the test dataset.
        
        Args:
        - model (nn.Module): Trained model.
        - test_loader (DataLoader): DataLoader for test data.
        
        Returns:
        - None (prints metrics and confusion matrix)
        """
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation
            for features, labels in test_loader:
                features, labels = features.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)  # Predicted class index
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        class_names = [f"Class {i}" for i in range(10)]

        # Terminal Output
        print(f"###################################################################\n{model_name} Model RESULTS:\n")
        print("Model Accuracy: ", accuracy, "\n")
        print("Confusion Matrix:\n", cm, "\n")
        print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names), "\n")

        # Save Confusion Matrix as a Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for the {model_name} Model')
        plt.savefig(f"./results/confusion_matrices/cm_{model_name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save Results to File
        output_file_path = "./results/evaluation_results.txt"
        with open(output_file_path, "a") as file:
            file.write(f"###################################################################\n{model_name} Model RESULTS:\n\n")
            file.write(f"Model Accuracy: {accuracy:.3f}\n\n")
            file.write(f"Confusion Matrix:\n{cm}\n\n")
            file.write(f"Classification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}\n\n")





# Run the training code only if the script is executed directly
if __name__ == "__main__":

    # Load train data and labels from .pt file
    images, labels = torch.load('./data/selected/selected_train_dataset.pt')    # Load the saved data
    train_dataset = TensorDataset(images, labels)                               # Create TensorDataset
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)       # Load into a DataLoader

    # Build the CNN Models
    cnn = CNN()                                                     # default architecture model AKA 3 layers AKA hidden_layers=[512, 512]
   
    # Train CNN Models (10 epochs for efficiency)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn.train_model(train_loader, device, epochs=45)

    # Save the models
    torch.save(cnn.state_dict(), './models/cnn/cnn.pth')

    # Display DONE message
    print("\nVGG11 CNN Models trained and saved successfully. See ./models or run the evaluate_models.py script\n")





    # Load test data and labels from .pt file
    images, labels = torch.load('./data/selected/selected_test_dataset.pt')     # Load the saved data
    test_dataset = TensorDataset(images, labels)                               # Create TensorDataset
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)       # Load into a DataLoader

    CNN.evaluate_model(cnn, test_loader)
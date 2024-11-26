import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Define the MLP model
class MLP(nn.Module):

    def __init__(self, hidden_layers=[512, 512]):
        """
        Create an MLP with a variable number of hidden layers.

        Args:
        - hidden_layers (list of int): List of sizes for the hidden layers.

        Important:
            The standard 3-layer architecture outlined in the project description is built when hidden_layers=[512, 512], which is the default value:
                - Linear(50, 512) - ReLU
                - Linear(512, 512) - BatchNorm(512) - ReLU
                - Linear(512, 10)
        """
        input_features=50
        output_size=10

        super(MLP, self).__init__()
        layers = []
    
        # Input Layer: Linear -> ReLU
        layers.append(nn.Linear(input_features, hidden_layers[0]))
        layers.append(nn.ReLU())
        input_features = hidden_layers[0]

        # Hidden Layers: Linear -> BatchNorm -> ReLU
        for h in hidden_layers[1:]:
            layers.append(nn.Linear(input_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            input_features = h
        
        # Output layer: Linear
        layers.append(nn.Linear(input_features, output_size))
        
        #print("Layers List:", layers)

        # Register the layers as a sequential model
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        # x = self.relu1(self.fc1(x))
        # x = self.relu2(self.batchnorm(self.fc2(x)))
        # x = self.fc3(x)
        # return x
        return self.model(x)


    def train_model(self, train_loader, epochs, lr=0.01, momentum=0.9):
            """
            Train the model using the given training data.
            
            Args:
            - train_loader (DataLoader): DataLoader for training data.
            - epochs (int): Number of training epochs.
            - lr (float): Learning rate.
            - momentum (float): Momentum for SGD optimizer.
            """
            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                for features, labels in train_loader:
                    # Forward pass
                    outputs = self(features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


    @staticmethod
    def evaluate_model(model, test_loader, model_name="MLP"):
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
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)  # Predicted class index
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        #----------------------------------------------------------------------
        # Output to Terminal
        print(f"###################################################################\n{model_name} Model RESULTS:\n")
        print("Model Accuracy: ", accuracy_score(all_labels, all_preds),"\n")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm)

        # Classification Report
        class_names = [f"Class {i}" for i in range(10)]
        print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))


        #-----------------------------------------------------------------------------------------------------------------
        # Saving Matrix heatmap as png images at ./results

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix for the MLP Model')
        plt.savefig(f"./results/confusion_matrices/cm_{model_name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()
        

        #-----------------------------------------------------------------------------------------------------------------
        # Outputs to evaluation_results.txt file

        # Define the file path for the results
        output_file_path = "./results/evaluation_results.txt"

        # Open the file in append mode
        with open(output_file_path, "a") as file:
            file.write(f"###################################################################\n{model_name} Model RESULTS:\n\n")
            file.write(f"Model Accuracy: {accuracy_score(all_labels, all_preds):.3f}\n\n")
            file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm}\n\n")
            file.write(f"Classification Report:\n {classification_report(all_labels, all_preds, target_names=class_names)}\n\n")



# Run the training code only if the script is executed directly
if __name__ == "__main__":

    # Load train data from .pt file
    train_features = torch.load('./data/features/train/train_features_pca.pt',  weights_only=True)

    # Load train labels from .npy file and convert it to a tensor
    train_labels = np.load('./data/features/train/train_labels.npy')
    train_labels = torch.tensor(train_labels, dtype=torch.long) #dtype=torch.long specifies that the labels are integers

    # Combine features and labels into one TensorDataset and load into a DataLoader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Build the MLP Models
    mlp = MLP()                                                     # default architecture model AKA 3 layers AKA hidden_layers=[512, 512]
    mlp_2_layers = MLP(hidden_layers=[512])                         # model with only 2 layers: Linear(50, 512) -> ReLU -> Linear(512, 10)
    mlp_6_layers = MLP(hidden_layers=[512, 512, 512, 512, 512])     # model with extra layers (6 total): Linear(50, 512) -> ReLU -> 4x[ Linear(512, 10) -> BatchNorm(512) -> ReLU ] -> Linear(512, 10)
    mlp_128_layer_size = MLP(hidden_layers=[128, 128])              # model with hidden layer size 128: Linear(50, 128) -> ReLU -> Linear(128, 10) -> BatchNorm(128) -> ReLU -> Linear(128, 10)
    mlp_2048_layer_size = MLP(hidden_layers=[2048, 2048])           # model with hidden layer size 2048: Linear(50, 2048) -> ReLU -> Linear(2048, 10) -> BatchNorm(2048) -> ReLU -> Linear(2048, 10)

    # Train MLP Models (10 epochs for efficiency)
    mlp.train_model(train_loader, epochs=10, lr=0.01, momentum=0.9)
    mlp_2_layers.train_model(train_loader, epochs=10, lr=0.01, momentum=0.9)
    mlp_6_layers.train_model(train_loader, epochs=10, lr=0.01, momentum=0.9)
    mlp_128_layer_size.train_model(train_loader, epochs=10, lr=0.01, momentum=0.9)
    mlp_2048_layer_size.train_model(train_loader, epochs=10, lr=0.01, momentum=0.9)


    # Save the models
    torch.save(mlp.state_dict(), './models/mlp/mlp.pth')
    torch.save(mlp_2_layers.state_dict(), './models/mlp/mlp_2_layers.pth')
    torch.save(mlp_6_layers.state_dict(), './models/mlp/mlp_6_layers.pth')
    torch.save(mlp_128_layer_size.state_dict(), './models/mlp/mlp_128_layer_size.pth')
    torch.save(mlp_2048_layer_size.state_dict(), './models/mlp/mlp_2048_layer_size.pth')

    # Display DONE message
    print("\nMulti-Layer Perceptron Models trained and saved successfully. See ./models or run the evaluate_models.py script\n")




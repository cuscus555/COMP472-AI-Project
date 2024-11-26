import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib                               # import for loading the saved models
import seaborn as sns                       # import for constructing confusion matrix heatmap
import matplotlib.pyplot as plt             # import for displaying confusion matrix heatmap

# Import the class definitions:
from naive_bayes import GaussianNaiveBayes  
from decision_tree import DecisionTree
from mlp import MLP
from cnn import CNN



#########################################################################################################################
# Load the test dataset

test_features = np.load('./data/features/test/test_features_pca.npy')
test_labels = np.load('./data/features/test/test_labels.npy')

# Load the test data tensors for the MLP models
test_features_tensor = torch.load('./data/features/test/test_features_pca.pt',  weights_only=True)
test_labels_tensor = torch.tensor(test_labels)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the test data tensors for the CNN models
images, labels = torch.load('./data/selected/selected_test_dataset.pt')             # Load the saved data
test_dataset_cnn = TensorDataset(images, labels)                                    # Create TensorDataset
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=128, shuffle=False)       # Load into a DataLoader


#########################################################################################################################
# Load the saved models


# Load the Gaussian Naive Bayes models
gnb = GaussianNaiveBayes.load_model("./models/naive_bayes/GNB.pkl")  # Loading my GNB model by using the static method load_model from the class definition
scikit_gnb = joblib.load("./models/naive_bayes/scikit_GNB.pkl")

# Load the Decision Tree models
dt = DecisionTree.load_model("./models/decision_tree/DT.pkl")  # Loading my DT model by using the static method load_model from the class definition
dt_100_layer = DecisionTree.load_model("./models/decision_tree/DT_100_layer.pkl")
dt_16_layer = DecisionTree.load_model("./models/decision_tree/DT_16_layer.pkl")
scikit_dt = joblib.load("./models/decision_tree/scikit_DT.pkl")

# Load the MLP models
mlp = MLP()
mlp_2_layers = MLP(hidden_layers=[512])                         
mlp_6_layers = MLP(hidden_layers=[512, 512, 512, 512, 512])    
mlp_128_layer_size = MLP(hidden_layers=[128, 128])              
mlp_2048_layer_size = MLP(hidden_layers=[2048, 2048])  

mlp.load_state_dict(torch.load('./models/mlp/mlp.pth', weights_only=True)) # the loading of the model is done like this for more robustness
mlp_2_layers.load_state_dict(torch.load('./models/mlp/mlp_2_layers.pth', weights_only=True)) 
mlp_6_layers.load_state_dict(torch.load('./models/mlp/mlp_6_layers.pth', weights_only=True)) 
mlp_128_layer_size.load_state_dict(torch.load('./models/mlp/mlp_128_layer_size.pth', weights_only=True)) 
mlp_2048_layer_size.load_state_dict(torch.load('./models/mlp/mlp_2048_layer_size.pth', weights_only=True)) 

# Load the CNN Models
cnn = CNN()
cnn.load_state_dict(torch.load('./models/cnn/cnn.pth', weights_only=True))


#########################################################################################################################
# Feed the test data to the loaded models

gnb_predictions = gnb.predict(test_features)
scikit_gnb_predictions = scikit_gnb.predict(test_features)

dt_predictions = dt.predict(test_features)
dt_100_predictions = dt_100_layer.predict(test_features)
dt_16_predictions = dt_16_layer.predict(test_features)
scikit_dt_predictions = scikit_dt.predict(test_features)




#########################################################################################################################
# Display the results

class_names = [f"Class {i}" for i in range(10)]     #get class names 

# Generate confusion matrices for each model
cm_gnb = confusion_matrix(test_labels, gnb_predictions)
cm_scikit_gnb = confusion_matrix(test_labels, scikit_gnb_predictions)
cm_dt = confusion_matrix(test_labels, dt_predictions)
cm_dt_100 = confusion_matrix(test_labels, dt_100_predictions)
cm_dt_16 = confusion_matrix(test_labels, dt_16_predictions)
cm_scikit_dt = confusion_matrix(test_labels, scikit_dt_predictions)


#-----------------------------------------------------------------------------------------------------------------
# Outputs to terminal

print("###################################################################\nGNB Model RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, gnb_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_gnb)
print("\nClassification Report:\n", classification_report(test_labels, gnb_predictions, target_names=class_names))

print("###################################################################\nScikit GNB Model RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, scikit_gnb_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_scikit_gnb)
print("\nClassification Report (Scikit-Learn):\n", classification_report(test_labels, scikit_gnb_predictions, target_names=class_names))

print("###################################################################\nDT Model (50 layers) RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, dt_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_dt)
print("\nClassification Report:\n", classification_report(test_labels, dt_predictions, target_names=class_names))

print("###################################################################\nDT Model (100 layers) RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, dt_100_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_dt_100)
print("\nClassification Report:\n", classification_report(test_labels, dt_100_predictions, target_names=class_names))

print("###################################################################\nDT Model (16 layers) RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, dt_16_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_dt_16)
print("\nClassification Report:\n", classification_report(test_labels, dt_16_predictions, target_names=class_names))

print("###################################################################\nScikit DT Model RESULTS:\n")
print("Model Accuracy: ", accuracy_score(test_labels, scikit_dt_predictions),"\n")
print("Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n", cm_scikit_dt)
print("\nClassification Report:\n", classification_report(test_labels, scikit_dt_predictions, target_names=class_names))



#-----------------------------------------------------------------------------------------------------------------
# Outputs to evaluation_results.txt file

# Define the file path for the results
output_file_path = "./results/evaluation_results.txt"

# Open the file in write mode
with open(output_file_path, "w") as file:
    # Write the output
    file.write("###################################################################\nGNB Model RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, gnb_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_gnb}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, gnb_predictions, target_names=class_names)}\n\n")

    file.write("###################################################################\nScikit GNB Model RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, scikit_gnb_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_scikit_gnb}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, scikit_gnb_predictions, target_names=class_names)}\n\n")

    file.write("###################################################################\nDT Model (50 layers) RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, dt_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_dt}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, dt_predictions, target_names=class_names)}\n\n")

    file.write("###################################################################\nDT Model (100 layers) RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, dt_100_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_dt_100}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, dt_100_predictions, target_names=class_names)}\n\n")

    file.write("###################################################################\nDT Model (16 layers) RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, dt_16_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_dt_16}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, dt_16_predictions, target_names=class_names)}\n\n")

    file.write("###################################################################\nScikit DT Model RESULTS:\n\n")
    file.write(f"Model Accuracy: {accuracy_score(test_labels, scikit_dt_predictions):.3f}\n\n")
    file.write(f"Confusion Matrix: Each column represents predicted labels (0-9) in left-to-right order, each row represents true labels (0-9) in top-to-bottom order\n {cm_scikit_dt}\n\n")
    file.write(f"Classification Report:\n {classification_report(test_labels, scikit_dt_predictions, target_names=class_names)}\n\n")


#-----------------------------------------------------------------------------------------------------------------
# Saving Matrices as png images at ./results

# Plot confusion matrix heatmap with labels and save it as png in ./results/confusion_matrices/
plt.figure(figsize=(10, 8))
sns.heatmap(cm_gnb, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for my GNB Model")
plt.savefig("./results/confusion_matrices/cm_gnb.png", dpi=300, bbox_inches="tight")  # Save as PNG
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_scikit_gnb, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for the Scikit GNB Model")
plt.savefig("./results/confusion_matrices/cm_scikit_gnb.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for the DT Model (50 layers)")
plt.savefig("./results/confusion_matrices/cm_dt.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_dt_100, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for the DT Model (16 layers)")
plt.savefig("./results/confusion_matrices/cm_dt_16_layer.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_dt_16, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for the DT Model (100 layers)")
plt.savefig("./results/confusion_matrices/cm_dt_100_layer.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(cm_scikit_dt, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for the Scikit DT Model")
plt.savefig("./results/confusion_matrices/cm_scikit_dt.png", dpi=300, bbox_inches="tight")
plt.close()



#------------------------------------------------------------------------------------------------------------------
# Calling the custom evaluation methods for MLP and CNN

# Evaluate the MLP Models
MLP.evaluate_model(mlp, test_loader, "Default_3_Layer_MLP")
MLP.evaluate_model(mlp_2_layers, test_loader, "2_Layer_MLP")
MLP.evaluate_model(mlp_6_layers, test_loader, "6_Layer_MLP")
MLP.evaluate_model(mlp_128_layer_size, test_loader, "128_Hidden_Layer_Size_MLP")
MLP.evaluate_model(mlp_2048_layer_size, test_loader, "2048_Hidden_Layer_Size_MLP")

# Evaluate the CNN Models
CNN.evaluate_model(cnn, test_loader_cnn, "Default_CNN")
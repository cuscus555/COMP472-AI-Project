import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



#######################################################################################
# FUNCTIONS
#######################################################################################

# Function to group images by class
def group_by_class(dataset):
    class_groups = {i: [] for i in range(10)}  # Dictionary to store images by class
    for idx, (image, label) in enumerate(dataset):
        class_groups[label].append(idx)  # Group by label
    return class_groups


# Save the dataset
def save_dataset(dataset, file_path):
    images, labels = [], []
    for img, label in dataset:
        images.append(img)
        labels.append(label)
    images = torch.stack(images)  # Combine all images into a single tensor
    labels = torch.tensor(labels)  # Combine all labels into a single tensor
    torch.save((images, labels), file_path)  # Save as a tuple
    print(f"Dataset saved to {file_path}")


# Function to extract features
def extract_features(loader, model, device):
    model = model.to(device)
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            output = model(images)
            features.append(output.view(output.size(0), -1))
    return torch.cat(features, dim=0)



# Save labels as .npy
def save_labels(loader, file_path):
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.cpu().numpy())  # Convert labels to NumPy array
    np.save(file_path, labels)
    return labels
    #print(f"Labels saved to {file_path}.npy")



#####################################################################################
#SELECT AND SAVE 5000+1000 CIFAR-10 images

# Define transformations (convert to tensors)
data_transforms_1 = transforms.Compose([
    transforms.ToTensor(), # Convert to PyTorch tensors
])

# Load CIFAR-10 training and testing datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms_1)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms_1)

# Group training and testing images by class
train_groups = group_by_class(train_dataset)
test_groups = group_by_class(test_dataset)

# Select first 500 training images and first 100 test images per class
selected_train_indices = [idx for class_idx in range(10) for idx in train_groups[class_idx][:500]]
selected_test_indices = [idx for class_idx in range(10) for idx in test_groups[class_idx][:100]]

# Create subsets using the selected indices
selected_train_dataset = Subset(train_dataset, selected_train_indices)
selected_test_dataset = Subset(test_dataset, selected_test_indices)

# Save the training and testing subsets
save_dataset(selected_train_dataset, './data/selected/selected_train_dataset.pt')
save_dataset(selected_test_dataset, './data/selected/selected_test_dataset.pt')




#####################################################################################
#SELECT AND SAVE 5000+1000 CIFAR-10 images + APPLY TRANSFORMATIONS: RESIZE AND NORMALIZE

data_transforms_2 = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to 224x224x3
    transforms.ToTensor(), # Convert to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet: values taken from ResNet documentation
])

# Load CIFAR-10 training and testing datasets
transformed_train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transforms_2)
transformed_test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=data_transforms_2)

# Group training and testing images by class
transformed_train_groups = group_by_class(transformed_train_dataset)
transformed_test_groups = group_by_class(transformed_test_dataset)

# Select first 500 training images and first 100 test images per class
transformed_train_indices = [idx for class_idx in range(10) for idx in transformed_train_groups[class_idx][:500]]
transformed_test_indices = [idx for class_idx in range(10) for idx in transformed_test_groups[class_idx][:100]]

# Create subsets using the selected indices
transformed_train_dataset = Subset(transformed_train_dataset, transformed_train_indices)
transformed_test_dataset = Subset(transformed_test_dataset, transformed_test_indices)

# Save the training and testing subsets
save_dataset(transformed_train_dataset, './data/resized&normalized/transformed_train_dataset.pt')
save_dataset(transformed_test_dataset, './data/resized&normalized/transformed_test_dataset.pt')





#####################################################################################
#PASS THROUGH RESNET-18

# Wrap subsets in DataLoader for batching   -->  AKA loading into memory for use
train_loader = DataLoader(transformed_train_dataset, batch_size=32, shuffle=False)  # I am not shuffling the train data here because it causes issues with the labeling, instead I will shuffle it right before the training step
test_loader = DataLoader(transformed_test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet-18 and remove last layer
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # or .IMAGENET1K_V1
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract features
train_features = extract_features(train_loader, resnet18, device)
test_features = extract_features(test_loader, resnet18, device)

# Save data
np.save('./data/features/train/train_features.npy', train_features.cpu().numpy())
np.save('./data/features/test/test_features.npy', test_features.cpu().numpy())
torch.save(train_features, './data/features/train/train_features.pt')
torch.save(test_features, './data/features/test/test_features.pt')

# Save labels
train_labels = save_labels(train_loader, './data/features/train/train_labels.npy')
test_labels = save_labels(test_loader, './data/features/test/test_labels.npy')


print(f"Train features shape: {train_features.shape}")  # Expect (5000, 512)
print(f"Test features shape: {test_features.shape}")    # Expect (1000, 512)




#####################################################################################
#PASS THROUGH PCA

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features.cpu().numpy())
test_features_pca = pca.transform(test_features.cpu().numpy())

# Save data
np.save('./data/features/train/train_features_pca.npy', train_features_pca)
np.save('./data/features/test/test_features_pca.npy', test_features_pca)
torch.save(torch.tensor(train_features_pca), './data/features/train/train_features_pca.pt')
torch.save(torch.tensor(test_features_pca), './data/features/test/test_features_pca.pt')

print(f"Train PCA features shape: {train_features_pca.shape}")
print(f"Test PCA features shape: {test_features_pca.shape}")



print(f"VALIDATION:")

print("Pre-processed data saved successfully!")

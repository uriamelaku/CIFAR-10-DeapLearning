#FCNN MODEL

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# 1. Loading images and preparing data (convert images to numerical arrays)
def load_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize the image to 32x32 pixels
    return np.array(img)  # Convert the image to a numerical array

# Augmentation for the training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a 50% chance
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image within a range of ±10 degrees
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # Randomly crop and resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the pixel values to [-1, 1]
])

# Simple augmentation for the test set
test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the pixel values to [-1, 1]
])


def process_images(dataframe, images_path, transform):
    images = []  # Initialize an empty list to store processed images
    for img_name in dataframe['id']:  # Loop through image IDs in the dataframe
        image_path = os.path.join(images_path, f"{img_name}.png")  # Construct the full image path
        img = Image.open(image_path).resize((32, 32))  # Open and resize the image to 32x32 pixels
        img_tensor = transform(img)  # Apply the specified transformations (e.g., normalization, augmentation)
        images.append(img_tensor)  # Add the transformed image tensor to the list
    return torch.stack(images)  # Stack all image tensors into a single tensor

# Paths to the training and test images
train_images_path = '/content/images'
test_images_path = '/content/images'

# Load the dataframes containing image IDs and labels
train_df = pd.read_csv('/content/train_split.csv')  # Training data
test_df = pd.read_csv('/content/test_split.csv')  # Test data

# Process images with transformations
X_train_tensor = process_images(train_df, train_images_path, train_transform)  # Training images
X_test_tensor = process_images(test_df, test_images_path, test_transform)  # Test images

# 2. Encoding labels (converting categorical labels to numerical labels)
label_encoder = LabelEncoder()  # Initialize the label encoder
train_labels_encoded = label_encoder.fit_transform(train_df['label'].values)  # Fit and transform training labels
test_labels_encoded = label_encoder.transform(test_df['label'].values)  # Transform test labels

# Convert encoded labels to PyTorch tensors
train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)  # Training labels tensor
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)  # Test labels tensor

# 3. The model class
class FNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNModel, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Fully connected layer: input -> hidden_dim
        self.bn1 = nn.BatchNorm1d(hidden_dim)       # Batch normalization for stabilizing training
        self.dropout1 = nn.Dropout(p=0.5)          # Dropout for regularization (50% chance to drop neurons)
        self.relu1 = nn.ReLU()                     # Activation function (ReLU)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer: hidden_dim -> hidden_dim
        self.bn2 = nn.BatchNorm1d(hidden_dim)         # Batch normalization
        self.dropout2 = nn.Dropout(p=0.5)            # Dropout
        self.relu2 = nn.ReLU()                       # Activation function (ReLU)

        # Third hidden layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer: hidden_dim -> hidden_dim
        self.bn3 = nn.BatchNorm1d(hidden_dim)         # Batch normalization
        self.dropout3 = nn.Dropout(p=0.5)            # Dropout
        self.relu3 = nn.ReLU()                       # Activation function (ReLU)

        # Output layer
        self.fc4 = nn.Linear(hidden_dim, output_dim)  # Fully connected layer: hidden_dim -> output_dim

    def forward(self, x):
        # Pass input through the first hidden layer
        x = self.relu1(self.dropout1(self.bn1(self.fc1(x))))

        # Pass through the second hidden layer
        x = self.relu2(self.dropout2(self.bn2(self.fc2(x))))

        # Pass through the third hidden layer
        x = self.relu3(self.dropout3(self.bn3(self.fc3(x))))

        # Output layer (no activation here, handled by CrossEntropyLoss)
        x = self.fc4(x)
        return x

# Model creation
input_dim = 32 * 32 * 3  # Images are RGB with 32x32 pixels, flattened to a single vector
hidden_dim = 256  # Number of neurons in each hidden layer
output_dim = len(np.unique(train_labels_encoded))  # Number of unique labels (output classes)
model = FNNModel(input_dim, hidden_dim, output_dim)  # Initialize the FNN model

# 4. Loss function and optimizer definition
criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Adam optimizer with a learning rate of 0.001 and weight decay for regularization

# 5. Early stopping setup
patience = 10  # Number of epochs to wait before stopping if no improvement
no_improvement_count = 0  # Counter for epochs without improvement
best_val_loss = float('inf')  # Initialize the best validation loss to infinity

# 6. Training setup
num_epochs = 200  # Maximum number of training epochs
train_losses, val_losses = [], []  # Lists to store training and validation losses
train_accuracies, val_accuracies = [], []  # Lists to store training and validation accuracies


for epoch in range(num_epochs):  # Loop through each epoch
    model.train()  # Set the model to training mode (activates Dropout, BatchNorm)
    optimizer.zero_grad()  # Clear gradients from the previous step

    # Forward pass: Compute the model's output for the training data
    output = model(X_train_tensor.view(X_train_tensor.size(0), -1))  # Flatten the input
    loss = criterion(output, train_labels_tensor)  # Compute the loss (CrossEntropyLoss)

    # Backward pass: Compute gradients
    loss.backward()
    optimizer.step()  # Update the model's parameters using the optimizer

    # Calculate training accuracy
    _, predicted_train = torch.max(output, 1)  # Get the predicted classes
    train_accuracy = accuracy_score(train_labels_tensor, predicted_train)  # Compare with true labels

    # Record training loss and accuracy
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)

    # Validation step
    model.eval()  # Set the model to evaluation mode (deactivates Dropout)
    with torch.no_grad():  # Disable gradient computation for efficiency
        # Forward pass on validation data
        output_val = model(X_test_tensor.view(X_test_tensor.size(0), -1))
        val_loss = criterion(output_val, test_labels_tensor)  # Compute validation loss
        _, predicted_val = torch.max(output_val, 1)  # Get predicted classes for validation
        val_accuracy = accuracy_score(test_labels_tensor, predicted_val)  # Compare with true labels

        # Record validation loss and accuracy
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)

    # Early stopping logic
    if val_loss.item() < best_val_loss:  # Check if validation loss improved
        best_val_loss = val_loss.item()  # Update the best validation loss
        no_improvement_count = 0  # Reset the counter
    else:
        no_improvement_count += 1  # Increment the counter if no improvement

    # Print progress for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")

    # Stop training if no improvement for 'patience' epochs
    if no_improvement_count >= patience:
        print("Early stopping triggered.")
        break


# Plotting graphs to visualize training and validation metrics

# Plotting loss for training and validation
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(train_losses, label="Train Loss")  # Plot training loss
plt.plot(val_losses, label="Validation Loss")  # Plot validation loss
plt.legend()  # Add a legend to differentiate the lines
plt.title("Train vs Validation Loss")  # Add a title to the plot
plt.show()  # Display the plot

# Plotting accuracy for training and validation
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(train_accuracies, label="Train Accuracy")  # Plot training accuracy
plt.plot(val_accuracies, label="Validation Accuracy")  # Plot validation accuracy
plt.legend()  # Add a legend to differentiate the lines
plt.title("Train vs Validation Accuracy")  # Add a title to the plot
plt.show()  # Display the plot


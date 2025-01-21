# Softmax Regression Model


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

# 1. Load images and prepare data
def load_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to 32x32
    return np.array(img).flatten()  # Convert to array and flatten

# Load image paths from training and testing CSV files
train_df = pd.read_csv('/content/train_split.csv')
test_df = pd.read_csv('/content/test_split.csv')

train_images = [load_image(os.path.join('/content/images', f"{img_name}.png")) for img_name in train_df['id']]
test_images = [load_image(os.path.join('/content/images', f"{img_name}.png")) for img_name in test_df['id']]

# Convert images into numerical arrays
X_train = np.array(train_images)
X_test = np.array(test_images)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df['label'].values)
test_labels_encoded = label_encoder.transform(test_df['label'].values)

train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)

# 2. Define the Softmax Regression model
class SoftmaxRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(SoftmaxRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Single linear layer
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer

    def forward(self, x):
        x = self.dropout(x)  # Apply dropout
        return self.fc(x)  # Linear transformation

# Model parameters
input_dim = X_train_tensor.shape[1]  # Number of input features (3072 for 32x32 RGB images)
output_dim = len(np.unique(train_labels_encoded))  # Number of categories
dropout_rate = 0.5

# Initialize the model
model = SoftmaxRegressionModel(input_dim, output_dim, dropout_rate)

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add weight decay for regularization

# 4. Early Stopping parameters
patience = 5  # Number of epochs without improvement before stopping
no_improvement_count = 0  # Counter for epochs without improvement
best_val_loss = float('inf')  # Best validation loss initialized to infinity

# 5. Training loop
num_epochs = 200
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, train_labels_tensor)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    _, predicted_train = torch.max(output, 1)
    train_accuracy = (predicted_train == train_labels_tensor).float().mean().item()

    # Store training loss and accuracy
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)

    # Validation step
    model.eval()
    with torch.no_grad():
        output_val = model(X_test_tensor)
        val_loss = criterion(output_val, test_labels_tensor)

        _, predicted_val = torch.max(output_val, 1)
        val_accuracy = (predicted_val == test_labels_tensor).float().mean().item()

        # Store validation loss and accuracy
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)

    # Check for improvement in validation loss
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    # Print epoch details
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss.item():.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if no_improvement_count >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# 6. Plot Loss and Accuracy graphs
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


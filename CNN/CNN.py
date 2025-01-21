# CNN Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt



############################################################
# Function to save Checkpoints
############################################################
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, filepath="content/checkpoint.pth"):
    '''
    Saves the model's current state, optimizer state, and training metrics
    (loss and accuracy for both train and validation) to a file.
    '''
    torch.save({
        'epoch': epoch,                                 # Current epoch number
        'model_state_dict': model.state_dict(),         # Model parameters
        'optimizer_state_dict': optimizer.state_dict(), # Optimizer parameters
        'train_loss': train_loss,                       # Training loss at the current epoch
        'val_loss': val_loss,                           # Validation loss at the current epoch
        'train_acc': train_acc,                         # Training accuracy at the current epoch
        'val_acc': val_acc,                             # Validation accuracy at the current epoch
    }, filepath)                                        # Filepath to save the checkpoint
    print(f"Checkpoint saved at epoch {epoch} in {filepath}") # Confirmation message


def load_checkpoint(model, optimizer, filepath="content/checkpoint.pth"):
    '''
    Loads a saved checkpoint, restoring the model's parameters,
    optimizer's parameters, and training metrics.
    '''
    checkpoint = torch.load(filepath)                      # Load the checkpoint file
    model.load_state_dict(checkpoint['model_state_dict'])  # Restore model parameters
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer parameters
    epoch = checkpoint['epoch']                            # Retrieve the epoch number
    train_loss = checkpoint['train_loss']                  # Retrieve the training loss
    val_loss = checkpoint['val_loss']                      # Retrieve the validation loss
    train_acc = checkpoint['train_acc']                    # Retrieve the training accuracy
    val_acc = checkpoint['val_acc']                        # Retrieve the validation accuracy

    print(f"Checkpoint loaded from epoch {epoch}")         # Confirmation message
    return epoch, train_loss, val_loss, train_acc, val_acc  # Return all retrieved values


# Convert labels to numerical values
label_encoder = LabelEncoder()  # Initialize a label encoder
train_df['label'] = label_encoder.fit_transform(train_df['label'])  # Fit and transform training labels
val_df['label'] = label_encoder.transform(val_df['label'])          # Transform validation labels
test_df['label'] = label_encoder.transform(test_df['label'])        # Transform test labels




# Custom Dataset class for handling image data
class ImageDataset(Dataset):
    '''
    A custom PyTorch Dataset to handle image data and their corresponding labels.
    '''
    def __init__(self, dataframe, images_folder, transform=None):
        '''
        Initializes the dataset.
        Args:
            dataframe: A pandas DataFrame containing image IDs and labels.
            images_folder: Path to the folder containing the images.
            transform: Optional transformations to apply to the images.
        '''
        self.dataframe = dataframe                # Store the dataframe
        self.images_folder = images_folder        # Store the images folder path
        self.transform = transform                # Store the transformations

    def __len__(self):
        '''
        Returns the total number of samples in the dataset.
        '''
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Retrieves an image and its corresponding label by index.
        Args:
            idx: Index of the sample to retrieve.
        Returns:
            image: Transformed image tensor.
            label: Corresponding label as a torch tensor.
        '''
        img_name = self.dataframe.iloc[idx]['id']  # Get the image ID
        label = self.dataframe.iloc[idx]['label']  # Get the label
        img_path = f"{self.images_folder}/{img_name}.png"  # Construct the image path
        image = Image.open(img_path).convert("RGB")  # Load the image and convert to RGB
        if self.transform:                          # Apply transformations if provided
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)  # Return image and label

############################################################
# 1) Data Augmentation (Transformations)
############################################################
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),                 # Resize images to 32x32 pixels
    transforms.RandomHorizontalFlip(p=0.5),     # Apply horizontal flip with 50% probability
    transforms.RandomRotation(10),              # Apply random rotation up to 10 degrees
    transforms.ToTensor(),                       # Convert image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize the image: mean and std for each channel
                         std=[0.5, 0.5, 0.5])
])


# For validation and testing, use "neutral" transformations (without augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),                 # Resize images to 32x32 pixels
    transforms.ToTensor(),                       # Convert image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize the image: mean and std for each channel
                         std=[0.5, 0.5, 0.5])
])


# Load training, validation, and testing data with the defined transformations
train_dataset = ImageDataset(train_df, '/content/images', transform=train_transform)  # Apply training transformations
val_dataset   = ImageDataset(val_df,   '/content/images', transform=val_test_transform)  # Apply validation transformations
test_dataset  = ImageDataset(test_df,  '/content/images', transform=val_test_transform)  # Apply testing transformations

# Create DataLoaders for batching and shuffling data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle training data for randomness
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)  # No shuffling for validation data
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)  # No shuffling for testing data



############################################################
# 2) Adding Dropout to the Model
############################################################
class CNNModel(nn.Module):
    '''
    A Convolutional Neural Network (CNN) model with Batch Normalization
    and Dropout to prevent overfitting.
    '''
    def __init__(self, num_classes):
        '''
        Initializes the CNN model.
        Args:
            num_classes: Number of output classes for classification.
        '''
        super(CNNModel, self).__init__()
        # First convolutional layer: 3 input channels, 32 output filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after Conv1

        # Second convolutional layer: 32 input filters, 64 output filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after Conv2

        # Third convolutional layer: 64 input filters, 128 output filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization after Conv3

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with 2x2 kernel

        self.flatten = nn.Flatten()  # Flatten the tensor for fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # First fully connected layer
        self.dropout_fc = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        '''
        Defines the forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output logits for the given input.
        '''
        x = torch.relu(self.bn1(self.conv1(x)))  # Conv1 -> BatchNorm -> ReLU
        x = torch.relu(self.bn2(self.conv2(x)))  # Conv2 -> BatchNorm -> ReLU
        x = torch.relu(self.bn3(self.conv3(x)))  # Conv3 -> BatchNorm -> ReLU
        x = self.pool(x)  # Max Pooling

        x = self.flatten(x)              # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))      # Fully Connected Layer 1 -> ReLU
        x = self.dropout_fc(x)           # Apply Dropout
        x = self.fc2(x)                  # Fully Connected Layer 2 (Output layer)
        return x


# Define the model, loss function, optimizer, and learning rate scheduler
num_classes = 10  # Number of output classes for classification
model = CNNModel(num_classes)  # Initialize the CNN model

# Initialize the optimizer with weight decay for regularization
# Uncomment the following line to use a different weight decay value
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam optimizer with weight decay

# Initialize a learning rate scheduler
# Reduces the learning rate by 30% every 10 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Define start_epoch based on whether a checkpoint is loaded
checkpoint_path = '/content/checkpoint.pth'
if os.path.exists(checkpoint_path):
    epoch, train_loss, val_loss, train_acc, val_acc = load_checkpoint(model, optimizer, checkpoint_path)
    print(f"Resumed training from epoch {epoch}")
    start_epoch = epoch + 1  # Continue from the next epoch
else:
    print("Starting training from scratch")
    start_epoch = 0  # Start from the first epoch

# Define the loss function
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification



# Training function with metrics tracking
def train_model_with_metrics(model, train_loader, val_loader, num_epochs, device, patience=10):
    '''
    Trains a model while tracking metrics such as loss and accuracy for both
    training and validation. Implements early stopping based on validation loss.
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        num_epochs: Total number of epochs for training.
        device: Device to use for training (CPU or GPU).
        patience: Number of epochs to wait without improvement before stopping.
    '''
    model.to(device)  # Move the model to the specified device
    train_losses, val_losses = [], []  # Lists to store training and validation losses
    train_accuracies, val_accuracies = [], []  # Lists to store training and validation accuracies

    best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    epochs_without_improvement = 0  # Counter for epochs without improvement in validation loss

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0  # Initialize training loss for the epoch
        train_correct = 0  # Initialize correct predictions counter for training

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()  # Clear previous gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_loss += loss.item() * images.size(0)  # Accumulate total training loss
            _, preds = torch.max(outputs, 1)  # Get predictions
            train_correct += torch.sum(preds == labels).item()  # Count correct predictions

        # Compute average training loss and accuracy for the epoch
        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(train_correct / len(train_loader.dataset))

        # Update learning rate using the scheduler
        scheduler.step()

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Initialize validation loss
        val_correct = 0  # Initialize correct predictions counter for validation

        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to the device
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                val_loss += loss.item() * images.size(0)  # Accumulate total validation loss
                _, preds = torch.max(outputs, 1)  # Get predictions
                val_correct += torch.sum(preds == labels).item()  # Count correct predictions

        # Compute average validation loss and accuracy for the epoch
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        val_losses.append(val_loss / len(val_loader.dataset))
        val_accuracies.append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            epochs_without_improvement = 0  # Reset the counter
            # Save a checkpoint when validation loss improves
            checkpoint_path = '/content/checkpoint.pth'
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, filepath=checkpoint_path)
        else:
            epochs_without_improvement += 1  # Increment the counter if no improvement

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")  # Stop training early if no improvement
            break

        # Print training and validation metrics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Plot loss and accuracy graphs
    plt.figure(figsize=(10, 4))

    # Plot training vs validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")

    # Plot training vs validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train vs Validation Accuracy")

    plt.tight_layout()
    plt.show()


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
train_model_with_metrics(model, train_loader, val_loader, num_epochs=80, device=device)  # Train the model with 80 epochs

# Evaluate the model on the test dataset
def evaluate_model_with_metrics(model, test_loader, device):
    '''
    Evaluates the model on the test dataset and computes performance metrics.
    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test dataset.
        device: Device to use for evaluation (CPU or GPU).
    '''
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode
    all_labels = []  # List to store true labels
    all_preds = []  # List to store predicted labels

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predictions with the highest probability
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())  # Store predicted labels

    # Calculate evaluation metrics
    precision = precision_score(all_labels, all_preds, average="weighted")  # Weighted precision
    recall = recall_score(all_labels, all_preds, average="weighted")  # Weighted recall
    f1 = f1_score(all_labels, all_preds, average="weighted")  # Weighted F1-score
    accuracy = accuracy_score(all_labels, all_preds)  # Overall accuracy

    # Print evaluation metrics
    print(f"Test Metrics:\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"F1-Score: {f1:.4f}\n"
          f"Accuracy: {accuracy:.4f}")

# Call the evaluation function
evaluate_model_with_metrics(model, test_loader, device)

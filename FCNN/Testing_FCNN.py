# Testing the FCNN Model

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Flatten the test data to match the input dimensions of the model
X_test_flattened = X_test_tensor.view(X_test_tensor.size(0), -1)

# Set the model to evaluation mode
model.eval()

# Initialize lists to store results
test_losses = []
test_accuracies = []

# Calculate loss and metrics on the test set
with torch.no_grad():  # Disable gradient computation during testing
    output_test = model(X_test_flattened)  # Compute predictions for the test set
    test_loss = criterion(output_test, test_labels_tensor)  # Calculate the test loss

    # Get the predicted labels
    _, predicted_test = torch.max(output_test, 1)

    # Calculate metrics
    test_accuracy = accuracy_score(test_labels_tensor.cpu(), predicted_test.cpu())
    test_precision = precision_score(test_labels_tensor.cpu(), predicted_test.cpu(), average='weighted')
    test_recall = recall_score(test_labels_tensor.cpu(), predicted_test.cpu(), average='weighted')
    test_f1 = f1_score(test_labels_tensor.cpu(), predicted_test.cpu(), average='weighted')

    # Store test results
    test_losses.append(test_loss.item())
    test_accuracies.append(test_accuracy)

# Print the test metrics
print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

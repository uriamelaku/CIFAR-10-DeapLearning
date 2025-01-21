# Testing the regression model

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # Import necessary metrics

#Perform a test on the model after training

# Set the model to evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():  # Disable gradient calculations during evaluation
    output_test = model(X_test_tensor)  # Compute predictions for the test set
    _, predicted_test = torch.max(output_test, 1)  # Get predicted classes

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(test_labels_tensor, predicted_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate Precision, Recall, and F1-Score on the test set
    test_precision = precision_score(test_labels_tensor, predicted_test, average='weighted')
    test_recall = recall_score(test_labels_tensor, predicted_test, average='weighted')
    test_f1 = f1_score(test_labels_tensor, predicted_test, average='weighted')

    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

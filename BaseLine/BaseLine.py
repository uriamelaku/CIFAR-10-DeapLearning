# BaseLine Model

import pandas as pd
from collections import Counter

# 1️⃣ Load the data
train_df = pd.read_csv('train_split.csv')  # Load the training dataset
test_df = pd.read_csv('test_split.csv')    # Load the test dataset

# 2️⃣ Find the most common label in the training set
label_counts = Counter(train_df['label'])  # Count the occurrences of each label
most_common_label = label_counts.most_common(1)[0][0]  # Get the most common label
most_common_count = label_counts.most_common(1)[0][1]  # Get the count of the most common label

print(f"📊 Most common label in Train: {most_common_label} (Count: {most_common_count})")

# 3️⃣ Calculate metrics for the Test Set
test_labels = test_df['label']  # Extract the labels from the test set

# Calculate True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
TP = sum(label == most_common_label for label in test_labels)  # Count correct predictions for the most common label
FP = len(test_labels) - TP  # False positives: Incorrect predictions for other labels
FN = sum((label == most_common_label) for label in train_df['label']) - TP  # False negatives: Missed predictions for the most common label
TN = len(test_labels) - TP - FP  # True negatives: Correct predictions for other labels

# Calculate Precision, Recall, Accuracy, and F1-Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision: TP / (TP + FP)
recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall: TP / (TP + FN)
accuracy = (TP + TN) / len(test_labels) if len(test_labels) > 0 else 0  # Accuracy: (TP + TN) / Total
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1-Score formula

# Print the metrics
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1-Score: {f1_score:.4f}")

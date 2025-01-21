# Checking the number of items in each split (train, validation, and test)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the split datasets
train_df = pd.read_csv('train_split.csv')  # Load the training set
val_df = pd.read_csv('val_split.csv')      # Load the validation set
test_df = pd.read_csv('test_split.csv')    # Load the test set

# Check the size of each set
print(f"Train set size: {len(train_df)}")       # Print the number of training samples
print(f"Validation set size: {len(val_df)}")    # Print the number of validation samples
print(f"Test set size: {len(test_df)}")         # Print the number of test samples

# Check label distribution in each set
print("\nTrain set label distribution:")        # Print label distribution in the training set
print(train_df['label'].value_counts())

print("\nValidation set label distribution:")   # Print label distribution in the validation set
print(val_df['label'].value_counts())

print("\nTest set label distribution:")         # Print label distribution in the test set
print(test_df['label'].value_counts())





# Show the split of the data using graphs

# Train set distribution
plt.figure(figsize=(12, 4))  # Set the figure size for the plot
sns.countplot(data=train_df, x='label')  # Create a bar plot for the label distribution in the train set
plt.title('Train Set Distribution')  # Add a title to the plot
plt.show()  # Display the plot

# Validation set distribution
plt.figure(figsize=(12, 4))  # Set the figure size for the plot
sns.countplot(data=val_df, x='label')  # Create a bar plot for the label distribution in the validation set
plt.title('Validation Set Distribution')  # Add a title to the plot
plt.show()  # Display the plot

# Test set distribution
plt.figure(figsize=(12, 4))  # Set the figure size for the plot
sns.countplot(data=test_df, x='label')  # Create a bar plot for the label distribution in the test set
plt.title('Test Set Distribution')  # Add a title to the plot
plt.show()  # Display the plot






# Splitting the data into train, validation, and test

import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load the labels file
df = pd.read_csv('Labels.csv')  # Load the CSV file containing the labels

# 2️⃣ First split: Train (70%) and Temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,          # 30% of the data will be split into Validation and Test
    random_state=42,        # Ensures the split is reproducible
    stratify=df['label']    # Ensures the label distribution remains balanced
)

# 3️⃣ Second split: Validation (15%) and Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,          # Split Temp into 50% Validation and 50% Test
    random_state=42,        # Ensures reproducibility
    stratify=temp_df['label']  # Maintains balanced label distribution
)

# 4️⃣ Save the splits as CSV files
train_df.to_csv('train_split.csv', index=False)  # Save training split
val_df.to_csv('val_split.csv', index=False)      # Save validation split
test_df.to_csv('test_split.csv', index=False)    # Save test split

# 5️⃣ Print a summary of the splits
print("Train size:", len(train_df))    # Print the number of training samples
print("Validation size:", len(val_df))  # Print the number of validation samples
print("Test size:", len(test_df))       # Print the number of test samples

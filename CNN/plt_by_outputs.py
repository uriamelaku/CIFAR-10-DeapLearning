# Printing the graph of the CNN model

"""
This script uses training logs to create graphs showing how the model performed during training.

Why use logs instead of checkpoints?
-------------------------------------
1. Long training sessions can stop because of time limits or disconnections.
2. Checkpoints only save the latest epoch, not the full training history.
3. Logs printed during training contain all the data (Train Loss, Validation Loss,
   Train Accuracy, Validation Accuracy) for every epoch.
4. By using these logs, we can recreate the full training history and plot graphs.

How to use:
-----------
1. Copy and paste the full training log (printed during training) into the `log` variable below.
2. Run the script to extract the data and plot the graphs.
"""


log = """
Starting training from scratch
Checkpoint saved at epoch 0 in /content/checkpoint.pth
Epoch [1/50], Train Loss: 2.3457, Train Accuracy: 0.1035, Val Loss: 2.2846, Val Accuracy: 0.1189
Checkpoint saved at epoch 1 in /content/checkpoint.pth
Epoch [2/50], Train Loss: 2.2560, Train Accuracy: 0.1221, Val Loss: 2.1225, Val Accuracy: 0.1796
Checkpoint saved at epoch 2 in /content/checkpoint.pth
Epoch [3/50], Train Loss: 2.2029, Train Accuracy: 0.1373, Val Loss: 2.0720, Val Accuracy: 0.1797
Checkpoint saved at epoch 3 in /content/checkpoint.pth
Epoch [4/50], Train Loss: 2.1880, Train Accuracy: 0.1437, Val Loss: 2.0709, Val Accuracy: 0.1833
Checkpoint saved at epoch 4 in /content/checkpoint.pth
Epoch [5/50], Train Loss: 2.1850, Train Accuracy: 0.1442, Val Loss: 2.0253, Val Accuracy: 0.2045
Epoch [6/50], Train Loss: 2.1810, Train Accuracy: 0.1438, Val Loss: 2.0436, Val Accuracy: 0.1824
Epoch [7/50], Train Loss: 2.1753, Train Accuracy: 0.1454, Val Loss: 2.0360, Val Accuracy: 0.1791
Epoch [8/50], Train Loss: 2.1726, Train Accuracy: 0.1467, Val Loss: 2.0406, Val Accuracy: 0.1961
Checkpoint saved at epoch 8 in /content/checkpoint.pth
Epoch [9/50], Train Loss: 2.1707, Train Accuracy: 0.1445, Val Loss: 2.0132, Val Accuracy: 0.1941
Checkpoint saved at epoch 9 in /content/checkpoint.pth
Epoch [10/50], Train Loss: 2.1666, Train Accuracy: 0.1478, Val Loss: 1.9972, Val Accuracy: 0.2073
Epoch [11/50], Train Loss: 2.1606, Train Accuracy: 0.1513, Val Loss: 2.0159, Val Accuracy: 0.1992
Epoch [12/50], Train Loss: 2.1603, Train Accuracy: 0.1511, Val Loss: 2.0127, Val Accuracy: 0.2029
Checkpoint saved at epoch 12 in /content/checkpoint.pth
Epoch [13/50], Train Loss: 2.1594, Train Accuracy: 0.1500, Val Loss: 1.9941, Val Accuracy: 0.1916
Checkpoint saved at epoch 13 in /content/checkpoint.pth
Epoch [14/50], Train Loss: 2.1553, Train Accuracy: 0.1509, Val Loss: 1.9847, Val Accuracy: 0.1996
Epoch [15/50], Train Loss: 2.1559, Train Accuracy: 0.1515, Val Loss: 2.0503, Val Accuracy: 0.1943
Checkpoint saved at epoch 15 in /content/checkpoint.pth
Epoch [16/50], Train Loss: 2.1120, Train Accuracy: 0.1620, Val Loss: 1.8492, Val Accuracy: 0.2276
Checkpoint saved at epoch 16 in /content/checkpoint.pth
Epoch [17/50], Train Loss: 2.0634, Train Accuracy: 0.1606, Val Loss: 1.8389, Val Accuracy: 0.2251
Checkpoint saved at epoch 17 in /content/checkpoint.pth
Epoch [18/50], Train Loss: 2.0544, Train Accuracy: 0.1555, Val Loss: 1.7975, Val Accuracy: 0.2111
Checkpoint saved at epoch 18 in /content/checkpoint.pth
Epoch [19/50], Train Loss: 2.0447, Train Accuracy: 0.1565, Val Loss: 1.7809, Val Accuracy: 0.2460
Checkpoint saved at epoch 19 in /content/checkpoint.pth
Epoch [20/50], Train Loss: 1.9805, Train Accuracy: 0.1892, Val Loss: 1.6607, Val Accuracy: 0.3319
Checkpoint saved at epoch 20 in /content/checkpoint.pth
Epoch [21/50], Train Loss: 1.9493, Train Accuracy: 0.2086, Val Loss: 1.6411, Val Accuracy: 0.3487
Epoch [22/50], Train Loss: 1.9334, Train Accuracy: 0.2132, Val Loss: 1.6493, Val Accuracy: 0.3623
Checkpoint saved at epoch 22 in /content/checkpoint.pth
Epoch [23/50], Train Loss: 1.8916, Train Accuracy: 0.2308, Val Loss: 1.5166, Val Accuracy: 0.4392
Checkpoint saved at epoch 23 in /content/checkpoint.pth
Epoch [24/50], Train Loss: 1.7767, Train Accuracy: 0.2767, Val Loss: 1.4304, Val Accuracy: 0.4897
Checkpoint saved at epoch 24 in /content/checkpoint.pth
Epoch [25/50], Train Loss: 1.7150, Train Accuracy: 0.3051, Val Loss: 1.3944, Val Accuracy: 0.5192
Checkpoint saved at epoch 25 in /content/checkpoint.pth
Epoch [26/50], Train Loss: 1.6230, Train Accuracy: 0.3452, Val Loss: 1.2568, Val Accuracy: 0.5687
Checkpoint saved at epoch 26 in /content/checkpoint.pth
Epoch [27/50], Train Loss: 1.5740, Train Accuracy: 0.3742, Val Loss: 1.2121, Val Accuracy: 0.5844
Checkpoint saved at epoch 27 in /content/checkpoint.pth
Epoch [28/50], Train Loss: 1.4787, Train Accuracy: 0.4236, Val Loss: 1.1405, Val Accuracy: 0.6183
Checkpoint saved at epoch 28 in /content/checkpoint.pth
Epoch [29/50], Train Loss: 1.4128, Train Accuracy: 0.4562, Val Loss: 1.0721, Val Accuracy: 0.6420
Checkpoint saved at epoch 29 in /content/checkpoint.pth
Epoch [30/50], Train Loss: 1.3700, Train Accuracy: 0.4737, Val Loss: 1.0323, Val Accuracy: 0.6665
Checkpoint saved at epoch 30 in /content/checkpoint.pth
Epoch [31/50], Train Loss: 1.3171, Train Accuracy: 0.4991, Val Loss: 0.9942, Val Accuracy: 0.6679
Checkpoint saved at epoch 31 in /content/checkpoint.pth
Epoch [32/50], Train Loss: 1.2551, Train Accuracy: 0.5187, Val Loss: 0.9468, Val Accuracy: 0.6765
Checkpoint saved at epoch 32 in /content/checkpoint.pth
Epoch [33/50], Train Loss: 1.2162, Train Accuracy: 0.5388, Val Loss: 0.9209, Val Accuracy: 0.6915
Checkpoint saved at epoch 33 in /content/checkpoint.pth
Epoch [34/50], Train Loss: 1.1783, Train Accuracy: 0.5531, Val Loss: 0.8849, Val Accuracy: 0.6967
Checkpoint saved at epoch 34 in /content/checkpoint.pth
Epoch [35/50], Train Loss: 1.1439, Train Accuracy: 0.5682, Val Loss: 0.8683, Val Accuracy: 0.6979
Epoch [36/50], Train Loss: 1.1044, Train Accuracy: 0.5849, Val Loss: 0.8572, Val Accuracy: 0.7099
Epoch [37/50], Train Loss: 1.0882, Train Accuracy: 0.5907, Val Loss: 0.8514, Val Accuracy: 0.7156
Epoch [38/50], Train Loss: 1.0870, Train Accuracy: 0.5907, Val Loss: 0.8251, Val Accuracy: 0.7184
Checkpoint saved at epoch 38 in /content/checkpoint.pth
Epoch [39/50], Train Loss: 1.0675, Train Accuracy: 0.5980, Val Loss: 0.8045, Val Accuracy: 0.7269
Checkpoint saved at epoch 39 in /content/checkpoint.pth
Epoch [40/50], Train Loss: 1.0584, Train Accuracy: 0.6042, Val Loss: 0.7975, Val Accuracy: 0.7275
Checkpoint saved at epoch 40 in /content/checkpoint.pth
Epoch [41/50], Train Loss: 1.0450, Train Accuracy: 0.6044, Val Loss: 0.7905, Val Accuracy: 0.7328
Checkpoint saved at epoch 41 in /content/checkpoint.pth
Epoch [42/50], Train Loss: 1.0286, Train Accuracy: 0.6134, Val Loss: 0.7759, Val Accuracy: 0.7371
Epoch [43/50], Train Loss: 1.0076, Train Accuracy: 0.6220, Val Loss: 0.7814, Val Accuracy: 0.7417
Checkpoint saved at epoch 43 in /content/checkpoint.pth
Epoch [44/50], Train Loss: 1.0023, Train Accuracy: 0.6248, Val Loss: 0.7728, Val Accuracy: 0.7339
Checkpoint saved at epoch 44 in /content/checkpoint.pth
Epoch [45/50], Train Loss: 0.9805, Train Accuracy: 0.6329, Val Loss: 0.7642, Val Accuracy: 0.7415
Checkpoint saved at epoch 45 in /content/checkpoint.pth
Epoch [46/50], Train Loss: 0.9606, Train Accuracy: 0.6402, Val Loss: 0.7484, Val Accuracy: 0.7463
Epoch [47/50], Train Loss: 0.9404, Train Accuracy: 0.6496, Val Loss: 0.7349, Val Accuracy: 0.7461
Checkpoint saved at epoch 47 in /content/checkpoint.pth
Epoch [48/50], Train Loss: 0.9118, Train Accuracy: 0.6578, Val Loss: 0.7290, Val Accuracy: 0.7535
Checkpoint saved at epoch 48 in /content/checkpoint.pth
Epoch [49/50], Train Loss: 0.9009, Train Accuracy: 0.6638, Val Loss: 0.7249, Val Accuracy: 0.7557
Checkpoint saved at epoch 49 in /content/checkpoint.pth
Epoch [50/50], Train Loss: 0.8910, Train Accuracy: 0.6621, Val Loss: 0.7180, Val Accuracy: 0.7539
Epoch [51/80], Train Loss: 0.8847, Train Accuracy: 0.6712, Val Loss: 0.7107, Val Accuracy: 0.7555
Epoch [52/80], Train Loss: 0.8668, Train Accuracy: 0.6726, Val Loss: 0.7139, Val Accuracy: 0.7587
Epoch [53/80], Train Loss: 0.8532, Train Accuracy: 0.6808, Val Loss: 0.7148, Val Accuracy: 0.7636
Checkpoint saved at epoch 53 in /content/checkpoint.pth
Epoch [54/80], Train Loss: 0.8381, Train Accuracy: 0.6881, Val Loss: 0.6928, Val Accuracy: 0.7640
Checkpoint saved at epoch 54 in /content/checkpoint.pth
Epoch [55/80], Train Loss: 0.8293, Train Accuracy: 0.6920, Val Loss: 0.6907, Val Accuracy: 0.7685
Epoch [56/80], Train Loss: 0.8160, Train Accuracy: 0.6994, Val Loss: 0.6971, Val Accuracy: 0.7668
Checkpoint saved at epoch 56 in /content/checkpoint.pth
Epoch [57/80], Train Loss: 0.8014, Train Accuracy: 0.7049, Val Loss: 0.6722, Val Accuracy: 0.7719
Checkpoint saved at epoch 57 in /content/checkpoint.pth
Epoch [58/80], Train Loss: 0.7828, Train Accuracy: 0.7139, Val Loss: 0.6699, Val Accuracy: 0.7725
Epoch [59/80], Train Loss: 0.7650, Train Accuracy: 0.7175, Val Loss: 0.6785, Val Accuracy: 0.7713
Epoch [60/80], Train Loss: 0.7521, Train Accuracy: 0.7247, Val Loss: 0.6709, Val Accuracy: 0.7732
Checkpoint saved at epoch 60 in /content/checkpoint.pth
Epoch [61/80], Train Loss: 0.7472, Train Accuracy: 0.7273, Val Loss: 0.6584, Val Accuracy: 0.7796
Checkpoint saved at epoch 61 in /content/checkpoint.pth
Epoch [62/80], Train Loss: 0.7286, Train Accuracy: 0.7321, Val Loss: 0.6550, Val Accuracy: 0.7827
Epoch [63/80], Train Loss: 0.7278, Train Accuracy: 0.7337, Val Loss: 0.6616, Val Accuracy: 0.7808
Epoch [64/80], Train Loss: 0.7165, Train Accuracy: 0.7415, Val Loss: 0.6653, Val Accuracy: 0.7777
Epoch [65/80], Train Loss: 0.7012, Train Accuracy: 0.7439, Val Loss: 0.6624, Val Accuracy: 0.7833
Checkpoint saved at epoch 65 in /content/checkpoint.pth
Epoch [66/80], Train Loss: 0.6791, Train Accuracy: 0.7512, Val Loss: 0.6411, Val Accuracy: 0.7892
Checkpoint saved at epoch 66 in /content/checkpoint.pth
Epoch [67/80], Train Loss: 0.6623, Train Accuracy: 0.7595, Val Loss: 0.6390, Val Accuracy: 0.7889
Checkpoint saved at epoch 67 in /content/checkpoint.pth
Epoch [68/90], Train Loss: 0.6624, Train Accuracy: 0.7593, Val Loss: 0.6407, Val Accuracy: 0.7900
Epoch [69/90], Train Loss: 0.6478, Train Accuracy: 0.7632, Val Loss: 0.6462, Val Accuracy: 0.7900
Checkpoint saved at epoch 69 in /content/checkpoint.pth
Epoch [70/90], Train Loss: 0.6369, Train Accuracy: 0.7677, Val Loss: 0.6381, Val Accuracy: 0.7919
Epoch [71/90], Train Loss: 0.6262, Train Accuracy: 0.7681, Val Loss: 0.6568, Val Accuracy: 0.7881
Checkpoint saved at epoch 71 in /content/checkpoint.pth
Epoch [72/90], Train Loss: 0.6265, Train Accuracy: 0.7716, Val Loss: 0.6332, Val Accuracy: 0.7951
Epoch [73/90], Train Loss: 0.6084, Train Accuracy: 0.7768, Val Loss: 0.6469, Val Accuracy: 0.7936
Epoch [74/90], Train Loss: 0.6129, Train Accuracy: 0.7756, Val Loss: 0.6490, Val Accuracy: 0.7903
Epoch [75/90], Train Loss: 0.5955, Train Accuracy: 0.7812, Val Loss: 0.6458, Val Accuracy: 0.7911
Epoch [76/90], Train Loss: 0.5901, Train Accuracy: 0.7838, Val Loss: 0.6425, Val Accuracy: 0.7940
Epoch [77/90], Train Loss: 0.5865, Train Accuracy: 0.7852, Val Loss: 0.6350, Val Accuracy: 0.7984
Epoch [78/90], Train Loss: 0.5661, Train Accuracy: 0.7936, Val Loss: 0.6351, Val Accuracy: 0.7987
Checkpoint saved at epoch 78 in /content/checkpoint.pth
Epoch [79/90], Train Loss: 0.5620, Train Accuracy: 0.7955, Val Loss: 0.6327, Val Accuracy: 0.8003
Checkpoint saved at epoch 79 in /content/checkpoint.pth
Epoch [80/90], Train Loss: 0.5535, Train Accuracy: 0.7997, Val Loss: 0.6285, Val Accuracy: 0.7987
Epoch [81/90], Train Loss: 0.5468, Train Accuracy: 0.7977, Val Loss: 0.6376, Val Accuracy: 0.8004
Epoch [82/90], Train Loss: 0.5443, Train Accuracy: 0.8022, Val Loss: 0.6441, Val Accuracy: 0.8005
Epoch [83/90], Train Loss: 0.5370, Train Accuracy: 0.8024, Val Loss: 0.6629, Val Accuracy: 0.7917
Epoch [84/90], Train Loss: 0.5305, Train Accuracy: 0.8064, Val Loss: 0.6413, Val Accuracy: 0.8000
Epoch [85/90], Train Loss: 0.5164, Train Accuracy: 0.8102, Val Loss: 0.6355, Val Accuracy: 0.8004
Epoch [86/90], Train Loss: 0.5199, Train Accuracy: 0.8116, Val Loss: 0.6336, Val Accuracy: 0.8039
Checkpoint saved at epoch 86 in /content/checkpoint.pth
Epoch [87/90], Train Loss: 0.5129, Train Accuracy: 0.8131, Val Loss: 0.6278, Val Accuracy: 0.8013
Epoch [88/90], Train Loss: 0.5022, Train Accuracy: 0.8159, Val Loss: 0.6379, Val Accuracy: 0.8012
Epoch [89/90], Train Loss: 0.4935, Train Accuracy: 0.8181, Val Loss: 0.6315, Val Accuracy: 0.8057
Epoch [90/90], Train Loss: 0.4912, Train Accuracy: 0.8197, Val Loss: 0.6351, Val Accuracy: 0.8007
"""

# Lists to store the extracted data
train_losses, val_losses = [], []  # Lists for train and validation losses
train_accuracies, val_accuracies = [], []  # Lists for train and validation accuracies

# Extract data from the training log
for line in log.strip().split("\n"):  # Iterate over each line in the log
    if "Epoch [" in line:  # Check if the line contains epoch information
        parts = line.split(", ")  # Split the line into parts using ', ' as a separator
        train_loss = float(parts[1].split(":")[1].strip())  # Extract train loss
        train_acc = float(parts[2].split(":")[1].strip())  # Extract train accuracy
        val_loss = float(parts[3].split(":")[1].strip())  # Extract validation loss
        val_acc = float(parts[4].split(":")[1].strip())  # Extract validation accuracy

        # Append the extracted values to the corresponding lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
# Check and fill missing epochs
for i in range(1, 50):  # Iterate through expected epoch numbers
    if i not in range(1, len(train_losses) + 1):  # Check if an epoch is missing in the data
        # Fill missing epoch using interpolation between neighboring epochs
        train_losses.insert(i - 1, (train_losses[i - 2] + train_losses[i]) / 2)  # Interpolate train loss
        val_losses.insert(i - 1, (val_losses[i - 2] + val_losses[i]) / 2)        # Interpolate validation loss
        train_accuracies.insert(i - 1, (train_accuracies[i - 2] + train_accuracies[i]) / 2)  # Interpolate train accuracy
        val_accuracies.insert(i - 1, (val_accuracies[i - 2] + val_accuracies[i]) / 2)        # Interpolate validation accuracy

import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Plotting the graph
plt.figure(figsize=(10, 4))  # Create a figure with specified size

# Loss graph
plt.subplot(1, 2, 1)  # Create the first subplot for losses
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')  # Plot train loss
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')  # Plot validation loss
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.legend()  # Add legend to differentiate between train and validation
plt.title('Loss vs Epochs')  # Title for the loss graph

# Accuracy graph
plt.subplot(1, 2, 2)  # Create the second subplot for accuracies
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')  # Plot train accuracy
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')  # Plot validation accuracy
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel('Accuracy')  # Label for the y-axis
plt.legend()  # Add legend to differentiate between train and validation
plt.title('Accuracy vs Epochs')  # Title for the accuracy graph

plt.tight_layout()  # Adjust layout to prevent overlap between plots
plt.show()  # Display the graphs


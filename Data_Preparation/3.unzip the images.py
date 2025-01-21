# Unzip all the images into a folder called "images"

import zipfile
import os
import shutil

# Create a new folder named 'images'
os.makedirs('/content/images', exist_ok=True)  # Ensure the folder exists (create if not)

# Extract the zip file into the 'images' folder
with zipfile.ZipFile('/content/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/images')  # Extract all files into the specified directory

# The extracted folder has a subfolder named 'train'. Move its content to 'images'.
train_folder = '/content/images/train'  # Path to the extracted 'train' folder
new_name = '/content/images/images'     # Final name of the main images folder

# Check if the 'train' folder exists
if os.path.exists(train_folder):
    # Move all contents from the 'train' folder to 'images'
    for item in os.listdir(train_folder):  # Iterate through all items in the 'train' folder
        s = os.path.join(train_folder, item)  # Source path of the item
        d = os.path.join('/content/images', item)  # Destination path
        if os.path.isdir(s):  # Check if the item is a folder
            shutil.copytree(s, d)  # Copy the entire folder and its contents
        else:  # If it's a file
            shutil.copy2(s, d)  # Copy the file

    # Remove the original 'train' folder to clean up
    shutil.rmtree(train_folder)

# Verify the new folder name
print(f"New folder name: {new_name}")






# Checking the number of images in the unzipped folder to ensure it contains 50K images

num_files = len(os.listdir('/content/images'))  # Count the number of files in the 'images' folder
print(f"✅ Number of images in '/content/images': {num_files}")  # Print the total number of images

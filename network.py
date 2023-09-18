import zipfile

import os

# Define the directory where the extracted files will be stored

extracted_dir = './data/mnist_data'

# Create the directory if it doesn't exist

if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

# Extract the ZIP file
with zipfile.ZipFile('./data/archive.zip', 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)


# List the extracted files
extracted_files = os.listdir(extracted_dir)

import os
import random
import shutil

# Path to the original image folder
original_folder = "flickr30k_images/flickr30k_images"

# Path to the destination folder for the test set
test_set_folder = "TestSet/"

# Create the test set folder if it doesn't exist
os.makedirs(test_set_folder, exist_ok=True)

# List all the image files in the original folder
image_files = os.listdir(original_folder)

# Select 25 random image files
random_images = random.sample(image_files, 25)

# Move the selected images to the test set folder
for image in random_images:
    source_path = os.path.join(original_folder, image)
    destination_path = os.path.join(test_set_folder, image)
    shutil.move(source_path, destination_path)

print("Test set created successfully.")

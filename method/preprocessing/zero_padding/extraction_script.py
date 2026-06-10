import os
import pathlib
import cv2
from tqdm import tqdm

from resize_pipeline import resize_and_pad

# Extract all images (enter your own paths)
bus_dataset_path = ""
resized_dataset_path = ""

# Create the main output directory if it doesn't exist
os.makedirs(resized_dataset_path, exist_ok=True)

image_paths = []
for dirpath, dirnames, filenames in os.walk(bus_dataset_path):
    for fn in filenames:
        fn = os.path.join(dirpath, fn)
        path = pathlib.Path(fn)
        image_paths.append(path)

# Iterate through each image and resize
for img_path in tqdm(image_paths):
    classification = img_path.parent.name    # needed for automatic classification of resized images
    filename = img_path.name                 # for writing file to disk
    
    # Read the image
    img = cv2.imread(str(img_path))
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        continue
        
    # Create output directory for this classification if it doesn't exist
    output_dir = os.path.join(resized_dataset_path, classification)
    os.makedirs(output_dir, exist_ok=True)
    
    # Resize the image
    resized_img = resize_and_pad(img, target_size=224)
    
    # Write the resized image
    output_path = os.path.join(resized_dataset_path, classification, filename)
    success = cv2.imwrite(output_path, resized_img)
    
    if not success:
        print(f"Failed to write image: {output_path}")
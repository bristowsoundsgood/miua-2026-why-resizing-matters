import os
import cv2
import pathlib
import numpy as np
from tqdm import tqdm

def resize_and_pad(image, target_size=224):
    """
    Resize image proportionally until one side reaches target_size,
    then zero-pad the smaller side to make it square.
   
    Args:
        image: Input image (numpy array)
        target_size: Target size for final square image (default: 224)
   
    Returns:
        Resized and padded square image
    """
    h, w = image.shape[:2]
   
    # Calculate scale factor to make the larger dimension equal to target_size
    scale = target_size / max(h, w)
   
    # Calculate new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
   
    # Resize image proportionally
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
   
    # Create square canvas with zeros
    if len(image.shape) == 3:  # Colour image
        padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    else:  # Grayscale image
        padded = np.zeros((target_size, target_size), dtype=image.dtype)
   
    # Center the resized image in the padded canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
   
    if len(image.shape) == 3:
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
   
    return padded

# Example usage
if __name__ == "__main__":
    # Extract all images (enter your own paths)
    bus_dataset_path = r''
    resized_dataset_path = r''
    
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
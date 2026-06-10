import pathlib
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mask_to_border(mask):
    height, width = mask.shape
    border = np.zeros((height, width))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def mask_to_bbox(mask):
    bboxes = []

    border = mask_to_border(mask)
    lbl = label(border)
    props = regionprops(lbl)

    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def expand_bbox_with_margin(bbox, image_height, image_width, margin_percent=0.08):
    """
    Expand bounding box by adding margin based on original image dimensions
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        image_height: Original image height
        image_width: Original image width
        margin_percent: Percentage of original image dimensions to add as margin (default 8%)
    
    Returns:
        Expanded bounding box coordinates [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate margin based on original image dimensions
    margin_x = int(image_width * margin_percent)
    margin_y = int(image_height * margin_percent)
    
    # Expand the bounding box
    expanded_x1 = max(0, x1 - margin_x)
    expanded_y1 = max(0, y1 - margin_y)
    expanded_x2 = min(image_width, x2 + margin_x)
    expanded_y2 = min(image_height, y2 + margin_y)
    
    return [expanded_x1, expanded_y1, expanded_x2, expanded_y2]

def extract_roi(target_dir: str, classification: str):
    # Create folder to save images
    boxed_dir = f'{classification}_boxed'
    roi_dir = f'{classification}_roi'
    create_dir(boxed_dir)
    create_dir(roi_dir)
   
    images = sorted(glob(os.path.join(target_dir, 'image', '*')))
    masks = sorted(glob(os.path.join(target_dir, 'mask', '*')))
    boxed_images = []
    
    """ Loop over the dataset """
    for image_path, mask_path in tqdm(zip(images, masks), total=len(images)):
       
        """ Extracting the name """
        name = image_path.split('\\')[-1]
        
        """ Read the image and mask """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get original image dimensions
        image_height, image_width = image.shape[:2]
        
        # Check if mask needs resizing (RODTOOK dataset has mismatching mask and image sizes. Fit mask to image for an accurate ROI!)
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            print(f"Resizing mask {name} from {mask.shape} to match image size {(image_height, image_width)}")
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        
        """ Detecting bounding boxes """
        bboxes = mask_to_bbox(mask)
        
        """ Marking bounding boxes on the image and extracting """
        for bbox in bboxes:
            # Expand bbox with 8% margin based on original image dimensions
            expanded_bbox = expand_bbox_with_margin(bbox, image_height, image_width, margin_percent=0.08)
            
            # Draw rectangle using expanded bbox
            image = cv2.rectangle(img=image, 
                                pt1=(expanded_bbox[0], expanded_bbox[1]), 
                                pt2=(expanded_bbox[2], expanded_bbox[3]), 
                                color=(255, 255, 255), 
                                thickness=1)  
           
            # Extract ROI using expanded bbox
            roi = image[expanded_bbox[1] : expanded_bbox[3], expanded_bbox[0] : expanded_bbox[2]]
            cv2.imwrite(f'{roi_dir}/{name}', roi)
            
        """ Save the images """
        boxed_images.append(image)
        cv2.imwrite(f'{boxed_dir}/{name}', image)
        
## INSERT DATASET DIRECTORIES HERE ##
# a. Split the dataset into malignant and benign folders.
# b. Each classification folder must follow the format 'class/image, class/mask' (e.g., 'malignant/image', 'malignant/mask')
# c. ...in other words, just separate the images and masks for each class.
# n.b. make sure they only contain images.
benign_dir = r''
malignant_dir = r''

# Running the code will extract the ROIs and bounding boxes into new folders: 'class_boxed' and 'class_roi'
extract_roi(target_dir=benign_dir, classification='benign')
extract_roi(target_dir=malignant_dir, classification='malignant')
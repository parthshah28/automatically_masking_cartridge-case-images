!pip install -q git+https://github.com/huggingface/transformers.git

import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import cv2
import gc

def show_mask(mask, ax, color, label=None):
    """
    Display a single mask on a given axis with a specified color and label.

    Parameters:
    - mask (numpy.ndarray): Binary mask array.
    - ax (matplotlib.axes.Axes): Matplotlib axis to display the mask.
    - color (numpy.ndarray): RGB color for the mask overlay.
    - label (str): Label for the mask.

    Returns:
    - matplotlib.image.AxesImage: Plotted image.
    """
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    im = ax.imshow(mask_image)
    del mask
    gc.collect()
    return im

def calculate_iou(mask, target):
    """
    Calculate Intersection over Union (IOU) between two binary masks.

    Parameters:
    - mask (numpy.ndarray): Binary mask array.
    - target (numpy.ndarray): Binary target mask array.

    Returns:
    - float: Intersection over Union (IOU) score.
    """
    intersection = np.logical_and(mask, target)
    union = np.logical_or(mask, target)
    iou = np.sum(intersection) / (np.sum(union) + 1e-5)  # Avoid division by zero
    return iou

def show_masks_on_image(raw_image, masks):
    """
    Display multiple masks on an image with bounding boxes and IOU scores.

    Parameters:
    - raw_image (PIL.Image.Image): Input image.
    - masks (list of numpy.ndarray): List of binary masks.

    Returns:
    - None
    """
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Specify the order of masking and corresponding colors
    mask_order = [
        ("The breech-face impression", np.array([1, 0, 0, 0.6])),
        ("The aperture shear", np.array([0, 1, 0, 0.6])),
        ("The firing pin impression", np.array([0.5, 0, 0.5, 0.6])),
        ("The firing pin drag", np.array([0, 0.5, 1, 0.6])),
    ]

    # Convert the list of masks to a dictionary
    masks_dict = {mask_info[0]: masks[i] for i, mask_info in enumerate(mask_order)}

    for label, color in mask_order:
        mask = masks_dict[label]
        show_mask(mask, ax=ax, color=color, label=label)

        # Calculate area, bbox, and predicted IOU for each mask
        area = np.sum(mask)
        # Convert the mask to uint8 before applying cv2.boundingRect
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(contour)
        else:
            bbox = (0, 0, 0, 0)
        target_mask = (label == "The breech-face impression")
        iou = calculate_iou(mask, target_mask)

        # Print information for each mask
        print(f"Label: {label}, Area: {area}, Bbox: {bbox}, Predicted IOU: {iou}")

        # Replace the following coordinates with actual bounding box coordinates
        x, y, w, h = bbox
        # Draw bounding box rectangle on the image
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    # Display separate legends for each mask
    legend_labels = [mask_info[0] for mask_info in mask_order]
    legend_colors = [mask_info[1] for mask_info in mask_order]

    for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
        ax.text(1.05, 0.95 - i * 0.1, label, color=color, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

# Load the model for mask generation
from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)

# Load the image
img_url = "unnamed.png" #insert image path here
raw_image = Image.open(img_url).convert("RGB")
image= cv2.imread(img_url)
image.shape

# Generate masks
outputs = generator(raw_image, points_per_batch=64)
masks = outputs["masks"]

# Show the masks on the image in the specified order
show_masks_on_image(raw_image, masks)
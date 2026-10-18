"""
Smart Gait AI - Preprocessing and Feature Extraction
Converts silhouette frames into a normalized Gait Energy Image (GEI)
"""

import os
import cv2
import numpy as np

def clean_silhouette(frame, kernel_size=(3, 3)):
    """Applies morphological filtering (closing) to remove background noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    cleaned = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    return cleaned

def generate_gei(silhouette_dir, target_size=(128, 128)):
    """
    Computes the temporal average Gait Energy Image (GEI) 
    from an aligned sequence of binary silhouettes.
    """
    frame_files = sorted([f for f in os.listdir(silhouette_dir) if f.endswith(('.png', '.jpg'))])
    if not frame_files:
        raise ValueError("No silhouette images found in the specified directory.")

    silhouette_stack = []
    for file in frame_files:
        img_path = os.path.join(silhouette_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_cleaned = clean_silhouette(img)
        img_resized = cv2.resize(img_cleaned, target_size, interpolation=cv2.INTER_AREA)
        
        # Binarize to {0, 1}
        _, bin_img = cv2.threshold(img_resized, 127, 1, cv2.THRESH_BINARY)
        silhouette_stack.append(bin_img)

    # Compute temporal mean across all gait frames
    gei = np.mean(silhouette_stack, axis=0) * 255.0
    return gei.astype(np.uint8)

if __name__ == "__main__":
    print("Smart Gait AI - Feature extraction pipeline initialized.")

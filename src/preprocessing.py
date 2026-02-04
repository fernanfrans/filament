import cv2
import matplotlib.pyplot as plt
import numpy as np 


def processing_steps(image_path):
    # Step 1: Image Acquisition and Display
    image = cv2.imread(image_path)

    # Step 2: Grayscale Conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian Blurring
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Thresholding to Obtain Binary Mask
    _, binary_mask = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 5: Retain Largest Connected Component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    isolated_component = np.where(labels == largest_component, 255, 0).astype('uint8')

    # Step 6: Hole Filling
    final_filament = cv2.morphologyEx(isolated_component, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    return final_filament
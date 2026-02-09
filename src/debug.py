import cv2
import numpy as np

from preprocessing import processing_steps


# Load your image and get the mask from your preprocessing script
image = cv2.imread(r"F:\FILAMENT\images\2.jpg")
mask = processing_steps(r"F:\FILAMENT\images\2.jpg")

# Create a visual overlay (Red mask over original image)
overlay = image.copy()
overlay[mask > 0] = [0, 0, 255] # Turn the detected filament Red
combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

cv2.imwrite(r"F:\FILAMENT\images\debug_mask_overlay.jpg", combined)
print("Check 'debug_mask_overlay.jpg' to see if the red area matches the filament.")
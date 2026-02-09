import cv2

# Load image
img = cv2.imread(r"F:\FILAMENT\images\piso.jpg", 0)

# 1. Find the 1 Peso coin (assuming it's the largest/only object for this example)
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coin_contour = max(contours, key=cv2.contourArea)

# 2. Get width in pixels
x, y, w_pixels, h_pixels = cv2.boundingRect(coin_contour)

# 3. Calculate cm_per_pixel
known_diameter_cm = 2.24
cm_per_pixel = known_diameter_cm / w_pixels

print(f"Pixel width: {w_pixels}px")
print(f"Scale: {cm_per_pixel:.5f} cm/pixel")

# Now use it!
# object_width_cm = other_object_pixels * cm_per_pixel
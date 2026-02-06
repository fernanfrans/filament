from skimage.morphology import skeletonize
import numpy as np
import cv2

def measurement_extraction(mask, PIXEL_TO_CM):
    """
    Objective: Obtain quantitative measurements of the filament.
    """
    # Step 1: Skeleton Extraction
    _, binary = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary > 0
    skeleton = skeletonize(binary)
    coords = np.column_stack(np.where(skeleton))
    if len(coords) < 2:
        return {"error": "Filament too small or not found"}
    
    # Step 2: Width Measurement
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    width_px = dist_transform[skeleton] * 2
    width_cm = width_px * PIXEL_TO_CM
    mean_width = np.mean(width_cm)
    std_width = np.std(width_cm)
    
    width_cm_smooth = np.convolve(width_cm, np.ones(5)/5, mode="valid")
    local_mean = (width_cm_smooth[:-1] + width_cm_smooth[1:]) / 2
    local_change = np.abs(np.diff(width_cm_smooth)) / local_mean
    width_max_local_change = np.max(local_change)

    # Step 3: Straightness Analysis
    mean = coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords - mean, full_matrices=False)
    best_fit_direction = Vt[0]
    coords_centered = coords - mean 
    dx, dy = best_fit_direction 
    distances_px = np.abs(
        coords_centered[:, 0] * dy -
        coords_centered[:, 1] * dx
    )
    distances_cm = distances_px * PIXEL_TO_CM
    straight_max_deviation = np.max(distances_cm)
    straight_mean_deviation = np.mean(distances_cm)

    return {
        "mean_width": mean_width,
        "std_width": std_width,
        "max_width_change": width_max_local_change,
        "straight_max_deviation": straight_max_deviation,
        "straight_mean_deviation": straight_mean_deviation

    }



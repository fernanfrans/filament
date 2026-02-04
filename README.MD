# Rule-Based Computer Vision Quality Assessment for Recycled 3D-Printer Filament

## Steps in Evaluating the Filament

### 1. Image Preprocessing
**Objective:** Isolate the filament from the background.

- **a. Image acquisition**
- **b. Convert the image to grayscale**
- **c. Apply thresholding** to obtain a binary mask
- **d. Noise removal**
- **e. Retain the largest connected component**
- **f. Hole filling**

### 2. Measurement Extraction
**Objective:** Obtain quantitative measurements of the filament.

- **a. Skeleton extraction**
- **b. Diameter measurement**
  - **i. Mean diameter**
  - **ii. Diameter standard deviation**
  - **iii. Local diameter variation**
- **c. Maximum straightness deviation**
- **d. Root Mean Square (RMS) deviation**

### 3. Evaluation
**Objective:** Evaluate filament quality using predefined manufacturing constraints.

**Metrics:**
- **a. Mean diameter**
- **b. Diameter standard deviation**
- **c. Maximum local diameter change**
- **d. Straightness deviation**

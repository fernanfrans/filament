# Rule-Based Computer Vision Quality Assessment for Recycled 3D-Printer Filament

## Steps In Evaluating the Filament
1. Input Acquisition
    - Objective: Capture filament images 
2. Image Preprocessing 
    - Objective: Isolate filament from background
        a. convert image to grayscale
        b. apply thresholding to obtain binary mask
        c. Noise removal
        d. Retain big island 
        d. hole filling
3. Filament Continuity Verification 
    - Objective: Ensure the filament is physically continuous
4. Measurement Extraction 
    - Objective: Obtain measuremnts of the filament
        a. skeleton extraction 
        b. diameter measurement
            i. mean diameter
            ii. diameter standard deviation 
            iii. local diameter variation
        c. maximum straightness deviation
        d. RMS deviation
5. Evaluation 
    - Objective: Evaluate filament quality using predefined manufacturing constraints
    - Metrics: 
        a. mean diameter
        b. diameter standard deviation
        c. maximum local diameter change
        d. straightness deviation
6. Output and Visualization

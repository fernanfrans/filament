from src.preprocessing import processing_steps
from src.measurement import measurement_extraction
import cv2


if __name__ == "__main__":
    image_path = "F:\FILAMENT\images\i.jpg"
    # 1: preprocessing of the filament image
    preprocessed_filament = processing_steps(image_path)
    
    # 2: measuring Extraction
    measurements = measurement_extraction(preprocessed_filament, PIXEL_TO_CM=0.0264583333)  # Example conversion factor
    print("Measurements:", measurements)

    # 3: Evaluation
    
    
    # # Display the processed filament
    # cv2.imshow("Processed Filament", preprocessed_filament)
    # # cv2.imwrite("F:\FILAMENT\images\processed_filament.jpg", preprocessed_filament)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
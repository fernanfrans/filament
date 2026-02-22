from src.preprocessing import processing_steps
from src.measurement import measurement_extraction
from src.evaluation import evaluation_filament
import cv2


if __name__ == "__main__":
    image_path = r"F:\FILAMENT\images\ideal4.jpg"
    # 1: preprocessing of the filament image
    preprocessed_filament = processing_steps(image_path)
    
    # 2: measuring Extraction
    measurements = measurement_extraction(preprocessed_filament, PIXEL_TO_MM=0.09248048067092896)

    # 3: Evaluation
    results, overall_pass = evaluation_filament(measurements)

    # 4: Print Results
    for metric, value in measurements.items():
        print(f'{metric}: {value:.2f} mm')
    print(f"The filament {image_path}: {'PASS' if overall_pass else 'FAIL'}")
    
    
    # # # Display the processed filament
    # cv2.imshow("Processed Filament", preprocessed_filament)
    # cv2.imwrite("F:\FILAMENT\images\sample_filament.jpg", preprocessed_filament)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
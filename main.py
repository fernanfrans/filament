import cv2 
import numpy as np 


def segment_strip(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

if __name__ == "__main__":
    img = r"F:\FILAMENT\images\ideal1.jpg"
    image = cv2.imread(img)
    mask = segment_strip(image)
    cv2.imshow("Segmented Strip", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

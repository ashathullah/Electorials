
import cv2
import numpy as np
from pathlib import Path

def preprocess_for_ocr_display_v2(bgr):
    """
    Revised logic: Adaptive Threshold + Gentle contrast
    """
    # 1) grayscale
    if len(bgr.shape) == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr

    # 2) background normalization
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)

    # 3) boost local contrast (Gentler)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    con = clahe.apply(norm)

    # 4) binarize (Adaptive)
    th = cv2.adaptiveThreshold(
        con, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, 
        C=10
    )

    return th

def create_comparison_v2():
    input_path = Path(r"c:\Users\ashat\persnal\projects\congress\Electorials\extracted\tamil_removed\crops\page-004\page-004-001.png")
    output_path = Path("preview_preprocessing_v2.jpg")

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    original = cv2.imread(str(input_path))
    if original is None:
        return

    processed_gray = preprocess_for_ocr_display_v2(original)
    processed_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, "Original", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(processed_bgr, "Adaptive Thresh", (10, 30), font, 1, (0, 0, 255), 2)

    # Stack
    h1, w1 = original.shape[:2]
    h2, w2 = processed_bgr.shape[:2]
    if h1 != h2:
        processed_bgr = cv2.resize(processed_bgr, (int(w2 * h1 / h2), h1))

    combined = np.hstack((original, processed_bgr))
    cv2.imwrite(str(output_path), combined)
    print(f"Comparison saved to {output_path.absolute()}")

if __name__ == "__main__":
    create_comparison_v2()

import cv2
import time
import numpy as np
from ocr_tamil.ocr import OCR
import os

# Create a dummy image with some text (or load one if available)
# generating a simple image with text might be hard, so let's try to load one from the project if possible.
# But for testing input types, we can use a random image or just check successful execution.

def test_ocr_inputs():
    print("Initializing OCR...")
    ocr = OCR(detect=True)
    
    # Create a dummy image (100x300, white background)
    img = np.zeros((100, 300, 3), dtype=np.uint8) + 255
    # Add some text if possible, but we just want to test if it accepts the input type
    # If it fails, it will raise an error.
    
    # Test 1: File Path (Baseline)
    print("\nTest 1: File Path")
    cv2.imwrite("test_temp.png", img)
    start = time.time()
    try:
        res = ocr.predict("test_temp.png")
        print(f"File Path Success. Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"File Path Failed: {e}")
    
    # Test 2: Numpy Array
    print("\nTest 2: Numpy Array")
    start = time.time()
    try:
        res = ocr.predict(img)
        print(f"Numpy Array Success. Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Numpy Array Failed: {e}")

    # Test 3: List of Arrays (Batch)
    print("\nTest 3: List of Arrays")
    batch = [img, img, img, img, img] # Batch of 5
    start = time.time()
    try:
        res = ocr.predict(batch)
        print(f"Batch List Success. Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Batch List Failed: {e}")

    if os.path.exists("test_temp.png"):
        os.remove("test_temp.png")

if __name__ == "__main__":
    test_ocr_inputs()

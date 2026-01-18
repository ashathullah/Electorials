def find_number_text_simple(image, pattern, roi_coords=None):
    """
    Simple approach: OCR the ROI normally, then use regex to find number patterns.
    """
    roi_offset_x, roi_offset_y = 0, 0
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        roi_offset_x, roi_offset_y = x1, y1
        image = image[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Try different preprocessing
    preprocessed = []
    
    # Otsu
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed.append(thresh1)
    
    # Adaptive
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    preprocessed.append(thresh2)
    
    # Inverted
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed.append(thresh3)
    
    # PSM modes
    psm_modes = [6, 7, 11]
    
    # Try combinations
    for thresh in preprocessed:
        for psm in psm_modes:
            try:
                # OCR without character whitelist
                config = f'--psm {psm}'
                text = pytesseract.image_to_string(thresh, config=config)
                
                # Find pattern matches using regex
                matches = re.findall(pattern, text)
                
                if matches:
                    return matches[0], (roi_offset_x, roi_offset_y, gray.shape[1], gray.shape[0])
                    
            except:
                continue
    
    return None, None

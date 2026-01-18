#!/usr/bin/env python3
"""
Simple debug image saver that just shows the full R

OIs.
"""
import cv2
from pathlib import Path
from rois import PINCODE_ROI, VOTERS_END_ROI


def denormalize_roi(roi, img_height, img_width):
    """Convert normalized ROI coordinates to pixel coordinates."""
    x1, y1, x2, y2 = roi
    return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)


def save_debug_image_simple(img, img_path, output_dir, pincode_text, voters_text):
    """Save stitched horizontal image showing the full pincode and voters_end ROIs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h_img, w_img = img.shape[:2]
    
    # Just show the full ROI - no cropping
    x1_pincode, y1_pincode, x2_pincode, y2_pincode = denormalize_roi(PINCODE_ROI, h_img, w_img)
    pincode_crop = img[y1_pincode:y2_pincode, x1_pincode:x2_pincode].copy()
    
    x1_voters, y1_voters, x2_voters, y2_voters = denormalize_roi(VOTERS_END_ROI, h_img, w_img)
    voters_crop = img[y1_voters:y2_voters, x1_voters:x2_voters].copy()
    
    # Stitch horizontally
    crops = [pincode_crop, voters_crop]
    max_h = max(c.shape[0] for c in crops)
    padded_crops = []
    for crop in crops:
        if crop.shape[0] < max_h:
            pad = max_h - crop.shape[0]
            crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded_crops.append(crop)
    
    stitched = cv2.hconcat(padded_crops)
    
    # Determine filename suffix
    if not pincode_text:
        suffix = 'missing_pincode'
    elif not voters_text:
        suffix = 'missing_voters'
    else:
        suffix = 'debug'
    
    output_path = output_dir / f"{img_path.stem}_{suffix}.png"
    cv2.imwrite(str(output_path), stitched)

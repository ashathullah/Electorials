#!/usr/bin/env python3
"""
Visualize the VOTERS_END_ROI to see what's being captured.
"""

import cv2
import numpy as np
from pathlib import Path
from rois import VOTERS_END_ROI

def denormalize_roi(roi, img_height, img_width):
    """Convert normalized ROI to pixel coordinates."""
    x1, y1, x2, y2 = roi
    return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)

# Test on document 10 and 100
test_docs = [
    'Tamil Nadu-(S22)_Thuraiyur-(AC146)_10',
    'Tamil Nadu-(S22)_Thuraiyur-(AC146)_100'
]

for doc_id in test_docs:
    img_path = Path(f'extracted_images/Thuraiyur/{doc_id}.png')
    if not img_path.exists():
        print(f"❌ {img_path} not found")
        continue
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Draw the current VOTERS_END_ROI
    x1, y1, x2, y2 = denormalize_roi(VOTERS_END_ROI, h, w)
    
    # Create visualization
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Add text showing coordinates
    cv2.putText(vis, f"ROI: ({x1},{y1}) to ({x2},{y2})", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save full image with ROI marked
    cv2.imwrite(f'vertical_stack_debug/{doc_id}_roi_marked.png', vis)
    
    # Extract and save the ROI itself
    roi_img = img[y1:y2, x1:x2]
    cv2.imwrite(f'vertical_stack_debug/{doc_id}_current_roi.png', roi_img)
    
    print(f"✅ {doc_id}:")
    print(f"   Image size: {w}x{h}")
    print(f"   ROI pixels: ({x1},{y1}) to ({x2},{y2})")
    print(f"   ROI size: {x2-x1}x{y2-y1}")
    print()

print("📁 Check vertical_stack_debug/ for:")
print("   *_roi_marked.png - Full page with ROI rectangle")
print("   *_current_roi.png - Extracted ROI region")

#!/usr/bin/env python3
"""Visualize ROIs on sample images to verify correctness."""

import cv2
import numpy as np
from pathlib import Path
import sys

# Import ROI definitions
from rois import PINCODE_ROI, VOTERS_END_ROI


def denormalize_roi(roi, img_height, img_width):
    """Convert normalized ROI coordinates to pixel coordinates."""
    x1, y1, x2, y2 = roi
    return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)


def visualize_rois(image_path):
    """Draw ROI rectangles on image and save visualization."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Create a copy for drawing
    vis = img.copy()
    
    # Draw PINCODE_ROI in RED
    x1, y1, x2, y2 = denormalize_roi(PINCODE_ROI, h, w)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(vis, "PINCODE_ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw VOTERS_END_ROI in BLUE
    x1, y1, x2, y2 = denormalize_roi(VOTERS_END_ROI, h, w)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(vis, "VOTERS_END_ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save visualization
    output_path = Path("roi_visualization.png")
    cv2.imwrite(str(output_path), vis)
    print(f"✓ Saved visualization to: {output_path.absolute()}")
    
    # Also show ROI coordinates
    print(f"\nImage size: {w}x{h}")
    print(f"PINCODE_ROI (normalized): {PINCODE_ROI}")
    print(f"PINCODE_ROI (pixels): {denormalize_roi(PINCODE_ROI, h, w)}")
    print(f"VOTERS_END_ROI (normalized): {VOTERS_END_ROI}")
    print(f"VOTERS_END_ROI (pixels): {denormalize_roi(VOTERS_END_ROI, h, w)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_rois.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: {image_path} does not exist")
        sys.exit(1)
    
    visualize_rois(image_path)

#!/usr/bin/env python3
"""
Standalone script to crop pincode and voters_end numbers from electoral roll images.
Stitches both numbers into a single horizontal image with a divider.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from pathlib import Path
import argparse
import sys

# Import ROI definitions
from rois import PINCODE_ROI, VOTER_END_ROI


def denormalize_roi(roi, img_height, img_width):
    """Convert normalized ROI coordinates to pixel coordinates."""
    x1, y1, x2, y2 = roi
    x1_px = int(x1 * img_width)
    y1_px = int(y1 * img_height)
    x2_px = int(x2 * img_width)
    y2_px = int(y2 * img_height)
    return x1_px, y1_px, x2_px, y2_px


def find_number_bbox(image, pattern, roi_coords=None, extract_longest=False):
    r"""
    Find bounding box of a number matching the pattern in the image.
    
    Args:
        image: Input image (numpy array)
        pattern: Regex pattern to match (e.g., r'\d{6}' for 6 digits)
        roi_coords: Optional (x1, y1, x2, y2) to search within a specific region
        extract_longest: If True, extract longest consecutive digit sequence from text
        
    Returns:
        (x, y, w, h) bounding box or None if not found
    """
    # If ROI specified, crop to that region first
    offset_x, offset_y = 0, 0
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        image = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    
    # Preprocess for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply multiple preprocessing techniques
    # 1. Simple threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Adaptive threshold for varying lighting
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # 3. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    
    # Try multiple OCR configurations
    configs = [
        '--psm 6 -c tessedit_char_whitelist=0123456789',  # Single block, digits only
        '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line, digits only
        '--psm 11 -c tessedit_char_whitelist=0123456789', # Sparse text, digits only
        '--psm 13 -c tessedit_char_whitelist=0123456789', # Raw line, digits only
    ]
    
    preprocessed_images = [thresh1, thresh2, thresh3]
    
    # Collect all candidates
    candidates = []
    
    # Try all combinations of preprocessing and config
    for thresh in preprocessed_images:
        for config in configs:
            try:
                ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                
                # Search for matching pattern
                for i, text in enumerate(ocr_data['text']):
                    original_text = text.strip()
                    
                    # Skip empty text
                    if not original_text:
                        continue
                    
                    # Remove any non-digit characters that might have slipped through
                    text = re.sub(r'\D', '', original_text)
                    
                    if not text:
                        continue
                    
                    # If extract_longest, find all digit sequences and use the longest one
                    if extract_longest:
                        digit_sequences = re.findall(r'\d+', text)
                        if digit_sequences:
                            text = max(digit_sequences, key=len)
                    
                    # Check if this matches our pattern (exact match)
                    if re.fullmatch(pattern, text):
                        x = ocr_data['left'][i] + offset_x
                        y = ocr_data['top'][i] + offset_y
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        
                        # Store candidate with its position (for selection)
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'text': text,
                            'conf': ocr_data['conf'][i],
                            'x': x,
                            'y': y
                        })
                    
                    # Also try: if text CONTAINS the required number of consecutive digits
                    elif re.search(pattern, text):
                        x = ocr_data['left'][i] + offset_x
                        y = ocr_data['top'][i] + offset_y
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'text': text,
                            'conf': ocr_data['conf'][i],
                            'x': x,
                            'y': y
                        })
            except Exception:
                continue
    
    # Select best candidate if we found any
    if candidates:
        # Strategy: Select the bottom-most and rightmost candidate
        # (voters_end number is typically at the bottom-right of the ROI)
        # Sort by: 1) bottom-most (y + h), 2) rightmost (x + w)
        best = max(candidates, key=lambda c: (c['y'] + c['bbox'][3], c['x'] + c['bbox'][2]))
        
        x, y, w, h = best['bbox']
        
        # Add generous padding to ensure full numbers are captured
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        return (x, y, w, h)
    
    return None


def create_stitched_image(img, pincode_bbox, voters_bbox, divider_width=3, spacing=10):
    """
    Create a stitched image with pincode and voters_end separated by a divider.
    
    Args:
        img: Original image
        pincode_bbox: (x, y, w, h) for pincode
        voters_bbox: (x, y, w, h) for voters_end
        divider_width: Width of the divider line
        spacing: Spacing around the divider
        
    Returns:
        Stitched image
    """
    # Extract crops
    p_x, p_y, p_w, p_h = pincode_bbox
    v_x, v_y, v_w, v_h = voters_bbox
    
    pincode_crop = img[p_y:p_y+p_h, p_x:p_x+p_w]
    voters_crop = img[v_y:v_y+v_h, v_x:v_x+v_w]
    
    # Determine final image dimensions
    max_height = max(p_h, v_h)
    total_width = p_w + spacing + divider_width + spacing + v_w
    
    # Create white canvas
    stitched = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    
    # Calculate vertical centering
    p_offset_y = (max_height - p_h) // 2
    v_offset_y = (max_height - v_h) // 2
    
    # Place pincode on the left
    stitched[p_offset_y:p_offset_y+p_h, 0:p_w] = pincode_crop
    
    # Draw divider
    divider_x = p_w + spacing
    cv2.line(stitched, 
             (divider_x, 0), 
             (divider_x, max_height), 
             (0, 0, 0), 
             divider_width)
    
    # Place voters_end on the right
    voters_x = divider_x + divider_width + spacing
    stitched[v_offset_y:v_offset_y+v_h, voters_x:voters_x+v_w] = voters_crop
    
    return stitched


def process_image(input_path, output_dir=None, visualize=False):
    """
    Process a single image to extract and stitch pincode and voters_end numbers.
    
    Args:
        input_path: Path to input image
        output_dir: Output directory (default: same as input)
        visualize: If True, show image with bounding boxes
        
    Returns:
        Path to output image or None if failed
    """
    input_path = Path(input_path)
    
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"❌ Error: Could not read image: {input_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Get ROI coordinates in pixels
    pincode_roi_px = denormalize_roi(PINCODE_ROI, height, width)
    voters_roi_px = denormalize_roi(VOTER_END_ROI, height, width)
    
    print(f"📄 Processing: {input_path.name}")
    print(f"   Image size: {width}x{height}")
    
    # Find pincode (6 digits)
    print(f"   🔍 Searching for pincode (6 digits)...")
    pincode_bbox = find_number_bbox(img, r'\d{6}', pincode_roi_px)
    
    if pincode_bbox is None:
        print(f"   ⚠️  Warning: Pincode not found via OCR, using full ROI")
        x1, y1, x2, y2 = pincode_roi_px
        pincode_bbox = (x1, y1, x2-x1, y2-y1)
    else:
        print(f"   ✓ Pincode found at {pincode_bbox}")
    
    # Find voters_end (1-4 digits)
    print(f"   🔍 Searching for voters_end (1-4 digits)...")
    voters_bbox = find_number_bbox(img, r'\d{1,4}', voters_roi_px)
    
    if voters_bbox is None:
        print(f"   ⚠️  Warning: Voters_end not found via OCR, using full ROI")
        x1, y1, x2, y2 = voters_roi_px
        voters_bbox = (x1, y1, x2-x1, y2-y1)
    else:
        print(f"   ✓ Voters_end found at {voters_bbox}")
    
    # Create stitched image
    stitched = create_stitched_image(img, pincode_bbox, voters_bbox)
    
    # Determine output path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
    else:
        output_path = input_path.parent / input_path.name
    
    # Save
    cv2.imwrite(str(output_path), stitched)
    print(f"   ✅ Saved to: {output_path}")
    
    # Visualization
    if visualize:
        # Draw bounding boxes on original image
        vis_img = img.copy()
        p_x, p_y, p_w, p_h = pincode_bbox
        v_x, v_y, v_w, v_h = voters_bbox
        
        cv2.rectangle(vis_img, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 255, 0), 2)
        cv2.putText(vis_img, "PINCODE", (p_x, p_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.rectangle(vis_img, (v_x, v_y), (v_x+v_w, v_y+v_h), (255, 0, 0), 2)
        cv2.putText(vis_img, "VOTERS_END", (v_x, v_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show both images
        cv2.imshow("Original with Bounding Boxes", vis_img)
        cv2.imshow("Stitched Result", stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Crop and stitch pincode and voters_end numbers from electoral roll images"
    )
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Process all subdirectories recursively (preserves folder structure by default)")
    parser.add_argument("--flatten", action="store_true",
                        help="Flatten output to single directory (only with -r, default is to preserve structure)")
    parser.add_argument("-v", "--visualize", action="store_true", 
                        help="Visualize bounding boxes")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Process single file or directory
    if input_path.is_file():
        process_image(input_path, args.output, args.visualize)
    elif input_path.is_dir():
        if args.recursive:
            # Process all subdirectories recursively
            all_images = list(input_path.rglob("*.png"))
            if not all_images:
                print(f"❌ No PNG files found in {input_path} or its subdirectories")
                return 1
            
            # Group by directory for better organization
            from collections import defaultdict
            images_by_dir = defaultdict(list)
            for img in all_images:
                images_by_dir[img.parent].append(img)
            
            total_images = len(all_images)
            total_dirs = len(images_by_dir)
            
            print(f"📁 Found {total_images} images across {total_dirs} directories")
            if args.flatten:
                print(f"📦 Output mode: Flattened (all in one directory)\n")
            else:
                print(f"📦 Output mode: Preserving directory structure\n")
            
            processed = 0
            for dir_path in sorted(images_by_dir.keys()):
                images = images_by_dir[dir_path]
                rel_path = dir_path.relative_to(input_path)
                
                print(f"📂 Processing directory: {rel_path} ({len(images)} images)")
                
                for img_file in images:
                    # Determine output path
                    if args.output:
                        output_dir = Path(args.output)
                        if not args.flatten:
                            # Preserve subdirectory structure (default behavior)
                            rel_dir = img_file.parent.relative_to(input_path)
                            output_dir = output_dir / rel_dir
                        process_image(img_file, str(output_dir), args.visualize)
                    else:
                        process_image(img_file, None, args.visualize)
                    
                    processed += 1
                    if processed % 10 == 0:
                        print(f"   Progress: {processed}/{total_images} ({100*processed//total_images}%)")
                
                print()
            
            print(f"✅ Completed! Processed {processed} images across {total_dirs} directories")
        else:
            # Process only current directory (non-recursive)
            image_files = list(input_path.glob("*.png"))
            if not image_files:
                print(f"❌ No PNG files found in {input_path}")
                print(f"💡 Tip: Use -r/--recursive to process subdirectories")
                return 1
            
            print(f"📁 Found {len(image_files)} images to process\n")
            
            for img_file in image_files:
                process_image(img_file, args.output, args.visualize)
                print()
    else:
        print(f"❌ Error: {input_path} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

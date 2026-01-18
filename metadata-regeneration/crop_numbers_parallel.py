#!/usr/bin/env python3
"""
Parallel version: Extract pincode and voters_end numbers from electoral roll images.
Uses multiprocessing for speed and extracts data to JSON.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from pathlib import Path
import argparse
import sys
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict
import time

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


def find_number_text(image, pattern, roi_coords=None):
    """
    Find text of a number matching the pattern in the image.
    Returns the actual text string, not bounding box.
    """
    # If ROI specified, crop to that region first
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        image = image[y1:y2, x1:x2]
    
    # Preprocess for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply multiple preprocessing techniques
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    
    # Try multiple OCR configurations
    configs = [
        '--psm 6 -c tessedit_char_whitelist=0123456789',
        '--psm 7 -c tessedit_char_whitelist=0123456789',
        '--psm 11 -c tessedit_char_whitelist=0123456789',
        '--psm 13 -c tessedit_char_whitelist=0123456789',
    ]
    
    preprocessed_images = [thresh1, thresh2, thresh3]
    candidates = []
    
    # Try all combinations
    for thresh in preprocessed_images:
        for config in configs:
            try:
                ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                
                for i, text in enumerate(ocr_data['text']):
                    original_text = text.strip()
                    if not original_text:
                        continue
                    
                    # Remove non-digits
                    text = re.sub(r'\D', '', original_text)
                    if not text:
                        continue
                    
                    # Check if matches pattern
                    if re.fullmatch(pattern, text) or re.search(pattern, text):
                        y = ocr_data['top'][i]
                        h = ocr_data['height'][i]
                        x = ocr_data['left'][i]
                        w = ocr_data['width'][i]
                        
                        candidates.append({
                            'text': text,
                            'conf': ocr_data['conf'][i],
                            'y': y,
                            'h': h,
                            'x': x,
                            'w': w
                        })
            except Exception:
                continue
    
    # Select best candidate (bottom-most, then rightmost)
    if candidates:
        best = max(candidates, key=lambda c: (c['y'] + c['h'], c['x'] + c['w']))
        # Extract just the matching digits
        match = re.search(pattern, best['text'])
        if match:
            return match.group()
        return best['text']
    
    return None


def find_number_bbox(image, pattern, roi_coords=None):
    """Find bounding box for cropping (same as before but returns bbox)."""
    # If ROI specified, crop to that region first
    offset_x, offset_y = 0, 0
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        image = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    
    configs = [
        '--psm 6 -c tessedit_char_whitelist=0123456789',
        '--psm 7 -c tessedit_char_whitelist=0123456789',
        '--psm 11 -c tessedit_char_whitelist=0123456789',
        '--psm 13 -c tessedit_char_whitelist=0123456789',
    ]
    
    preprocessed_images = [thresh1, thresh2, thresh3]
    candidates = []
    
    for thresh in preprocessed_images:
        for config in configs:
            try:
                ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                
                for i, text in enumerate(ocr_data['text']):
                    original_text = text.strip()
                    if not original_text:
                        continue
                    
                    text = re.sub(r'\D', '', original_text)
                    if not text:
                        continue
                    
                    if re.fullmatch(pattern, text) or re.search(pattern, text):
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
    
    if candidates:
        best = max(candidates, key=lambda c: (c['y'] + c['bbox'][3], c['x'] + c['bbox'][2]))
        x, y, w, h = best['bbox']
        
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        return (x, y, w, h)
    
    return None


def create_stitched_image(img, pincode_bbox, voters_bbox, divider_width=3, spacing=10):
    """Create stitched image with pincode and voters_end."""
    p_x, p_y, p_w, p_h = pincode_bbox
    v_x, v_y, v_w, v_h = voters_bbox
    
    pincode_crop = img[p_y:p_y+p_h, p_x:p_x+p_w]
    voters_crop = img[v_y:v_y+v_h, v_x:v_x+v_w]
    
    max_height = max(p_h, v_h)
    total_width = p_w + spacing + divider_width + spacing + v_w
    
    stitched = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    
    p_offset_y = (max_height - p_h) // 2
    v_offset_y = (max_height - v_h) // 2
    
    stitched[p_offset_y:p_offset_y+p_h, 0:p_w] = pincode_crop
    
    divider_x = p_w + spacing
    cv2.line(stitched, (divider_x, 0), (divider_x, max_height), (0, 0, 0), divider_width)
    
    voters_x = divider_x + divider_width + spacing
    stitched[v_offset_y:v_offset_y+v_h, voters_x:voters_x+v_w] = voters_crop
    
    return stitched


def process_single_image(img_path, output_dir=None, save_images=True):
    """
    Process a single image and return extracted data.
    
    Args:
        img_path: Path to input image
        output_dir: Output directory for cropped images (if save_images=True)
        save_images: Whether to save cropped images
        
    Returns:
        dict with document_id, pincode, voters_end, status
    """
    img_path = Path(img_path)
    
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return {
                'document_id': img_path.stem,
                'pincode': None,
                'voters_end': None,
                'status': 'error_read_image'
            }
        
        height, width = img.shape[:2]
        
        # Get ROI coordinates
        pincode_roi_px = denormalize_roi(PINCODE_ROI, height, width)
        voters_roi_px = denormalize_roi(VOTER_END_ROI, height, width)
        
        # Extract text numbers
        pincode_text = find_number_text(img, r'\d{6}', pincode_roi_px)
        voters_text = find_number_text(img, r'\d{1,4}', voters_roi_px)
        
        result = {
            'document_id': img_path.stem,
            'pincode': pincode_text,
            'voters_end': voters_text,
            'status': 'success'
        }
        
        # Optionally save cropped images
        if save_images and output_dir:
            # Find bounding boxes for cropping
            pincode_bbox = find_number_bbox(img, r'\d{6}', pincode_roi_px)
            voters_bbox = find_number_bbox(img, r'\d{1,4}', voters_roi_px)
            
            if pincode_bbox is None:
                x1, y1, x2, y2 = pincode_roi_px
                pincode_bbox = (x1, y1, x2-x1, y2-y1)
            
            if voters_bbox is None:
                x1, y1, x2, y2 = voters_roi_px
                voters_bbox = (x1, y1, x2-x1, y2-y1)
            
            stitched = create_stitched_image(img, pincode_bbox, voters_bbox)
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / img_path.name
            
            cv2.imwrite(str(output_path), stitched)
        
        return result
        
    except Exception as e:
        return {
            'document_id': img_path.stem,
            'pincode': None,
            'voters_end': None,
            'status': f'error: {str(e)}'
        }


def process_directory_batch(args_tuple):
    """Process a directory's worth of images (for multiprocessing)."""
    dir_path, images, output_dir, save_images = args_tuple
    
    results = []
    for img_path in images:
        if output_dir:
            # Preserve directory structure
            rel_dir = img_path.parent.name
            img_output_dir = Path(output_dir) / rel_dir if save_images else None
        else:
            img_output_dir = None
        
        result = process_single_image(img_path, img_output_dir, save_images)
        results.append(result)
    
    return dir_path, results


def main():
    parser = argparse.ArgumentParser(
        description="Extract pincode and voters_end numbers (parallel version)"
    )
    parser.add_argument("input", help="Input directory")
    parser.add_argument("-o", "--output", help="Output directory for cropped images")
    parser.add_argument("--json-output", default="extracted_numbers",
                        help="Output directory for JSON files (default: extracted_numbers)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving cropped images (JSON only)")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help=f"Number of parallel jobs (default: {cpu_count()})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.is_dir():
        print(f"❌ Error: {input_path} is not a directory")
        return 1
    
    # Find all images recursively
    all_images = list(input_path.rglob("*.png"))
    if not all_images:
        print(f"❌ No PNG files found in {input_path}")
        return 1
    
    # Group by directory
    images_by_dir = defaultdict(list)
    for img in all_images:
        images_by_dir[img.parent].append(img)
    
    total_images = len(all_images)
    total_dirs = len(images_by_dir)
    num_workers = args.jobs or cpu_count()
    
    print(f"📁 Found {total_images} images across {total_dirs} directories")
    print(f"🚀 Using {num_workers} parallel workers")
    print(f"💾 JSON output: {args.json_output}/")
    if not args.no_images:
        print(f"🖼️  Image output: {args.output or 'same as input'}/")
    else:
        print(f"🖼️  Skipping image generation (JSON only)")
    print()
    
    # Prepare work batches
    work_batches = []
    for dir_path, images in sorted(images_by_dir.items()):
        work_batches.append((dir_path, images, args.output, not args.no_images))
    
    # Process in parallel
    start_time = time.time()
    all_results_by_dir = {}
    
    with Pool(num_workers) as pool:
        for i, (dir_path, results) in enumerate(pool.imap_unordered(process_directory_batch, work_batches)):
            rel_path = dir_path.relative_to(input_path)
            all_results_by_dir[str(rel_path)] = results
            
            processed = sum(len(r) for r in all_results_by_dir.values())
            print(f"✓ [{i+1}/{total_dirs}] {rel_path}: {len(results)} images | Total: {processed}/{total_images} ({100*processed//total_images}%)")
    
    # Save JSON files (one per directory)
    json_output_path = Path(args.json_output)
    json_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving JSON files...")
    for dir_name, results in all_results_by_dir.items():
        # Sanitize directory name for filename
        json_filename = dir_name.replace('/', '_').replace(' ', '_') + '.json'
        json_path = json_output_path / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if args.verbose:
            print(f"  ✓ {json_path}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f}s ({total_images/elapsed:.1f} imgs/sec)")
    print(f"📊 Processed {total_images} images across {total_dirs} directories")
    print(f"💾 JSON files saved to: {json_output_path}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

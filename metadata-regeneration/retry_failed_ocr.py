#!/usr/bin/env python3
"""
Second-pass OCR script with aggressive preprocessing for failed extractions.
Can run in two modes:
1. Normal: Process all images with intensive OCR
2. Retry-failed: Only reprocess images that failed in the first pass
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import argparse
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
from tqdm import tqdm

# Import ROIs from the main script
from rois import PINCODE_ROI, VOTER_END_ROI


def extract_digits_only(text):
    """Extract only digits from text."""
    return re.sub(r'\D', '', text)


def denormalize_roi(roi, img_h, img_w):
    """Convert normalized ROI to pixel coordinates."""
    x1_norm, y1_norm, x2_norm, y2_norm = roi
    x1 = int(x1_norm * img_w)
    y1 = int(y1_norm * img_h)
    x2 = int(x2_norm * img_w)
    y2 = int(y2_norm * img_h)
    return x1, y1, x2, y2


def aggressive_ocr(image, pattern, roi_coords=None, expected_digits=6):
    """
    Aggressive OCR with multiple preprocessing methods.
    Tries more combinations than the standard pipeline.
    """
    roi_offset_x, roi_offset_y = 0, 0
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        roi_offset_x, roi_offset_y = x1, y1
        image = image[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Expanded PSM modes
    configs = [
        '--psm 8 -c tessedit_char_whitelist=0123456789',
        '--psm 7 -c tessedit_char_whitelist=0123456789',
        '--psm 6 -c tessedit_char_whitelist=0123456789',
        '--psm 13 -c tessedit_char_whitelist=0123456789',
        '--psm 11 -c tessedit_char_whitelist=0123456789',
    ]
    
    candidates = []
    
    # AGGRESSIVE STRATEGY: Try morphological isolation for pincodes
    if expected_digits >= 5:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)  # More dilation
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        number_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Relaxed criteria for pincode detection
            if aspect_ratio > 2.0 and area > 300 and h > 8:  # More lenient
                number_regions.append((x, y, w, h, area))
        
        number_regions.sort(key=lambda r: r[4], reverse=True)
        
        # Try top 3 regions with multiple preprocessing
        for x, y, w, h, _ in number_regions[:3]:
            pad = 7
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
            cropped = gray[y1:y2, x1:x2]
            
            # Try 3 different preprocessing methods on the crop
            _, thresh1 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh2 = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            # Inverted
            _, thresh3 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            for thresh_img in [thresh1, thresh2, thresh3]:
                for config in configs[:3]:
                    try:
                        text = pytesseract.image_to_string(thresh_img, config=config).strip()
                        digits = extract_digits_only(text)
                        
                        if digits and re.fullmatch(pattern, digits):
                            candidates.append({
                                'text': digits,
                                'y_bottom': y + h,
                                'x_right': x + w,
                                'confidence': 90,
                                'bbox': (x, y, w, h)
                            })
                    except:
                        continue
    
    # AGGRESSIVE FULL ROI OCR - More preprocessing combinations
    if not candidates:
        # Preprocessing variations
        preprocessed = []
        
        # 1. Standard Otsu
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh1)
        
        # 2. Adaptive Gaussian
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(thresh2)
        
        # 3. Adaptive Mean
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(thresh3)
        
        # 4. Bilateral filtering + Otsu
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh4)
        
        # 5. Sharpening + Otsu
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        _, thresh5 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh5)
        
        # 6. Morphological closing
        kernel = np.ones((2, 2), np.uint8)
        thresh6 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        preprocessed.append(thresh6)
        
        # 7. Inverted
        _, thresh7 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed.append(thresh7)
        
        for thresh in preprocessed:
            for config in configs:
                try:
                    ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                    
                    for i, text in enumerate(ocr_data['text']):
                        if not text.strip():
                            continue
                        
                        digits = extract_digits_only(text)
                        if not digits:
                            continue
                        
                        if re.fullmatch(pattern, digits):
                            y = ocr_data['top'][i]
                            h = ocr_data['height'][i]
                            x = ocr_data['left'][i]
                            w = ocr_data['width'][i]
                            conf = ocr_data['conf'][i]
                            
                            candidates.append({
                                'text': digits,
                                'y_bottom': y + h,
                                'x_right': x + w,
                                'confidence': int(conf) if conf != '-1' else 0,
                                'bbox': (x, y, w, h)
                            })
                except:
                    continue
    
    if candidates:
        best = max(candidates, key=lambda c: (c['confidence'], c['y_bottom'], c['x_right']))
        x, y, w, h = best['bbox']
        return best['text'], (roi_offset_x + x, roi_offset_y + y, w, h)
    
    return None, None


def save_debug_image(img, img_path, output_dir, pincode_data, voters_data):
    """Save stitched horizontal image of OCR-detected pincode and voters_end regions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pincode_text, pincode_bbox = pincode_data
    voters_text, voters_bbox = voters_data
    
    h_img, w_img = img.shape[:2]
    crops = []
    
    # For pincode
    if pincode_text and pincode_bbox:
        x, y, w, h = pincode_bbox
        pad_x, pad_y = int(w * 0.2), int(h * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        pincode_crop = img[y1:y2, x1:x2].copy()
    else:
        x1_roi, y1_roi, x2_roi, y2_roi = denormalize_roi(PINCODE_ROI, h_img, w_img)
        pincode_crop = img[y1_roi:y2_roi, x1_roi:x2_roi].copy()
    
    crops.append(pincode_crop)
    
    # For voters_end
    if voters_text and voters_bbox:
        x, y, w, h = voters_bbox
        pad_x, pad_y = int(w * 0.2), int(h * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        voters_crop = img[y1:y2, x1:x2].copy()
    else:
        x1_roi, y1_roi, x2_roi, y2_roi = denormalize_roi(VOTER_END_ROI, h_img, w_img)
        voters_crop = img[y1_roi:y2_roi, x1_roi:x2_roi].copy()
    
    crops.append(voters_crop)
    
    # Stitch horizontally
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


def process_single_image(img_path, retry_mode=False, existing_result=None, debug_dir=None):
    """Process a single image with aggressive OCR."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        h_img, w_img = img.shape[:2]
        
        # Skip if already successful in retry mode
        if retry_mode and existing_result:
            if existing_result.get('status') == 'success':
                return existing_result
        
        # Extract pincode
        pincode_roi = denormalize_roi(PINCODE_ROI, h_img, w_img)
        pincode_text, pincode_bbox = aggressive_ocr(img, r'\d{6}', pincode_roi, expected_digits=6)
        
        # Extract voters_end
        voters_roi = denormalize_roi(VOTER_END_ROI, h_img, w_img)
        voters_text, voters_bbox = aggressive_ocr(img, r'\d{1,4}', voters_roi, expected_digits=3)
        
        # Determine status
        status = 'success'
        if not pincode_text:
            status = 'missing_pincode'
        elif not voters_text:
            status = 'missing_voters_end'
        
        # Save debug image if failed and debug_dir is specified
        if status != 'success' and debug_dir:
            dir_name = img_path.parent.name
            debug_output = Path(debug_dir) / dir_name
            save_debug_image(img, img_path, debug_output, (pincode_text, pincode_bbox), (voters_text, voters_bbox))
        
        return {
            'document_id': img_path.stem,
            'pincode': pincode_text,
            'voters_end': voters_text,
            'status': status,
        }
    
    except Exception as e:
        return {
            'document_id': img_path.stem if img_path else 'unknown',
            'pincode': None,
            'voters_end': None,
            'status': 'error',
            'error': str(e)
        }


def load_existing_results(json_dir):
    """Load existing JSON results to identify failed images."""
    json_dir = Path(json_dir)
    failed_images = {}
    all_results = {}
    
    for json_file in json_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            directory_name = data['metadata']['directory']
            
            for result in data['results']:
                doc_id = result['document_id']
                all_results[doc_id] = result
                
                if result['status'] != 'success':
                    failed_images[doc_id] = {
                        'directory': directory_name,
                        'result': result
                    }
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return failed_images, all_results


def main():
    parser = argparse.ArgumentParser(description='Aggressive second-pass OCR for failed extractions')
    parser.add_argument('image_dir', help='Directory containing extracted images')
    parser.add_argument('--json-dir', required=True, help='Directory containing JSON results')
    parser.add_argument('--output-dir', required=True, help='Output directory for updated JSONs')
    parser.add_argument('--debug-dir', help='Output directory for debug images (optional)')
    parser.add_argument('--retry-failed', action='store_true', 
                        help='Only retry images that failed in the first pass')
    parser.add_argument('-j', '--jobs', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Loading existing results...")
    failed_images, all_results = load_existing_results(json_dir)
    
    if args.retry_failed:
        print(f"🔄 RETRY MODE: Found {len(failed_images)} failed images to reprocess")
        images_to_process = []
        
        for doc_id, info in failed_images.items():
            img_path = image_dir / info['directory'] / f"{doc_id}.png"
            if img_path.exists():
                images_to_process.append((img_path, info['result']))
        
        print(f"📁 {len(images_to_process)} images found on disk")
    else:
        print(f"🔄 NORMAL MODE: Processing all images with aggressive OCR")
        images_to_process = [(img, None) for img in image_dir.rglob('*.png')]
        print(f"📁 Found {len(images_to_process)} images")
    
    if not images_to_process:
        print("❌ No images to process!")
        return
    
    # Group by directory
    results_by_dir = {}
    
    print(f"\n🚀 Processing with {args.jobs} workers...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_single_image, img_path, args.retry_failed, existing, debug_dir): img_path
            for img_path, existing in images_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                img_path = futures[future]
                dir_name = img_path.parent.name
                
                if dir_name not in results_by_dir:
                    results_by_dir[dir_name] = []
                results_by_dir[dir_name].append(result)
    
    elapsed = time.time() - start_time
    
    # Save results
    print("\n💾 Saving results...")
    total_success = 0
    total_missing_pincode = 0
    total_missing_voters = 0
    
    for dir_name, results in results_by_dir.items():
        # Sort by document_id
        results.sort(key=lambda r: r['document_id'])
        
        # Count statuses
        success = sum(1 for r in results if r['status'] == 'success')
        missing_pc = sum(1 for r in results if r['status'] == 'missing_pincode')
        missing_vt = sum(1 for r in results if r['status'] == 'missing_voters_end')
        
        total_success += success
        total_missing_pincode += missing_pc
        total_missing_voters += missing_vt
        
        output_file = output_dir / f"{dir_name}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'directory': dir_name,
                    'total_images': len(results),
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'retry_failed' if args.retry_failed else 'aggressive_full',
                    'status': 'completed'
                },
                'results': results
            }, f, indent=2)
        
        print(f"  ✓ {dir_name}.json: {len(results)} images | ✓{success} ❌PC:{missing_pc} ❌VT:{missing_vt}")
    
    total_images = sum(len(r) for r in results_by_dir.values())
    print(f"\n{'='*60}")
    print(f"✅ COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Total images: {total_images}")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print(f"⚡ Speed: {total_images/elapsed:.2f} images/second")
    print(f"\n📈 Status Summary:")
    print(f"  ✅ Success: {total_success} ({100*total_success/total_images:.1f}%)")
    print(f"  ❌ Missing pincode: {total_missing_pincode} ({100*total_missing_pincode/total_images:.1f}%)")
    print(f"  ❌ Missing voters_end: {total_missing_voters} ({100*total_missing_voters/total_images:.1f}%)")
    print(f"\n💾 Output saved to: {output_dir}")


if __name__ == '__main__':
    main()

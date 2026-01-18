#!/usr/bin/env python3
"""
Optimized parallel extraction of pincode and voters_end numbers.
JSON output only, with timing and debug images for missing values.
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
import time
from datetime import datetime

# Import ROI definitions
from rois import PINCODE_ROI, VOTERS_END_ROI


def denormalize_roi(roi, img_height, img_width):
    """Convert normalized ROI coordinates to pixel coordinates."""
    x1, y1, x2, y2 = roi
    return int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)


def remove_table_lines_inpaint(bgr_img):
    """Remove table/grid lines via masking + inpainting; tends to preserve digits better."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    horiz_len = max(25, min(120, bgr_img.shape[1] // 2))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horizontal = cv2.erode(bw, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    vert_len = max(25, min(120, bgr_img.shape[0] // 2))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.erode(bw, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    cleaned = cv2.inpaint(bgr_img, line_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, line_mask


def stacked_digits_ocr(img, pincode_roi_px, voters_roi_px, debug_dir=None, doc_id=None):
    """OCR digits from a horizontal stitch: [pincode_roi | sep | voters_roi(cleaned)].

    Picks pincode from the left region (6 digits starting with 6) and voters_end
    from the right region (3-4 digits, bottom-most by y).
    """
    x1_pc, y1_pc, x2_pc, y2_pc = pincode_roi_px
    x1_ve, y1_ve, x2_ve, y2_ve = voters_roi_px

    pincode_roi = img[y1_pc:y2_pc, x1_pc:x2_pc]
    voters_roi = img[y1_ve:y2_ve, x1_ve:x2_ve]
    voters_cleaned, voters_line_mask = remove_table_lines_inpaint(voters_roi)

    # Pad to same height for hconcat
    target_h = max(pincode_roi.shape[0], voters_cleaned.shape[0])
    def pad_to_h(bgr, h):
        if bgr.shape[0] == h:
            return bgr
        pad = h - bgr.shape[0]
        return cv2.copyMakeBorder(bgr, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    pincode_roi = pad_to_h(pincode_roi, target_h)
    voters_cleaned = pad_to_h(voters_cleaned, target_h)

    sep_w = 30
    separator = np.ones((target_h, sep_w, 3), dtype=np.uint8) * 255
    stitched = cv2.hconcat([pincode_roi, separator, voters_cleaned])

    if debug_dir and doc_id:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{doc_id}_stacked_h.png"), stitched)
        cv2.imwrite(str(out_dir / f"{doc_id}_voters_line_mask.png"), voters_line_mask)

    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    config = '--psm 6 -c tessedit_char_whitelist=0123456789'
    ocr = pytesseract.image_to_data(gray, output_type=Output.DICT, lang='eng', config=config)

    boundary_x = (pincode_roi.shape[1] + sep_w) * 3  # scaled
    best_pincode = None
    best_pincode_score = (-1, -1)  # (conf, x)

    best_voters = None
    best_voters_score = (-1, -1)  # (y, conf)

    for i, raw_text in enumerate(ocr['text']):
        text = (raw_text or '').strip()
        if not text:
            continue
        if not text.isdigit():
            continue

        try:
            conf = int(float(ocr['conf'][i])) if ocr['conf'][i] != '-1' else 0
        except Exception:
            conf = 0

        x = ocr['left'][i]
        y = ocr['top'][i]
        w = ocr['width'][i]
        h = ocr['height'][i]
        cx = x + w // 2
        cy = y + h // 2

        if len(text) == 6 and text.startswith('6') and cx < boundary_x:
            score = (conf, cx)
            if score > best_pincode_score:
                best_pincode_score = score
                best_pincode = text
        elif 3 <= len(text) <= 4 and cx > boundary_x:
            # Bottom-most wins; tie-break on confidence
            score = (cy, conf)
            if score > best_voters_score:
                best_voters_score = score
                best_voters = text

    # Secondary pass for pincode: sometimes one digit drops in the stitched OCR.
    if best_pincode is None:
        pc_gray = cv2.cvtColor(pincode_roi, cv2.COLOR_BGR2GRAY)
        pc_gray = cv2.resize(pc_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        pc_imgs = [
            cv2.threshold(pc_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(pc_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            cv2.threshold(pc_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        ]
        for pc_img in pc_imgs:
            for psm in (6, 7, 8, 11, 13):
                txt = pytesseract.image_to_string(pc_img, lang='eng', config=f'--psm {psm} -c tessedit_char_whitelist=0123456789')
                m = re.search(r'6\d{5}', txt)
                if m:
                    best_pincode = m.group(0)
                    break
            if best_pincode:
                break

    return best_pincode, best_voters


def extract_digits_only(text):
    """Extract only digits from text."""
    return re.sub(r'\D', '', text)


def aggressive_ocr(image, pattern, roi_coords=None, expected_digits=6):
    """
    Aggressive OCR with extensive preprocessing.
    Used as fallback when standard OCR fails.
    """
    roi_offset_x, roi_offset_y = 0, 0
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        roi_offset_x, roi_offset_y = x1, y1
        image = image[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    configs = [
        '--psm 8 -c tessedit_char_whitelist=0123456789',
        '--psm 7 -c tessedit_char_whitelist=0123456789',
        '--psm 6 -c tessedit_char_whitelist=0123456789',
        '--psm 13 -c tessedit_char_whitelist=0123456789',
        '--psm 11 -c tessedit_char_whitelist=0123456789',
    ]
    
    candidates = []
    
    # Try morphological isolation for pincodes with relaxed criteria
    if expected_digits >= 5:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        number_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 2.0 and area > 300 and h > 8:
                number_regions.append((x, y, w, h, area))
        
        number_regions.sort(key=lambda r: r[4], reverse=True)
        
        for x, y, w, h, _ in number_regions[:3]:
            pad = 7
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            cropped = gray[y1:y2, x1:x2]
            
            _, thresh1 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh2 = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            for thresh_img in [thresh1, thresh2]:
                for config in configs[:3]:
                    try:
                        text = pytesseract.image_to_string(thresh_img, lang='tam+eng', config=config).strip()
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
    
    # Aggressive full ROI OCR with 7 preprocessing methods
    if not candidates:
        preprocessed = []
        
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh1)
        
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(thresh2)
        
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(thresh3)
        
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh4)
        
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        _, thresh5 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(thresh5)
        
        kernel = np.ones((2, 2), np.uint8)
        thresh6 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        preprocessed.append(thresh6)
        
        _, thresh7 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed.append(thresh7)
        
        for thresh in preprocessed:
            for config in configs:
                try:
                    ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, lang='tam+eng', config=config)
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


def find_number_text(image, pattern, roi_coords=None):
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
                text = pytesseract.image_to_string(thresh, lang='tam+eng', config=config)
                
                # Find pattern matches using regex
                matches = re.findall(pattern, text)
                
                if matches:
                    # For 6-digit numbers (pincodes), filter to those starting with '6' (Tamil Nadu)
                    if r'\d{6}' in pattern:
                        valid_matches = [m for m in matches if m.startswith('6')]
                        if valid_matches:
                            return valid_matches[0], (roi_offset_x, roi_offset_y, gray.shape[1], gray.shape[0])
                    else:
                        # For voters_end, return first match
                        return matches[0], (roi_offset_x, roi_offset_y, gray.shape[1], gray.shape[0])
                    
            except:
                continue
    
    return None, None


def save_debug_image(img, img_path, output_dir, pincode_data, voters_data):
    """Save full ROI regions without cropping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pincode_text, pincode_bbox = pincode_data
    voters_text, voters_bbox = voters_data
    
    h_img, w_img = img.shape[:2]
    
    # Just show full ROIs
    x1_pc, y1_pc, x2_pc, y2_pc = denormalize_roi(PINCODE_ROI, h_img, w_img)
    pincode_crop = img[y1_pc:y2_pc, x1_pc:x2_pc].copy()
    
    x1_ve, y1_ve, x2_ve, y2_ve = denormalize_roi(VOTERS_END_ROI, h_img, w_img)
    voters_crop = img[y1_ve:y2_ve, x1_ve:x2_ve].copy()
    
    # Stitch
    crops = [pincode_crop, voters_crop]
    max_h = max(c.shape[0] for c in crops)
    padded = []
    for crop in crops:
        if crop.shape[0] < max_h:
            pad = max_h - crop.shape[0]
            crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        padded.append(crop)
    stitched = cv2.hconcat(padded)
    
    suffix = "missing_pincode" if not pincode_text else ("missing_voters" if not voters_text else "debug")
    output_path = output_dir / f"{img_path.stem}_{suffix}.png"
    cv2.imwrite(str(output_path), stitched)


def process_single_image(args_tuple):
    """Process a single image and return extracted data with timing."""
    img_path, debug_dir, use_aggressive = args_tuple
    
    start_time = time.time()
    img_path = Path(img_path)
    
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return {
                'document_id': img_path.stem,
                'pincode': None,
                'voters_end': None,
                'status': 'error_read_image',
                'time_ms': int((time.time() - start_time) * 1000)
            }
        
        height, width = img.shape[:2]
        
        pincode_roi_px = denormalize_roi(PINCODE_ROI, height, width)
        voters_roi_px = denormalize_roi(VOTERS_END_ROI, height, width)
        
        # Choose OCR method based on flag
        if use_aggressive:
            # Use aggressive OCR with extensive preprocessing
            pincode_text, pincode_bbox = aggressive_ocr(img, r'\d{6}', pincode_roi_px, expected_digits=6)
            voters_text, voters_bbox = aggressive_ocr(img, r'\d{3,4}', voters_roi_px, expected_digits=3)
        else:
            # Use fast standard OCR
            pincode_text, pincode_bbox = find_number_text(img, r'\d{6}', pincode_roi_px)
            voters_text, voters_bbox = find_number_text(img, r'\d{3,4}', voters_roi_px)

        # Robust fallback: horizontal stacked OCR with inpaint line removal
        # Triggers when something is missing OR voters_end is not 3-4 digits.
        voters_valid = bool(re.match(r'^\d{3,4}$', str(voters_text or '')))
        needs_fallback = (not pincode_text) or (not voters_text) or (not voters_valid)
        if needs_fallback:
            stacked_pc, stacked_ve = stacked_digits_ocr(
                img,
                pincode_roi_px,
                voters_roi_px,
                debug_dir=debug_dir,
                doc_id=img_path.stem,
            )
            if not pincode_text and stacked_pc:
                pincode_text = stacked_pc
                pincode_bbox = (pincode_roi_px[0], pincode_roi_px[1], pincode_roi_px[2] - pincode_roi_px[0], pincode_roi_px[3] - pincode_roi_px[1])

            # If we had to fallback due to missing pincode, the earlier voters_end can be a false-positive.
            # Prefer the stacked method when it finds a plausible voters_end.
            override_voters = (not voters_text) or (not voters_valid) or (not pincode_text)
            if override_voters and stacked_ve and re.match(r'^\d{3,4}$', str(stacked_ve)):
                voters_text = stacked_ve
                voters_bbox = (voters_roi_px[0], voters_roi_px[1], voters_roi_px[2] - voters_roi_px[0], voters_roi_px[3] - voters_roi_px[1])
        
        # Validate: pincode is REQUIRED, voters_end is optional
        status = 'success'
        if not pincode_text:
            status = 'missing_pincode'
        elif not voters_text:
            status = 'missing_voters_end'
        
        # Save debug image if any value is missing
        if status != 'success' and debug_dir:
            rel_dir = img_path.parent.name
            debug_output = Path(debug_dir) / rel_dir
            save_debug_image(img, img_path, debug_output, (pincode_text, pincode_bbox), (voters_text, voters_bbox))
        
        result = {
            'document_id': img_path.stem,
            'pincode': pincode_text,
            'voters_end': voters_text,
            'status': status,
            'time_ms': int((time.time() - start_time) * 1000)
        }
        
        return result
        
    except Exception as e:
        return {
            'document_id': img_path.stem,
            'pincode': None,
            'voters_end': None,
            'status': f'error: {str(e)}',
            'time_ms': int((time.time() - start_time) * 1000)
        }


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Extract pincode and voters_end numbers (optimized parallel)"
    )
    parser.add_argument("input", help="Input directory")
    parser.add_argument("--json-output", default="extracted_numbers",
                        help="Output directory for JSON files")
    parser.add_argument("--debug-dir", default="debug_images",
                        help="Directory for debug images (missing values)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only reprocess images that failed in previous run (reads from --json-output)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive OCR with extensive preprocessing (slower but more accurate)")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help=f"Number of parallel jobs (default: {cpu_count()})")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark with different worker counts")
    parser.add_argument("--limit", type=int, help="Limit number of images to process (for testing)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)

    # Resolve output dirs inside the project by default.
    # - If user passes a relative path, make it relative to this script folder.
    # - If user passes an absolute path (e.g. /tmp/...), respect it.
    json_output_path = Path(args.json_output)
    if not json_output_path.is_absolute():
        json_output_path = script_dir / json_output_path

    debug_dir_path = Path(args.debug_dir)
    if not debug_dir_path.is_absolute():
        debug_dir_path = script_dir / debug_dir_path
    
    if not input_path.is_dir():
        print(f"❌ Error: {input_path} is not a directory")
        return 1
    
    # Load existing results if retry mode
    existing_results_by_dir = {}
    failed_image_paths = set()
    
    if args.retry_failed:
        print("🔄 RETRY MODE: Loading existing results...")
        if not json_output_path.exists():
            print(f"❌ Error: {json_output_path} does not exist. Run normal mode first.")
            return 1
        
        for json_file in json_output_path.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                dir_name = data['metadata']['directory']
                existing_results_by_dir[dir_name] = data
                
                # Find failed images (includes invalid voters_end)
                for result in data['results']:
                    is_failed = False
                    
                    # Explicit failures
                    if result['status'] != 'success':
                        is_failed = True
                    
                    # Also check if voters_end is invalid (1-2 digits instead of 3-4)
                    elif result.get('voters_end'):
                        voters_val = str(result['voters_end'])
                        if not re.match(r'^\d{3,4}$', voters_val):
                            is_failed = True
                            print(f"  ⚠️  {result['document_id']}: Invalid voters_end '{voters_val}' (needs 3-4 digits)")
                    
                    if is_failed:
                        # Construct image path
                        img_path = input_path / dir_name / f"{result['document_id']}.png"
                        if img_path.exists():
                            failed_image_paths.add(img_path)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {json_file}: {e}")
        
        print(f"📊 Found {len(failed_image_paths)} images to retry (failures + invalid voters_end)")
        
        if not failed_image_paths:
            print("✅ No failed images found! All extractions were successful.")
            return 0
    
    # Find all images
    if args.retry_failed:
        all_images = sorted(list(failed_image_paths))
    else:
        all_images = sorted(list(input_path.rglob("*.png")))
    if not all_images:
        print(f"❌ No PNG files found in {input_path}")
        return 1
    
    if args.limit:
        all_images = all_images[:args.limit]
        print(f"⚠️  Limited to {args.limit} images for testing")
    
    # Group by directory
    from collections import defaultdict
    images_by_dir = defaultdict(list)
    for img in all_images:
        images_by_dir[img.parent].append(img)
    
    total_images = len(all_images)
    total_dirs = len(images_by_dir)
    
    if args.benchmark:
        # Benchmark mode: test different worker counts
        test_counts = [1, 2, 4, 6, 8, 12, 16]
        test_counts = [n for n in test_counts if n <= cpu_count() * 2]
        
        print(f"🔬 BENCHMARK MODE")
        print(f"📁 Testing with {total_images} images across {total_dirs} directories")
        print(f"🖥️  CPU cores: {cpu_count()}")
        print(f"🧪 Testing worker counts: {test_counts}\n")
        
        results = []
        for num_workers in test_counts:
            print(f"Testing {num_workers} workers...", end=' ', flush=True)
            
            start = time.time()
            work_items = [(img_path, None, False) for img_path in all_images]
            
            with Pool(num_workers) as pool:
                _ = list(pool.imap_unordered(process_single_image, work_items, chunksize=1))
            
            elapsed = time.time() - start
            imgs_per_sec = total_images / elapsed
            results.append((num_workers, elapsed, imgs_per_sec))
            
            print(f"{elapsed:.1f}s ({imgs_per_sec:.1f} img/s)")
        
        print(f"\n📊 Benchmark Results:")
        print(f"{'Workers':<10} {'Time (s)':<12} {'Speed (img/s)':<15} {'Speedup':<10}")
        print("-" * 50)
        baseline = results[0][1]
        for workers, elapsed, speed in results:
            speedup = baseline / elapsed
            print(f"{workers:<10} {elapsed:<12.1f} {speed:<15.1f} {speedup:<10.2f}x")
        
        best = max(results, key=lambda x: x[2])
        print(f"\n✨ Best: {best[0]} workers ({best[2]:.1f} img/s)")
        
        return 0
    
    # Normal processing mode
    num_workers = args.jobs or cpu_count()
    
    print(f"📁 Found {total_images} images across {total_dirs} directories")
    print(f"🚀 Using {num_workers} parallel workers")
    print(f"💾 JSON output: {json_output_path}/")
    print(f"🐛 Debug images: {debug_dir_path}/")
    print(f"⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    overall_start = time.time()
    
    # Prepare work items
    work_items = [(img_path, str(debug_dir_path), args.aggressive) for img_path in all_images]
    
    # Process in parallel with progress
    all_results = []
    processed = 0
    
    # Track dirty directories to save incrementally
    results_by_dir = defaultdict(list)
    
    # Save helper
    def save_incremental(dir_name, results, final=False):
        safe_dir = dir_name
        if safe_dir in ('.', ''):
            safe_dir = input_path.name
        json_filename = safe_dir.replace('/', '_').replace(' ', '_') + '.json'
        json_path = json_output_path / json_filename
        
        # In retry mode, merge with existing results
        if args.retry_failed and dir_name in existing_results_by_dir:
            existing_data = existing_results_by_dir[dir_name]
            existing_results = {r['document_id']: r for r in existing_data['results']}
            
            # Update with new results
            for new_result in results:
                existing_results[new_result['document_id']] = new_result
            
            # Combine all results
            all_results = sorted(existing_results.values(), key=lambda r: r['document_id'])
        else:
            all_results = results
        
        # Determine stats
        times = [r['time_ms'] for r in all_results]
        stats = {
            'count': len(all_results),
            'total_time_ms': sum(times),
            'avg_time_ms': sum(times) / len(all_results) if all_results else 0,
            'min_time_ms': min(times) if times else 0,
            'max_time_ms': max(times) if times else 0
        }
        
        output_data = {
            'metadata': {
                'directory': dir_name,
                'total_images': len(all_results),
                'timestamp': datetime.now().isoformat(),
                'timing': stats,
                'status': 'in_progress' if not final else 'completed',
                'retry_mode': args.retry_failed
            },
            'results': all_results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return json_path
    
    # Create output dir first
    json_output_path.mkdir(parents=True, exist_ok=True)
    
    dirty_dirs = set()
    
    # Build id_to_dir map before loop for immediate logging
    id_to_dir = {}
    for dir_path, images in images_by_dir.items():
        rel_dir_str = str(dir_path.relative_to(input_path))
        for img in images:
            id_to_dir[img.stem] = rel_dir_str
    
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(process_single_image, work_items, chunksize=1):
            all_results.append(result)
            processed += 1
            
            # Find directory for this image and add to local aggregation
            # We must map document_id back to directory. 
            # Looking up in images_by_dir is slow if done linearly.
            # Efficient way: Pre-build a map or infer from document_id if possible. 
            # With multiprocessing inputs, we lost the direct link unless we pass it through.
            # But we can recover it by iterating, or better, change process_single_image to return directory.
            # Since we can't easily change process_single_image signature without breaking existing tool usage logic 
            # (though I am replacing the file content so I could), let's just do the lookup we did before but optimized.
            # Actually, we can assume images_by_dir is available.
            
            # Optimized lookup: Pre-build map
            # We'll do this "just in time" or simply cache the map before loop
            # But wait, I can just build a map: {stem: dir_name}
            
            dir_name = id_to_dir.get(result['document_id'], "unknown")
            results_by_dir[dir_name].append(result)
            dirty_dirs.add(dir_name)
            
            # Per-image logging with real-time feedback
            status_emoji = "✅" if result['status'] == 'success' else ("⚠️" if 'missing_voters' in result['status'] else "❌")
            elapsed_total = time.time() - overall_start
            speed = processed / elapsed_total if elapsed_total > 0 else 0
            eta = (total_images - processed) / speed if speed > 0 else 0
            
            print(f"{status_emoji} [{processed}/{total_images}] {result['document_id']} | "
                  f"PC:{result.get('pincode', 'N/A')} VE:{result.get('voters_end', 'N/A')} | "
                  f"{result['time_ms']}ms | {speed:.1f} img/s | ETA:{eta:.0f}s", flush=True)
            
            # Aggregate stats every 50 images
            if processed % 50 == 0:
                success_count = sum(1 for r in all_results if r['status'] == 'success')
                print(f"\n📊 Summary: {processed}/{total_images} | "
                      f"Success: {success_count}/{processed} ({100*success_count//processed}%) | "
                      f"Speed: {speed:.1f} img/s\n", flush=True)
                      
            # Save more frequently in retry mode (every 5 images vs 20 for normal)
            save_freq = 5 if args.retry_failed else 20
            if processed % save_freq == 0:
                for dirty_dir in dirty_dirs:
                    save_incremental(dirty_dir, results_by_dir[dirty_dir])
                dirty_dirs.clear()

    # Final Save
    print(f"\n💾 Saving final JSON files...")
    for dir_name, results in results_by_dir.items():
        path = save_incremental(dir_name, results, final=True)
        
        # Stats for printing
        success = sum(1 for r in results if r['status'] == 'success')
        missing_voters = sum(1 for r in results if r['status'] == 'missing_voters_end')
        missing_pincode = sum(1 for r in results if r['status'] == 'missing_pincode')
        
        print(f"  ✓ {path.name}: {len(results)} images | "
              f"✓{success} ⚠️{missing_voters} ❌{missing_pincode}")
    
    overall_elapsed = time.time() - overall_start
    
    # Final statistics
    print(f"\n" + "="*60)
    print(f"✅ COMPLETED")
    print(f"="*60)
    print(f"📊 Total images: {total_images}")
    print(f"📁 Directories: {total_dirs}")
    print(f"⏱️  Total time: {overall_elapsed:.1f}s")
    print(f"⚡ Speed: {total_images/overall_elapsed:.1f} images/second")
    print(f"💾 JSON files: {len(results_by_dir)}")
    
    # Overall stats
    all_success = sum(1 for r in all_results if r['status'] == 'success')
    all_missing_voters = sum(1 for r in all_results if r['status'] == 'missing_voters_end')
    all_missing_pincode = sum(1 for r in all_results if r['status'] == 'missing_pincode')
    all_errors = sum(1 for r in all_results if 'error' in r['status'])
    
    print(f"\n📈 Status Summary:")
    print(f"  ✅ Success: {all_success} ({100*all_success//total_images}%)")
    print(f"  ⚠️  Missing voters_end: {all_missing_voters} ({100*all_missing_voters//total_images}%)")
    print(f"  ❌ Missing pincode: {all_missing_pincode} ({100*all_missing_pincode//total_images}%)")
    if all_errors > 0:
        print(f"  💥 Errors: {all_errors}")
    
    print(f"\n💾 Output saved to: {json_output_path.absolute()}")
    if all_missing_voters + all_missing_pincode > 0:
        print(f"🐛 Debug images: {debug_dir_path.absolute()}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

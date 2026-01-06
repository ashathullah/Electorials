"""
ID Field Cropper processor.

Crops serial_no, epic_no, and house_no from voter images and stitches them
side-by-side with separators.

ROIs:
EPIC_ROI = (0.449227, 0.009029, 0.839956, 0.162528)
HOUSE_ROI = (0.303532, 0.410835, 0.728477, 0.559819)
SERIAL_NO_ROI = (0.152318, 0.002257, 0.373068, 0.160271)
"""

from __future__ import annotations

import json
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..logger import get_logger

logger = get_logger("id_field_cropper")

# Field ROIs match USER request
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
EPIC_ROI = (0.694260, 0.047244, 0.969095, 0.207349)
HOUSE_ROI = (0.242826, 0.430446, 0.416115, 0.540682)
SERIAL_NO_ROI = (0.152318, 0.002257, 0.373068, 0.160271)

# Stitching settings
SEPARATOR_WIDTH = 4
SEPARATOR_COLOR = 0  # Black
BG_COLOR = 255       # White
PADDING = 10          # Padding around fields if needed

@dataclass
class IdCropResult:
    """Result of ID field cropping for a single voter image."""
    image_name: str
    image_name: str
    output_path: Optional[Path] = None
    error: Optional[str] = None
    epic_no: str = ""
    house_no: str = ""


@dataclass
class IdFieldCropperSummary:
    """Summary of ID field cropping."""
    total_images: int = 0
    successful_crops: int = 0
    failed_crops: int = 0


class IdFieldCropper(BaseProcessor):
    """
    Crops specific ID fields (Serial, Epic, House) and merges them horizontally.
    """
    
    name = "IdFieldCropper"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.summary = IdFieldCropperSummary()
        self.use_paddle = PADDLE_AVAILABLE
        self.use_tesseract = TESSERACT_AVAILABLE and not self.use_paddle
        self.perform_ocr = self.use_paddle or self.use_tesseract
        
        self.paddle_ocr = None
        if self.use_paddle:
            try:
                # Initialize PaddleOCR (English, disabled angle classification for speed)
                self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
                self.log_info("Initialized PaddleOCR for ID field extraction")
            except Exception as e:
                self.log_warning(f"Failed to initialize PaddleOCR: {e}. Fallback to Tesseract.")
                self.use_paddle = False
                self.use_tesseract = TESSERACT_AVAILABLE

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.crops_dir:
            self.log_error("Crops directory not set")
            return False
        
        if not self.context.crops_dir.exists():
            self.log_error(f"Crops directory not found: {self.context.crops_dir}")
            return False
            
        if not self.context.id_crops_dir:
            self.log_error("ID Crops directory not set")
            return False
            
        return True

    def process(self) -> bool:
        """Process all cropped voter images."""
        crops_dir = self.context.crops_dir
        
        # Find page directories
        page_dirs = sorted([
            d for d in crops_dir.iterdir()
            if d.is_dir() and d.name.startswith("page-")
        ])
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        # Process pages
        total_images = 0
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(os.cpu_count() or 4, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page_dir in page_dirs:
                # Create output dir for page
                page_output_dir = self.context.id_crops_dir / page_dir.name
                page_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get images
                images = sorted([
                    p for p in page_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ])
                
                total_images += len(images)
                page_results = []
                futures = []
                
                for img_path in images:
                    futures.append(
                        executor.submit(self._process_image, img_path, page_output_dir)
                    )
                
                # Collect page results
                for future in as_completed(futures):
                    result = future.result()
                    if result.error:
                        failed += 1
                        page_results.append({
                            "image": result.image_name,
                            "error": result.error
                        })
                    else:
                        successful += 1
                        page_results.append({
                            "image": result.image_name,
                            "epic_no": result.epic_no,
                            "house_no": result.house_no
                        })
                
                # Save extraction results for this page
                if self.perform_ocr:
                    extraction_path = page_output_dir / "id_extraction.json"
                    try:
                        with open(extraction_path, "w", encoding="utf-8") as f:
                            json.dump(page_results, f, indent=2, ensure_ascii=False)
                        self.log_info(f"Saved {len(page_results)} ID records to {extraction_path}")
                    except Exception as e:
                        self.log_error(f"Failed to save extraction results for {page_dir.name}: {e}")
        
        self.summary.total_images = total_images
        self.summary.successful_crops = successful
        self.summary.failed_crops = failed
        
        self.log_info(
            f"ID Field Cropping complete: {successful}/{total_images} successful, {failed} failed"
        )
        
        return True

    def _process_image(self, img_path: Path, output_dir: Path) -> IdCropResult:
        """Process a single voter image."""
        try:
            # Read image
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            
            if img is None:
                return IdCropResult(img_path.name, error="Failed to load image")
            
            H, W = img.shape[:2]
            
            # Helper to extract ROI
            def get_roi(roi_def):
                x1 = int(W * roi_def[0])
                y1 = int(H * roi_def[1])
                x2 = int(W * roi_def[2])
                y2 = int(H * roi_def[3])
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                return img[y1:y2, x1:x2]

            serial_crop = get_roi(SERIAL_NO_ROI)
            epic_crop = get_roi(EPIC_ROI)
            house_crop = get_roi(HOUSE_ROI)
            
            # Validate crops
            if serial_crop.size == 0 or epic_crop.size == 0 or house_crop.size == 0:
                 return IdCropResult(img_path.name, error="Invalid ROI dimensions")

            # OCR Extraction (if enabled)
            epic_no, house_no = "", ""
            if self.perform_ocr:
                try:
                    if self.use_paddle and self.paddle_ocr:
                        # PaddleOCR expects path or numpy array
                        # Extract EPIC
                        res_epic = self.paddle_ocr.ocr(epic_crop, cls=False)
                        if res_epic and res_epic[0]:
                            # res is [[[[x,y],..], ("text", conf)], ...]
                            # We just want the text with highest confidence or concatenate all?
                            # Usually simple field, so take first line
                            epic_no = res_epic[0][0][1][0].strip()
                            
                        # Extract House
                        res_house = self.paddle_ocr.ocr(house_crop, cls=False)
                        if res_house and res_house[0]:
                            house_no = res_house[0][0][1][0].strip()
                            
                    elif self.use_tesseract:
                       # Use Tesseract with English configuration as requested
                       # For House, assuming numeric/alphanumeric. EPIC is definitely alphanumeric.
                       # psm 7 = Treat the image as a single text line.
                       # User requested "With only english configuration" -> lang='eng'
                       # Serial No extraction removed as per request
                       epic_no = pytesseract.image_to_string(epic_crop, lang='eng', config='--psm 7').strip()
                       house_no = pytesseract.image_to_string(house_crop, lang='eng', config='--psm 7').strip()
                except Exception as e:
                    # Don't fail the whole process just because OCR failed
                    pass

            # Resize to same height for clean stitching?
            # Or just center them vertically?
            # Let's target the max height of the three
            h1, w1 = serial_crop.shape
            h2, w2 = epic_crop.shape
            h3, w3 = house_crop.shape
            
            target_h = max(h1, h2, h3)
            
            def pad_to_height(crop, target_h):
                h, w = crop.shape
                if h == target_h:
                    return crop
                if h > target_h:
                    # Should not happen since target_h is max
                    return crop[:target_h, :]
                
                pad_top = (target_h - h) // 2
                pad_bottom = target_h - h - pad_top
                return cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, 0, 0, 
                    cv2.BORDER_CONSTANT, value=BG_COLOR
                )

            serial_crop = pad_to_height(serial_crop, target_h)
            epic_crop = pad_to_height(epic_crop, target_h)
            house_crop = pad_to_height(house_crop, target_h)
            
            # Create separator and padding
            separator = np.full((target_h, SEPARATOR_WIDTH), SEPARATOR_COLOR, dtype=np.uint8)
            padding = np.full((target_h, PADDING), BG_COLOR, dtype=np.uint8)
            
            # Stitch: Serial <pad> | <pad> Epic <pad> | <pad> House
            stitched = np.hstack([
                serial_crop,
                padding, separator, padding,
                epic_crop,
                padding, separator, padding,
                house_crop
            ])
            
            # Save
            output_path = output_dir / img_path.name
            success, encoded = cv2.imencode(".png", stitched)
            if success:
                encoded.tofile(str(output_path))
                return IdCropResult(
                    img_path.name, 
                    output_path=output_path,
                    epic_no=epic_no,
                    house_no=house_no
                )
            else:
                return IdCropResult(img_path.name, error="Failed to encode image")

        except Exception as e:
            return IdCropResult(img_path.name, error=str(e))

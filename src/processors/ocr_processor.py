"""
OCR Processor for voter information extraction.

Extracts structured voter data from cropped images using Tamil OCR:
- EPIC number (via ROI extraction)
- Serial number (via ROI extraction)
- Name, relation, house number, age, gender (via line-based text extraction)

Uses ocr_tamil (https://github.com/gnana70/tamil_ocr) for better Tamil text recognition.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image

# Tamil OCR - better accuracy for Tamil text
try:
    from ocr_tamil.ocr import OCR as TamilOCR
    TAMIL_OCR_AVAILABLE = True
except ImportError:
    TAMIL_OCR_AVAILABLE = False

# Tesseract as fallback
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .base import BaseProcessor, ProcessingContext
from ..config import ROIConfig
from ..models import Voter
from ..exceptions import OCRProcessingError


# Character whitelists
WL_EPIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WL_DIGITS = "0123456789"
WL_HOUSE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"

# Label variants for field extraction
NAME_LABELS = ["name", "பெயர்", "பெயர்‌"]

RELATION_PATTERNS = {
    "father": [
        r"தந்தையின்\s*பெயர்",
        r"தந்தை\s*பெயர்",
        r"father'?s?\s*name",
        r"father",
    ],
    "mother": [
        r"தாயின்\s*பெயர்",
        r"தாய்\s*பெயர்",
        r"mother'?s?\s*name",
        r"mother",
    ],
    "husband": [
        r"கணவர்\s*பெயர்",
        r"கணவர்",
        r"husband'?s?\s*name",
        r"husband",
    ],
}

HOUSE_PATTERNS = [
    r"வீட்டு\s*எண்",
    r"வீட்டுஎண்",
    r"ட்டு\s*எண்",
    r"house\s*(?:no|number)?",
]

AGE_PATTERNS = [r"வயது", r"age"]
GENDER_PATTERNS = [r"பாலினம்", r"gender"]

GENDER_MAP = {
    "ஆண்": "Male", "ஆண": "Male", "male": "Male", "m": "Male",
    "பெண்": "Female", "பெண": "Female", "female": "Female", "f": "Female",
    "திருநங்கை": "Other", "other": "Other", "o": "Other",
}

PHOTO_MARKERS = ["photo", "phot", "available", "புகைப்பட", "படம்"]

# Voter separator markers (from voter_end.jpg)
VOTER_END_MARKERS = ["voter_end", "voter-end", "voterend", "VOTER_END", "VOTER-END", "VOTEREND"]


@dataclass
class WordBox:
    """OCR word bounding box."""
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int
    line_num: int
    block_num: int
    par_num: int


@dataclass
class OCRResult:
    """Result of OCR processing for a single voter image."""
    image_name: str
    serial_no: str = ""
    epic_no: str = ""
    epic_valid: bool = False
    name: str = ""
    relation_type: str = ""
    relation_name: str = ""
    house_no: str = ""
    age: str = ""
    gender: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_voter(self, sequence_in_page: int = 0) -> Voter:
        """Convert to Voter model."""
        # Use sequence_in_page as serial_no since OCR struggles with reading it
        serial_no = str(sequence_in_page) if sequence_in_page > 0 else self.serial_no
        
        # Calculate extraction confidence based on field completeness
        fields_present = sum([
            bool(self.epic_no and self.epic_valid),  # 30%
            bool(self.name),  # 25%
            bool(self.relation_type and self.relation_name),  # 20%
            bool(self.house_no),  # 10%
            bool(self.age),  # 10%
            bool(self.gender),  # 5%
        ])
        confidence_weights = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
        extraction_confidence = sum(confidence_weights[i] for i in range(fields_present))
        
        return Voter(
            serial_no=serial_no,
            epic_no=self.epic_no,
            name=self.name,
            relation_type=self.relation_type,
            relation_name=self.relation_name,
            house_no=self.house_no,
            age=self.age,
            gender=self.gender,
            sequence_in_page=sequence_in_page,
            image_file=self.image_name,
            epic_valid=self.epic_valid,
            processing_time_ms=round(self.elapsed_seconds * 1000, 2),
            extraction_confidence=round(extraction_confidence, 2),
        )


@dataclass
class PageOCRResult:
    """OCR results for a page."""
    page_id: str
    images_processed: int
    total_seconds: float
    records: List[OCRResult] = field(default_factory=list)


class OCRProcessor(BaseProcessor):
    """
    Extract voter information from cropped images using Tamil OCR.
    
    Uses hybrid extraction:
    - EPIC and Serial: ROI-based extraction with character whitelisting
    - Other fields: Line-based text pattern matching via Tamil OCR
    """
    
    name = "OCRProcessor"
    
    # Singleton OCR instance to avoid reloading models
    _ocr_instance: Optional['TamilOCR'] = None
    
    def __init__(
        self,
        context: ProcessingContext,
        languages: str = "eng+tam",
        allow_next_line: bool = True,
        dump_raw_ocr: bool = False,
        use_cuda: bool = True,
        use_merged: bool = False,
    ):
        """
        Initialize OCR processor.
        
        Args:
            context: Processing context
            languages: Language codes (for compatibility, Tamil OCR uses both)
            allow_next_line: Allow value on next line if not found on label line
            dump_raw_ocr: Dump raw OCR text for debugging
            use_cuda: Enable CUDA for Tamil OCR (if available)
            use_merged: Use merged images from /merged folder for faster processing
        """
        super().__init__(context)
        self.languages = languages
        self.allow_next_line = allow_next_line
        self.dump_raw_ocr = dump_raw_ocr
        self.use_cuda = use_cuda
        self.use_merged = use_merged
        self.ocr_config = self.config.ocr  # Contains ROI configs
        self.page_results: List[PageOCRResult] = []
        self._ocr_initialized = False
        
        # Merged images directory - inside extracted folder
        self.merged_dir = self.context.extracted_dir / "merged" if self.context.extracted_dir else None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not TAMIL_OCR_AVAILABLE:
            self.log_error("Tamil OCR not available. Install: pip install ocr-tamil")
            return False
        
        # For merged mode, check merged directory
        if self.use_merged:
            if not self.merged_dir or not self.merged_dir.exists():
                self.log_warning(f"Merged directory not found, falling back to individual crops")
                self.use_merged = False
        
        # Check crops directory for fallback or non-merged mode
        if not self.use_merged:
            if not self.context.crops_dir:
                self.log_error("Crops directory not set")
                return False
            
            if not self.context.crops_dir.exists():
                self.log_error(f"Crops directory not found: {self.context.crops_dir}")
                return False
        
        return True
    
    def _initialize_ocr(self) -> None:
        """Initialize Tamil OCR engine (singleton pattern)."""
        if self._ocr_initialized:
            return
        
        if OCRProcessor._ocr_instance is None:
            self.log_info("Initializing Tamil OCR engine (first time, may download models)...")
            try:
                # Initialize Tamil OCR with text detection enabled
                # detect=True enables CRAFT text detection
                # details=2 gives us text, confidence, and bbox info
                OCRProcessor._ocr_instance = TamilOCR(
                    detect=True,
                    enable_cuda=self.use_cuda,
                    batch_size=8,
                    details=0,  # Just text output for simplicity
                    lang=["tamil", "english"],
                    recognize_thres=0.5,  # Lower threshold for voter card text
                )
                self.log_info("Tamil OCR initialized successfully")
            except Exception as e:
                raise OCRProcessingError(f"Failed to initialize Tamil OCR: {e}")
        
        self._ocr_initialized = True
    
    @property
    def ocr(self) -> 'TamilOCR':
        """Get the OCR instance."""
        if OCRProcessor._ocr_instance is None:
            self._initialize_ocr()
        return OCRProcessor._ocr_instance
    
    def process(self) -> bool:
        """
        Process all cropped images in the document.
        
        If use_merged is True and merged images exist, process those for faster
        extraction. Otherwise, fall back to individual cropped images.
        
        Returns:
            True if processing succeeded
        """
        self._initialize_ocr()
        
        # Try merged image processing if enabled
        if self.use_merged and self.merged_dir and self.merged_dir.exists():
            return self._process_merged_images()
        
        # Fall back to individual crop processing
        return self._process_individual_crops()
    
    def _process_individual_crops(self) -> bool:
        """
        Process individual cropped images (original method).
        
        Returns:
            True if processing succeeded
        """
        crops_dir = self.context.crops_dir
        page_dirs = self._get_page_dirs(crops_dir)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} page(s) with crops")
        
        total_images = 0
        
        for page_dir in page_dirs:
            result = self._process_page(page_dir)
            self.page_results.append(result)
            total_images += result.images_processed
            self.context.total_voters_found += len(result.records)
        
        self.log_info(f"OCR complete: {total_images} images processed")
        
        return True
    
    def _process_merged_images(self) -> bool:
        """
        Process merged batch images for faster OCR.
        
        Each batch image contains up to 10 voter images separated by voter_end markers.
        Uses template matching to split the batch into individual voter images,
        then runs OCR on each segment.
        
        Merged structure: merged/<page-XXX>/<batch-001.png, batch-002.png, ...>
        
        Returns:
            True if processing succeeded
        """
        if not self.merged_dir or not self.merged_dir.exists():
            self.log_warning(f"Merged directory not found at {self.merged_dir}, falling back to individual crops")
            return self._process_individual_crops()
        
        # Load voter_end template for template matching
        voter_end_path = self.config.base_dir / "voter_end.jpg"
        if not voter_end_path.exists():
            self.log_warning("voter_end.jpg not found, falling back to individual crops")
            return self._process_individual_crops()
        
        voter_end_template = cv2.imdecode(
            np.fromfile(str(voter_end_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if voter_end_template is None:
            self.log_warning("Failed to load voter_end.jpg, falling back to individual crops")
            return self._process_individual_crops()
        
        # Get page directories (same structure as crops)
        page_dirs = self._get_page_dirs(self.merged_dir)
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {self.merged_dir}, falling back to individual crops")
            return self._process_individual_crops()
        
        self.log_info(f"Processing {len(page_dirs)} page(s) with merged batches")
        
        total_voters = 0
        total_batches = 0
        
        for page_dir in page_dirs:
            page_id = page_dir.name
            batch_images = self._get_batch_images(page_dir)
            
            if not batch_images:
                self.log_debug(f"No batch images in {page_dir}")
                continue
            
            self.log_debug(f"Processing {len(batch_images)} batches from {page_id}")
            
            page_records: List[OCRResult] = []
            page_start_time = time.perf_counter()
            
            for batch_path in batch_images:
                batch_records = self._process_batch_image(batch_path, page_id, voter_end_template)
                page_records.extend(batch_records)
                total_batches += 1
            
            page_time = time.perf_counter() - page_start_time
            
            result = PageOCRResult(
                page_id=page_id,
                images_processed=len(page_records),
                total_seconds=page_time,
                records=page_records,
            )
            
            self.page_results.append(result)
            total_voters += len(page_records)
            self.context.total_voters_found += len(page_records)
            
            self.log_info(f"Processed {page_id}: {len(page_records)} voters from {len(batch_images)} batch(es)")
        
        self.log_info(f"OCR complete (merged batches): {total_voters} voters from {total_batches} batches")
        
        return True
    
    def _get_batch_images(self, page_dir: Path) -> List[Path]:
        """Get sorted batch images from a page directory."""
        if not page_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        images = [p for p in page_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    
    def _process_batch_image(
        self, 
        batch_path: Path, 
        page_id: str,
        voter_end_template: np.ndarray
    ) -> List[OCRResult]:
        """
        Process a single batch image containing multiple voters.
        
        Uses template matching to split by voter_end markers, then runs OCR on each segment.
        
        Args:
            batch_path: Path to batch image
            page_id: Parent page ID for naming
            voter_end_template: Template image for matching separator
        
        Returns:
            List of OCRResult for each voter in the batch
        """
        self.log_debug(f"Processing batch: {batch_path.name}")
        batch_start = time.perf_counter()
        
        # Load batch image
        img_bgr = cv2.imdecode(
            np.fromfile(str(batch_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_bgr is None:
            self.log_error(f"Failed to load batch image: {batch_path}")
            return []
        
        # Find voter_end separator positions using template matching
        separator_positions = self._find_voter_end_positions(img_bgr, voter_end_template)
        
        if not separator_positions:
            self.log_warning(f"No voter_end separators found in {batch_path.name}")
            return []
        
        self.log_debug(f"Found {len(separator_positions)} voter_end separators in {batch_path.name}")
        
        # Split image into voter segments
        template_height = voter_end_template.shape[0]
        voter_segments = self._split_by_separators(img_bgr, separator_positions, template_height)
        
        # Extract batch number for naming
        batch_name = batch_path.stem  # e.g., "batch-001"
        batch_num = batch_name.replace("batch-", "").replace("batch", "")
        try:
            batch_offset = (int(batch_num) - 1) * 10  # Assuming max 10 per batch
        except ValueError:
            batch_offset = 0
        
        records: List[OCRResult] = []
        
        for idx, segment in enumerate(voter_segments, start=1):
            if segment is None or segment.size == 0:
                continue
            
            # Skip very small segments (likely just noise)
            if segment.shape[0] < 50 or segment.shape[1] < 50:
                continue
            
            voter_num = batch_offset + idx
            image_name = f"{page_id}-{voter_num:03d}.png"
            
            result = self._process_voter_segment(segment, image_name)
            records.append(result)
        
        return records
    
    def _find_voter_end_positions(
        self, 
        img: np.ndarray, 
        template: np.ndarray,
        threshold: float = 0.7
    ) -> List[int]:
        """
        Find vertical positions of voter_end separators using template matching.
        
        Args:
            img: The merged image
            template: The voter_end template image
            threshold: Matching threshold (0-1)
        
        Returns:
            List of y-coordinates where separators were found
        """
        # Convert to grayscale for matching
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # In ImageMerger, the voter_end image is padded, not scaled.
        # So we should primarily look for the template at its original scale.
        
        positions = []
        scales = [1.0]
        
        # Try slight variations just in case of minor resizing artifacts
        if img.shape[1] > template.shape[1] * 2:
             scales = [1.0, 0.95, 1.05]
        
        best_score = 0.0
        
        for scale in scales:
            if scale != 1.0:
                new_width = int(template.shape[1] * scale)
                new_height = int(template.shape[0] * scale)
                if new_width > img.shape[1] or new_height > img.shape[0]:
                    continue
                scaled_template = cv2.resize(template_gray, (new_width, new_height))
            else:
                scaled_template = template_gray
            
            # Template matching
            try:
                result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                
                # Find all locations above threshold
                locations = np.where(result >= threshold)
                
                for y in locations[0]:
                    # Check if this position is not too close to an existing one
                    is_duplicate = False
                    for existing_y in positions:
                        if abs(y - existing_y) < template.shape[0] * 2:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        positions.append(int(y))
                        
            except cv2.error as e:
                self.log_debug(f"Template matching error at scale {scale}: {e}")
                continue
        
        if not positions:
            self.log_debug(f"No separators found. Best score was {best_score:.4f} (threshold: {threshold})")
        
        # Sort positions by y-coordinate
        return sorted(positions)
    
    def _split_by_separators(
        self,
        img: np.ndarray,
        separator_positions: List[int],
        template_height: int
    ) -> List[np.ndarray]:
        """
        Split image into segments based on separator positions.
        
        Args:
            img: The merged image
            separator_positions: Y-coordinates of separators
            template_height: Height of the separator template
        
        Returns:
            List of image segments (one per voter)
        """
        segments = []
        
        # Add start and end boundaries
        start_y = 0
        
        for sep_y in separator_positions:
            # Segment from start_y to separator
            if sep_y > start_y:
                segment = img[start_y:sep_y, :]
                segments.append(segment)
            
            # Move start to after the separator
            start_y = sep_y + template_height
        
        # Don't add remaining segment after last separator 
        # (it's just the last voter_end with nothing after)
        
        return segments
    
    def _process_voter_segment(self, segment: np.ndarray, image_name: str) -> OCRResult:
        """
        Process a single voter segment extracted from merged image.
        
        Args:
            segment: Image segment for one voter
            image_name: Virtual image name for this voter
        
        Returns:
            OCRResult with extracted fields
        """
        start_time = time.perf_counter()
        result = OCRResult(image_name=image_name)
        
        try:
            # Save segment temporarily for OCR (Tamil OCR expects file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, segment)
            
            try:
                # Run Tamil OCR on the segment
                lines = self._run_tamil_ocr(Path(tmp_path), segment)
                
                # Extract EPIC from ROI
                result.epic_no = self._extract_epic(segment)
                result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", result.epic_no))
                
                # Extract Serial from ROI
                result.serial_no = self._extract_serial(segment)
                
                # Extract fields from lines
                result.name = self._extract_name(lines)
                result.relation_type, result.relation_name = self._extract_relation(lines)
                result.house_no = self._extract_house(lines, segment)
                result.age = self._extract_age(lines)
                result.gender = self._extract_gender(lines)
                
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"OCR error for {image_name}: {e}")
        
        result.elapsed_seconds = time.perf_counter() - start_time
        return result

    def _extract_house_from_text(self, text: str) -> str:
        """Extract house number from text segment."""
        for pattern in HOUSE_PATTERNS:
            match = re.search(
                rf"(?:{pattern})\s*[:\-–—]?\s*([A-Za-z0-9\-/]+)",
                text,
                re.IGNORECASE | re.UNICODE
            )
            if match:
                return match.group(1).strip()
        return ""
    
    def _get_merged_images(self) -> List[Path]:
        """Legacy method - get all merged page images (deprecated)."""
        if not self.merged_dir or not self.merged_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        images = [p for p in self.merged_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    

    def _get_page_dirs(self, crops_dir: Path) -> List[Path]:
        """Get sorted page directories."""
        if not crops_dir.exists():
            return []
        return sorted([p for p in crops_dir.iterdir() if p.is_dir()])
    
    def _get_images(self, images_dir: Path) -> List[Path]:
        """Get sorted images from directory."""
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(images)
    
    def _process_page(self, page_dir: Path) -> PageOCRResult:
        """Process all images in a page directory."""
        page_id = page_dir.name
        
        # Check for images subdirectory
        images_dir = page_dir / "images" if (page_dir / "images").exists() else page_dir
        images = self._get_images(images_dir)
        
        if not images:
            self.log_debug(f"No images in {page_dir}")
            return PageOCRResult(page_id=page_id, images_processed=0, total_seconds=0.0)
        
        start_time = time.perf_counter()
        records: List[OCRResult] = []
        
        for idx, image_path in enumerate(images, start=1):
            result = self._process_image(image_path)
            records.append(result)
            
            self.log_debug(
                f"[{idx}/{len(images)}] {image_path.name}",
                epic=result.epic_no,
                time=f"{result.elapsed_seconds:.2f}s"
            )
        
        total_time = time.perf_counter() - start_time
        
        return PageOCRResult(
            page_id=page_id,
            images_processed=len(images),
            total_seconds=total_time,
            records=records,
        )
    
    def _process_image(self, image_path: Path) -> OCRResult:
        """
        Process a single voter image.
        
        Uses hybrid extraction:
        1. EPIC via ROI extraction
        2. Serial via ROI extraction
        3. Other fields via Tamil OCR for better Tamil text recognition
        """
        start_time = time.perf_counter()
        
        result = OCRResult(image_name=image_path.name)
        
        # Load image
        img_bgr = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_bgr is None:
            result.error = "Failed to read image"
            result.elapsed_seconds = time.perf_counter() - start_time
            return result
        
        try:
            # Full OCR using Tamil OCR for better Tamil text
            lines = self._run_tamil_ocr(image_path, img_bgr)
            
            # Extract EPIC from ROI
            result.epic_no = self._extract_epic(img_bgr)
            result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", result.epic_no))
            
            # Extract Serial from ROI
            result.serial_no = self._extract_serial(img_bgr)
            
            # Extract fields from lines
            result.name = self._extract_name(lines)
            result.relation_type, result.relation_name = self._extract_relation(lines)
            result.house_no = self._extract_house(lines, img_bgr)
            result.age = self._extract_age(lines)
            result.gender = self._extract_gender(lines)
            
        except Exception as e:
            result.error = str(e)
            self.log_debug(f"OCR error for {image_path.name}: {e}")
        
        result.elapsed_seconds = time.perf_counter() - start_time
        return result
    
    # ==================== ROI Extraction ====================
    
    def _crop_roi(self, img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """Crop image by relative ROI coordinates."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = roi
        X1 = max(0, min(w - 1, int(round(x1 * w))))
        Y1 = max(0, min(h - 1, int(round(y1 * h))))
        X2 = max(1, min(w, int(round(x2 * w))))
        Y2 = max(1, min(h, int(round(y2 * h))))
        return img[Y1:Y2, X1:X2]
    
    def _extract_epic(self, img_bgr: np.ndarray) -> str:
        """Extract EPIC number from ROI."""
        roi = self.ocr_config.epic_roi.as_tuple()
        epic_crop = self._crop_roi(img_bgr, roi)
        
        # Preprocess
        gray = cv2.cvtColor(epic_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Use Tesseract if available (better for alphanumeric with whitelist)
        if TESSERACT_AVAILABLE:
            try:
                config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_EPIC}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
                return self._clean_epic(txt.strip())
            except Exception:
                pass
        
        # Use Tamil OCR and filter for EPIC pattern
        try:
            # Convert grayscale to BGR for Tamil OCR
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = self.ocr.predict(gray_bgr)
            if result and isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    text = " ".join(str(item) for item in result[0] if item)
                else:
                    text = str(result[0]) if result[0] else ""
                return self._clean_epic(text)
        except Exception:
            pass
        
        return ""
    
    def _clean_epic(self, raw: str) -> str:
        """Clean EPIC number."""
        s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
        if len(s) >= 3:
            prefix, rest = s[:3], s[3:]
            # Fix common OCR confusions in prefix (should be letters)
            prefix = prefix.replace("0", "O").replace("1", "I").replace("2", "Z").replace("5", "S").replace("8", "B")
            # Fix common OCR confusions in rest (should be digits)
            rest = rest.replace("O", "0").replace("I", "1").replace("Z", "2").replace("S", "5").replace("B", "8")
            s = prefix + rest
        return s
    
    def _extract_serial(self, img_bgr: np.ndarray) -> str:
        """Extract serial number from ROI."""
        roi = self.ocr_config.serial_roi.as_tuple()
        serial_crop = self._crop_roi(img_bgr, roi)
        
        # Preprocess
        gray = cv2.cvtColor(serial_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )
        
        # Invert if needed
        if float(np.mean(gray)) > 180:
            gray = 255 - gray
        
        # Use Tesseract if available (better for digits with whitelist)
        if TESSERACT_AVAILABLE:
            try:
                config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_DIGITS}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
                digits = re.sub(r"[^0-9]", "", txt)
                if digits:
                    if len(digits) > 4:
                        digits = digits[-4:]
                    return digits.lstrip("0") or digits
            except Exception:
                pass
        
        # Use Tamil OCR and extract digits
        try:
            # Convert grayscale to BGR for Tamil OCR
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = self.ocr.predict(gray_bgr)
            if result and isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    text = " ".join(str(item) for item in result[0] if item)
                else:
                    text = str(result[0]) if result[0] else ""
                digits = re.sub(r"[^0-9]", "", text)
                if digits:
                    if len(digits) > 4:
                        digits = digits[-4:]
                    return digits.lstrip("0") or digits
        except Exception:
            pass
        
        return ""
    
    # ==================== Full OCR ====================
    
    def _run_tamil_ocr(self, image_path: Path, img_bgr: np.ndarray) -> List[str]:
        """
        Run Tamil OCR on image and return lines.
        
        Tamil OCR provides better accuracy for Tamil text compared to Tesseract.
        Tamil OCR with detect=True returns a list of detected words.
        We combine them into a single text blob and parse as lines.
        """
        try:
            # Tamil OCR expects the image path or numpy array
            result = self.ocr.predict(str(image_path))
            
            # Process result - Tamil OCR returns [[word1, word2, ...]]
            lines = []
            raw_words = []  # Keep raw words for alternative parsing
            
            if result:
                if isinstance(result, list):
                    if len(result) > 0:
                        # Result is typically [[word1, word2, ...]]
                        if isinstance(result[0], list):
                            # Get all words
                            raw_words = [str(w).strip() for w in result[0] if w and str(w).strip()]
                            if raw_words:
                                # Create combined text
                                combined_text = " ".join(raw_words)
                                
                                # Also store raw words as a special "raw" line for direct parsing
                                lines.append("__RAW__:" + "|".join(raw_words))
                                
                                # Try to reconstruct lines based on field markers
                                field_markers = [
                                    r"பெயர்",        # Name
                                    r"தந்தை",        # Father
                                    r"தாய்",         # Mother
                                    r"கணவர்",        # Husband
                                    r"வீட்டு",       # House
                                    r"எண்",          # Number
                                    r"வயது",        # Age
                                    r"பாலினம்",      # Gender
                                    r"Photo",       # Photo marker
                                ]
                                
                                # Split by field markers while keeping the markers
                                pattern = r'(' + '|'.join(field_markers) + r')'
                                parts = re.split(pattern, combined_text, flags=re.IGNORECASE | re.UNICODE)
                                
                                # Reconstruct lines by pairing markers with their values
                                current_line = ""
                                for part in parts:
                                    part = part.strip()
                                    if not part:
                                        continue
                                    if re.search(r'^(' + '|'.join(field_markers) + r')', part, re.IGNORECASE | re.UNICODE):
                                        # This is a field marker - start a new line
                                        if current_line:
                                            lines.append(current_line.strip())
                                        current_line = part
                                    else:
                                        # This is a value - append to current line
                                        current_line += " " + part
                                
                                if current_line:
                                    lines.append(current_line.strip())
                                
                                # Also add the combined text as a line for fallback parsing
                                lines.append(combined_text)
                        else:
                            # Simple list of words
                            for item in result:
                                if item and str(item).strip():
                                    lines.append(str(item).strip())
                elif isinstance(result, str):
                    lines = [line.strip() for line in result.split('\n') if line.strip()]
            
            if self.dump_raw_ocr:
                self.log_debug(f"Tamil OCR result for {image_path.name}: {lines}")
            
            return lines
            
        except Exception as e:
            self.log_debug(f"Tamil OCR error for {image_path.name}: {e}")
            # Fallback to Tesseract if available
            if TESSERACT_AVAILABLE:
                return self._run_tesseract_fallback(image_path)
            return []
    
    def _run_tesseract_fallback(self, image_path: Path) -> List[str]:
        """Fallback to Tesseract OCR if Tamil OCR fails."""
        try:
            img = Image.open(image_path)
            config = "--oem 1 --psm 3"
            ocr_data = pytesseract.image_to_data(
                img,
                lang=self.languages,
                config=config,
                output_type=Output.DICT
            )
            img.close()
            return self._reconstruct_lines_from_tesseract(ocr_data)
        except Exception:
            return []
    
    def _reconstruct_lines_from_tesseract(self, ocr_data: Dict[str, Any]) -> List[str]:
        """Reconstruct text lines from Tesseract OCR data."""
        n = len(ocr_data.get("text", []))
        lines_dict: Dict[Tuple[int, int, int], List[Tuple[int, str]]] = {}
        
        for i in range(n):
            txt = (ocr_data["text"][i] or "").strip()
            if not txt:
                continue
            
            block = int(ocr_data.get("block_num", [0] * n)[i])
            par = int(ocr_data.get("par_num", [0] * n)[i])
            line = int(ocr_data.get("line_num", [0] * n)[i])
            x = int(ocr_data["left"][i])
            
            key = (block, par, line)
            if key not in lines_dict:
                lines_dict[key] = []
            lines_dict[key].append((x, txt))
        
        result = []
        for key in sorted(lines_dict.keys()):
            words = sorted(lines_dict[key], key=lambda t: t[0])
            line_text = " ".join(w[1] for w in words)
            result.append(line_text)
        
        return result
    
    # ==================== Field Extraction ====================
    
    def _normalize_line(self, line: str) -> str:
        """Normalize line for matching."""
        s = line.strip()
        s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
        # Tamil OCR sometimes outputs 'ஃ' or other chars as separator - treat them as colon
        s = s.replace("ஃ", ":").replace(";", ":").replace(",", ":")
        s = re.sub(r"\s+", " ", s)
        return s
    
    def _clean_extracted_value(self, value: str) -> str:
        """Clean extracted value - remove leading colons, Tamil chars in English, etc."""
        if not value:
            return ""
        
        # Remove leading colons, dashes, and special chars
        value = re.sub(r"^[:\-–—;,ஃ\s]+", "", value)
        
        # Remove trailing colons, dashes, and special chars
        value = re.sub(r"[:\-–—;,ஃ\s]+$", "", value)
        
        # If the language context is English, remove Tamil characters
        # Check if value is predominantly English (Latin chars)
        has_english = len(re.findall(r'[A-Za-z]', value))
        has_tamil = len(re.findall(r'[\u0B80-\u0BFF]', value))
        
        # If it has English letters and some Tamil mixed in, keep only English
        if has_english > 0 and has_tamil > 0:
            # Check ratio - if more English than Tamil, filter out Tamil
            if has_english >= has_tamil:
                value = re.sub(r'[\u0B80-\u0BFF]', '', value)
                value = re.sub(r'\s+', ' ', value).strip()
        
        # If it's purely Tamil (no English), it might be a placeholder like இடன் (blank)
        # These should be returned as empty
        if has_tamil > 0 and has_english == 0:
            # Check for common Tamil placeholder words
            tamil_placeholders = ["இடன்", "நடது", "கடம்", "வெற்று"]
            if any(p in value for p in tamil_placeholders):
                return ""
        
        return value.strip()
    
    def _extract_value_after_colon(self, line: str, label_pattern: str) -> str:
        """Extract value after label and separator."""
        norm_line = self._normalize_line(line)
        
        # Try pattern with separator (colon, dash, space etc.) - more flexible for Tamil OCR output
        # Allow any separator: :, -, space, or nothing after label
        pattern = rf"(?:{label_pattern})\s*[:\-–—\s]*(.+?)(?:\s*[\-–—]\s*$|\s*$)"
        match = re.search(pattern, norm_line, re.IGNORECASE | re.UNICODE)
        
        if match:
            value = match.group(1).strip()
            value = re.sub(r"\s*[\-–—]\s*$", "", value)
            value = re.sub(r"\s*(Photo|photo|is|available|புகைப்பட|படம்).*$", "", value, flags=re.IGNORECASE)
            return value.strip()
        
        return ""
    
    def _extract_name(self, lines: List[str]) -> str:
        """Extract name from lines."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                name = self._extract_name_from_words(words)
                if name:
                    # Clean the extracted name
                    name = self._clean_extracted_value(name)
                    if name:
                        return name
        
        # Patterns for name label - flexible to handle Tamil OCR variations
        name_patterns = [r"பெயர்", r"name\b"]
        exclude_patterns = [r"தந்தை", r"தாய்", r"கணவர்", r"father", r"mother", r"husband"]
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            norm_lower = norm.lower()
            
            # Skip if this is a relation line (father/mother/husband name)
            if any(re.search(p, norm_lower, re.IGNORECASE | re.UNICODE) for p in exclude_patterns):
                continue
            
            for pattern in name_patterns:
                if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                    # Extract value after the label
                    value = self._extract_value_after_colon(line, r"(?:பெயர்|name)")
                    if value:
                        # Clean up any additional noise
                        value = re.sub(r"^[:\-–—\s]+", "", value)
                        value = re.sub(r"(Photo|photo|is|available|புகைப்பட|படம்).*$", "", value, flags=re.IGNORECASE)
                        # Remove house number or other fields that might have leaked in
                        value = re.sub(r"\s*-\s*வீட்டு.*$", "", value)  # Remove - வீட்டு... (house no pattern)
                        value = re.sub(r"\s*எண்.*$", "", value)
                        value = re.sub(r"\s*வயது.*$", "", value)
                        value = re.sub(r"\s*பாலினம்.*$", "", value)
                        # Remove any trailing dash/hyphen pattern
                        value = re.sub(r"\s*[-–—]\s*$", "", value)
                        # Clean pipe characters
                        value = value.replace("|", " ").strip()
                        # Apply final cleaning
                        value = self._clean_extracted_value(value)
                        if value and len(value) > 1:
                            return value
        
        return ""
    
    def _extract_name_from_words(self, words: List[str]) -> str:
        """Extract name from raw word list - first பெயர் or Name marker."""
        # Pattern: பெயர்; பழனிவேல் or Name: John or name: John
        # OCR output example: ['Name', ':', 'Rangaraj', 'Father', 'Name:', 'Makali', ...]
        # Or: ['Name', ':Lakshmi', 'Husband', 'Name:', 'Rangaraj', ...]
        # We want to get 'Rangaraj' or 'Lakshmi' (the first name after first 'Name' marker)
        
        # Markers that indicate we've passed to relation section
        relation_markers = ["father", "mother", "husband", "தந்தை", "தாய்", "கணவர்"]
        skip_markers = ["வீட்டு", "house", "photo", "age", "gender", "எண்", "வயது", "பாலினம்"]
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            word_lower = word_clean.lower()
            
            # Check if this word contains பெயர் (Tamil name marker) or "name" (English marker)
            is_tamil_name_marker = "பெயர்" in word_clean or "பெயர" in word_clean
            is_english_name_marker = word_lower == "name" or word_lower == "name:" or word_lower.startswith("name:")
            
            if is_tamil_name_marker or is_english_name_marker:
                # Look for the name value
                # Skip colon if it's a separate token
                name_start_idx = i + 1
                if name_start_idx >= len(words):
                    continue
                    
                next_word = words[name_start_idx].strip()
                
                # Handle case where colon is separate: ['Name', ':', 'John', 'Father', ...]
                if next_word in [":", "-", ";", ","]:
                    name_start_idx += 1
                    if name_start_idx >= len(words):
                        continue
                    next_word = words[name_start_idx].strip()
                
                # Handle ":Lakshmi" case - name attached to colon
                if next_word.startswith(":"):
                    name_val = next_word.lstrip(":").strip()
                    if name_val and not any(m in name_val.lower() for m in skip_markers + relation_markers):
                        # This is the name! Clean and return
                        name_val = re.sub(r'^[:\-–—;,ஃ\s]+', '', name_val)
                        if name_val:
                            return name_val.strip()
                    name_start_idx += 1
                    if name_start_idx >= len(words):
                        continue
                    next_word = words[name_start_idx].strip()
                
                # Check if next word is a relation marker - if so, skip this Name marker
                # because this is "Father Name:" or "Husband Name:", not the person's name
                next_word_lower = next_word.lower().rstrip(":,;")
                if next_word_lower in relation_markers or any(rm in next_word_lower for rm in relation_markers):
                    continue  # This is a relation prefix like "Husband" before "Name:"
                
                # Now check if the word at name_start_idx is valid
                name_word = next_word.rstrip(",;:ஃ%&")
                name_word = name_word.lstrip(":").strip()  # Remove leading colon
                
                # Skip if it's a marker or empty
                if not name_word or any(m in name_word.lower() for m in skip_markers + relation_markers):
                    continue
                
                # Check if it's a valid name (has letters)
                has_english = re.search(r'[A-Za-z]', name_word)
                has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                
                if has_english or has_tamil:
                    # For English names, try to get full name (multiple words until we hit a marker)
                    if has_english and not has_tamil:
                        full_name = name_word
                        for j in range(name_start_idx + 1, min(name_start_idx + 4, len(words))):
                            extra_word = words[j].strip().rstrip(",;:ஃ%&")
                            extra_lower = extra_word.lower()
                            # Stop if we hit a marker
                            if any(m in extra_lower for m in skip_markers + relation_markers):
                                break
                            if extra_word.startswith("-") or extra_word.startswith(":"):
                                break
                            if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                full_name += " " + extra_word
                            else:
                                break
                        # Clean the name before returning
                        full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                        return full_name.strip()
                    # Tamil name
                    name_word = re.sub(r'^[:\-–—;,ஃ\s]+', '', name_word)
                    return name_word.strip()
        
        return ""
    
    def _extract_relation(self, lines: List[str]) -> Tuple[str, str]:
        """Extract relation type and name."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                return self._extract_relation_from_words(words)
        
        # Fallback to line-based extraction
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            for rel_type, patterns in RELATION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                        label_pattern = r"(?:" + "|".join(patterns) + r")"
                        value = self._extract_value_after_colon(line, label_pattern)
                        
                        # Clean up residual label text
                        value = re.sub(r"^பெயர்\s*[:\-–—]?\s*", "", value)
                        value = re.sub(r"^name\s*[:\-–—]?\s*", "", value, flags=re.IGNORECASE)
                        value = re.sub(r"^[:\-–—\s]+", "", value)
                        
                        # Remove trailing content that's not part of the name
                        value = re.sub(r"(Photo|photo|is|available|புகைப்பட|படம்).*$", "", value, flags=re.IGNORECASE)
                        value = re.sub(r"\s*எண்.*$", "", value)
                        value = re.sub(r"\s*வீட்டு.*$", "", value)
                        value = re.sub(r"\s*வயது.*$", "", value)
                        value = re.sub(r"\s*பாலினம்.*$", "", value)
                        value = value.strip()
                        # Apply final cleaning to remove leading colons
                        value = self._clean_extracted_value(value)
                        
                        if value and len(value) > 1:
                            return rel_type, value
        
        return "", ""
    
    def _extract_relation_from_words(self, words: List[str]) -> Tuple[str, str]:
        """Extract relation from raw word list - handles both Tamil and English OCR output."""
        # Pattern: ... கணவர் பெயர்; ராமசாமி ... (Tamil)
        # or: ... Father's Name: John ... (English)
        # or: ... father name: John ... (English)
        
        relation_type = ""
        relation_name = ""
        
        # Find relation type markers (Tamil and English)
        relation_markers = {
            "father": ["தந்தை", "தந்தையின்", "father", "father's", "fathers"],
            "mother": ["தாய்", "தாயின்", "mother", "mother's", "mothers"],
            "husband": ["கணவர்", "கணவரின்", "husband", "husband's", "husbands"],
        }
        
        # First, find if there's a relation type in the words
        found_relation_idx = -1
        for i, word in enumerate(words):
            word_clean = word.strip().rstrip(",;:ஃ%&").lower()
            for rel_type, markers in relation_markers.items():
                for marker in markers:
                    if marker.lower() in word_clean or word_clean == marker.lower():
                        relation_type = rel_type
                        found_relation_idx = i
                        break
                if relation_type:
                    break
            if relation_type:
                break
        
        # Skip markers for both Tamil and English
        skip_markers_tamil = ["வீட்டு", "எண்", "வயது", "பாலினம்"]
        skip_markers_english = ["photo", "house", "age", "gender", "address", "no", "number"]
        skip_markers = skip_markers_tamil + skip_markers_english
        
        # If we found a relation type, look for the name after the name marker
        if relation_type and found_relation_idx >= 0:
            # Look for பெயர் or "name" after the relation marker
            for i in range(found_relation_idx + 1, len(words)):
                word = words[i].strip()
                word_lower = word.lower()
                is_tamil_name = "பெயர்" in word or "பெயர" in word
                is_english_name = "name" in word_lower
                
                if is_tamil_name or is_english_name:
                    # The next word(s) should be the name
                    if i + 1 < len(words):
                        name_word = words[i + 1].strip().rstrip(",;:ஃ%&")
                        # Skip if it's another marker
                        if name_word and not any(m.lower() in name_word.lower() for m in skip_markers):
                            # For English names, try to get full name (multiple words)
                            has_english = re.search(r'[A-Za-z]', name_word)
                            has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                            
                            if has_english and not has_tamil:
                                full_name = name_word
                                for j in range(i + 2, min(i + 5, len(words))):
                                    extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                        full_name += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from the name
                                full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                                relation_name = full_name.strip()
                            else:
                                # Clean leading colons from Tamil name
                                name_word = re.sub(r'^[:\-–—;,ஃ\s]+', '', name_word)
                                relation_name = name_word.strip()
                            break
            
            # If no name found after name marker, try direct value after relation type
            if not relation_name and found_relation_idx + 1 < len(words):
                # Check for pattern like "Father: John" or "Father's John"
                for i in range(found_relation_idx + 1, min(found_relation_idx + 4, len(words))):
                    word = words[i].strip().rstrip(",;:ஃ%&")
                    word_lower = word.lower()
                    # Skip "name" keyword and markers
                    if word_lower in ["name", "name:", "'s"] or any(m.lower() in word_lower for m in skip_markers):
                        continue
                    # Check if it's a valid name (has letters, not a marker)
                    if word and re.match(r'^[A-Za-z\u0B80-\u0BFF]', word):
                        has_english = re.search(r'[A-Za-z]', word)
                        has_tamil = re.search(r'[\u0B80-\u0BFF]', word)
                        if has_english or has_tamil:
                            if has_english and not has_tamil:
                                full_name = word
                                for j in range(i + 1, min(i + 4, len(words))):
                                    extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                    if any(m.lower() in extra_word.lower() for m in skip_markers):
                                        break
                                    if extra_word.startswith("-") or extra_word.startswith(":"):
                                        break
                                    if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                        full_name += " " + extra_word
                                    else:
                                        break
                                # Clean leading colons from the name
                                full_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', full_name)
                                relation_name = full_name.strip()
                            else:
                                # Clean leading colons from Tamil name
                                word = re.sub(r'^[:\-–—;,ஃ\s]+', '', word)
                                relation_name = word.strip()
                            break
        
        # If no explicit relation type found, check if there's a second name pattern
        # (indicates relation name even without explicit type marker)
        if not relation_type:
            name_count = 0
            name_indices = []
            for i, word in enumerate(words):
                word_lower = word.lower()
                if "பெயர்" in word or "பெயர" in word or word_lower.startswith("name"):
                    name_count += 1
                    name_indices.append(i)
            
            # If there are two name markers, the second one is likely relation
            if name_count >= 2 and len(name_indices) >= 2:
                second_name_idx = name_indices[1]
                if second_name_idx + 1 < len(words):
                    name_word = words[second_name_idx + 1].strip().rstrip(",;:ஃ%&")
                    if name_word and not any(m.lower() in name_word.lower() for m in skip_markers):
                        # For English names, get full name
                        has_english = re.search(r'[A-Za-z]', name_word)
                        has_tamil = re.search(r'[\u0B80-\u0BFF]', name_word)
                        
                        if has_english and not has_tamil:
                            full_name = name_word
                            for j in range(second_name_idx + 2, min(second_name_idx + 5, len(words))):
                                extra_word = words[j].strip().rstrip(",;:ஃ%&")
                                if any(m.lower() in extra_word.lower() for m in skip_markers):
                                    break
                                if extra_word.startswith("-") or extra_word.startswith(":"):
                                    break
                                if extra_word and re.match(r'^[A-Za-z]', extra_word):
                                    full_name += " " + extra_word
                                else:
                                    break
                            relation_name = full_name.strip()
                        else:
                            relation_name = name_word
                        # Default to father if no type specified
                        relation_type = "father"
        
        # Clean the relation_name before returning
        if relation_name:
            # Remove leading colons, dashes, etc.
            relation_name = re.sub(r'^[:\-–—;,ஃ\s]+', '', relation_name)
            relation_name = relation_name.strip()
        
        return relation_type, relation_name
    
    def _extract_house(self, lines: List[str], img_bgr: np.ndarray) -> str:
        """Extract house number from lines or ROI fallback."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                house = self._extract_house_from_words(words)
                if house:
                    return house
        
        combined_pattern = r"(?:" + "|".join(HOUSE_PATTERNS) + r")"
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                # Try direct extraction with single colon/separator
                # Pattern: வீட்டு எண் : 1 or house no : 1
                direct_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]\s*(\d[\dA-Za-z/\-]*)",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if direct_match:
                    house_val = direct_match.group(1).strip()
                    # Clean up any trailing noise
                    house_val = re.sub(r"(Photo|photo|is|available|புகைப்பட|படம்).*", "", house_val, flags=re.IGNORECASE)
                    house_val = house_val.strip()
                    if house_val and len(house_val) <= 15:
                        return house_val
                
                # Fallback to _extract_value_after_colon
                value = self._extract_value_after_colon(line, combined_pattern)
                value = re.sub(r"(Photo|photo|is|available|புகைப்பட|படம்).*", "", value, flags=re.IGNORECASE)
                value = re.sub(r"\s+", "", value)
                
                house_match = re.search(r"^(\d[\dA-Za-z/\-]{0,15})", value)
                if house_match:
                    return house_match.group(1)
                
                cleaned = re.sub(r"[^\dA-Za-z/\-]", "", value)
                if cleaned and len(cleaned) <= 15:
                    return cleaned
        
        # Fallback to ROI extraction
        return self._extract_house_from_roi(img_bgr)
    
    def _extract_house_from_words(self, words: List[str]) -> str:
        """Extract house number from raw word list."""
        # Pattern: ... வீட்டு ... எண்ஃ1 ... or எண் 1 or எண்:1
        # Look for எண் followed by a number AFTER வீட்டு marker
        
        found_veedu = False  # வீட்டு marker
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            
            # Track if we've seen வீட்டு (house) marker
            if "வீட்டு" in word_clean:
                found_veedu = True
                continue
            
            # Only look for எண் (number) AFTER வீட்டு marker
            if found_veedu and ("எண்" in word_clean or "எண" in word_clean):
                # Try to extract number from the same word (e.g., எண்ஃ1, எண்:1)
                num_match = re.search(r"எண்?\s*[ஃ;:,&]?\s*(\d+[A-Za-z/\-]*)", word_clean)
                if num_match:
                    house_val = num_match.group(1)
                    # Validate - house numbers are typically short
                    if len(house_val) <= 10:
                        return house_val
                
                # Try to extract from next word
                if i + 1 < len(words):
                    next_word = words[i + 1].strip()
                    # Check if next word starts with a digit
                    if next_word and len(next_word) > 0 and next_word[0].isdigit():
                        num_match = re.match(r"(\d+[A-Za-z/\-]*)", next_word)
                        if num_match:
                            house_val = num_match.group(1)
                            if len(house_val) <= 10:
                                return house_val
        
        # Also look for standalone number pattern after வீட்டுஎண் combined
        for i, word in enumerate(words):
            word_clean = word.strip()
            if "வீட்டுஎண்" in word_clean or "வீட்டு" in word_clean:
                # Check if house number is embedded like வீட்டுஎயின் or similar with number
                num_match = re.search(r"[வீட்டு|எண்]+[ஃ;:,&]?\s*(\d+[A-Za-z/\-]*)", word_clean)
                if num_match:
                    house_val = num_match.group(1)
                    if len(house_val) <= 10:
                        return house_val
        
        return ""
    
    def _extract_house_from_roi(self, img_bgr: np.ndarray) -> str:
        """Extract house number from ROI as fallback using Tesseract."""
        roi = self.ocr_config.house_roi.as_tuple()
        house_crop = self._crop_roi(img_bgr, roi)
        
        gray = cv2.cvtColor(house_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        txt = ""
        
        # Use Tesseract for house number (alphanumeric with whitelist) if available
        if TESSERACT_AVAILABLE:
            try:
                config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_HOUSE}"
                txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
            except Exception:
                pass
        
        # Fallback to Tamil OCR
        if not txt:
            try:
                # Convert grayscale to BGR for Tamil OCR
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = self.ocr.predict(gray_bgr)
                if result and isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        txt = " ".join(str(item) for item in result[0] if item)
                    else:
                        txt = str(result[0]) if result[0] else ""
            except Exception:
                txt = ""
        
        # Clean house value
        s = txt.upper()
        s = re.sub(r"[^A-Z0-9/\-\s]", " ", s)
        s = s.replace("O", "0").replace("I", "1")
        
        tokens = [t.strip("/-") for t in s.split() if t.strip("/-")]
        
        for t in tokens:
            if not re.search(r"\d", t):
                continue
            # House numbers are typically short - max 6 chars for ROI extraction
            # Reject values that look like PIN codes (6 digits starting with 6)
            if len(t) >= 6 and t.isdigit() and t.startswith("6"):
                continue  # Skip PIN codes
            if len(t) <= 6 and re.fullmatch(r"^\d[\dA-Z/\-]{0,5}$", t):
                return t
        
        return ""
    
    def _extract_age(self, lines: List[str]) -> str:
        """Extract age from lines."""
        # First, try to extract from raw words if available
        for line in lines:
            if line.startswith("__RAW__:"):
                words = line[8:].split("|")
                age = self._extract_age_from_words(words)
                if age:
                    return age
        
        combined_pattern = r"(?:" + "|".join(AGE_PATTERNS) + r")"
        
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                age_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\d{{1,3}})",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if age_match:
                    return age_match.group(1)
        
        # Also try to find any age value in the text (fallback)
        for line in lines:
            if line.startswith("__RAW__:"):
                continue
            norm = self._normalize_line(line)
            # Look for pattern like "79" after any text containing age indicator
            age_pattern = re.search(r"(?:வயது|age)\s*[:\-–—,]?\s*(\d{1,3})", norm, re.IGNORECASE | re.UNICODE)
            if age_pattern:
                age_val = age_pattern.group(1)
                if 1 <= int(age_val) <= 120:
                    return age_val
        
        return ""
    
    def _extract_age_from_words(self, words: List[str]) -> str:
        """Extract age from raw word list."""
        # Pattern: வயது,77 or வயது&77 or வயது:77 or வயது 77
        # Also handles English: age:77, Age: 77, age 77
        # OCR may output: Age இடன் 57 (with Tamil placeholder between Age and number)
        
        # Tamil placeholder words that may appear between Age and number
        tamil_placeholders = ["இடன்", "கடம்", "நடது", "வெற்று"]
        
        for i, word in enumerate(words):
            word_clean = word.strip()
            word_lower = word_clean.lower()
            
            # Check if this word contains வயது (Tamil age marker) or "age" (English marker)
            is_tamil_age = "வயது" in word_clean
            is_english_age = word_lower.startswith("age") or word_lower == "age" or word_lower == "age:"
            
            if is_tamil_age or is_english_age:
                # Try to extract age from the same word (e.g., வயது,77, வயது&77, age:77)
                # Handle various separators: , & ; : - and Tamil special chars
                if is_tamil_age:
                    age_match = re.search(r"வயது\s*[,&;:\-ஃ]?\s*(\d{1,3})", word_clean)
                else:
                    age_match = re.search(r"age\s*[,&;:\-ஃ]?\s*(\d{1,3})", word_clean, re.IGNORECASE)
                
                if age_match:
                    age_val = age_match.group(1)
                    if 1 <= int(age_val) <= 120:
                        return age_val
                
                # Look for age in next few words (skip Tamil placeholders)
                for j in range(i + 1, min(i + 4, len(words))):
                    next_word = words[j].strip().lstrip(",&;:-ஃ:")
                    
                    # Skip Tamil placeholder words
                    if any(p in next_word for p in tamil_placeholders):
                        continue
                    
                    # Check if this word contains a number
                    age_match = re.search(r"(\d{1,3})", next_word)
                    if age_match:
                        age_val = age_match.group(1)
                        if 1 <= int(age_val) <= 120:
                            return age_val
                    
                    # If it's not a number and not a placeholder, stop looking
                    if next_word and not re.search(r'[\u0B80-\u0BFF]', next_word):
                        break
        
        # Also look for age pattern anywhere in words (fallback)
        all_text = " ".join(words)
        # Try Tamil pattern first
        age_match = re.search(r"வயது\s*[,&;:\-ஃ]?\s*(\d{1,3})", all_text)
        if age_match:
            age_val = age_match.group(1)
            if 1 <= int(age_val) <= 120:
                return age_val
        
        # Try English pattern - allow Tamil placeholder words between Age and number
        # Pattern: Age [optional Tamil placeholder] number
        age_match = re.search(r"\bage\s*[,&;:\-ஃ:]?\s*(?:[\u0B80-\u0BFF]+\s*)?(\d{1,3})", all_text, re.IGNORECASE)
        if age_match:
            age_val = age_match.group(1)
            if 1 <= int(age_val) <= 120:
                return age_val
        
        return ""
    
    def _extract_gender(self, lines: List[str]) -> str:
        """Extract gender from lines."""
        combined_pattern = r"(?:" + "|".join(GENDER_PATTERNS) + r")"
        
        # First, look for gender in the combined text of all lines
        all_text = " ".join(lines)
        
        # Look for explicit gender words anywhere
        if re.search(r"ஆண்|ஆண\b", all_text):
            return "Male"
        if re.search(r"பெண்|பெண\b", all_text):
            return "Female"
        if re.search(r"\bmale\b", all_text, re.IGNORECASE):
            return "Male"
        if re.search(r"\bfemale\b", all_text, re.IGNORECASE):
            return "Female"
        
        for line in lines:
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                # Look for gender value after the label
                gender_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\S+)",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if gender_match:
                    raw_gender = gender_match.group(1).strip()
                    
                    # Map to standard gender values
                    for key, value in GENDER_MAP.items():
                        if key.lower() in raw_gender.lower():
                            return value
                    
                    # Check if it's a Tamil gender word
                    if "ஆண" in raw_gender:
                        return "Male"
                    if "பெண" in raw_gender:
                        return "Female"
        
        return ""
    
    def get_all_voters(self) -> List[Voter]:
        """Get all extracted voters as Voter models."""
        voters = []
        sequence_in_doc = 0
        
        for page_result in self.page_results:
            for idx, ocr_result in enumerate(page_result.records, start=1):
                if ocr_result.error:
                    continue
                
                sequence_in_doc += 1
                voter = ocr_result.to_voter(sequence_in_page=idx)
                voter.page_id = page_result.page_id
                voter.sequence_in_document = sequence_in_doc
                voters.append(voter)
        
        return voters


def process_ocr(
    extracted_dir: Path,
    languages: str = "eng+tam",
    allow_next_line: bool = True,
) -> List[Voter]:
    """
    Convenience function to run OCR on an extracted folder.
    
    Args:
        extracted_dir: Path to extracted folder
        languages: Tesseract language codes
        allow_next_line: Allow value on next line
    
    Returns:
        List of extracted voters
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    processor = OCRProcessor(
        context,
        languages=languages,
        allow_next_line=allow_next_line,
    )
    
    if not processor.run():
        raise OCRProcessingError("OCR processing failed", str(extracted_dir))
    
    return processor.get_all_voters()

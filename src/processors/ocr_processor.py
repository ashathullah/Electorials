"""
OCR Processor for voter information extraction.

Extracts structured voter data from cropped images using Tesseract OCR:
- EPIC number (via ROI extraction)
- Serial number (via ROI extraction)
- Name, relation, house number, age, gender (via line-based text extraction)
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
        return Voter(
            serial_no=self.serial_no,
            epic_no=self.epic_no,
            name=self.name,
            relation_type=self.relation_type,
            relation_name=self.relation_name,
            house_no=self.house_no,
            age=int(self.age) if self.age.isdigit() else None,
            gender=self.gender,
            sequence_in_page=sequence_in_page,
            source_image=self.image_name,
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
    Extract voter information from cropped images using Tesseract OCR.
    
    Uses hybrid extraction:
    - EPIC and Serial: ROI-based extraction with character whitelisting
    - Other fields: Line-based text pattern matching
    """
    
    name = "OCRProcessor"
    
    def __init__(
        self,
        context: ProcessingContext,
        languages: str = "eng+tam",
        allow_next_line: bool = True,
        dump_raw_ocr: bool = False,
    ):
        """
        Initialize OCR processor.
        
        Args:
            context: Processing context
            languages: Tesseract language codes (default: eng+tam)
            allow_next_line: Allow value on next line if not found on label line
            dump_raw_ocr: Dump raw OCR text for debugging
        """
        super().__init__(context)
        self.languages = languages
        self.allow_next_line = allow_next_line
        self.dump_raw_ocr = dump_raw_ocr
        self.ocr_config = self.config.ocr  # Contains ROI configs
        self.page_results: List[PageOCRResult] = []
        self._tesseract_initialized = False
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not TESSERACT_AVAILABLE:
            self.log_error("Tesseract not available. Install: pip install pytesseract pillow")
            return False
        
        if not self.context.crops_dir:
            self.log_error("Crops directory not set")
            return False
        
        if not self.context.crops_dir.exists():
            self.log_error(f"Crops directory not found: {self.context.crops_dir}")
            return False
        
        return True
    
    def _initialize_tesseract(self) -> None:
        """Initialize Tesseract OCR engine."""
        if self._tesseract_initialized:
            return
        
        import os
        
        # Windows-specific path
        if os.name == "nt":
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(tesseract_path).exists():
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.log_debug(f"Using Tesseract from: {tesseract_path}")
        
        try:
            version = pytesseract.get_tesseract_version()
            self.log_info(f"Tesseract version: {version}")
        except Exception as e:
            raise OCRProcessingError(f"Tesseract not available: {e}")
        
        self._tesseract_initialized = True
    
    def process(self) -> bool:
        """
        Process all cropped images in the document.
        
        Returns:
            True if processing succeeded
        """
        self._initialize_tesseract()
        
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
        2. Serial via enhanced ROI extraction
        3. Other fields via line-based text extraction
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
            # Extract EPIC (ROI-based)
            result.epic_no = self._extract_epic(img_bgr)
            result.epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", result.epic_no))
            
            # Extract Serial (ROI-based)
            result.serial_no = self._extract_serial(img_bgr)
            
            # Full OCR for other fields
            ocr_data = self._run_full_ocr(image_path)
            lines = self._reconstruct_lines(ocr_data)
            
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
        
        # OCR
        config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_EPIC}"
        txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
        
        return self._clean_epic(txt.strip())
    
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
        
        # OCR
        config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_DIGITS}"
        txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
        
        digits = re.sub(r"[^0-9]", "", txt)
        if not digits:
            return ""
        if len(digits) > 4:
            digits = digits[-4:]
        return digits.lstrip("0") or digits
    
    # ==================== Full OCR ====================
    
    def _run_full_ocr(self, image_path: Path) -> Dict[str, Any]:
        """Run full OCR on image."""
        img = Image.open(image_path)
        try:
            config = "--oem 1 --psm 6"
            return pytesseract.image_to_data(
                img,
                lang=self.languages,
                config=config,
                output_type=Output.DICT
            )
        finally:
            img.close()
    
    def _reconstruct_lines(self, ocr_data: Dict[str, Any]) -> List[str]:
        """Reconstruct text lines from OCR data."""
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
        s = re.sub(r"\s+", " ", s)
        return s
    
    def _extract_value_after_colon(self, line: str, label_pattern: str) -> str:
        """Extract value after label and separator."""
        norm_line = self._normalize_line(line)
        
        pattern = rf"(?:{label_pattern})\s*[:\-–—]?\s*[:\-–—]\s*(.+?)(?:\s*[\-–—]\s*$|\s*$)"
        match = re.search(pattern, norm_line, re.IGNORECASE | re.UNICODE)
        
        if match:
            value = match.group(1).strip()
            value = re.sub(r"\s*[\-–—]\s*$", "", value)
            value = re.sub(r"\s*(Photo|photo|is|available|புகைப்பட|படம்).*$", "", value, flags=re.IGNORECASE)
            return value.strip()
        
        return ""
    
    def _extract_name(self, lines: List[str]) -> str:
        """Extract name from lines."""
        name_patterns = [r"^பெயர்", r"^name\b"]
        exclude_patterns = [r"தந்தை", r"தாய்", r"கணவர்", r"father", r"mother", r"husband"]
        
        for line in lines:
            norm = self._normalize_line(line).lower()
            
            if any(re.search(p, norm, re.IGNORECASE) for p in exclude_patterns):
                continue
            
            for pattern in name_patterns:
                if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                    value = self._extract_value_after_colon(line, r"(?:பெயர்|name)")
                    if value:
                        return value
        
        return ""
    
    def _extract_relation(self, lines: List[str]) -> Tuple[str, str]:
        """Extract relation type and name."""
        for line in lines:
            norm = self._normalize_line(line)
            
            for rel_type, patterns in RELATION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                        label_pattern = r"(?:" + "|".join(patterns) + r")"
                        value = self._extract_value_after_colon(line, label_pattern)
                        
                        # Clean up residual label text
                        value = re.sub(r"^பெயர்\s*[:\-–—]?\s*", "", value)
                        value = re.sub(r"^name\s*[:\-–—]?\s*", "", value, flags=re.IGNORECASE)
                        
                        if value:
                            return rel_type, value.strip()
        
        return "", ""
    
    def _extract_house(self, lines: List[str], img_bgr: np.ndarray) -> str:
        """Extract house number from lines or ROI fallback."""
        combined_pattern = r"(?:" + "|".join(HOUSE_PATTERNS) + r")"
        
        for line in lines:
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
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
    
    def _extract_house_from_roi(self, img_bgr: np.ndarray) -> str:
        """Extract house number from ROI as fallback."""
        roi = self.ocr_config.house_roi.as_tuple()
        house_crop = self._crop_roi(img_bgr, roi)
        
        gray = cv2.cvtColor(house_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_HOUSE}"
        txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
        
        # Clean house value
        s = txt.upper()
        s = re.sub(r"[^A-Z0-9/\-\s]", " ", s)
        s = s.replace("O", "0").replace("I", "1")
        
        tokens = [t.strip("/-") for t in s.split() if t.strip("/-")]
        
        for t in tokens:
            if not re.search(r"\d", t):
                continue
            if len(t) <= 12 and re.fullmatch(r"^\d[\dA-Z/\-]{0,11}$", t):
                return t
        
        return ""
    
    def _extract_age(self, lines: List[str]) -> str:
        """Extract age from lines."""
        combined_pattern = r"(?:" + "|".join(AGE_PATTERNS) + r")"
        
        for line in lines:
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                age_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\d{{1,3}})",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if age_match:
                    return age_match.group(1)
        
        return ""
    
    def _extract_gender(self, lines: List[str]) -> str:
        """Extract gender from lines."""
        combined_pattern = r"(?:" + "|".join(GENDER_PATTERNS) + r")"
        
        for line in lines:
            norm = self._normalize_line(line)
            
            if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
                gender_match = re.search(
                    rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\S+)",
                    norm,
                    re.IGNORECASE | re.UNICODE
                )
                if gender_match:
                    raw_gender = gender_match.group(1).strip()
                    
                    for key, value in GENDER_MAP.items():
                        if key.lower() in raw_gender.lower():
                            return value
                    
                    if raw_gender and len(raw_gender) < 15:
                        return raw_gender
        
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

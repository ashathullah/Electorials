"""
OCR Processor for extracted voter-crop images using Tesseract.

Goal (your latest requirements):
- FAST (single-pass for most fields)
- EPIC extracted with ROI + whitelist + format correction (most accurate)
- Other fields extracted from ONE full-page OCR pass using image_to_data() (word boxes)
- Output structured JSON

Fields extracted:
- epic_no (ROI)
- name (Name/பெயர்)
- relation_type + relation_name (Father/Mother/Husband + Tamil equivalents)
- house_no (House Number/வீட்டு எண்)
- age (Age/வயது)
- gender (Gender/பாலினம்)

Input:
    extracted/<file_name>/crops/<page_name>/*.(png/jpg/...)
    (also supports extracted/<file_name>/crops/<page_name>/images/* if present)

Output:
    extracted/<file_name>/output/page_wise/<page_name>.json
    extracted/<file_name>/output/<file_name>.json

Install:
  pip install pytesseract pillow opencv-python numpy

Notes:
- This assumes the crop layout is consistent:
  serial top-left, epic top-right, followed by labeled lines.
- Extraction uses label detection + "value on same line to the right".
  If your value is on next line for some fields, enable --allow-next-line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from markdown_preview import write_file_markdown, write_page_markdown
from raw_ocr_dump import append_raw_ocr_text_only, init_raw_ocr_markdown, write_raw_ocr_detailed_for_image

try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image
    import cv2
    import numpy as np  # noqa: F401

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ----------------------------- Configuration ----------------------------

# EPIC ROI ratios (x1, y1, x2, y2) relative to crop size
# epic is top-right
# EPIC_ROI = (0.55, 0.05, 0.98, 0.18)
EPIC_ROI = (0.65, 0.05, 0.98, 0.20)

# Serial number ROI ratios (x1, y1, x2, y2) relative to crop size
# serial is in top-left boxed area (includes the small number box to the right)
# SERIAL_ROI = (0.005, 0.01, 0.35, 0.20)
SERIAL_ROI = (0.15, 0.06, 0.329, 0.2)

# House line is below relation line and above age/gender
# HOUSE_ROI = (0.05, 0.38, 0.78, 0.58)
HOUSE_ROI = (0.02, 0.42, 0.40, 0.54)

# Label variants (English/Tamil). We match by "contains" after normalization.
NAME_LABELS = [
    "name",        # English
    "பெயர்",       # Tamil
    "பெயர்‌",      # Tamil with ZWJ/ZWSP variants in OCR
]

RELATION_LABELS = {
    "father": [
        "father", "தந்தையின்", "தந்தை", "தந்தையின்‌", "தந்தையின்‌ பெயர்",
        "தந்தை பெயர்‌", "தந்தை பெயர்", "தந்தையின் பெயர்", "தந்தையின் பெயர்"
    ],
    "mother": [
        "mother", "தாயின்", "தாய்", "தாயின்‌", "தாயின்‌ பெயர்",
        "தாய் பெயர்‌", "தாய் பெயர்", "தாயின் பெயர்", "தாயின் பெயர்"
    ],
    "husband": [
        "husband", "கணவர்", "கணவர்‌", "கணவர்‌ பெயர்", "கணவர்‌ பெயர்‌",
        "கணவர் பெயர்", "கணவர் பெயர்", "கணவர் பெயர்"
    ],
}

PHOTO_MARKERS = [
    "photo", "phot", "available",
    "புகைப்பட", "படம்"
]

# Tokens that indicate a new label is starting (so stop capturing value)
STOP_LABELS = [
    # English
    "name", "father", "mother", "husband", "house", "number", "age", "gender",
    # Tamil (common OCR variants)
    "பெயர்", "தந்தை", "தந்தையின்", "தாய்", "தாயின்", "கணவர்", "வீட்டு", "எண்", "வயது", "பாலினம்",
    # Photo line words
    "photo", "available", "புகைப்பட", "படம்",
]

# IMPORTANT: House label is often multi-token (Tamil: "வீட்டு" + "எண்", Eng: "House" + "No/Number")
# We'll match multi-token spans in _extract_value_after_label(), so keep full phrases prioritized.
HOUSE_LABELS = [
    "house no", "house number", "house",         # English
    "வீட்டு எண்", "வீட்டு எண்‌", "வீடு", "ட்டு எண்", "வீட்டுஎண்"        # Tamil
]

AGE_LABELS = ["age", "வயது", "வயது‌"]
GENDER_LABELS = ["gender", "பாலினம்", "பாலினம்‌"]

# Common separators that show up near labels
LABEL_SEPARATORS = {":", "-", "—", "|", "=", "№"}

# Whitelists
WL_EPIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WL_DIGITS = "0123456789"
WL_HOUSE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"


# Enable to save ROI overlays for quick coordinate debugging.
# Prefer enabling via CLI: --debug-rois 1
DEBUG_DRAW_ROIS = False


# ------------------------------ Data types ------------------------------

@dataclass
class WordBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int
    line_num: int
    block_num: int
    par_num: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


# ------------------------------ IO helpers ------------------------------

def _get_extracted_folders(extracted_dir: Path) -> list[Path]:
    folders: list[Path] = []
    for folder in sorted(extracted_dir.iterdir()):
        if folder.is_dir():
            crops_dir = folder / "crops"
            if crops_dir.exists() and crops_dir.is_dir():
                folders.append(folder)
    return folders


def _get_sorted_page_dirs(crops_dir: Path) -> list[Path]:
    if not crops_dir.exists() or not crops_dir.is_dir():
        return []
    pages = [p for p in crops_dir.iterdir() if p.is_dir()]
    return sorted(pages)


def _resolve_page_images_dir(page_dir: Path) -> Path:
    """Some extractions store crops directly under page_dir; some under page_dir/images."""
    images_dir = page_dir / "images"
    if images_dir.exists() and images_dir.is_dir():
        return images_dir
    return page_dir


def _get_sorted_images(images_dir: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    images = [
        img for img in images_dir.iterdir()
        if img.is_file() and img.suffix.lower() in extensions
    ]
    return sorted(images)


# ------------------------- normalize text helpers ---------------------------

def _normalize_text(s: str) -> str:
    """
    Normalize OCR token to improve matching:
    - lower
    - remove spaces
    - remove common punctuation
    - keep Tamil letters intact
    """
    if not s:
        return ""
    s = s.strip().lower()
    # remove zero-width / odd spaces some OCR outputs
    s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # remove common separators/punct
    s = re.sub(r"[^\w\u0B80-\u0BFF]+", "", s, flags=re.UNICODE)  # keep Tamil block
    return s


def _norm_for_match(s: str) -> str:
    # normalized + also normalize 0->o so PH0T0 matches PHOTO
    t = _normalize_text(s)
    return t.replace("0", "o")


def _line_key(w: WordBox) -> tuple[int, int, int]:
    return (w.block_num, w.par_num, w.line_num)


def _looks_like_photo_line(line_words: list[WordBox]) -> bool:
    joined = " ".join(w.text for w in line_words)
    j = _norm_for_match(joined)
    return any(_norm_for_match(m) in j for m in PHOTO_MARKERS)


def _is_stop_token(token: str) -> bool:
    t = _norm_for_match(token)
    if not t:
        return False
    for s in STOP_LABELS:
        ss = _norm_for_match(s)
        if ss and (ss == t or ss in t or t in ss):
            return True
    return False


def _strip_leading_name_label(v: str) -> str:
    if not v:
        return ""
    v2 = v.strip()
    # remove leading English/Tamil "Name:" / "பெயர்:"
    v2 = re.sub(r"^(name)\s*[:\-–—]?\s*", "", v2, flags=re.IGNORECASE)
    v2 = re.sub(r"^(பெயர்)\s*[:\-–—]?\s*", "", v2)
    return v2.strip()


# -------------------------- Tesseract init ------------------------------

def _initialize_tesseract(languages: str) -> None:
    if not TESSERACT_AVAILABLE:
        print("[ERROR] Missing deps. Please run: pip install pytesseract pillow opencv-python numpy")
        raise ImportError("pytesseract/pillow/opencv-python/numpy required")

    if os.name == "nt":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"[INFO] Using Tesseract from: {tesseract_path}")

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract version: {version}")
    except Exception as e:
        print("[ERROR] Tesseract not found. Please install Tesseract OCR.")
        print("        Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        raise RuntimeError(f"Tesseract not available: {e}")

    print(f"[INFO] Using languages for full OCR: {languages}")
    print("[SUCCESS] Tesseract initialized (CPU mode)")


# ----------------------- EPIC extraction helpers ------------------------

def _crop_rel(img_bgr, x1: float, y1: float, x2: float, y2: float):
    h, w = img_bgr.shape[:2]
    X1 = int(round(x1 * w)); Y1 = int(round(y1 * h))
    X2 = int(round(x2 * w)); Y2 = int(round(y2 * h))
    X1 = max(0, min(w - 1, X1)); X2 = max(1, min(w, X2))
    Y1 = max(0, min(h - 1, Y1)); Y2 = max(1, min(h, Y2))
    return img_bgr[Y1:Y2, X1:X2]


def _normalize_for_epic(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return gray


def _normalize_for_serial(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    # If digits are light on dark, invert (heuristic)
    if float(np.mean(gray)) > 180:
        gray = 255 - gray
    return gray


def _debug_draw_rois(image_path: Path, out_dir: Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return
    h, w = bgr.shape[:2]

    def rect(roi: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = roi
        X1, Y1 = int(x1 * w), int(y1 * h)
        X2, Y2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(bgr, (X1, Y1), (X2, Y2), color, 2)

    rect(EPIC_ROI, (0, 255, 0))
    rect(SERIAL_ROI, (255, 0, 0))
    rect(HOUSE_ROI, (0, 0, 255))

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"debug_{image_path.name}"), bgr)


def _ocr_epic(gray) -> str:
    config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_EPIC}"
    txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
    return " ".join(txt.strip().split())


def _ocr_serial(gray) -> str:
    config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_DIGITS}"
    txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
    return " ".join(txt.strip().split())


def _clean_epic(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if len(s) >= 3:
        prefix, rest = s[:3], s[3:]
        # NOTE: your original had replace("0","D") which is almost certainly a bug.
        # I'm not changing it here unless you want; leaving as-is can damage EPIC.
        prefix = (prefix.replace("0", "O").replace("1", "I").replace("2", "Z").replace("5", "S").replace("8", "B"))
        rest = (rest.replace("O", "0").replace("I", "1").replace("Z", "2").replace("S", "5").replace("B", "8"))
        s = prefix + rest
    return s


def _extract_epic_from_image(image_path: Path) -> str:
    voter_bgr = cv2.imread(str(image_path))
    if voter_bgr is None:
        return ""
    epic_bgr = _crop_rel(voter_bgr, *EPIC_ROI)
    epic_gray = _normalize_for_epic(epic_bgr)
    epic_raw = _ocr_epic(epic_gray)
    return _clean_epic(epic_raw)


def _extract_serial_from_image(image_path: Path) -> str:
    voter_bgr = cv2.imread(str(image_path))
    if voter_bgr is None:
        return ""
    serial_bgr = _crop_rel(voter_bgr, *SERIAL_ROI)
    serial_gray = _normalize_for_serial(serial_bgr)
    raw = _ocr_serial(serial_gray)
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        return ""
    # serial should be small (1-3 digits) in most rolls
    if len(digits) > 4:
        digits = digits[-4:]
    return digits.lstrip("0") or digits




# ---------------------- Full OCR to word boxes --------------------------

def _ocr_words(image_path: Path, languages: str) -> list[WordBox]:
    """
    Single-pass OCR producing word boxes.
    """
    img = Image.open(image_path)
    try:
        config = "--oem 1 --psm 6"
        data = pytesseract.image_to_data(img, lang=languages, config=config, output_type=Output.DICT)
    finally:
        img.close()

    n = len(data.get("text", []))
    out: list[WordBox] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1

        # Confidence for non-Latin scripts (Tamil) is often reported low; keep Tamil tokens
        # so label matching still works.
        if conf != -1 and conf < 20:
            if not re.search(r"[\u0B80-\u0BFF]", txt):
                continue

        out.append(
            WordBox(
                text=txt,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=conf,
                line_num=int(data.get("line_num", [0] * n)[i]),
                block_num=int(data.get("block_num", [0] * n)[i]),
                par_num=int(data.get("par_num", [0] * n)[i]),
            )
        )
    return out


def _ocr_data_dict(image_path: Path, languages: str, config: str = "--oem 1 --psm 6") -> dict[str, Any]:
    img = Image.open(image_path)
    try:
        return pytesseract.image_to_data(img, lang=languages, config=config, output_type=Output.DICT)
    finally:
        img.close()


def _words_from_ocr_data(data: dict[str, Any], *, filter_low_conf: bool = True) -> list[WordBox]:
    n = len(data.get("text", []))
    out: list[WordBox] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1

        if filter_low_conf:
            # Confidence for non-Latin scripts (Tamil) is often reported low; keep Tamil tokens
            # so label matching still works.
            if conf != -1 and conf < 20:
                if not re.search(r"[\u0B80-\u0BFF]", txt):
                    continue

        out.append(
            WordBox(
                text=txt,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=conf,
                line_num=int(data.get("line_num", [0] * n)[i]),
                block_num=int(data.get("block_num", [0] * n)[i]),
                par_num=int(data.get("par_num", [0] * n)[i]),
            )
        )
    return out


# =====================================================================
# LINE-BASED TEXT EXTRACTION (NEW APPROACH)
# =====================================================================
# Instead of relying on word-box token matching (which is fragile with
# Tamil OCR), this approach reconstructs full lines from OCR data and
# uses regex patterns to extract values. This is much more robust.
# =====================================================================


def _reconstruct_lines_from_ocr_data(data: dict[str, Any]) -> list[str]:
    """
    Reconstruct full text lines from Tesseract image_to_data output.
    Groups words by (block, par, line) and joins them with spaces.
    Returns list of line strings in order.
    """
    n = len(data.get("text", []))
    lines_dict: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        
        block = int(data.get("block_num", [0] * n)[i])
        par = int(data.get("par_num", [0] * n)[i])
        line = int(data.get("line_num", [0] * n)[i])
        x = int(data["left"][i])
        
        key = (block, par, line)
        if key not in lines_dict:
            lines_dict[key] = []
        lines_dict[key].append((x, txt))
    
    # Sort keys to maintain reading order, then join words in each line by x position
    result = []
    for key in sorted(lines_dict.keys()):
        words = sorted(lines_dict[key], key=lambda t: t[0])
        line_text = " ".join(w[1] for w in words)
        result.append(line_text)
    
    return result


def _normalize_line_for_matching(line: str) -> str:
    """
    Normalize a line for pattern matching:
    - Remove zero-width chars
    - Normalize whitespace
    - Keep Tamil and ASCII chars
    """
    s = line.strip()
    # Remove zero-width chars
    s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_value_after_colon(line: str, label_pattern: str) -> str:
    """
    Generic extractor: finds label pattern and extracts value after : or - separator.
    Returns the value portion cleaned up.
    """
    norm_line = _normalize_line_for_matching(line)
    
    # Try to match label followed by separator and value
    # Pattern: label ... [:|-|—] ... value ... [-|end]
    pattern = rf"(?:{label_pattern})\s*[:\-–—]?\s*[:\-–—]\s*(.+?)(?:\s*[\-–—]\s*$|\s*$)"
    match = re.search(pattern, norm_line, re.IGNORECASE | re.UNICODE)
    if match:
        value = match.group(1).strip()
        # Remove trailing separators and "Photo is available" type text
        value = re.sub(r"\s*[\-–—]\s*$", "", value)
        value = re.sub(r"\s*(Photo|photo|is|available|புகைப்பட|படம்).*$", "", value, flags=re.IGNORECASE)
        return value.strip()
    
    return ""


def _extract_name_from_lines(lines: list[str]) -> str:
    """Extract name from line containing பெயர் : or Name :"""
    # Pattern for name line - but NOT relation lines (father/mother/husband)
    name_patterns = [
        r"^பெயர்",                    # Tamil: starts with பெயர்
        r"^name\b",                   # English: starts with name
    ]
    
    # Patterns to EXCLUDE (relation lines that also contain பெயர்)
    exclude_patterns = [
        r"தந்தை",      # Father
        r"தாய்",       # Mother  
        r"கணவர்",      # Husband
        r"father",
        r"mother", 
        r"husband",
    ]
    
    for line in lines:
        norm = _normalize_line_for_matching(line).lower()
        
        # Skip relation lines
        if any(re.search(p, norm, re.IGNORECASE) for p in exclude_patterns):
            continue
        
        # Check if it's a name line
        for pattern in name_patterns:
            if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                # Extract value after the label
                value = _extract_value_after_colon(line, r"(?:பெயர்|name)")
                if value:
                    return value
    
    return ""


def _extract_relation_from_lines(lines: list[str]) -> tuple[str, str]:
    """
    Extract relation_type and relation_name from lines.
    Detects: Father (தந்தையின் பெயர்), Mother (தாயின் பெயர்), Husband (கணவர் பெயர்)
    Returns: (relation_type, relation_name)
    """
    relation_patterns = {
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
    
    for line in lines:
        norm = _normalize_line_for_matching(line)
        
        for rel_type, patterns in relation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, norm, re.IGNORECASE | re.UNICODE):
                    # Build full label pattern for extraction
                    label_pattern = r"(?:" + "|".join(patterns) + r")"
                    value = _extract_value_after_colon(line, label_pattern)
                    
                    # Clean up - remove any "பெயர்" that might be left over
                    value = re.sub(r"^பெயர்\s*[:\-–—]?\s*", "", value)
                    value = re.sub(r"^name\s*[:\-–—]?\s*", "", value, flags=re.IGNORECASE)
                    
                    if value:
                        return rel_type, value.strip()
    
    return "", ""


def _extract_house_from_lines(lines: list[str]) -> str:
    """
    Extract house number from lines containing வீட்டு எண் or House No.
    """
    house_patterns = [
        r"வீட்டு\s*எண்",
        r"வீட்டுஎண்",
        r"ட்டு\s*எண்",     # Sometimes OCR cuts off first char
        r"house\s*(?:no|number)?",
    ]
    
    combined_pattern = r"(?:" + "|".join(house_patterns) + r")"
    
    for line in lines:
        norm = _normalize_line_for_matching(line)
        
        if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
            # Extract what comes after the label
            value = _extract_value_after_colon(line, combined_pattern)
            
            # House numbers are typically simple: digits, maybe with letters
            # Clean out common garbage
            value = re.sub(r"(Photo|photo|is|available|புகைப்பட|படம்).*", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\s+", "", value)  # Remove spaces
            
            # Extract just the house number pattern (digit-based, may have letters)
            house_match = re.search(r"^(\d[\dA-Za-z/\-]{0,15})", value)
            if house_match:
                return house_match.group(1)
            
            # If value looks reasonable, return it
            cleaned = re.sub(r"[^\dA-Za-z/\-]", "", value)
            if cleaned and len(cleaned) <= 15:
                return cleaned
    
    return ""


def _extract_age_from_lines(lines: list[str]) -> str:
    """Extract age from lines containing வயது or Age."""
    age_patterns = [
        r"வயது",
        r"age",
    ]
    
    combined_pattern = r"(?:" + "|".join(age_patterns) + r")"
    
    for line in lines:
        norm = _normalize_line_for_matching(line)
        
        if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
            # Age is on the same line, typically "வயது : 79 பாலினம் : ஆண்"
            # Extract the number right after age label
            age_match = re.search(
                rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\d{{1,3}})",
                norm,
                re.IGNORECASE | re.UNICODE
            )
            if age_match:
                return age_match.group(1)
    
    return ""


def _extract_gender_from_lines(lines: list[str]) -> str:
    """Extract gender from lines containing பாலினம் or Gender."""
    gender_patterns = [
        r"பாலினம்",
        r"gender",
    ]
    
    combined_pattern = r"(?:" + "|".join(gender_patterns) + r")"
    
    # Gender value mappings
    gender_map = {
        "ஆண்": "Male",
        "ஆண": "Male",
        "male": "Male",
        "m": "Male",
        "பெண்": "Female",
        "பெண": "Female",
        "female": "Female",
        "f": "Female",
        "திருநங்கை": "Other",
        "other": "Other",
        "o": "Other",
    }
    
    for line in lines:
        norm = _normalize_line_for_matching(line)
        
        if re.search(combined_pattern, norm, re.IGNORECASE | re.UNICODE):
            # Gender value comes after the label
            gender_match = re.search(
                rf"(?:{combined_pattern})\s*[:\-–—]?\s*(\S+)",
                norm,
                re.IGNORECASE | re.UNICODE
            )
            if gender_match:
                raw_gender = gender_match.group(1).strip()
                # Map to standard value
                for key, value in gender_map.items():
                    if key.lower() in raw_gender.lower() or raw_gender.lower() in key.lower():
                        return value
                # If no mapping found but looks like gender, return as-is
                if raw_gender and len(raw_gender) < 15:
                    return raw_gender
    
    return ""


# =====================================================================
# ENHANCED ROI-BASED EXTRACTION (for Serial Number)
# =====================================================================


def _extract_serial_number_enhanced(image_path: Path) -> str:
    """
    Enhanced serial number extraction using multiple approaches:
    1. Direct ROI with improved preprocessing
    2. Contour-based digit detection
    """
    voter_bgr = cv2.imread(str(image_path))
    if voter_bgr is None:
        return ""
    
    h, w = voter_bgr.shape[:2]
    
    # Try multiple ROI regions - the serial box can vary slightly
    rois_to_try = [
        SERIAL_ROI,                           # Default
        (0.12, 0.04, 0.35, 0.22),             # Slightly wider/taller
        (0.15, 0.08, 0.32, 0.18),             # Slightly narrower
        (0.10, 0.02, 0.40, 0.25),             # Much wider fallback
    ]
    
    best_result = ""
    
    for roi in rois_to_try:
        try:
            serial_bgr = _crop_rel(voter_bgr, *roi)
            
            # Multiple preprocessing attempts
            results = []
            
            # Method 1: Standard preprocessing
            gray1 = cv2.cvtColor(serial_bgr, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.resize(gray1, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
            _, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive threshold
            gray2 = cv2.cvtColor(serial_bgr, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.resize(gray2, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 8)
            
            # Method 3: Inverted (for dark text on light bg)
            gray3 = cv2.cvtColor(serial_bgr, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.resize(gray3, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            gray3 = cv2.bitwise_not(gray3)
            _, thresh3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            for preprocessed in [thresh1, thresh2, thresh3]:
                config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_DIGITS}"
                txt = pytesseract.image_to_string(Image.fromarray(preprocessed), lang="eng", config=config)
                digits = re.sub(r"[^0-9]", "", txt)
                if digits:
                    results.append(digits)
            
            # Pick the most reasonable result (1-4 digits for serial)
            for r in results:
                if 1 <= len(r) <= 4:
                    if not best_result or len(r) > len(best_result):
                        best_result = r.lstrip("0") or r
                        
        except Exception:
            continue
    
    return best_result


# =====================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# =====================================================================


# ---------------------- Label-based field extraction ---------------------

def _same_line(a: WordBox, b: WordBox) -> bool:
    return (a.block_num, a.par_num, a.line_num) == (b.block_num, b.par_num, b.line_num)


def _line_words(words: list[WordBox], ref: WordBox) -> list[WordBox]:
    lw = [w for w in words if _same_line(w, ref)]
    return sorted(lw, key=lambda w: w.x)


def _join_value_words(words: list[WordBox], start_x: int, max_chars: int = 64) -> str:
    parts: list[str] = []
    length = 0

    for w in words:
        if w.x < start_x:
            continue
        t = w.text.strip()
        if not t:
            continue
        if t in LABEL_SEPARATORS:
            continue

        # STOP if another label begins (fixes "Name:" / "பெயர்:" getting included)
        if _is_stop_token(t) or (t.endswith(":") and _is_stop_token(t[:-1])):
            break

        if length + len(t) + 1 > max_chars:
            break
        parts.append(t)
        length += len(t) + 1

    return " ".join(parts).strip()


def _tokenize_label_variant(v: str) -> list[str]:
    v = v.strip()
    if not v:
        return []
    parts = re.split(r"\s+|[:\-–—|=]+", v)
    return [_normalize_text(p) for p in parts if _normalize_text(p)]


def _find_label_span(words: list[WordBox], label_variants: list[str]) -> Optional[tuple[WordBox, WordBox]]:
    """
    Find label as a *span* (start_word, end_word). Supports multi-token labels like:
    - "House Number"
    - "வீட்டு எண்"
    """
    if not words:
        return None

    # group by line
    lines: dict[tuple[int, int, int], list[WordBox]] = {}
    for w in words:
        lines.setdefault(_line_key(w), []).append(w)

    variants_tok = [_tokenize_label_variant(v) for v in label_variants]
    variants_tok = [vt for vt in variants_tok if vt]

    for _, lw in lines.items():
        lw = sorted(lw, key=lambda w: w.x)
        norm_tokens = [_normalize_text(w.text) for w in lw]

        for vt in variants_tok:
            L = len(vt)
            for i in range(0, len(lw) - L + 1):
                window = norm_tokens[i:i + L]

                ok = True
                for a, b in zip(window, vt):
                    if not a or not b or not (b in a or a in b):
                        ok = False
                        break

                if ok:
                    return (lw[i], lw[i + L - 1])

    return None


def _extract_value_after_label(
    words: list[WordBox],
    label_variants: list[str],
    allow_next_line: bool = True,
    max_chars: int = 64,
) -> str:
    span = _find_label_span(words, label_variants)
    if not span:
        return ""

    label_start, label_end = span

    lw = sorted([w for w in words if _line_key(w) == _line_key(label_start)], key=lambda w: w.x)
    if _looks_like_photo_line(lw):
        return ""

    # start AFTER the last token in the label span (e.g., after "Number")
    value = _join_value_words(lw, start_x=max(0, label_end.x2 - 5), max_chars=max_chars)
    if value:
        return value

    if not allow_next_line:
        return ""

    next_line = _get_next_visual_line(words, label_start)

    if _looks_like_photo_line(next_line):
        return ""

    return _join_value_words(next_line, start_x=0, max_chars=max_chars)


def _get_next_visual_line(words: list[WordBox], label: WordBox, max_dy: int = 60) -> list[WordBox]:
    # Find the closest visual line below the label using y proximity.
    label_cutoff_y = label.y + int(label.h * 0.6)
    below = [
        w
        for w in words
        if (w.y > label_cutoff_y) and (w.y <= label_cutoff_y + max_dy)
    ]
    if not below:
        return []

    min_y = min(w.y for w in below)
    line = [w for w in below if abs(w.y - min_y) <= 10]
    return sorted(line, key=lambda w: w.x)



def _extract_relation(words: list[WordBox], allow_next_line: bool) -> tuple[str, str]:
    """
    Only one relation label will exist: father/mother/husband (English/Tamil).
    Returns: (relation_type, relation_name) with label prefixes removed.
    """
    for rtype, variants in RELATION_LABELS.items():
        v = _extract_value_after_label(words, variants, allow_next_line=allow_next_line, max_chars=80)
        v = _strip_leading_name_label(v)
        if v:
            return rtype, v
    return "", ""


# ---------------------- House ROI extraction (fallback) ------------------

def _normalize_for_house(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return gray


def _ocr_house_roi(gray) -> str:
    # single line / whitelist (fast)
    config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_HOUSE}"
    txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
    return " ".join(txt.strip().split())


def _extract_house_from_roi(image_path: Path) -> str:
    voter_bgr = cv2.imread(str(image_path))
    if voter_bgr is None:
        return ""
    house_bgr = _crop_rel(voter_bgr, *HOUSE_ROI)
    house_gray = _normalize_for_house(house_bgr)
    raw = _ocr_house_roi(house_gray)
    return _clean_house_value(raw)


HOUSE_BAD_SUBSTRINGS = [
    "GEN", "GENDER", "MALE", "FEMALE", "AVAILABLE",
    "AGE", "SEX", "GM",
    "வயது", "பாலினம்", "ஆண்", "பெண்",
    "PHOTO", "PHOT",
]


HOUSE_TOKEN_RE = re.compile(r"^\d[\dA-Z/\-]{0,11}$")  # starts with digit, max length ~12


def _clean_house_value(s: str) -> str:
    if not s:
        return ""

    # Cut at photo markers if present
    raw_norm = _norm_for_match(s)
    cut_idx = None
    for m in PHOTO_MARKERS:
        mm = _norm_for_match(m)
        pos = raw_norm.find(mm)
        if pos != -1:
            cut_idx = pos if cut_idx is None else min(cut_idx, pos)
    if cut_idx is not None:
        s = s[:cut_idx].strip()

    s = s.upper()
    # keep only house-ish characters; convert everything else to spaces
    s = re.sub(r"[^A-Z0-9/\-\s]", " ", s)

    # confusion fixes
    s = s.replace("O", "0").replace("I", "1")

    tokens = [t.strip("/-") for t in s.split() if t.strip("/-")]

    # Filter tokens
    candidates = []
    for t in tokens:
        if not re.search(r"\d", t):
            continue
        if any(b in t for b in HOUSE_BAD_SUBSTRINGS):
            continue
        if len(t) > 12:
            continue
        if HOUSE_TOKEN_RE.fullmatch(t):
            candidates.append(t)

    if candidates:
        # Prefer tokens with more digits, then longer tokens
        candidates.sort(key=lambda t: (sum(ch.isdigit() for ch in t), len(t)), reverse=True)
        return candidates[0]

    return ""


# ---------------------- Age & gender cleaning ---------------------------

def _clean_age_value(s: str) -> str:
    if not s:
        return ""
    digits = re.sub(r"[^0-9]", "", s)
    if digits and digits.isdigit():
        n = int(digits)
        if 1 <= n <= 120:
            return digits
    return digits


def _map_gender_value(s: str) -> str:
    if not s:
        return ""

    # If it looks like PHOTO line, discard
    norm = _norm_for_match(s)
    if any(_norm_for_match(m) in norm for m in PHOTO_MARKERS):
        return ""

    g = s.lower()

    if any(k in g for k in ["male", "ஆண்", "ஆண"]):
        return "Male"
    if any(k in g for k in ["female", "பெண்", "பெண"]):
        return "Female"

    # sometimes OCR returns just M/F
    if g.strip() == "m":
        return "Male"
    if g.strip() == "f":
        return "Female"

    return ""


# ---------------------------- Per-image OCR -----------------------------

def _process_image(
    image_path: Path,
    languages: str,
    allow_next_line: bool,
    raw_ocr_md_path: Optional[Path] = None,
    raw_ocr_detail_out_dir: Optional[Path] = None,
    file_name: str = "",
    page_name: str = "",
) -> tuple[dict[str, Any], float]:
    """
    Hybrid extraction approach:
    - EPIC via ROI (most accurate)
    - Serial via enhanced ROI extraction
    - All other fields via LINE-BASED extraction (new, more robust approach)
    Returns structured record.
    """
    start = time.time()

    if DEBUG_DRAW_ROIS:
        try:
            _debug_draw_rois(image_path, image_path.parent / "_debug_rois")
        except Exception:
            pass

    # ==================== EPIC EXTRACTION (ROI) ====================
    epic_no = ""
    epic_valid = False
    try:
        epic_no = _extract_epic_from_image(image_path)
        epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", epic_no)) if epic_no else False
    except Exception:
        epic_no = ""
        epic_valid = False

    # ==================== SERIAL NUMBER EXTRACTION (Enhanced ROI) ====================
    serial_no = ""
    try:
        # Try enhanced extraction first
        serial_no = _extract_serial_number_enhanced(image_path)
        if not serial_no:
            # Fallback to original method
            serial_no = _extract_serial_from_image(image_path)
    except Exception:
        serial_no = ""

    # ==================== FULL OCR + LINE RECONSTRUCTION ====================
    try:
        tesseract_config = "--oem 1 --psm 6"
        ocr_data = _ocr_data_dict(image_path, languages, config=tesseract_config)
        
        # Raw OCR dump for debugging (if enabled)
        if raw_ocr_md_path is not None:
            append_raw_ocr_text_only(
                md_path=raw_ocr_md_path,
                file=file_name,
                page=page_name,
                image=image_path.name,
                languages=languages,
                tesseract_config=tesseract_config,
                ocr_data=ocr_data,
            )
        if raw_ocr_detail_out_dir is not None:
            detail_md_path = raw_ocr_detail_out_dir / f"{image_path.stem}.md"
            write_raw_ocr_detailed_for_image(
                out_path=detail_md_path,
                file=file_name,
                page=page_name,
                image=image_path.name,
                languages=languages,
                tesseract_config=tesseract_config,
                ocr_data=ocr_data,
            )
        
        # Reconstruct full lines for line-based extraction
        lines = _reconstruct_lines_from_ocr_data(ocr_data)
        
        # Keep word boxes for fallback (legacy method)
        words = _words_from_ocr_data(ocr_data, filter_low_conf=True)
        
    except Exception as e:
        lines = []
        words = []
        elapsed = time.time() - start
        return {
            "image": image_path.name,
            "serial_no": serial_no,
            "epic_no": epic_no,
            "epic_valid": epic_valid,
            "error": f"OCR failed: {e}",
        }, elapsed

    # ==================== LINE-BASED FIELD EXTRACTION (NEW) ====================
    
    # NAME: Extract from name line (not relation lines)
    name_val = _extract_name_from_lines(lines)
    
    # Fallback to old method if line-based fails
    if not name_val:
        name_val = _extract_value_after_label(words, NAME_LABELS, allow_next_line=allow_next_line, max_chars=80)
    
    # RELATION: Extract type and name
    rel_type, rel_name = _extract_relation_from_lines(lines)
    
    # Fallback to old method
    if not rel_name:
        rel_type, rel_name = _extract_relation(words, allow_next_line=allow_next_line)
    
    # HOUSE NUMBER: Extract from house line
    house_val = _extract_house_from_lines(lines)
    
    # Fallback chain for house
    if not house_val:
        house_raw = _extract_value_after_label(words, HOUSE_LABELS, allow_next_line=allow_next_line, max_chars=80)
        house_val = _clean_house_value(house_raw)
    
    if not house_val:
        house_val = _extract_house_from_roi(image_path)
    
    # AGE: Extract from age line
    age_val = _extract_age_from_lines(lines)
    
    # Fallback to old method
    if not age_val:
        age_val = _extract_value_after_label(words, AGE_LABELS, allow_next_line=allow_next_line, max_chars=6)
        age_val = _clean_age_value(age_val)
    
    # GENDER: Extract from gender line
    gender_val = _extract_gender_from_lines(lines)
    
    # Fallback to old method  
    if not gender_val:
        gender_val = _extract_value_after_label(words, GENDER_LABELS, allow_next_line=allow_next_line, max_chars=12)
        gender_val = _map_gender_value(gender_val)

    # ==================== FINAL CLEANUP ====================
    rel_name = _strip_leading_name_label(rel_name)

    # Safety: reject if photo markers detected in relation name
    if rel_name and any(_norm_for_match(m) in _norm_for_match(rel_name) for m in PHOTO_MARKERS):
        rel_name = ""
        rel_type = ""

    elapsed = time.time() - start
    return {
        "image": image_path.name,
        "serial_no": serial_no,
        "epic_no": epic_no,
        "epic_valid": epic_valid,

        "name": name_val,

        "relation_type": rel_type,
        "relation_name": rel_name,

        "house_no": house_val,
        "age": age_val,
        "gender": gender_val,
    }, elapsed


# ---------------------------- File/page process ----------------------------

def _process_page(
    file_folder: Path,
    page_dir: Path,
    languages: str,
    allow_next_line: bool,
    raw_ocr_md_path: Optional[Path] = None,
    raw_ocr_detail_root: Optional[Path] = None,
) -> dict[str, Any]:
    page_name = page_dir.name
    images_dir = _resolve_page_images_dir(page_dir)
    images = _get_sorted_images(images_dir)

    if not images:
        print(f"    [WARNING] No images found in {images_dir}")
        return {
            "page": page_name,
            "images_processed": 0,
            "error": f"No images found in {images_dir}",
        }

    page_start = time.time()
    records: list[dict[str, Any]] = []
    image_times: list[float] = []

    output_dir = file_folder / "output"
    page_wise_dir = output_dir / "page_wise"
    page_wise_dir.mkdir(parents=True, exist_ok=True)

    raw_ocr_detail_out_dir: Optional[Path] = None
    if raw_ocr_detail_root is not None:
        raw_ocr_detail_out_dir = raw_ocr_detail_root / page_name
        raw_ocr_detail_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        rec, t = _process_image(
            image_path,
            languages,
            allow_next_line,
            raw_ocr_md_path=raw_ocr_md_path,
            raw_ocr_detail_out_dir=raw_ocr_detail_out_dir,
            file_name=file_folder.name,
            page_name=page_name,
        )
        records.append(rec)
        image_times.append(t)
        print(f"      [{idx}/{len(images)}] {image_path.name} [EPIC+DATA] - {t:.2f}s")

    page_elapsed = time.time() - page_start
    avg_image = (sum(image_times) / len(image_times)) if image_times else 0.0

    page_output_path = page_wise_dir / f"{page_name}.json"
    page_md_path = page_wise_dir / f"{page_name}.md"

    payload = {
        "file": file_folder.name,
        "page": page_name,
        "images_count": len(images),
        "generated_at_epoch": int(time.time()),
        "languages": languages,
        "allow_next_line": allow_next_line,
        "epic_roi": {"x1": EPIC_ROI[0], "y1": EPIC_ROI[1], "x2": EPIC_ROI[2], "y2": EPIC_ROI[3]},
        "serial_roi": {"x1": SERIAL_ROI[0], "y1": SERIAL_ROI[1], "x2": SERIAL_ROI[2], "y2": SERIAL_ROI[3]},
        "timing": {
            "page_total_seconds": round(page_elapsed, 4),
            "avg_seconds_per_image": round(avg_image, 4),
            "per_image_seconds": [round(x, 4) for x in image_times],
        },
        "records": records,
    }

    write_page_markdown(page_md_path, payload)
    print(f"    [SUCCESS] Saved page Markdown: {page_md_path.name}")

    page_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [SUCCESS] Saved page JSON: {page_output_path.name}")
    print(f"    [TIMING] Page total: {page_elapsed:.2f}s | Avg per image: {avg_image:.2f}s")

    return {
        "page": page_name,
        "images_processed": len(images),
        "page_output_path": str(page_output_path),
        "page_total_seconds": page_elapsed,
        "avg_seconds_per_image": avg_image,
        "records": records,
    }


def _process_file_folder(
    folder: Path,
    languages: str,
    allow_next_line: bool,
    limit_pages: int,
    only_page: str,
    dump_raw_ocr_md: bool,
) -> dict[str, Any]:
    folder_name = folder.name
    crops_dir = folder / "crops"
    print(f"\nProcessing file: {folder_name}")

    pages = _get_sorted_page_dirs(crops_dir)

    if only_page:
        pages = [p for p in pages if p.name == only_page]

    if limit_pages and limit_pages > 0:
        pages = pages[:limit_pages]

    if not pages:
        print(f"  [WARNING] No pages found in {crops_dir}")
        return {"folder": folder_name, "pages_processed": 0, "images_processed": 0, "error": "No pages found"}

    print(f"  Found {len(pages)} page(s)")
    file_start = time.time()

    output_dir = folder / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_ocr_md_path: Optional[Path] = None
    raw_ocr_detail_root: Optional[Path] = None
    if dump_raw_ocr_md:
        raw_ocr_md_path = output_dir / "raw_ocr.md"
        raw_ocr_detail_root = output_dir / "raw_ocr"
        init_raw_ocr_markdown(
            raw_ocr_md_path,
            file=folder_name,
            languages=languages,
            tesseract_config="--oem 1 --psm 6",
        )
        print(f"  [INFO] Raw OCR text dump enabled: {raw_ocr_md_path.name}")
        print("  [INFO] Raw OCR detailed dump enabled: output/raw_ocr/<page>/<image>.md")

    page_summaries: list[dict[str, Any]] = []
    total_images = 0

    for idx, page_dir in enumerate(pages, start=1):
        print(f"  [PAGE {idx}/{len(pages)}] {page_dir.name}")
        page_summary = _process_page(
            folder,
            page_dir,
            languages,
            allow_next_line,
            raw_ocr_md_path=raw_ocr_md_path,
            raw_ocr_detail_root=raw_ocr_detail_root,
        )
        page_summaries.append(page_summary)
        total_images += int(page_summary.get("images_processed", 0) or 0)

    file_elapsed = time.time() - file_start

    file_output_path = output_dir / f"{folder_name}.json"
    file_md_path = output_dir / f"{folder_name}.md"

    file_payload = {
        "folder": folder_name,
        "pages_count": len(pages),
        "images_count": total_images,
        "generated_at_epoch": int(time.time()),
        "languages": languages,
        "allow_next_line": allow_next_line,
        "epic_roi": {"x1": EPIC_ROI[0], "y1": EPIC_ROI[1], "x2": EPIC_ROI[2], "y2": EPIC_ROI[3]},
        "timing": {
            "file_total_seconds": round(file_elapsed, 4),
            "avg_seconds_per_image": round((file_elapsed / total_images) if total_images else 0.0, 4),
        },
        "pages": [
            {
                "page": p.get("page"),
                "images_processed": p.get("images_processed"),
                "page_output_path": p.get("page_output_path"),
                "page_total_seconds": round(float(p.get("page_total_seconds", 0.0) or 0.0), 4),
                "avg_seconds_per_image": round(float(p.get("avg_seconds_per_image", 0.0) or 0.0), 4),
                "records": p.get("records", []),
            }
            for p in page_summaries
            if int(p.get("images_processed", 0) or 0) > 0
        ],
    }

    write_file_markdown(file_md_path, file_payload)
    print(f"  [SUCCESS] Saved file Markdown: {file_md_path.name}")

    file_output_path.write_text(json.dumps(file_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  [SUCCESS] Saved file JSON: {file_output_path.name}")
    print(f"  [TIMING] File total: {file_elapsed:.2f}s | Images: {total_images} | Avg/image: {(file_elapsed / total_images) if total_images else 0.0:.2f}s")

    return {
        "folder": folder_name,
        "pages_processed": len([p for p in page_summaries if int(p.get('images_processed', 0) or 0) > 0]),
        "images_processed": total_images,
        "file_output_path": str(file_output_path),
        "file_total_seconds": file_elapsed,
    }


# --------------------------------- CLI ----------------------------------

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_extracted = script_dir / "extracted"

    parser = argparse.ArgumentParser(
        description="Process extracted voter crops with Tesseract (EPIC ROI + single-pass label extraction)."
    )
    parser.add_argument("--extracted", type=Path, default=default_extracted,
                        help="Directory containing extracted image folders.")
    parser.add_argument("--languages", type=str, default="eng+tam",
                        help="Languages for OCR in Tesseract format (e.g., eng+tam).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N files/folders under extracted (0 = all).")
    parser.add_argument("--limit-pages", type=int, default=0,
                        help="Process only first N pages per file (0 = all).")
    parser.add_argument("--only-folder", type=str, default="",
                        help="Process only a specific folder under extracted (exact name match).")
    parser.add_argument("--only-page", type=str, default="",
                        help="Process only a specific page folder under crops (e.g., page-004).")
    parser.add_argument("--allow-next-line", type=int, default=1,
                        help="If 1, when value isn't on same line as label, try next line in same block/par.")
    parser.add_argument(
        "--dump-raw-ocr-md",
        action="store_true",
        help="Write raw OCR text-only dump (single file) to extracted/<file>/output/raw_ocr.md",
    )
    parser.add_argument("--debug-rois", type=int, default=0,
                        help="If 1, saves ROI overlay images to a _debug_rois folder near each input image.")

    args = parser.parse_args()

    global DEBUG_DRAW_ROIS
    DEBUG_DRAW_ROIS = bool(args.debug_rois)

    extracted_dir: Path = args.extracted
    allow_next_line = bool(args.allow_next_line)

    if not extracted_dir.exists():
        print(f"[ERROR] Extracted directory not found: {extracted_dir}")
        return 1

    folders = _get_extracted_folders(extracted_dir)

    if args.only_folder:
        folders = [f for f in folders if f.name == args.only_folder]

    if args.limit and args.limit > 0:
        folders = folders[:args.limit]

    if not folders:
        print(f"[ERROR] No folders with images found in: {extracted_dir}")
        return 2

    print(f"Found {len(folders)} folder(s) to process")
    print(f"Languages: {args.languages}")
    print(f"Allow next line: {allow_next_line}")
    if args.dump_raw_ocr_md:
        print("Raw OCR Markdown: enabled")
    if args.only_folder:
        print(f"Only folder: {args.only_folder}")
    if args.only_page:
        print(f"Only page: {args.only_page}")
    if args.limit_pages and args.limit_pages > 0:
        print(f"Limit pages per file: {args.limit_pages}")

    print("\nInitializing Tesseract OCR...")
    init_start = time.time()
    _initialize_tesseract(args.languages)
    init_time = time.time() - init_start
    print(f"[TIMING] Initialization took: {init_time:.2f}s")

    results = []
    total_start = time.time()
    for folder in folders:
        results.append(
            _process_file_folder(
                folder,
                args.languages,
                allow_next_line,
                args.limit_pages,
                args.only_page,
                dump_raw_ocr_md=bool(args.dump_raw_ocr_md),
            )
        )
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("SUMMARY (TESSERACT EPIC ROI + PAGE-WISE DATA EXTRACTION)")
    print("=" * 60)
    total_images = sum(int(r.get("images_processed", 0) or 0) for r in results)
    total_ocr_time = sum(float(r.get("file_total_seconds", 0.0) or 0.0) for r in results)
    successful = len([r for r in results if int(r.get("images_processed", 0) or 0) > 0])

    print(f"[SUCCESS] Processed {successful}/{len(folders)} folders")
    print(f"[SUCCESS] Total images processed: {total_images}")
    print("[SUCCESS] Output: extracted/<file>/output/<file>.json + extracted/<file>/output/page_wise/<page>.json")
    print("[INFO] Processing mode: EPIC ROI + image_to_data label extraction")
    print(f"[TIMING] Initialization: {init_time:.2f}s")
    print(f"[TIMING] OCR Processing: {total_ocr_time:.2f}s")
    print(f"[TIMING] Total elapsed: {total_elapsed:.2f}s")
    if total_images > 0:
        print(f"[TIMING] Average per image: {total_ocr_time / total_images:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

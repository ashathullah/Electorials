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
  pdf-converter/extracted/*/images/*.(png/jpg/...)

Output:
  pdf-converter/json-files-tesseract/<folder>.json

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
from typing import Any, Iterable, Optional

try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image
    import cv2
    import numpy as np

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ----------------------------- Configuration ----------------------------

# EPIC ROI ratios (x1, y1, x2, y2) relative to crop size
# You confirmed: epic is top-right
EPIC_ROI = (0.55, 0.05, 0.98, 0.18)

# Label variants (English/Tamil). We match by "contains" after normalization.
NAME_LABELS = [
    "name",        # English
    "பெயர்",       # Tamil
    "பெயர்‌",      # Tamil with ZWJ/ZWSP variants in OCR
]

RELATION_LABELS = {
    "father": ["father", "தந்தையின்", "தந்தை", "தந்தையின்‌", "தந்தையின்‌ பெயர்", "தந்தை பெயர்‌", "தந்தை பெயர்", "தந்தையின் பெயர்", "தந்தையின் பெயர்"],
    "mother": ["mother", "தாயின்", "தாய்", "தாயின்‌", "தாயின்‌ பெயர்", "தாய் பெயர்‌", "தாய் பெயர்", "தாயின் பெயர்", "தாயின் பெயர்"],
    "husband": ["husband", "கணவர்", "கணவர்‌", "கணவர்‌ பெயர்", "கணவர்‌ பெயர்‌", "கணவர் பெயர்", "கணவர் பெயர்", "கணவர் பெயர்"],
}

HOUSE_LABELS = ["house", "house no", "house number", "வீட்டு", "வீட்டு எண்", "வீட்டு எண்‌", "வீடு"]
AGE_LABELS = ["age", "வயது", "வயது‌"]
GENDER_LABELS = ["gender", "பாலினம்", "பாலினம்‌"]

# Common separators that show up near labels
LABEL_SEPARATORS = {":", "-", "—", "|", "=", "№"}

# Whitelists
WL_EPIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WL_DIGITS = "0123456789"
WL_HOUSE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"


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
            images_dir = folder / "images"
            if images_dir.exists() and images_dir.is_dir():
                folders.append(folder)
    return folders


def _get_sorted_images(images_dir: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    images = [
        img for img in images_dir.iterdir()
        if img.is_file() and img.suffix.lower() in extensions
    ]
    return sorted(images)


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


def _ocr_epic(gray) -> str:
    config = f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL_EPIC}"
    txt = pytesseract.image_to_string(Image.fromarray(gray), lang="eng", config=config)
    return " ".join(txt.strip().split())


def _clean_epic(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if len(s) >= 3:
        prefix, rest = s[:3], s[3:]
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


# ---------------------- Full OCR to word boxes --------------------------

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

        # You can raise this if you want more strictness (at risk of missing)
        if conf != -1 and conf < 20:
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


# ---------------------- Label-based field extraction ---------------------

def _same_line(a: WordBox, b: WordBox) -> bool:
    # Using line_num + block + paragraph gives stable grouping when available
    return (a.block_num, a.par_num, a.line_num) == (b.block_num, b.par_num, b.line_num)


def _line_words(words: list[WordBox], ref: WordBox) -> list[WordBox]:
    lw = [w for w in words if _same_line(w, ref)]
    return sorted(lw, key=lambda w: w.x)


def _join_value_words(words: list[WordBox], start_x: int, max_chars: int = 64) -> str:
    """
    Join words to the right of a label on same line.
    """
    parts: list[str] = []
    length = 0
    for w in words:
        if w.x < start_x:
            continue
        t = w.text.strip()
        if not t:
            continue
        # skip pure separators
        if t in LABEL_SEPARATORS:
            continue
        if length + len(t) + 1 > max_chars:
            break
        parts.append(t)
        length += len(t) + 1
    return " ".join(parts).strip()


def _find_label_token(words: list[WordBox], label_variants: list[str]) -> Optional[WordBox]:
    """
    Find the first token whose normalized text contains any normalized label variant.
    """
    norm_variants = [_normalize_text(v) for v in label_variants if v.strip()]
    for w in words:
        nw = _normalize_text(w.text)
        if not nw:
            continue
        for v in norm_variants:
            if v and (v in nw or nw in v):
                return w
    return None


def _extract_value_after_label(
    words: list[WordBox],
    label_variants: list[str],
    allow_next_line: bool = True,
    max_chars: int = 64,
) -> str:
    """
    1) Locate a label token.
    2) Take words to the RIGHT on same line as value.
    3) If empty and allow_next_line: take next line's words (same block/par) from left.
    """
    label = _find_label_token(words, label_variants)
    if not label:
        return ""

    lw = _line_words(words, label)
    # start after label's right edge (with small margin)
    value = _join_value_words(lw, start_x=label.x2 + 5, max_chars=max_chars)

    if value:
        return value

    if not allow_next_line:
        return ""

    # Try next line in same block/par
    target_block_par = (label.block_num, label.par_num)
    # candidate lines with same block/par and line_num = label.line_num + 1
    next_line = [
        w for w in words
        if (w.block_num, w.par_num) == target_block_par and w.line_num == label.line_num + 1
    ]
    next_line = sorted(next_line, key=lambda w: w.x)
    # join from beginning
    value2 = _join_value_words(next_line, start_x=0, max_chars=max_chars)
    return value2


def _extract_relation(words: list[WordBox], allow_next_line: bool) -> tuple[str, str]:
    """
    Only one relation label will exist: father/mother/husband (English/Tamil).
    Returns: (relation_type, relation_name)
    """
    for rtype, variants in RELATION_LABELS.items():
        v = _extract_value_after_label(words, variants, allow_next_line=allow_next_line, max_chars=80)
        if v:
            return rtype, v
    return "", ""


def _clean_house_value(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9/\-]", "", s)
    # confusion fixes
    s = s.replace("O", "0").replace("I", "1")
    return s


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
    g = s.lower()
    if any(k in g for k in ["male", "ஆண்", "ஆண"]):
        return "Male"
    if any(k in g for k in ["female", "பெண்", "பெண"]):
        return "Female"
    # sometimes OCR returns just M/F
    if g.strip() in {"m", "f"}:
        return "Male" if g.strip() == "m" else "Female"
    return s.strip()


# ---------------------------- Per-image OCR -----------------------------

def _process_image(image_path: Path, languages: str, allow_next_line: bool) -> tuple[dict[str, Any], float]:
    """
    Hybrid:
    - EPIC via ROI
    - Other fields via image_to_data (single-pass)
    Returns structured record.
    """
    start = time.time()

    epic_no = ""
    epic_valid = False
    try:
        epic_no = _extract_epic_from_image(image_path)
        epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", epic_no)) if epic_no else False
    except Exception:
        epic_no = ""
        epic_valid = False

    # single-pass OCR to words
    try:
        words = _ocr_words(image_path, languages)
    except Exception as e:
        words = []
        # still output epic if we have it
        elapsed = time.time() - start
        return {
            "image": image_path.name,
            "epic_no": epic_no,
            "epic_valid": epic_valid,
            "error": f"image_to_data failed: {e}",
        }, elapsed

    # Extract fields from labels
    name_val = _extract_value_after_label(words, NAME_LABELS, allow_next_line=allow_next_line, max_chars=80)

    rel_type, rel_name = _extract_relation(words, allow_next_line=allow_next_line)

    house_val = _extract_value_after_label(words, HOUSE_LABELS, allow_next_line=allow_next_line, max_chars=40)
    house_val = _clean_house_value(house_val)

    age_val = _extract_value_after_label(words, AGE_LABELS, allow_next_line=allow_next_line, max_chars=6)
    age_val = _clean_age_value(age_val)

    gender_val = _extract_value_after_label(words, GENDER_LABELS, allow_next_line=allow_next_line, max_chars=12)
    gender_val = _map_gender_value(gender_val)

    elapsed = time.time() - start
    return {
        "image": image_path.name,
        "epic_no": epic_no,
        "epic_valid": epic_valid,

        "name": name_val,

        "relation_type": rel_type,      # father/mother/husband (or "")
        "relation_name": rel_name,      # extracted string (or "")

        "house_no": house_val,
        "age": age_val,
        "gender": gender_val,
    }, elapsed


# ---------------------------- Folder process ----------------------------

def _process_folder(folder: Path, languages: str, json_output_dir: Path, allow_next_line: bool) -> dict[str, Any]:
    folder_name = folder.name
    images_dir = folder / "images"

    print(f"\nProcessing: {folder_name}")

    images = _get_sorted_images(images_dir)
    if not images:
        print(f"  [WARNING] No images found in {images_dir}")
        return {"folder": folder_name, "images_processed": 0, "error": "No images found"}

    print(f"  Found {len(images)} images")

    records: list[dict[str, Any]] = []
    total_time = 0.0

    for idx, image_path in enumerate(images, start=1):
        rec, t = _process_image(image_path, languages, allow_next_line)
        total_time += t
        records.append(rec)
        print(f"  [{idx}/{len(images)}] {image_path.name} [EPIC+DATA] - {t:.2f}s")

    json_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = json_output_dir / f"{folder_name}.json"

    payload = {
        "folder": folder_name,
        "images_count": len(images),
        "generated_at_epoch": int(time.time()),
        "languages": languages,
        "allow_next_line": allow_next_line,
        "epic_roi": {"x1": EPIC_ROI[0], "y1": EPIC_ROI[1], "x2": EPIC_ROI[2], "y2": EPIC_ROI[3]},
        "records": records,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    avg = total_time / len(images) if images else 0.0
    print(f"  [SUCCESS] Saved to: {output_path.name}")
    print(f"  [TIMING] Total: {total_time:.2f}s | Avg per image: {avg:.2f}s")

    return {
        "folder": folder_name,
        "images_processed": len(images),
        "output_path": str(output_path.as_posix()),
        "total_time": total_time,
        "avg_time": avg,
    }


# --------------------------------- CLI ----------------------------------

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_extracted = script_dir / "extracted"
    default_json_output = script_dir / "json-files-tesseract"

    parser = argparse.ArgumentParser(
        description="Process extracted voter crops with Tesseract (EPIC ROI + single-pass label extraction)."
    )
    parser.add_argument("--extracted", type=Path, default=default_extracted,
                        help="Directory containing extracted image folders.")
    parser.add_argument("--output", type=Path, default=default_json_output,
                        help="Output directory for JSON files.")
    parser.add_argument("--languages", type=str, default="eng+tam",
                        help="Languages for OCR in Tesseract format (e.g., eng+tam).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N folders (0 = all).")
    parser.add_argument("--allow-next-line", type=int, default=1,
                        help="If 1, when value isn't on same line as label, try next line in same block/par.")

    args = parser.parse_args()

    extracted_dir: Path = args.extracted
    json_output_dir: Path = args.output
    allow_next_line = bool(args.allow_next_line)

    if not extracted_dir.exists():
        print(f"[ERROR] Extracted directory not found: {extracted_dir}")
        return 1

    folders = _get_extracted_folders(extracted_dir)
    if args.limit and args.limit > 0:
        folders = folders[:args.limit]

    if not folders:
        print(f"[ERROR] No folders with images found in: {extracted_dir}")
        return 2

    print(f"Found {len(folders)} folder(s) to process")
    print(f"Languages: {args.languages}")
    print(f"Allow next line: {allow_next_line}")

    print("\nInitializing Tesseract OCR...")
    init_start = time.time()
    _initialize_tesseract(args.languages)
    init_time = time.time() - init_start
    print(f"[TIMING] Initialization took: {init_time:.2f}s")

    results = []
    total_start = time.time()
    for folder in folders:
        results.append(_process_folder(folder, args.languages, json_output_dir, allow_next_line))
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("SUMMARY (TESSERACT EPIC ROI + DATA EXTRACTION)")
    print("=" * 60)
    total_images = sum(r.get("images_processed", 0) for r in results)
    total_ocr_time = sum(r.get("total_time", 0) for r in results)
    successful = len([r for r in results if r.get("images_processed", 0) > 0])

    print(f"[SUCCESS] Processed {successful}/{len(folders)} folders")
    print(f"[SUCCESS] Total images processed: {total_images}")
    print(f"[SUCCESS] Output directory: {json_output_dir}")
    print("[INFO] Processing mode: EPIC ROI + image_to_data label extraction")
    print(f"[TIMING] Initialization: {init_time:.2f}s")
    print(f"[TIMING] OCR Processing: {total_ocr_time:.2f}s")
    print(f"[TIMING] Total elapsed: {total_elapsed:.2f}s")
    if total_images > 0:
        print(f"[TIMING] Average per image: {total_ocr_time / total_images:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

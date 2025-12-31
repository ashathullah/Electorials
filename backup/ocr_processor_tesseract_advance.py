"""
OCR Processor for extracted PDF images using Tesseract.

Hybrid strategy (per your request):
- EPIC number: field-wise ROI + whitelist + format corrections (fast & accurate)
- Everything else: OCR "as usual" on the full image (single Tesseract call)

Output:
- Structured JSON per folder (one JSON file per folder)
  pdf-converter/json-files-tesseract/<folder>.json

Defaults:
  - Input:    pdf-converter/extracted/*/images/*.(png/jpg/...)
  - Output:   pdf-converter/json-files-tesseract/<folder>.json

Examples:
  python pdf-converter/ocr_processor_tesseract.py
  python pdf-converter/ocr_processor_tesseract.py --languages eng+tam
  python pdf-converter/ocr_processor_tesseract.py --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ----------------------------------------------------------------------
# You confirmed:
# serial : top-left
# epic   : top-right
# name
# father_name
# house_no
# age     gender
#
# We'll ONLY use ROI for EPIC (top-right).
# Ratios are (x1, y1, x2, y2) relative to voter crop width/height.
# ----------------------------------------------------------------------
EPIC_ROI = (0.55, 0.05, 0.98, 0.18)


def _get_extracted_folders(extracted_dir: Path) -> list[Path]:
    """Get all folders in extracted directory that contain images."""
    folders: list[Path] = []
    for folder in sorted(extracted_dir.iterdir()):
        if folder.is_dir():
            images_dir = folder / "images"
            if images_dir.exists() and images_dir.is_dir():
                folders.append(folder)
    return folders


def _get_sorted_images(images_dir: Path) -> list[Path]:
    """Get all images from a directory, sorted by name."""
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    images = [
        img for img in images_dir.iterdir()
        if img.is_file() and img.suffix.lower() in extensions
    ]
    return sorted(images)


def _initialize_tesseract(languages: str) -> None:
    """Initialize Tesseract and check if it's available."""
    if not TESSERACT_AVAILABLE:
        print("[ERROR] Missing deps. Please run: pip install pytesseract pillow opencv-python numpy")
        raise ImportError("pytesseract/pillow/opencv-python/numpy required")

    # Set Tesseract path for Windows
    if os.name == "nt":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"[INFO] Using Tesseract from: {tesseract_path}")

    # Verify installation
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract version: {version}")
    except Exception as e:
        print("[ERROR] Tesseract not found. Please install Tesseract OCR.")
        print("        Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        raise RuntimeError(f"Tesseract not available: {e}")

    print(f"[INFO] Using languages for FULL OCR: {languages}")
    print("[SUCCESS] Tesseract initialized (CPU mode)")


# ----------------------- EPIC extraction helpers ------------------------

def _crop_rel(img_bgr, x1: float, y1: float, x2: float, y2: float):
    """Crop using relative ratios."""
    h, w = img_bgr.shape[:2]
    X1 = int(round(x1 * w)); Y1 = int(round(y1 * h))
    X2 = int(round(x2 * w)); Y2 = int(round(y2 * h))
    X1 = max(0, min(w - 1, X1)); X2 = max(1, min(w, X2))
    Y1 = max(0, min(h - 1, Y1)); Y2 = max(1, min(h, Y2))
    return img_bgr[Y1:Y2, X1:X2]


def _normalize_for_epic(bgr):
    """
    Lightweight preprocessing for EPIC ROI:
    - grayscale
    - upscale 2x
    - light denoise (fast)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return gray


def _pil_from_gray(gray):
    return Image.fromarray(gray)


def _ocr_epic(gray) -> str:
    # single line, whitelist A-Z0-9, LSTM only
    config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    txt = pytesseract.image_to_string(_pil_from_gray(gray), lang="eng", config=config)
    return " ".join(txt.strip().split())


def _clean_epic(raw: str) -> str:
    """
    EPIC format: first 3 letters + digits.
    Targeted confusion fixes.
    """
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    if len(s) >= 3:
        prefix, rest = s[:3], s[3:]

        # prefix should be letters
        prefix = (prefix
                  .replace("0", "O")
                  .replace("1", "I")
                  .replace("2", "Z")
                  .replace("5", "S")
                  .replace("8", "B"))

        # tail should be digits
        rest = (rest
                .replace("O", "0")
                .replace("I", "1")
                .replace("Z", "2")
                .replace("S", "5")
                .replace("B", "8"))

        s = prefix + rest

    return s


def _extract_epic_from_image(image_path: Path) -> str:
    """
    Read image via OpenCV -> crop EPIC ROI -> OCR -> clean.
    Falls back to empty string if anything fails.
    """
    voter_bgr = cv2.imread(str(image_path))
    if voter_bgr is None:
        return ""

    epic_bgr = _crop_rel(voter_bgr, *EPIC_ROI)
    epic_gray = _normalize_for_epic(epic_bgr)
    epic_raw = _ocr_epic(epic_gray)
    return _clean_epic(epic_raw)


# ----------------------- Full OCR (as usual) ----------------------------

def _full_ocr_text(image_path: Path, languages: str) -> str:
    """
    Your original approach: PIL + pytesseract.image_to_string on full image.
    Kept fast and simple.
    """
    img = Image.open(image_path)
    try:
        # Mildly better defaults than plain call, without slowing much
        config = "--oem 1 --psm 6"
        txt = pytesseract.image_to_string(img, lang=languages, config=config)
        return txt.strip() if txt.strip() else ""
    finally:
        img.close()


# ---------------------------- Processing --------------------------------

def _process_image_with_ocr(image_path: Path, languages: str) -> tuple[dict[str, Any], float]:
    """
    Returns:
      {
        "image": "<filename>",
        "epic_no": "<cleaned>",
        "text": "<full ocr text>",
        "epic_valid": true/false
      }
    """
    start_time = time.time()

    epic_no = ""
    try:
        epic_no = _extract_epic_from_image(image_path)
    except Exception:
        epic_no = ""

    try:
        text = _full_ocr_text(image_path, languages)
    except Exception as e:
        text = f"[Error processing image: {e}]"

    epic_valid = bool(re.fullmatch(r"[A-Z]{3}\d+", epic_no)) if epic_no else False

    elapsed = time.time() - start_time
    return {
        "image": image_path.name,
        "epic_no": epic_no,
        "epic_valid": epic_valid,
        "text": text,
    }, elapsed


def _process_folder(folder: Path, languages: str, json_output_dir: Path) -> dict[str, Any]:
    """Process all images in a folder and create a JSON file."""
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
        rec, img_time = _process_image_with_ocr(image_path, languages)
        total_time += img_time
        records.append(rec)
        print(f"  [{idx}/{len(images)}] {image_path.name} [EPIC+FULL] - {img_time:.2f}s")

    # Write JSON
    json_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = json_output_dir / f"{folder_name}.json"

    payload = {
        "folder": folder_name,
        "images_count": len(images),
        "generated_at_epoch": int(time.time()),
        "languages_full": languages,
        "epic_roi": {"x1": EPIC_ROI[0], "y1": EPIC_ROI[1], "x2": EPIC_ROI[2], "y2": EPIC_ROI[3]},
        "records": records,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_time = total_time / len(images) if images else 0.0
    print(f"  [SUCCESS] Saved to: {output_path.name}")
    print(f"  [TIMING] Total: {total_time:.2f}s | Avg per image: {avg_time:.2f}s")

    return {
        "folder": folder_name,
        "images_processed": len(images),
        "output_path": str(output_path.as_posix()),
        "total_time": total_time,
        "avg_time": avg_time,
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_extracted = script_dir / "extracted"
    default_json_output = script_dir / "json-files-tesseract"

    parser = argparse.ArgumentParser(description="Process extracted images with Tesseract OCR (EPIC ROI + full OCR).")
    parser.add_argument("--extracted", type=Path, default=default_extracted,
                        help="Directory containing extracted image folders.")
    parser.add_argument("--output", type=Path, default=default_json_output,
                        help="Output directory for JSON files.")
    parser.add_argument("--languages", type=str, default="eng+tam",
                        help="Languages for FULL OCR in Tesseract format (e.g., eng+tam).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N folders (0 = all).")

    args = parser.parse_args()

    extracted_dir: Path = args.extracted
    json_output_dir: Path = args.output

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
    print(f"Languages (FULL OCR): {args.languages}")

    # Initialize Tesseract
    print("\nInitializing Tesseract OCR...")
    init_start = time.time()
    _initialize_tesseract(args.languages)
    init_time = time.time() - init_start
    print(f"[TIMING] Initialization took: {init_time:.2f}s")

    results = []
    total_start = time.time()
    for folder in folders:
        result = _process_folder(folder, args.languages, json_output_dir)
        results.append(result)
    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (TESSERACT EPIC+FULL OCR)")
    print("=" * 60)
    total_images = sum(r.get("images_processed", 0) for r in results)
    total_ocr_time = sum(r.get("total_time", 0) for r in results)
    successful = len([r for r in results if r.get("images_processed", 0) > 0])

    print(f"[SUCCESS] Processed {successful}/{len(folders)} folders")
    print(f"[SUCCESS] Total images processed: {total_images}")
    print(f"[SUCCESS] Output directory: {json_output_dir}")
    print("[INFO] Processing mode: EPIC ROI + Full OCR text")
    print(f"[TIMING] Initialization: {init_time:.2f}s")
    print(f"[TIMING] OCR Processing: {total_ocr_time:.2f}s")
    print(f"[TIMING] Total elapsed: {total_elapsed:.2f}s")
    if total_images > 0:
        print(f"[TIMING] Average per image: {total_ocr_time / total_images:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

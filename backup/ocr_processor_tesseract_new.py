"""
OCR Processor for extracted PDF images using Tesseract (field-wise).

What changed vs your original:
- Uses OpenCV to read each voter-crop image.
- Crops each FIELD by relative ROI (works across 610x256, 758x314, +/- 2px, etc.).
- Runs Tesseract per-field with the right language + whitelist.
- Enforces EPIC format (first 3 letters + digits) with targeted confusion fixes.
- Writes a markdown file per folder (same as your pipeline).

Defaults:
  - Input:    pdf-converter/extracted/*/images/*.(png/jpg/...)
  - Output:   pdf-converter/md-files-tesseract/<folder>.md

Examples:
  python pdf-converter/ocr_processor_tesseract.py
  python pdf-converter/ocr_processor_tesseract.py --languages eng+tam
  python pdf-converter/ocr_processor_tesseract.py --limit 2
  python pdf-converter/ocr_processor_tesseract.py --strip-labels 0.18
"""

from __future__ import annotations

import argparse
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
# Field layout (you confirmed):
# serial : top-left
# epic   : top-right
# name
# father_name
# house_no
# age (left)  gender (right)
#
# All ratios are (x1, y1, x2, y2) relative to voter crop width/height.
# ----------------------------------------------------------------------
FIELD_ROIS = {
    "serial":      (0.05, 0.05, 0.35, 0.18),  # top-left
    "epic":        (0.55, 0.05, 0.98, 0.18),  # top-right

    "name":        (0.05, 0.20, 0.98, 0.38),
    "father_name": (0.05, 0.38, 0.98, 0.56),
    "house_no":    (0.05, 0.56, 0.98, 0.72),

    "age":         (0.05, 0.72, 0.55, 0.92),  # bottom-left
    "gender":      (0.55, 0.72, 0.98, 0.92),  # bottom-right
}


# ----------------------------- File helpers -----------------------------

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


# -------------------------- Tesseract init ------------------------------

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

    print(f"[INFO] Using languages arg (for text fields): {languages}")
    print("[SUCCESS] Tesseract initialized (CPU mode)")


# -------------------------- OCR utilities -------------------------------

def _crop_rel(img_bgr, x1: float, y1: float, x2: float, y2: float):
    """Crop using relative ratios, robust to small size differences."""
    h, w = img_bgr.shape[:2]
    X1 = int(round(x1 * w)); Y1 = int(round(y1 * h))
    X2 = int(round(x2 * w)); Y2 = int(round(y2 * h))
    X1 = max(0, min(w - 1, X1)); X2 = max(1, min(w, X2))
    Y1 = max(0, min(h - 1, Y1)); Y2 = max(1, min(h, Y2))
    return img_bgr[Y1:Y2, X1:X2]


def _normalize_for_ocr(bgr):
    """
    OCR-friendly preprocessing:
    - grayscale
    - upscale 2x (helps Tesseract)
    - light denoise (doesn't thicken strokes)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return gray


def _pil_from_gray(gray):
    return Image.fromarray(gray)


def _ocr_line(gray, lang: str, whitelist: str | None = None) -> str:
    """
    OCR a single-line region.
    psm 7 = treat image as a single text line.
    oem 1 = LSTM only.
    """
    config = "--oem 1 --psm 7"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(_pil_from_gray(gray), lang=lang, config=config)
    return " ".join(txt.strip().split())


def _clean_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s)


def _clean_house(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9/\-]", "", s)
    # light confusion fixes
    s = s.replace("O", "0").replace("I", "1")
    return s


def _clean_epic(raw: str) -> str:
    """
    EPIC format: first 3 letters + digits.
    Apply targeted fixes:
      - Prefix: digits -> letters (0->O, 1->I, 5->S, 2->Z, 8->B)
      - Tail: letters -> digits (O->0, I->1, S->5, Z->2, B->8)
    """
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    if len(s) >= 3:
        prefix, rest = s[:3], s[3:]

        prefix = (prefix
                  .replace("0", "O")
                  .replace("1", "I")
                  .replace("2", "Z")
                  .replace("5", "S")
                  .replace("8", "B"))

        rest = (rest
                .replace("O", "0")
                .replace("I", "1")
                .replace("Z", "2")
                .replace("S", "5")
                .replace("B", "8"))

        s = prefix + rest

    return s


def _map_gender(raw: str) -> str:
    g = raw.lower()
    if any(k in g for k in ["male", "ஆண்", "ஆண"]):
        return "Male"
    if any(k in g for k in ["female", "பெண்", "பெண"]):
        return "Female"
    return raw.strip()


def _strip_label_area(bgr, strip_left_ratio: float) -> Any:
    """
    Optionally remove a left label column if your ROI includes label text.
    Example: strip_left_ratio=0.18 removes first 18% of width.
    """
    if strip_left_ratio <= 0:
        return bgr
    h, w = bgr.shape[:2]
    x = int(w * strip_left_ratio)
    x = max(0, min(w - 1, x))
    return bgr[:, x:]


def _extract_voter_fields(voter_bgr, text_languages: str, strip_labels_ratio: float) -> dict[str, str]:
    """
    Extract all fields from a single voter crop image (BGR).
    text_languages is used for Name/Father/Gender where Tamil+English appears.
    """
    def roi(field: str):
        x1, y1, x2, y2 = FIELD_ROIS[field]
        region = _crop_rel(voter_bgr, x1, y1, x2, y2)
        # strip label area only for stacked text lines (not serial/epic)
        if field in {"name", "father_name", "house_no", "age", "gender"}:
            region = _strip_label_area(region, strip_labels_ratio)
        return region

    fields: dict[str, str] = {}

    # Serial: digits only
    serial_img = _normalize_for_ocr(roi("serial"))
    fields["serial_no"] = _clean_digits(_ocr_line(serial_img, "eng", whitelist="0123456789"))

    # EPIC: 3 letters + digits
    epic_img = _normalize_for_ocr(roi("epic"))
    epic_raw = _ocr_line(epic_img, "eng", whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    fields["epic_no"] = _clean_epic(epic_raw)

    # Name: Tamil + English
    name_img = _normalize_for_ocr(roi("name"))
    fields["name"] = _ocr_line(name_img, text_languages)

    # Father Name: Tamil + English
    father_img = _normalize_for_ocr(roi("father_name"))
    fields["father_name"] = _ocr_line(father_img, text_languages)

    # House No: alphanum + / -
    house_img = _normalize_for_ocr(roi("house_no"))
    house_raw = _ocr_line(house_img, "eng", whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-")
    fields["house_no"] = _clean_house(house_raw)

    # Age: digits only (+ sanity check)
    age_img = _normalize_for_ocr(roi("age"))
    age = _clean_digits(_ocr_line(age_img, "eng", whitelist="0123456789"))
    if age.isdigit() and not (1 <= int(age) <= 120):
        # keep it but it's suspicious; you can blank it if you prefer
        pass
    fields["age"] = age

    # Gender: map common Tamil/English
    gender_img = _normalize_for_ocr(roi("gender"))
    gender_raw = _ocr_line(gender_img, text_languages)
    fields["gender"] = _map_gender(gender_raw)

    return fields


def _format_fields_as_markdown(fields: dict[str, str]) -> str:
    # EPIC validation note (optional)
    epic = fields.get("epic_no", "")
    epic_ok = bool(re.fullmatch(r"[A-Z]{3}\d+", epic))
    epic_note = "" if epic_ok or not epic else "  *(check EPIC)*"

    return (
        f"- Serial No: **{fields.get('serial_no', '')}**\n"
        f"- EPIC No: **{epic}**{epic_note}\n"
        f"- Name: {fields.get('name', '')}\n"
        f"- Father Name: {fields.get('father_name', '')}\n"
        f"- House No: {fields.get('house_no', '')}\n"
        f"- Age: {fields.get('age', '')}\n"
        f"- Gender: {fields.get('gender', '')}\n"
    )


# ----------------------------- Main OCR ---------------------------------

def _process_image_with_ocr(image_path: Path, languages: str, strip_labels_ratio: float) -> tuple[str, float]:
    """
    Process a single voter crop image:
    - Read with OpenCV
    - Extract fields by ROIs
    - Field-wise OCR
    - Return markdown block
    """
    try:
        start_time = time.time()

        voter_bgr = cv2.imread(str(image_path))
        if voter_bgr is None:
            return f"[Error: unreadable image {image_path.name}]", 0.0

        fields = _extract_voter_fields(voter_bgr, text_languages=languages, strip_labels_ratio=strip_labels_ratio)
        text = _format_fields_as_markdown(fields)

        return text.strip(), time.time() - start_time

    except Exception as e:
        print(f"  [WARNING] Error processing {image_path.name}: {e}")
        return f"[Error processing image: {e}]", 0.0


def _process_folder(folder: Path, languages: str, md_output_dir: Path, strip_labels_ratio: float) -> dict[str, Any]:
    """Process all images in a folder and create a markdown file."""
    folder_name = folder.name
    images_dir = folder / "images"

    print(f"\nProcessing: {folder_name}")

    images = _get_sorted_images(images_dir)
    if not images:
        print(f"  [WARNING] No images found in {images_dir}")
        return {"folder": folder_name, "images_processed": 0, "error": "No images found"}

    print(f"  Found {len(images)} images")

    md_content: list[str] = []
    md_content.append(f"# {folder_name}\n")
    md_content.append(f"Extracted from PDF with {len(images)} crops\n")
    md_content.append("---\n")

    total_time = 0.0

    for idx, image_path in enumerate(images, start=1):
        text, img_time = _process_image_with_ocr(image_path, languages, strip_labels_ratio)
        total_time += img_time
        print(f"  [{idx}/{len(images)}] {image_path.name} [TESSERACT-FIELDS] - {img_time:.2f}s")

        md_content.append(text)
        md_content.append("\n---\n")

    md_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = md_output_dir / f"{folder_name}.md"
    output_path.write_text("\n".join(md_content), encoding="utf-8")

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
    default_md_output = script_dir / "md-files-tesseract"

    parser = argparse.ArgumentParser(description="Process extracted voter-crop images with Tesseract OCR (field-wise).")
    parser.add_argument("--extracted", type=Path, default=default_extracted,
                        help="Directory containing extracted image folders.")
    parser.add_argument("--output", type=Path, default=default_md_output,
                        help="Output directory for markdown files.")
    parser.add_argument("--languages", type=str, default="tam+eng",
                        help="Languages for text fields (e.g., tam+eng).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N folders (0 = all).")
    parser.add_argument("--strip-labels", type=float, default=0.0,
                        help="Strip left label column inside text ROIs (ratio, e.g. 0.18). Use 0 to disable.")

    args = parser.parse_args()

    extracted_dir: Path = args.extracted
    md_output_dir: Path = args.output

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
    print(f"Text languages: {args.languages}")
    print(f"Strip label ratio: {args.strip_labels}")

    print("\nInitializing Tesseract OCR...")
    init_start = time.time()
    _initialize_tesseract(args.languages)
    init_time = time.time() - init_start
    print(f"[TIMING] Initialization took: {init_time:.2f}s")

    results = []
    total_start = time.time()
    for folder in folders:
        result = _process_folder(folder, args.languages, md_output_dir, args.strip_labels)
        results.append(result)
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("SUMMARY (TESSERACT-FIELDS)")
    print("=" * 60)
    total_images = sum(r.get("images_processed", 0) for r in results)
    total_ocr_time = sum(r.get("total_time", 0) for r in results)
    successful = len([r for r in results if r.get("images_processed", 0) > 0])

    print(f"[SUCCESS] Processed {successful}/{len(folders)} folders")
    print(f"[SUCCESS] Total images processed: {total_images}")
    print(f"[SUCCESS] Output directory: {md_output_dir}")
    print("[INFO] Processing mode: Tesseract (CPU) field-wise")
    print(f"[TIMING] Initialization: {init_time:.2f}s")
    print(f"[TIMING] OCR Processing: {total_ocr_time:.2f}s")
    print(f"[TIMING] Total elapsed: {total_elapsed:.2f}s")
    if total_images > 0:
        print(f"[TIMING] Average per image: {total_ocr_time / total_images:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

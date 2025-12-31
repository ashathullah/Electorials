"""
OCR Processor for extracted PDF images using Tesseract.

Uses pytesseract (Tesseract OCR) for performance comparison.
Processes images from extracted PDFs and generates markdown files.

Defaults:
  - Input:    pdf-converter/extracted/*/images/*.png
  - Output:   pdf-converter/md-files-tesseract/<pdf-stem>.md

Examples:
  python pdf-converter/ocr_processor_tesseract.py
  python pdf-converter/ocr_processor_tesseract.py --languages eng+tam
  python pdf-converter/ocr_processor_tesseract.py --limit 2

Note: Requires Tesseract OCR to be installed on the system.
  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  pip install pytesseract pillow
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def _get_extracted_folders(extracted_dir: Path) -> list[Path]:
    """Get all folders in extracted directory that contain images."""
    folders = []
    for folder in sorted(extracted_dir.iterdir()):
        if folder.is_dir():
            images_dir = folder / "images"
            if images_dir.exists() and images_dir.is_dir():
                folders.append(folder)
    return folders


def _get_sorted_images(images_dir: Path) -> list[Path]:
    """Get all images from a directory, sorted by name."""
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    images = [
        img for img in images_dir.iterdir()
        if img.is_file() and img.suffix.lower() in extensions
    ]
    return sorted(images)


def _initialize_tesseract(languages: str) -> None:
    """Initialize Tesseract and check if it's available."""
    if not TESSERACT_AVAILABLE:
        print("[ERROR] pytesseract not installed. Please run: pip install pytesseract pillow")
        raise ImportError("pytesseract is required but not installed")
    
    # Set Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(tesseract_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"[INFO] Using Tesseract from: {tesseract_path}")
    
    # Try to get Tesseract version to verify installation
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract version: {version}")
    except Exception as e:
        print(f"[ERROR] Tesseract not found. Please install Tesseract OCR.")
        print(f"        Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        raise RuntimeError(f"Tesseract not available: {e}")
    
    print(f"[INFO] Using languages: {languages}")
    print("[SUCCESS] Tesseract initialized (CPU mode)")


def _process_image_with_ocr(image_path: Path, languages: str) -> tuple[str, float]:
    """Process a single image with OCR and return extracted text and processing time."""
    try:
        start_time = time.time()
        
        # Open image with PIL
        img = Image.open(image_path)
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img, lang=languages)
        
        processing_time = time.time() - start_time
        
        # Clean up
        img.close()
        
        text = text.strip() if text.strip() else "[No text detected]"
        return text, processing_time
    except Exception as e:
        print(f"  [WARNING] Error processing {image_path.name}: {e}")
        return f"[Error processing image: {e}]", 0.0


def _process_folder(
    folder: Path,
    languages: str,
    md_output_dir: Path
) -> dict[str, Any]:
    """Process all images in a folder and create a markdown file."""
    folder_name = folder.name
    images_dir = folder / "images"
    
    print(f"\nProcessing: {folder_name}")
    
    # Get sorted images
    images = _get_sorted_images(images_dir)
    if not images:
        print(f"  [WARNING] No images found in {images_dir}")
        return {"folder": folder_name, "images_processed": 0, "error": "No images found"}
    
    print(f"  Found {len(images)} images")
    
    # Process each image
    md_content = []
    md_content.append(f"# {folder_name}\n")
    md_content.append(f"Extracted from PDF with {len(images)} pages\n")
    md_content.append("---\n")
    
    total_time = 0.0
    
    for idx, image_path in enumerate(images, start=1):
        # Extract text from image
        text, img_time = _process_image_with_ocr(image_path, languages)
        total_time += img_time
        print(f"  [{idx}/{len(images)}] {image_path.name} [TESSERACT] - {img_time:.2f}s")
        
        # Add page header
        # md_content.append(f"\n## Page {idx}\n")
        # md_content.append(f"*Source: {image_path.name}*\n")
        md_content.append(text)
        md_content.append("\n---\n")
    
    # Write markdown file
    md_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = md_output_dir / f"{folder_name}.md"
    output_path.write_text("\n".join(md_content), encoding="utf-8")
    
    avg_time = total_time / len(images) if images else 0
    print(f"  [SUCCESS] Saved to: {output_path.name}")
    print(f"  [TIMING] Total: {total_time:.2f}s | Avg per image: {avg_time:.2f}s")
    
    return {
        "folder": folder_name,
        "images_processed": len(images),
        "output_path": str(output_path.as_posix()),
        "total_time": total_time,
        "avg_time": avg_time
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_extracted = script_dir / "extracted"
    default_md_output = script_dir / "md-files-tesseract"
    
    parser = argparse.ArgumentParser(
        description="Process extracted PDF images with Tesseract OCR."
    )
    parser.add_argument(
        "--extracted",
        type=Path,
        default=default_extracted,
        help="Directory containing extracted image folders."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_md_output,
        help="Output directory for markdown files."
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="eng+tam",
        help="Languages for OCR in Tesseract format (e.g., eng, eng+tam, eng+hin, tam)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N folders (0 = all)."
    )
    
    args = parser.parse_args()
    
    extracted_dir: Path = args.extracted
    md_output_dir: Path = args.output
    
    if not extracted_dir.exists():
        print(f"[ERROR] Extracted directory not found: {extracted_dir}")
        return 1
    
    # Get all folders to process
    folders = _get_extracted_folders(extracted_dir)
    if args.limit and args.limit > 0:
        folders = folders[:args.limit]
    
    if not folders:
        print(f"[ERROR] No folders with images found in: {extracted_dir}")
        return 2
    
    print(f"Found {len(folders)} folder(s) to process")
    print(f"Languages: {args.languages}")
    
    # Initialize Tesseract
    print("\nInitializing Tesseract OCR...")
    init_start = time.time()
    _initialize_tesseract(args.languages)
    init_time = time.time() - init_start
    print(f"[TIMING] Initialization took: {init_time:.2f}s")
    
    # Process each folder
    results = []
    total_start = time.time()
    for folder in folders:
        result = _process_folder(folder, args.languages, md_output_dir)
        results.append(result)
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (TESSERACT)")
    print("=" * 60)
    total_images = sum(r.get("images_processed", 0) for r in results)
    total_ocr_time = sum(r.get("total_time", 0) for r in results)
    successful = len([r for r in results if r.get("images_processed", 0) > 0])
    print(f"[SUCCESS] Processed {successful}/{len(folders)} folders")
    print(f"[SUCCESS] Total images processed: {total_images}")
    print(f"[SUCCESS] Output directory: {md_output_dir}")
    print(f"[INFO] Processing mode: Tesseract (CPU)")
    print(f"[TIMING] Initialization: {init_time:.2f}s")
    print(f"[TIMING] OCR Processing: {total_ocr_time:.2f}s")
    print(f"[TIMING] Total elapsed: {total_elapsed:.2f}s")
    if total_images > 0:
        print(f"[TIMING] Average per image: {total_ocr_time/total_images:.2f}s")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

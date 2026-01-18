#!/usr/bin/env python3
"""
Extract images from PDFs in the first_pages folder.
This script processes PDFs in a specified district folder and extracts images in parallel.
"""

import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import fitz  # PyMuPDF


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFImageExtractor:
    """Handles extracting images from PDFs in parallel."""
    
    def __init__(self, folder_name: str, output_dir: str = "extracted_images", max_workers: int = 10):
        """
        Initialize the image extractor.
        
        Args:
            folder_name: Name of the district folder inside first_pages/
            output_dir: Directory to save extracted images (default: extracted_images)
            max_workers: Number of parallel workers (default: 10)
        """
        self.script_dir = Path(__file__).parent
        self.first_pages_dir = self.script_dir / "first_pages"
        self.source_folder = self.first_pages_dir / folder_name
        self.output_dir = self.script_dir / output_dir / folder_name
        self.max_workers = max_workers
        
        # Validate source folder exists
        if not self.source_folder.exists():
            raise ValueError(f"Folder not found: {self.source_folder}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source folder: {self.source_folder}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max workers: {self.max_workers}")
    
    def _get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the source folder."""
        pdf_files = list(self.source_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.source_folder.name}")
        return pdf_files
    
    def _extract_images_from_pdf(self, pdf_path: Path) -> Tuple[str, int, bool]:
        """
        Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (pdf_name, image_count, success)
        """
        pdf_name = pdf_path.stem
        image_count = 0
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Iterate through each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get images on the page
                image_list = page.get_images(full=True)
                
                # Extract each image
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # Image reference number
                    
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate output filename
                    # Format: {pdf_name}.{ext}
                    # Note: We keep the first image only since PDFs typically have one image per page
                    if img_index == 0 and page_num == 0:
                        output_filename = f"{pdf_name}.{image_ext}"
                    else:
                        # For any additional images (rare), use indexed format
                        output_filename = f"{pdf_name}_img{image_count + 1}.{image_ext}"
                    output_path = self.output_dir / output_filename
                    
                    # Save image
                    with open(output_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_count += 1
                    logger.debug(f"Extracted: {output_filename}")
            
            doc.close()
            
            if image_count > 0:
                logger.info(f"✓ {pdf_name}: Extracted {image_count} images")
            else:
                logger.warning(f"⚠ {pdf_name}: No images found")
            
            return (pdf_name, image_count, True)
            
        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_name}: {e}")
            return (pdf_name, 0, False)
    
    def extract_all_images(self) -> dict:
        """
        Extract images from all PDFs in parallel.
        
        Returns:
            Dictionary with processing statistics
        """
        pdf_files = self._get_pdf_files()
        
        if not pdf_files:
            logger.warning("No PDF files to process")
            return {
                "total_pdfs": 0,
                "successful": 0,
                "failed": 0,
                "total_images": 0
            }
        
        logger.info("=" * 60)
        logger.info(f"Starting parallel image extraction with {self.max_workers} workers")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        total_images = 0
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self._extract_images_from_pdf, pdf_file): pdf_file
                for pdf_file in pdf_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                processed += 1
                
                try:
                    pdf_name, image_count, success = future.result()
                    
                    if success:
                        successful += 1
                        total_images += image_count
                    else:
                        failed += 1
                    
                    # Log progress
                    logger.info(f"Progress: {processed}/{len(pdf_files)} "
                              f"(✓ {successful}, ✗ {failed}, Images: {total_images})")
                    
                except Exception as e:
                    logger.error(f"Task exception for {pdf_file.name}: {e}")
                    failed += 1
        
        # Final summary
        stats = {
            "total_pdfs": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "total_images": total_images
        }
        
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total PDFs processed: {stats['total_pdfs']}")
        logger.info(f"Successfully processed: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total images extracted: {stats['total_images']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)
        
        return stats


def list_available_folders():
    """List all available district folders in first_pages directory."""
    script_dir = Path(__file__).parent
    first_pages_dir = script_dir / "first_pages"
    
    if not first_pages_dir.exists():
        logger.error(f"first_pages directory not found: {first_pages_dir}")
        return []
    
    folders = [f.name for f in first_pages_dir.iterdir() if f.is_dir()]
    folders.sort()
    
    return folders


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract images from PDFs in the first_pages folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract images from ALL district folders (parallel processing)
  python extract_image.py
  
  # Extract images from a specific district folder
  python extract_image.py --folder "Coimbatore (North)"
  
  # List all available folders
  python extract_image.py --list
  
  # Extract with custom output directory and workers
  python extract_image.py --folder "Harbour" --output "my_images" --workers 20
        """
    )
    
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        help="Name of the district folder inside first_pages/ to process (if not specified, processes all folders)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="extracted_images",
        help="Output directory for extracted images (default: extracted_images)"
    )
    
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )
    
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available district folders"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        folders = list_available_folders()
        print("\nAvailable district folders:")
        print("=" * 60)
        for folder in folders:
            print(f"  - {folder}")
        print("=" * 60)
        print(f"Total: {len(folders)} folders")
        return
    
    # Process all folders if no specific folder is provided
    if not args.folder:
        folders = list_available_folders()
        if not folders:
            logger.error("No folders found in first_pages/")
            return
        
        logger.info(f"No folder specified - processing ALL {len(folders)} folders in parallel")
        logger.info("=" * 60)
        
        total_stats = {
            "total_pdfs": 0,
            "successful": 0,
            "failed": 0,
            "total_images": 0
        }
        
        for i, folder in enumerate(folders, 1):
            logger.info(f"\n[{i}/{len(folders)}] Processing folder: {folder}")
            logger.info("-" * 60)
            
            try:
                extractor = PDFImageExtractor(
                    folder_name=folder,
                    output_dir=args.output,
                    max_workers=args.workers
                )
                
                stats = extractor.extract_all_images()
                
                # Accumulate stats
                total_stats["total_pdfs"] += stats["total_pdfs"]
                total_stats["successful"] += stats["successful"]
                total_stats["failed"] += stats["failed"]
                total_stats["total_images"] += stats["total_images"]
                
            except Exception as e:
                logger.error(f"Failed to process folder {folder}: {e}")
                continue
        
        # Print overall summary
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL SUMMARY - ALL FOLDERS")
        logger.info("=" * 60)
        logger.info(f"Total folders processed: {len(folders)}")
        logger.info(f"Total PDFs processed: {total_stats['total_pdfs']}")
        logger.info(f"Successfully processed: {total_stats['successful']}")
        logger.info(f"Failed: {total_stats['failed']}")
        logger.info(f"Total images extracted: {total_stats['total_images']}")
        logger.info(f"Output directory: {Path(__file__).parent / args.output}")
        logger.info("=" * 60)
        logger.info("Script completed successfully!")
        return
    
    # Run extraction for single folder
    try:
        extractor = PDFImageExtractor(
            folder_name=args.folder,
            output_dir=args.output,
            max_workers=args.workers
        )
        
        extractor.extract_all_images()
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()

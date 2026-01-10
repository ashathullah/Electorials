
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Set

import cv2
import numpy as np

# Add src to path if needed (though usually running from root works)
import sys
sys.path.append(os.getcwd())

from src.config import Config
from src.processors.base import ProcessingContext, BaseProcessor
from src.processors.pdf_extractor import PDFExtractor
from src.processors.image_cropper import ImageCropper
from src.processors.ocr_processor import OCRProcessor
from src.processors.ai_ocr_processor import AIOCRProcessor
from src.models import Voter
from src.logger import get_logger

logger = get_logger("process_separate_voters")

def main():
    parser = argparse.ArgumentParser(description="Process specific missing voters")
    parser.add_argument("--skip-pages", type=int, default=2, help="Number of pages to skip during cropping")
    args = parser.parse_args()
    
    # Paths
    base_dir = Path.cwd()
    missing_pdfs_dir = base_dir / "missing_voters_pdfs"
    json_path = base_dir / "voters_missing_names.json"
    
    # Output dir - separate from main extraction
    extracted_separate_base = base_dir / "extracted_separate"
    extracted_separate_base.mkdir(parents=True, exist_ok=True)
    
    if not json_path.exists():
        logger.error(f"Input JSON not found: {json_path}")
        return

    # Read input JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        missing_data = json.load(f)
        
    logger.info(f"Loaded {len(missing_data)} files to process from {json_path}")
        
    # Process each file
    for entry in missing_data:
        file_name = entry.get("file_name")
        missing_serials = entry.get("missing_name_serial_numbers", [])
        
        if not file_name or not missing_serials:
            continue
            
        pdf_path = missing_pdfs_dir / f"{file_name}.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            continue
            
        logger.info(f"Processing {file_name} for serials: {missing_serials}")
        
        # Setup Context
        # We need a unique directory for this file's extraction
        # This will create extracted_separate/<file_name>/...
        
        config = Config()
        config.extracted_dir = extracted_separate_base
        
        context = ProcessingContext(config=config)
        context.setup_paths_from_pdf(pdf_path)
        
        # Verify paths are correct
        # context.extracted_dir should be extracted_separate_base / file_name
        logger.info(f"Working directory: {context.extracted_dir}")
        
        # 1. Extract PDF
        # Check if already extracted
        images_exist = False
        if context.images_dir and context.images_dir.exists():
            if len(list(context.images_dir.glob("*.png"))) > 0:
                images_exist = True
                
        if not images_exist:
             logger.info("Extracting PDF...")
             extractor = PDFExtractor(context)
             if not extractor.run():
                 logger.error("Extraction failed")
                 continue
        else:
             logger.info("Using existing extracted images")

        # 2. Crop Images
        # We need to configure ImageCropper to skip pages as requested.
        logger.info(f"Cropping images (Skipping first {args.skip_pages} pages)...")
        cropper = ImageCropper(context)
        
        # Monkeypatch skip pages
        cropper._get_skip_pages_count = lambda: args.skip_pages
        
        if not cropper.run():
             logger.error("Cropping failed")
             continue
             
        # 3. Identify Match and Process
        crops_dir = context.crops_dir # extracted_separate/file_name/crops
        if not crops_dir or not crops_dir.exists():
            logger.error(f"Crops directory missing: {crops_dir}")
            continue
            
        # Get processed voters list
        final_voters_list = []
        
        # Prepare processors
        ocr_processor = OCRProcessor(context, use_cuda=True, languages="tam+eng")
        ai_processor = AIOCRProcessor(context)
        
        # Initialize AI client early to fail fast if config missing
        try:
            ai_processor._initialize_client()
        except Exception as e:
            logger.warning(f"AI Processor initialization warning: {e}")
        
        # Set of target serials for lookup
        target_serials = set(str(s) for s in missing_serials)
        
        # Iterate page directories in order to maintain correct serial count
        page_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir() and d.name.startswith("page-")])
        
        total_voters_count = 0
        voters_found = 0
        
        for page_dir in page_dirs:
             # Get crops sorted
            crops = sorted([p for p in page_dir.iterdir() if p.suffix == '.png'])
            
            for crop_path in crops:
                total_voters_count += 1
                current_serial = str(total_voters_count)
                
                if current_serial in target_serials:
                    logger.info(f"Found target serial {current_serial} at {crop_path.name}")
                    voters_found += 1
                    
                    # Process this crop
                    
                    # Read Image
                    img = cv2.imdecode(np.fromfile(str(crop_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        logger.error(f"Failed to read image {crop_path}")
                        continue
                    
                    # 1. Run OCR
                    if not ocr_processor._ocr_initialized:
                        ocr_processor._initialize_ocr()
                        
                    ocr_results = ocr_processor.ocr.predict([img])
                    if not ocr_results:
                        logger.warning(f"OCR returned no results for {current_serial}")
                        continue
                        
                    # Parse OCR Result
                    ocr_result_obj = ocr_processor._process_voter_result_from_ocr(img, ocr_results[0], crop_path.name)
                    
                    # Check if Name or Relation is missing/invalid
                    missing_name = not ocr_result_obj.name or len(ocr_result_obj.name.strip()) < 3
                    missing_relation = not ocr_result_obj.relation_name or len(ocr_result_obj.relation_name.strip()) < 3
                    
                    # 2. Retry with AI if needed
                    if (missing_name or missing_relation):
                        logger.info(f"Missing details for {current_serial} (Name: {not missing_name}, Rel: {not missing_relation}), retrying with AI...")
                        
                        try:
                            # Use crop path for AI
                            ai_results = ai_processor._call_ai_api_single(crop_path)
                            
                            if ai_results:
                                ai_data = ai_results[0]
                                if missing_name and ai_data.get('name'):
                                    ocr_result_obj.name = ai_data.get('name')
                                    logger.info(f"AI Recovered Name: {ocr_result_obj.name}")
                                    ocr_result_obj.error = None # Clear error if fixed
                                    
                                if missing_relation and ai_data.get('relation_name'):
                                    ocr_result_obj.relation_name = ai_data.get('relation_name')
                                    logger.info(f"AI Recovered Relation: {ocr_result_obj.relation_name}")
                                    ocr_result_obj.error = None # Clear error if fixed

                                # Update others if AI has them
                                for f in ['relation_type', 'house_no', 'age', 'gender', 'epic_no']:
                                    if ai_data.get(f):
                                        setattr(ocr_result_obj, f, ai_data[f])
                            else:
                                logger.warning(f"AI returned no results for {current_serial}")
                        except Exception as e:
                            logger.error(f"AI processing failed: {e}")
                    
                    # Construct Voter object
                    voter = ocr_result_obj.to_voter(sequence_in_document=int(current_serial))
                    # Force serial no
                    voter.serial_no = current_serial
                    
                    final_voters_list.append(voter.to_dict())
                    
        # Save results to JSON
        # Create output file path: extracted_separate/file_name/file_name.json
        results_dir = extracted_separate_base / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = results_dir / f"{file_name}.json"
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_voters_list, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Completed {file_name}: Processed {voters_found}/{len(missing_serials)} targets. Saved to {output_file_path}")

if __name__ == "__main__":
    main()

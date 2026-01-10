import sys
import os
import json
import time
import shutil
import logging
import traceback
from pathlib import Path
import re
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import fitz
from types import SimpleNamespace
from src.config import Config, S3Config
from src.processors.metadata_extractor import extract_metadata
from src.utils.s3_utils import download_from_s3
from src.models.metadata import DocumentMetadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MissingDataProcessor")

# S3 Mappings
S3_MAPPINGS = {
    "Tamil Nadu-(S22)_Sriperumbudur": "2026/1/S22/pdfs/Tamil Nadu/Sriperumbudur/",
    "Tamil Nadu-(S22)_Manachanallur": "2026/1/S22/pdfs/Tamil Nadu/Manachanallur/",
    "Tamil Nadu-(S22)_Tiruppur (South)": "2026/1/S22/pdfs/Tamil Nadu/Tiruppur (South)/",
    "Tamil Nadu-(S22)_Sivaganga": "2026/1/S22/pdfs/Tamil Nadu/Sivaganga/",
    "Tamil Nadu-(S22)_Coimbatore (North)": "2026/1/S22/pdfs/Tamil Nadu/Coimbatore (South)/",
    "Tamil Nadu-(S22)_Coimbatore (South)": "2026/1/S22/pdfs/Tamil Nadu/Coimbatore (South)/",
}
BUCKET = "264676382451-eci-download"

CORRECTION_DIR = Path(__file__).parent
MISSING_DATA_FILE = CORRECTION_DIR / "missing_data.txt"
STATUS_FILE = CORRECTION_DIR / "missing_data_status.json"
METADATA_DIR = CORRECTION_DIR / "missing_metadata"

def load_status():
    """Load processing status from JSON file."""
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_status(status):
    """Save processing status to JSON file."""
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')

def parse_missing_data_file():
    """Parse missing_data.txt to extract document IDs."""
    if not MISSING_DATA_FILE.exists():
        logger.error(f"Missing data file not found: {MISSING_DATA_FILE}")
        return []
    
    doc_ids = []
    lines = MISSING_DATA_FILE.read_text(encoding='utf-8').strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove _metadata.json suffix if present
        if line.endswith('_metadata.json'):
            doc_id = line[:-14]  # Remove '_metadata.json'
        elif line.endswith('.json'):
            doc_id = line[:-5]  # Remove '.json'
        else:
            doc_id = line
        
        doc_ids.append(doc_id)
    
    logger.info(f"Loaded {len(doc_ids)} document IDs from {MISSING_DATA_FILE}")
    return doc_ids

def get_s3_url(document_id):
    """Construct S3 URL based on document ID prefix mapping."""
    for prefix_key, s3_prefix in S3_MAPPINGS.items():
        if document_id.startswith(prefix_key):
            # Apply filename correction for AC codes
            filename = re.sub(r'-\(AC(\d+)_', r'-(AC\1)_', document_id)
            return f"s3://{BUCKET}/{s3_prefix}{filename}.pdf"
    return None

def initialize_status(doc_ids):
    """Initialize status file with document IDs from missing_data.txt."""
    status_db = load_status()
    updates_made = False
    
    for doc_id in doc_ids:
        s3_url = get_s3_url(doc_id)
        
        if doc_id not in status_db:
            status_db[doc_id] = {
                "document_id": doc_id,
                "s3_url": s3_url,
                "progress_status": "pending",
                "error": None if s3_url else "No S3 mapping found"
            }
            updates_made = True
    
    if updates_made:
        save_status(status_db)
        logger.info(f"Status file initialized with {len(doc_ids)} documents")
    
    return status_db

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and process PDFs from missing_data.txt")
    parser.add_argument("--retry-failed", action="store_true", help="Retry documents that previously failed")
    parser.add_argument("--force-all", action="store_true", help="Force reprocess all documents, ignoring status")
    args = parser.parse_args()

    config = Config()
    
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = CORRECTION_DIR / "temp_missing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse missing_data.txt
    doc_ids = parse_missing_data_file()
    if not doc_ids:
        logger.error("No document IDs found in missing_data.txt")
        return
    
    # 2. Initialize status
    status_db = initialize_status(doc_ids)
    
    # 3. Reset 'processing' status to 'pending' (in case of crash)
    for k, v in status_db.items():
        if v.get("progress_status") == "processing":
            v["progress_status"] = "pending"
            status_db[k] = v
    
    # 4. Filter items to process
    pending_items = []
    skipped_count = 0
    
    for doc_id, v in status_db.items():
        if doc_id not in doc_ids:
            continue  # Only process items from current missing_data.txt
        
        if not v.get("s3_url"):
            skipped_count += 1
            continue
        
        status = v.get("progress_status")
        
        should_process = False
        if args.force_all:
            should_process = True
        elif status == "pending":
            should_process = True
        elif status == "failed" and args.retry_failed:
            should_process = True
        elif status == "completed":
            should_process = False
        else:
            should_process = False
        
        if should_process:
            pending_items.append(doc_id)
        else:
            skipped_count += 1
    
    if not pending_items:
        logger.info(f"No documents to process. (Skipped {skipped_count}). Use --retry-failed or --force-all to override.")
        return
    
    logger.info(f"Starting processing for {len(pending_items)} documents (Skipped {skipped_count})...")
    
    # 5. Process each document with progress bar
    with tqdm(total=len(pending_items), desc="Processing Documents", unit="doc") as pbar:
        for i, doc_id in enumerate(pending_items):
            item_status = status_db[doc_id]
            s3_url = item_status["s3_url"]
            
            pbar.set_description(f"Processing {doc_id[:50]}...")
            logger.info(f"\n[{i+1}/{len(pending_items)}] Processing {doc_id}")
        
            # Update status to processing
            status_db[doc_id]["progress_status"] = "processing"
            save_status(status_db)
            
            start_time = time.time()
            pdf_path = None
            
            try:
                # Check for existing metadata JSON
                meta_json_path = METADATA_DIR / f"{doc_id}_metadata.json"
                
                # Check for variations of the filename
                if not meta_json_path.exists():
                    corrected_id = re.sub(r'-\(AC(\d+)_', r'-(AC\1)_', doc_id)
                    corrected_path = METADATA_DIR / f"{corrected_id}_metadata.json"
                    if corrected_path.exists():
                        logger.info(f"Found existing metadata: {corrected_path.name}")
                        meta_json_path = corrected_path
                
                meta_result = None
                extraction_result = None
                
                # Try to load existing metadata
                if meta_json_path.exists():
                    try:
                        logger.info(f"Found existing metadata at {meta_json_path}. Skipping Download & AI.")
                        data = json.loads(meta_json_path.read_text(encoding='utf-8'))
                        ai_stats = data.get("ai_metadata", {})
                        meta_result = DocumentMetadata.from_ai_response(
                            data,
                            ai_provider=ai_stats.get("provider", ""),
                            ai_model=ai_stats.get("model", ""),
                            input_tokens=ai_stats.get("input_tokens", 0),
                            output_tokens=ai_stats.get("output_tokens", 0),
                            cost_usd=ai_stats.get("cost_usd"),
                            extraction_time_sec=ai_stats.get("extraction_time_sec", 0.0)
                        )
                        meta_result.document_id = doc_id
                        
                        filename = s3_url.split('/')[-1]
                        pdf_name = filename[:-4] if filename.endswith('.pdf') else filename
                        extraction_result = SimpleNamespace(pdf_name=pdf_name)
                    except Exception as ex:
                        logger.warning(f"Failed to load existing metadata, will reprocess: {ex}")
                        meta_result = None
                
                # Download and process if no existing metadata
                if not meta_result:
                    # Download PDF from S3
                    logger.info(f"Downloading from {s3_url}")
                    pdf_path = download_from_s3(s3_url, config.s3, download_dir=temp_dir)
                    
                    # Extract first page only
                    logger.info("Extracting PDF first page...")
                    
                    pdf_name = pdf_path.stem
                    extracted_base = temp_dir / "extracted" / doc_id / pdf_name
                    images_dir = extracted_base / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Extract first page as PNG
                        doc = fitz.open(pdf_path)
                        page = doc.load_page(0)
                        pix = page.get_pixmap(dpi=200)
                        pix.save(str(images_dir / "page-001.png"))
                        doc.close()
                    except Exception as e:
                        raise Exception(f"Failed to extract first page: {e}")
                    
                    extracted_path = extracted_base
                    extraction_result = SimpleNamespace(pdf_name=pdf_name)
                    
                    if not (extracted_path / "images").exists():
                        raise Exception("Images not extracted correctly")
                    
                    # Run metadata extraction (AI) - Simple extraction without detailed_elector_summary
                    logger.info("Running Metadata Extraction (AI - Basic Mode)...")
                    meta_result = extract_metadata(extracted_path, force=True)
                    
                    if not meta_result:
                        raise Exception("Metadata extraction returned None")
                    
                    if not meta_result.document_id:
                        meta_result.document_id = doc_id
                    
                    # Save metadata JSON
                    meta_dict = meta_result.to_dict()
                    meta_json_path.write_text(json.dumps(meta_dict, indent=2, ensure_ascii=False), encoding='utf-8')
                    logger.info(f"Saved metadata to {meta_json_path}")
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Update status
                status_db[doc_id]["progress_status"] = "completed"
                status_db[doc_id]["error"] = None
                status_db[doc_id]["time_taken"] = f"{duration:.2f}s"
                status_db[doc_id]["ai_token_usage"] = {
                    "input": meta_result.ai_input_tokens,
                    "output": meta_result.ai_output_tokens,
                    "cost": meta_result.ai_cost_usd
                }
                save_status(status_db)
                pbar.update(1)
                logger.info(f"✓ Successfully completed {doc_id}")
                    
            except Exception as e:
                logger.error(f"✗ Error processing {doc_id}: {e}")
                logger.error(traceback.format_exc())
                
                status_db[doc_id]["progress_status"] = "failed"
                status_db[doc_id]["error"] = str(e)
                save_status(status_db)
                pbar.update(1)
            
            finally:
                # Cleanup
                if pdf_path and pdf_path.exists():
                    try:
                        pdf_path.unlink()
                    except Exception:
                        pass
                
                extract_base = temp_dir / "extracted" / doc_id
                if extract_base.exists():
                    try:
                        shutil.rmtree(extract_base)
                    except Exception:
                        pass
    
    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    
    logger.info("\n" + "="*50)
    logger.info("Processing complete!")
    
    # Summary
    completed = sum(1 for v in status_db.values() if v.get("progress_status") == "completed")
    failed = sum(1 for v in status_db.values() if v.get("progress_status") == "failed")
    pending = sum(1 for v in status_db.values() if v.get("progress_status") == "pending")
    
    logger.info(f"Summary: Completed={completed}, Failed={failed}, Pending={pending}")
    logger.info(f"Metadata saved to: {METADATA_DIR}")
    logger.info(f"Status file: {STATUS_FILE}")
    logger.info("="*50)

if __name__ == "__main__":
    main()

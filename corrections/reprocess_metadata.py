import sys
import os
import json
import time
import shutil
import logging
import traceback
from pathlib import Path

import re

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import fitz
from types import SimpleNamespace
from src.config import Config, S3Config
# from src.processors.pdf_extractor import extract_pdf
from src.processors.metadata_extractor import extract_metadata
from src.persistence.postgres import PostgresRepository
from src.models import ProcessedDocument
from src.utils.s3_utils import download_from_s3
from src.models.metadata import DocumentMetadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetadataReprocess")

# Mappings
# State-(Code)_Constituency -> S3 Prefix path relative to bucket
S3_MAPPINGS = {
    "Tamil Nadu-(S22)_Sriperumbudur": "2026/1/S22/pdfs/Tamil Nadu/Sriperumbudur/",
    "Tamil Nadu-(S22)_Manachanallur": "2026/1/S22/pdfs/Tamil Nadu/Manachanallur/",
    "Tamil Nadu-(S22)_Tiruppur (South)": "2026/1/S22/pdfs/Tamil Nadu/Tiruppur (South)/",
    "Tamil Nadu-(S22)_Sivaganga": "2026/1/S22/pdfs/Tamil Nadu/Sivaganga/",
    "Tamil Nadu-(S22)_Coimbatore (North)": "2026/1/S22/pdfs/Tamil Nadu/Coimbatore (South)/", # Mapped as per previous request
    "Tamil Nadu-(S22)_Coimbatore (South)": "2026/1/S22/pdfs/Tamil Nadu/Coimbatore (South)/",
}
BUCKET = "264676382451-eci-download"

CORRECTION_DIR = Path(__file__).parent
STATUS_FILE = CORRECTION_DIR / "correction_status.json"
METADATA_DIR = CORRECTION_DIR / "metadata"

def load_status():
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_status(status):
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')

def get_s3_url(document_id):
    """Construct S3 URL based on document ID prefix mapping."""
    for prefix_key, s3_prefix in S3_MAPPINGS.items():
        if document_id.startswith(prefix_key):
            # Patch for Manachanallur: DB has `...-(AC144_...`
            # S3 actual files are `...-(AC144)_...` (Closing paren needed)
            filename = document_id
            # Generic patch for missing closing parenthesis in AC part of ID
            # e.g. ...-(AC144_... -> ...-(AC144)_...
            filename = re.sub(r'-\(AC(\d+)_', r'-(AC\1)_', filename)
            
            return f"s3://{BUCKET}/{s3_prefix}{filename}.pdf"
    return None

def sync_db_to_status(repo):
    """Query DB for missing metadata and initialize status file."""
    logger.info("checking database for documents needing update...")
    conn = repo._get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT document_id 
            FROM metadata 
            WHERE detailed_elector_summary->'serial_number_range'->'end' IS NULL;
        """)
        results = cur.fetchall()
        missing_ids = [r[0] for r in results]
    
    logger.info(f"Found {len(missing_ids)} documents in database needing update.")
    
    status_db = load_status()
    updates_made = False
    
    for doc_id in missing_ids:
        # Always update S3 URL to ensure latest logic is applied
        current_s3_url = get_s3_url(doc_id)
        
        if doc_id not in status_db:
            status_db[doc_id] = {
                "document_id": doc_id,
                "s3_folder_link": current_s3_url,
                "progress_status": "pending",
                "updated_to_database": False,
                "error": None if current_s3_url else "No S3 mapping found"
            }
            updates_made = True
        else:
            # Update URL if it changed
            if status_db[doc_id].get("s3_folder_link") != current_s3_url:
                status_db[doc_id]["s3_folder_link"] = current_s3_url
                updates_made = True
                # Optional: Reset status if URL changed? Maybe not, allow user to control via flags.
    
    if updates_made:
        save_status(status_db)
        logger.info("Status file updated with new pending documents.")
    else:
        logger.info("Status file is up to date with database.")
        
    return status_db

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reprocess metadata for missing fields.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry documents that previously failed.")
    parser.add_argument("--force-all", action="store_true", help="Force reprocess all documents found in DB query, ignoring status.")
    args = parser.parse_args()

    config = Config()
    repo = PostgresRepository(config.db)
    
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = CORRECTION_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sync DB to Status File
    status_db = sync_db_to_status(repo)
    
    # 2. Process Pending Items
    # Determine what to process based on status and flags
    # Reset "processing" to "pending" in case of previous crash
    for k, v in status_db.items():
        if v.get("progress_status") == "processing":
            v["progress_status"] = "pending"
            status_db[k] = v
            
    # Filter items
    pending_items = []
    skipped_count = 0
    
    for doc_id, v in status_db.items():
        if not v.get("s3_folder_link"):
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
        logger.info(f"No documents to process. (Skipped {skipped_count} processed/failed documents). Use --retry-failed or --force-all to override.")
        if skipped_count > 0 and not (args.retry_failed or args.force_all):
             logger.info("Tip: Some documents were skipped because they are failed/completed. check correction_status.json")
        return
    
    logger.info(f"Starting processing for {len(pending_items)} documents (Skipped {skipped_count})...")
    
    for i, doc_id in enumerate(pending_items):
        item_status = status_db[doc_id]
        s3_url = item_status["s3_folder_link"]
        
        logger.info(f"[{i+1}/{len(pending_items)}] Processing {doc_id}")
        
        # Update status to processing
        status_db[doc_id]["progress_status"] = "processing"
        save_status(status_db)
        
        start_time = time.time()
        pdf_path = None
        
        try:
            # Check for existing metadata (Skip AI if found)
            meta_json_path = METADATA_DIR / f"{doc_id}.json"
            
            # If explicit file doesn't exist, check for corrected filename (missing paren fix)
            if not meta_json_path.exists():
                corrected_id = re.sub(r'-\(AC(\d+)_', r'-(AC\1)_', doc_id)
                corrected_path = METADATA_DIR / f"{corrected_id}.json"
                if corrected_path.exists():
                    logger.info(f"Found existing metadata with corrected name: {corrected_path.name}")
                    meta_json_path = corrected_path

            meta_result = None
            extraction_result = None

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

            if not meta_result:
                # Download
                logger.info(f"Downloading from {s3_url}")
                pdf_path = download_from_s3(s3_url, config.s3, download_dir=temp_dir)
                
                # Extract PDF - First Page Only
                logger.info("Extracting PDF first page only...")
                
                pdf_name = pdf_path.stem
                extracted_base = temp_dir / "extracted" / doc_id / pdf_name
                images_dir = extracted_base / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Open PDF and extract only the first page
                    doc = fitz.open(pdf_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=200)
                    pix.save(str(images_dir / "page-001.png"))
                    doc.close()
                except Exception as e:
                    raise Exception(f"Failed to extract first page: {e}")
    
                extracted_path = extracted_base
                # create mock result for compatibility
                extraction_result = SimpleNamespace(pdf_name=pdf_name)
                
                if not (extracted_path / "images").exists():
                    raise Exception("Images not extracted correctly")
    
                # Run Metadata Extraction
                logger.info("Running Metadata Extraction (AI)...")
                meta_result = extract_metadata(extracted_path, force=True)
                
                if not meta_result:
                    raise Exception("Metadata extraction returned None")
                    
                if not meta_result.document_id:
                    meta_result.document_id = doc_id
                
                # Save Metadata JSON to file
                meta_dict = meta_result.to_dict()
                meta_json_path.write_text(json.dumps(meta_dict, indent=2, ensure_ascii=False), encoding='utf-8')
                logger.info(f"Saved metadata to {meta_json_path}")

            # Update Database
            logger.info("Updating Database...")
            doc = ProcessedDocument(
                id=doc_id,
                pdf_name=extraction_result.pdf_name,
                pdf_path=str(pdf_path),
                metadata=meta_result,
                pages=[] 
            )
            
            success = repo.save_document(doc)
            
            if success:
                end_time = time.time()
                duration = end_time - start_time
                
                # Update status
                status_db[doc_id]["progress_status"] = "completed"
                status_db[doc_id]["updated_to_database"] = True
                status_db[doc_id]["error"] = None
                status_db[doc_id]["time_taken"] = f"{duration:.2f}s"
                status_db[doc_id]["ai_token_usage"] = {
                    "input": meta_result.ai_input_tokens,
                    "output": meta_result.ai_output_tokens,
                    "cost": meta_result.ai_cost_usd
                }
                save_status(status_db)
                logger.info(f"Successfully completed {doc_id}")
            else:
                raise Exception("Database save returned False")
                
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            logger.error(traceback.format_exc())
            
            status_db[doc_id]["progress_status"] = "failed"
            status_db[doc_id]["updated_to_database"] = False
            status_db[doc_id]["error"] = str(e)
            save_status(status_db)
            
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

    # Cleanup temp dir
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
        
    logger.info("All processing complete.")

if __name__ == "__main__":
    main()

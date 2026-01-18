#!/usr/bin/env python3
"""
Download PDFs from S3, extract first page, and track progress.
This script downloads PDFs from S3 for specified districts, extracts only the first page,
deletes the original PDF, and maintains a JSON status file for resumability.
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter

# Load environment variables
load_dotenv()

# Configuration
S3_BUCKET = "264676382451-eci-download"
S3_BASE_PATH = "2026/1/S22/pdfs/Tamil Nadu"
DISTRICTS_FILE = "districts.txt"
STATUS_FILE = "download_status.json"
OUTPUT_DIR = "first_pages"
MAX_WORKERS = 10  # Number of parallel downloads

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFDownloader:
    """Handles downloading PDFs from S3 and extracting first pages."""
    
    def __init__(self):
        """Initialize S3 client and load status."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-south-1')
        )
        
        self.script_dir = Path(__file__).parent
        self.districts_file = self.script_dir / DISTRICTS_FILE
        self.status_file = self.script_dir / STATUS_FILE
        self.output_dir = self.script_dir / OUTPUT_DIR
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load or initialize status
        self.status = self._load_status()
        
    def _load_status(self) -> Dict:
        """Load download status from JSON file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    logger.info(f"Loaded status: {len(status.get('completed', []))} completed, "
                               f"{len(status.get('failed', []))} failed")
                    return status
            except Exception as e:
                logger.warning(f"Failed to load status file: {e}")
        
        return {
            'completed': [],
            'failed': [],
            'in_progress': []
        }
    
    def _save_status(self):
        """Save download status to JSON file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, indent=2, fp=f)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def _load_districts(self) -> List[str]:
        """Load districts from file."""
        if not self.districts_file.exists():
            logger.error(f"Districts file not found: {self.districts_file}")
            return []
        
        with open(self.districts_file, 'r') as f:
            districts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(districts)} districts")
        return districts
    
    def _list_pdfs_in_district(self, district: str) -> List[str]:
        """List all PDF files in a district folder on S3."""
        prefix = f"{S3_BASE_PATH}/{district}/"
        pdf_files = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.pdf'):
                            pdf_files.append(key)
            
            logger.info(f"Found {len(pdf_files)} PDFs in {district}")
            
        except ClientError as e:
            logger.error(f"Failed to list PDFs in {district}: {e}")
        
        return pdf_files
    
    def _extract_first_page(self, pdf_path: Path, output_path: Path) -> bool:
        """Extract first page from PDF and save it."""
        try:
            reader = PdfReader(pdf_path)
            
            if len(reader.pages) == 0:
                logger.warning(f"PDF has no pages: {pdf_path}")
                return False
            
            writer = PdfWriter()
            writer.add_page(reader.pages[0])
            
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            logger.debug(f"Extracted first page: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract first page from {pdf_path}: {e}")
            return False
    
    def _process_pdf(self, s3_key: str) -> bool:
        """Download PDF, extract first page, and delete original."""
        # Check if already processed
        if s3_key in self.status['completed']:
            logger.debug(f"Skipping already completed: {s3_key}")
            return True
        
        # Create temporary download path
        filename = Path(s3_key).name
        district = s3_key.split('/')[-2]
        
        # Create district subfolder in output
        district_output_dir = self.output_dir / district
        district_output_dir.mkdir(exist_ok=True)
        
        temp_pdf = self.script_dir / f"temp_{filename}"
        output_pdf = district_output_dir / filename
        
        try:
            # Mark as in progress
            if s3_key not in self.status['in_progress']:
                self.status['in_progress'].append(s3_key)
                self._save_status()
            
            # Download PDF
            logger.info(f"Downloading: {s3_key}")
            self.s3_client.download_file(S3_BUCKET, s3_key, str(temp_pdf))
            
            # Extract first page
            if self._extract_first_page(temp_pdf, output_pdf):
                # Delete temporary PDF
                temp_pdf.unlink()
                
                # Update status
                self.status['in_progress'].remove(s3_key)
                if s3_key not in self.status['completed']:
                    self.status['completed'].append(s3_key)
                if s3_key in self.status['failed']:
                    self.status['failed'].remove(s3_key)
                self._save_status()
                
                logger.info(f"✓ Completed: {filename}")
                return True
            else:
                raise Exception("Failed to extract first page")
                
        except Exception as e:
            logger.error(f"✗ Failed to process {s3_key}: {e}")
            
            # Clean up temp file if it exists
            if temp_pdf.exists():
                temp_pdf.unlink()
            
            # Update status
            if s3_key in self.status['in_progress']:
                self.status['in_progress'].remove(s3_key)
            if s3_key not in self.status['failed']:
                self.status['failed'].append(s3_key)
            self._save_status()
            
            return False
    
    def process_all_districts(self):
        """Process all PDFs from all districts in parallel."""
        districts = self._load_districts()
        
        if not districts:
            logger.error("No districts to process")
            return
        
        # Collect all PDF keys from all districts
        all_pdf_keys = []
        for district in districts:
            logger.info(f"Scanning district: {district}")
            pdf_keys = self._list_pdfs_in_district(district)
            all_pdf_keys.extend(pdf_keys)
        
        logger.info(f"Total PDFs to process: {len(all_pdf_keys)}")
        
        # Filter out already completed
        pdf_keys_to_process = [
            key for key in all_pdf_keys 
            if key not in self.status['completed']
        ]
        
        logger.info(f"PDFs remaining to process: {len(pdf_keys_to_process)}")
        
        if not pdf_keys_to_process:
            logger.info("All PDFs already processed!")
            return
        
        # Process PDFs in parallel
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(self._process_pdf, key): key 
                for key in pdf_keys_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Task exception for {key}: {e}")
                    failed_count += 1
                
                # Log progress
                total_processed = completed_count + failed_count
                logger.info(f"Progress: {total_processed}/{len(pdf_keys_to_process)} "
                           f"(✓ {completed_count}, ✗ {failed_count})")
        
        # Final summary
        logger.info("="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total PDFs found: {len(all_pdf_keys)}")
        logger.info(f"Previously completed: {len(all_pdf_keys) - len(pdf_keys_to_process)}")
        logger.info(f"Newly completed: {completed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Total completed: {len(self.status['completed'])}")
        logger.info(f"Total failed: {len(self.status['failed'])}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60)


def main():
    """Main entry point."""
    logger.info("Starting PDF download and extraction script")
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"S3 Base Path: {S3_BASE_PATH}")
    logger.info(f"Parallel workers: {MAX_WORKERS}")
    logger.info("="*60)
    
    downloader = PDFDownloader()
    downloader.process_all_districts()
    
    logger.info("Script completed!")


if __name__ == "__main__":
    main()

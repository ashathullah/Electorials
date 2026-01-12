#!/usr/bin/env python3
"""
Script to download PDFs for voters with missing names.
Tracks download status in the JSON file and supports resume functionality.
Uses AWS credentials from .env file for authenticated S3 downloads.
"""

import json
import os
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse, unquote_plus
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
JSON_FILE = "voters_missing_names.json"
OUTPUT_DIR = "missing_voters_pdfs"
CHUNK_SIZE = 8192  # 8KB chunks for download
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Status constants
STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def load_json_data(file_path):
    """Load the JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        exit(1)


def save_json_data(file_path, data):
    """Save the JSON data to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON: {e}")


def parse_s3_url(url):
    """
    Parse S3 URL to extract bucket and key.
    
    Supports formats:
    - https://bucket-name.s3.region.amazonaws.com/path/to/file
    - https://s3.region.amazonaws.com/bucket-name/path/to/file
    
    Returns:
        tuple: (bucket_name, object_key) or (None, None) if invalid
    """
    try:
        parsed = urlparse(url)
        
        # Format: bucket-name.s3.region.amazonaws.com
        if '.s3.' in parsed.netloc and '.amazonaws.com' in parsed.netloc:
            bucket = parsed.netloc.split('.s3.')[0]
            # Remove leading slash and decode URL-encoded characters (+ becomes space)
            key = unquote_plus(parsed.path.lstrip('/'))
            return bucket, key
        
        # Format: s3.region.amazonaws.com/bucket-name (less common)
        elif parsed.netloc.startswith('s3.') and parsed.path:
            parts = parsed.path.lstrip('/').split('/', 1)
            if len(parts) == 2:
                bucket, key = parts
                return bucket, unquote_plus(key)
        
        return None, None
    except Exception as e:
        return None, None


def download_pdf(url, output_path, retries=MAX_RETRIES, s3_client=None):
    """
    Download a PDF file from S3 using boto3.
    
    Args:
        url: S3 URL to download from
        output_path: Path where to save the file
        retries: Number of retry attempts
        s3_client: boto3 S3 client (optional, will create if not provided)
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    # Parse S3 URL to get bucket and key
    bucket, key = parse_s3_url(url)
    
    if not bucket or not key:
        return False, f"Invalid S3 URL format: {url}"
    
    # Create S3 client if not provided
    if s3_client is None:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'ap-southeast-2')
            )
        except NoCredentialsError:
            return False, "AWS credentials not found. Please check your .env file."
    
    for attempt in range(retries):
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download file from S3
            s3_client.download_file(bucket, key, output_path)
            
            # Verify file was written
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True, None
            else:
                error_msg = "File was not written properly"
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = f"S3 Error ({error_code}): {str(e)} (attempt {attempt + 1}/{retries})"
            
            # Don't retry on certain errors
            if error_code in ['NoSuchKey', 'NoSuchBucket', '404']:
                return False, error_msg
                
        except NoCredentialsError:
            error_msg = "AWS credentials not found or invalid"
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)} (attempt {attempt + 1}/{retries})"
        
        # If not last attempt, wait before retrying
        if attempt < retries - 1:
            time.sleep(RETRY_DELAY)
    
    # All retries failed
    return False, error_msg


def get_download_list(data, retry_failed=False, output_dir=OUTPUT_DIR):
    """
    Get list of items to download based on their status.
    Also updates status based on existing files on disk.
    
    Args:
        data: JSON data
        retry_failed: If True, retry failed downloads; otherwise skip them
        output_dir: Output directory for checking file existence
    
    Returns:
        list: Items to download
    """
    download_list = []
    updated_count = 0
    
    print("Checking existing files in output directory...")
    
    for item in data:
        # Initialize download_status if not present
        if 'download_status' not in item:
            item['download_status'] = STATUS_PENDING
            
        # Check actual file existence
        file_name_base = item['file_name']
        pdf_filename = file_name_base + '.pdf'
        file_path = os.path.join(output_dir, pdf_filename)
        
        is_file_valid = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        
        # Logic to update status based on file existence
        if is_file_valid:
            if item['download_status'] != STATUS_SUCCESS:
                item['download_status'] = STATUS_SUCCESS
                item['download_error'] = None
                updated_count += 1
        elif item['download_status'] == STATUS_SUCCESS:
            # File missing but marked success - reset to pending
            item['download_status'] = STATUS_PENDING
            updated_count += 1
        
        status = item['download_status']
        
        # Determine if we should download this item
        should_download = False
        
        if status == STATUS_PENDING:
            should_download = True
        elif status == STATUS_FAILED and retry_failed:
            should_download = True
        
        if should_download:
            download_list.append(item)
            
    if updated_count > 0:
        print(f"Updated status for {updated_count} items based on existing files.")
    
    return download_list


def main():
    parser = argparse.ArgumentParser(
        description='Download PDFs for voters with missing names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all pending PDFs
  python download_missing_voters_pdfs.py
  
  # Retry failed downloads
  python download_missing_voters_pdfs.py --retry-failed
  
  # Use custom paths
  python download_missing_voters_pdfs.py --json custom.json --output-dir custom_pdfs
        """
    )
    
    parser.add_argument(
        '--json',
        default=JSON_FILE,
        help=f'Path to JSON file (default: {JSON_FILE})'
    )
    
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory for PDFs (default: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry failed downloads'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=MAX_RETRIES,
        help=f'Maximum retry attempts per file (default: {MAX_RETRIES})'
    )
    
    args = parser.parse_args()
    
    # Use local variables instead of globals
    output_dir = args.output_dir
    max_retries = args.max_retries
    
    print(f"Loading data from {args.json}...")
    data = load_json_data(args.json)
    
    print(f"Total entries in JSON: {len(data)}")
    
    # Get list of items to download and update status based on existing files
    download_list = get_download_list(data, retry_failed=args.retry_failed, output_dir=output_dir)
    
    # Save the updated status immediately to persist "found" files
    save_json_data(args.json, data)
    
    if not download_list:
        print("\n✓ No files to download. All done!")
        print("\nStatus summary:")
        status_counts = {}
        for item in data:
            status = item.get('download_status', STATUS_PENDING)
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
        return
    
    print(f"Files to download: {len(download_list)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize S3 client with credentials from .env
    print("\nInitializing AWS S3 client...")
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-southeast-2')
        )
        # Test connection by listing buckets (or any simple operation)
        s3_client.list_buckets()
        print("✓ AWS S3 connection successful!")
    except NoCredentialsError:
        print("✗ Error: AWS credentials not found!")
        print("Please ensure your .env file contains:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION")
        return
    except ClientError as e:
        print(f"✗ Error connecting to AWS S3: {e}")
        print("Please check your AWS credentials and permissions.")
        return
    except Exception as e:
        print(f"✗ Unexpected error initializing S3 client: {e}")
        return
    
    # Initialize counters
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Download files with progress bar
    print(f"\nDownloading PDFs to {output_dir}/\n")
    
    with tqdm(total=len(download_list), desc="Overall Progress", unit="file") as pbar:
        for item in download_list:
            file_name = item['file_name']
            download_url = item['download_link']
            
            # Create PDF filename
            pdf_filename = file_name + '.pdf'
            output_path = os.path.join(output_dir, pdf_filename)
            
            # Update progress bar description
            pbar.set_description(f"Downloading {pdf_filename[:50]}")
            
            # Download the file using S3 client
            success, error = download_pdf(download_url, output_path, retries=max_retries, s3_client=s3_client)
            
            if success:
                item['download_status'] = STATUS_SUCCESS
                item['download_error'] = None
                success_count += 1
            else:
                item['download_status'] = STATUS_FAILED
                item['download_error'] = error
                failed_count += 1
                tqdm.write(f"✗ Failed: {pdf_filename} - {error}")
            
            # Update progress bar
            pbar.update(1)
            
            # Save progress periodically (every 10 files)
            if (success_count + failed_count) % 10 == 0:
                save_json_data(args.json, data)
    
    # Final save
    print("\nSaving final status to JSON...")
    save_json_data(args.json, data)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total processed: {len(download_list)}")
    print(f"✓ Successful:    {success_count}")
    print(f"✗ Failed:        {failed_count}")
    print("="*60)
    
    if failed_count > 0:
        print(f"\n⚠ {failed_count} downloads failed. Run with --retry-failed to retry them.")
    
    # Show overall status
    print("\nOverall status:")
    status_counts = {}
    for item in data:
        status = item.get('download_status', STATUS_PENDING)
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\nPDFs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

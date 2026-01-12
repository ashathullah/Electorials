"""
Download CSV files from S3 bucket to local directory.

This script downloads all CSV files from the specified S3 bucket and prefix,
organizing them in the extraction_results directory with progress tracking
and resumable downloads.
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


class S3CSVDownloader:
    """Download CSV files from S3 bucket with progress tracking and resume capability."""
    
    # Download status constants
    STATUS_PENDING = "pending"
    STATUS_DOWNLOADED = "downloaded"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"
    
    def __init__(self, bucket_name: str, prefix: str, download_dir: str, tracking_dir: str):
        """
        Initialize the S3 CSV downloader.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: S3 prefix/path to search for CSV files
            download_dir: Local directory to save downloaded files
            tracking_dir: Directory to store progress tracking JSON
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.download_dir = Path(download_dir)
        self.tracking_dir = Path(tracking_dir)
        
        # Initialize S3 client
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'ap-southeast-2')
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "AWS credentials not found in .env file. "
                "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Create directories if they don't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking file path
        safe_prefix = self.prefix.replace('/', '_').strip('_')
        self.tracking_file = self.tracking_dir / f"download_status_{safe_prefix}.json"
        
        # Load or initialize tracking data
        self.tracking_data = self._load_tracking_data()
        
        print(f"✓ S3 client initialized for bucket: {bucket_name}")
        print(f"✓ Download directory: {self.download_dir}")
        print(f"✓ Tracking file: {self.tracking_file}")
    
    def _load_tracking_data(self) -> Dict:
        """Load tracking data from JSON file or create new one."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✓ Loaded existing tracking data with {len(data.get('files', {}))} files")
                    return data
            except Exception as e:
                print(f"⚠️  Error loading tracking file: {e}. Creating new one.")
        
        return {
            'bucket': self.bucket_name,
            'prefix': self.prefix,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'files': {}
        }
    
    def _save_tracking_data(self):
        """Save tracking data to JSON file."""
        self.tracking_data['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error saving tracking file: {e}")
    
    def _get_file_status(self, s3_key: str) -> Optional[str]:
        """Get current status of a file from tracking data."""
        return self.tracking_data['files'].get(s3_key, {}).get('status')
    
    def _update_file_status(self, s3_key: str, status: str, message: str = "", file_size: int = 0):
        """Update file status in tracking data."""
        if s3_key not in self.tracking_data['files']:
            self.tracking_data['files'][s3_key] = {}
        
        self.tracking_data['files'][s3_key].update({
            'status': status,
            'message': message,
            'file_size': file_size,
            'last_updated': datetime.now().isoformat()
        })
    
    def list_csv_files(self) -> List[Dict]:
        """
        List all CSV files in the S3 bucket with the given prefix.
        
        Returns:
            List of dictionaries with S3 file metadata
        """
        print(f"\n🔍 Searching for CSV files in s3://{self.bucket_name}/{self.prefix}")
        
        csv_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        try:
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv'):
                        csv_files.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat()
                        })
                        
            print(f"✓ Found {len(csv_files)} CSV files")
            return csv_files
            
        except ClientError as e:
            print(f"❌ Error listing S3 objects: {e}")
            raise
    
    def download_file(self, s3_key: str, s3_size: int) -> tuple[bool, str, int]:
        """
        Download a single file from S3.
        
        Args:
            s3_key: S3 key of the file to download
            s3_size: Size of the file in S3
            
        Returns:
            Tuple of (success: bool, message: str, file_size: int)
        """
        # Create local file path maintaining S3 structure
        relative_path = s3_key.replace(self.prefix, '', 1).lstrip('/')
        local_file_path = self.download_dir / relative_path
        
        # Create parent directories if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and has correct size
        if local_file_path.exists():
            file_size = local_file_path.stat().st_size
            if file_size == s3_size:
                return True, f"Already exists ({file_size:,} bytes)", file_size
            else:
                # Size mismatch, re-download
                pass
        
        try:
            # Download the file
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_file_path))
            file_size = local_file_path.stat().st_size
            return True, f"Downloaded ({file_size:,} bytes)", file_size
            
        except ClientError as e:
            return False, f"Error: {str(e)[:100]}", 0
        except Exception as e:
            return False, f"Error: {str(e)[:100]}", 0
    
    def download_all_csvs(self, retry_failed: bool = False, force: bool = False) -> Dict:
        """
        Download all CSV files from S3.
        
        Args:
            retry_failed: Only retry files that previously failed
            force: Re-download all files regardless of status
            
        Returns:
            Dictionary with download statistics
        """
        csv_files = self.list_csv_files()
        
        if not csv_files:
            print("\n⚠️  No CSV files found!")
            return {'total': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        # Filter files based on mode
        files_to_process = []
        for file_info in csv_files:
            s3_key = file_info['key']
            current_status = self._get_file_status(s3_key)
            
            if retry_failed:
                # Only process failed files
                if current_status == self.STATUS_FAILED:
                    files_to_process.append(file_info)
            elif force:
                # Process all files
                files_to_process.append(file_info)
            else:
                # Skip already downloaded files
                if current_status != self.STATUS_DOWNLOADED:
                    files_to_process.append(file_info)
        
        if not files_to_process:
            if retry_failed:
                print("\n✓ No failed files to retry!")
            else:
                print("\n✓ All files already downloaded!")
            return {'total': len(csv_files), 'downloaded': 0, 'skipped': len(csv_files), 'failed': 0}
        
        mode_text = "failed files" if retry_failed else ("all files (force mode)" if force else "pending files")
        print(f"\n📥 Downloading {len(files_to_process)} {mode_text}...\n")
        
        stats = {
            'total': len(csv_files),
            'downloaded': 0,
            'skipped': 0,
            'failed': 0
        }
        
        # Create progress bar
        with tqdm(total=len(files_to_process), 
                  desc="Downloading CSVs",
                  unit="file",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for file_info in files_to_process:
                s3_key = file_info['key']
                s3_size = file_info['size']
                filename = os.path.basename(s3_key)
                
                # Update progress bar description with current file
                pbar.set_description(f"Downloading {filename[:40]}")
                
                try:
                    success, message, file_size = self.download_file(s3_key, s3_size)
                    
                    # Update stats and tracking
                    if success:
                        if "Already exists" in message:
                            status = self.STATUS_SKIPPED
                            stats['skipped'] += 1
                        else:
                            status = self.STATUS_DOWNLOADED
                            stats['downloaded'] += 1
                    else:
                        status = self.STATUS_FAILED
                        stats['failed'] += 1
                    
                    self._update_file_status(s3_key, status, message, file_size)
                    
                    # Save tracking data periodically (every 10 files)
                    if (stats['downloaded'] + stats['failed']) % 10 == 0:
                        self._save_tracking_data()
                    
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)[:100]}"
                    self._update_file_status(s3_key, self.STATUS_FAILED, error_msg, 0)
                    stats['failed'] += 1
                
                # Update progress bar
                pbar.update(1)
        
        # Final save of tracking data
        self._save_tracking_data()
        
        return stats
    
    def print_summary(self, stats: Dict):
        """Print download summary statistics."""
        print("\n" + "=" * 70)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Total files found:     {stats['total']:4d}")
        print(f"Downloaded:            {stats['downloaded']:4d}")
        print(f"Skipped (existing):    {stats['skipped']:4d}")
        print(f"Failed:                {stats['failed']:4d}")
        print("=" * 70)
        
        if stats['failed'] > 0:
            print(f"\n⚠️  {stats['failed']} file(s) failed to download.")
            print(f"💡 Run with --retry to retry failed downloads")
        elif stats['downloaded'] == 0 and stats['skipped'] > 0:
            print("\n✓ All files were already downloaded!")
        else:
            print("\n✓ Download complete!")
        
        print(f"\n📄 Download log: {self.tracking_file}")
    
    def show_status(self):
        """Show current download status from tracking file."""
        if not self.tracking_data['files']:
            print("\n⚠️  No tracking data found. Run download first.")
            return
        
        status_counts = {
            self.STATUS_DOWNLOADED: 0,
            self.STATUS_FAILED: 0,
            self.STATUS_SKIPPED: 0,
            self.STATUS_PENDING: 0
        }
        
        for file_data in self.tracking_data['files'].values():
            status = file_data.get('status', self.STATUS_PENDING)
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n" + "=" * 70)
        print("📊 CURRENT STATUS")
        print("=" * 70)
        print(f"Total tracked files:   {len(self.tracking_data['files']):4d}")
        print(f"Downloaded:            {status_counts[self.STATUS_DOWNLOADED]:4d}")
        print(f"Skipped:               {status_counts[self.STATUS_SKIPPED]:4d}")
        print(f"Failed:                {status_counts[self.STATUS_FAILED]:4d}")
        print(f"Pending:               {status_counts[self.STATUS_PENDING]:4d}")
        print("=" * 70)
        
        if status_counts[self.STATUS_FAILED] > 0:
            print(f"\n⚠️  {status_counts[self.STATUS_FAILED]} failed file(s). Run with --retry to retry.")
        
        print(f"\n📄 Tracking file: {self.tracking_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download CSV files from S3 bucket with resume capability'
    )
    parser.add_argument(
        '--bucket',
        default='264676382451-eci-download',
        help='S3 bucket name (default: 264676382451-eci-download)'
    )
    parser.add_argument(
        '--prefix',
        default='2026/1/S22/extraction_results/',
        help='S3 prefix/path (default: 2026/1/S22/extraction_results/)'
    )
    parser.add_argument(
        '--output',
        default='../../extraction_results',
        help='Output directory (default: ../../extraction_results)'
    )
    parser.add_argument(
        '--tracking',
        default='../../meta_logs',
        help='Tracking directory for progress JSON (default: ../../meta_logs)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download all files even if they already exist'
    )
    parser.add_argument(
        '--retry',
        action='store_true',
        help='Retry only failed downloads from previous runs'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current download status and exit'
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths relative to script location
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output).resolve()
    tracking_dir = (script_dir / args.tracking).resolve()
    
    try:
        print("=" * 70)
        print("S3 CSV DOWNLOADER")
        print("=" * 70)
        print(f"Bucket:   {args.bucket}")
        print(f"Prefix:   {args.prefix}")
        print(f"Output:   {output_dir}")
        print(f"Tracking: {tracking_dir}")
        if args.retry:
            print(f"Mode:     RETRY FAILED")
        elif args.force:
            print(f"Mode:     FORCE RE-DOWNLOAD")
        else:
            print(f"Mode:     RESUME (skip existing)")
        print("=" * 70)
        
        # Initialize downloader
        downloader = S3CSVDownloader(
            bucket_name=args.bucket,
            prefix=args.prefix,
            download_dir=str(output_dir),
            tracking_dir=str(tracking_dir)
        )
        
        # Show status and exit if requested
        if args.status:
            downloader.show_status()
            return
        
        # Download CSV files
        stats = downloader.download_all_csvs(
            retry_failed=args.retry,
            force=args.force
        )
        
        # Print summary
        downloader.print_summary(stats)
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    except NoCredentialsError:
        print("\n❌ AWS credentials not found or invalid!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user. Progress has been saved.")
        print("💡 Run the script again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

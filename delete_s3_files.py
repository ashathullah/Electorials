"""
Script to delete metadata and voters CSV files from S3 based on names in invalid.txt
Includes progress tracking, resumability, and detailed logging.
"""

import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
INVALID_FILE = "invalid.txt"
STATUS_FILE = "deletion_status.json"
S3_BUCKET = "264676382451-eci-download"
S3_BASE_PATH = "2026/1/S22/extraction_results"

class S3FileDeleter:
    def __init__(self):
        """Initialize S3 client and load configuration"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-south-1')
        )
        
        self.bucket = S3_BUCKET
        self.base_path = S3_BASE_PATH
        self.status = self.load_status()
        
        # Counters
        self.metadata_deleted = 0
        self.voters_deleted = 0
        self.metadata_failed = 0
        self.voters_failed = 0
        self.metadata_not_found = 0
        self.voters_not_found = 0
        
    def load_status(self):
        """Load status from JSON file for resumability"""
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        return {
            "completed": [],
            "failed": [],
            "last_updated": None,
            "started_at": datetime.now().isoformat()
        }
    
    def save_status(self):
        """Save current status to JSON file"""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, indent=2, fp=f)
    
    def load_invalid_names(self):
        """Load file names from invalid.txt"""
        with open(INVALID_FILE, 'r') as f:
            # Strip whitespace and filter out empty lines
            names = [line.strip() for line in f if line.strip()]
        return names
    
    def delete_file(self, s3_key):
        """
        Delete a single file from S3
        Returns: ('success', None) or ('not_found', None) or ('error', error_message)
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            return ('success', None)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404' or error_code == 'NoSuchKey':
                return ('not_found', None)
            return ('error', str(e))
        except Exception as e:
            return ('error', str(e))
    
    def delete_files_for_name(self, name):
        """
        Delete both metadata and voters CSV files for a given name
        Returns: dict with deletion results
        """
        # Extract constituency folder from filename
        # Format: "Tamil Nadu-(S22)_Sivaganga-(AC186)_329"
        # Split by underscore and take the second part as the subdirectory
        parts = name.split('_')
        if len(parts) >= 2:
            constituency_folder = parts[1]
        else:
            # Fallback: if format is unexpected, use name as-is
            constituency_folder = ""
            print(f"Warning: Unexpected filename format: {name}")
        
        # Construct S3 keys with constituency subdirectory
        if constituency_folder:
            metadata_key = f"{self.base_path}/{constituency_folder}/{name}_metadata.csv"
            voters_key = f"{self.base_path}/{constituency_folder}/{name}_voters.csv"
        else:
            # Fallback to old structure if constituency couldn't be extracted
            metadata_key = f"{self.base_path}/{name}_metadata.csv"
            voters_key = f"{self.base_path}/{name}_voters.csv"
        
        result = {
            "name": name,
            "metadata": {},
            "voters": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Delete metadata file
        status, error = self.delete_file(metadata_key)
        result["metadata"]["status"] = status
        result["metadata"]["s3_key"] = metadata_key
        if error:
            result["metadata"]["error"] = error
        
        if status == 'success':
            self.metadata_deleted += 1
        elif status == 'not_found':
            self.metadata_not_found += 1
        else:
            self.metadata_failed += 1
        
        # Delete voters file
        status, error = self.delete_file(voters_key)
        result["voters"]["status"] = status
        result["voters"]["s3_key"] = voters_key
        if error:
            result["voters"]["error"] = error
        
        if status == 'success':
            self.voters_deleted += 1
        elif status == 'not_found':
            self.voters_not_found += 1
        else:
            self.voters_failed += 1
        
        return result
    
    def process_deletions(self):
        """Main processing logic"""
        print("=" * 80)
        print("S3 File Deletion Script")
        print("=" * 80)
        print(f"Bucket: {self.bucket}")
        print(f"Base Path: {self.base_path}")
        print(f"Status File: {STATUS_FILE}")
        print("=" * 80)
        
        # Load invalid names
        all_names = self.load_invalid_names()
        print(f"\nTotal files in invalid.txt: {len(all_names)}")
        
        # Filter out already completed names
        completed_names = set(self.status.get("completed", []))
        names_to_process = [name for name in all_names if name not in completed_names]
        
        if completed_names:
            print(f"Already completed: {len(completed_names)}")
            print(f"Remaining to process: {len(names_to_process)}")
        
        if not names_to_process:
            print("\n✓ All files already processed!")
            return
        
        print("\nStarting deletion process...\n")
        
        # Process each name with progress bar
        with tqdm(total=len(names_to_process), desc="Deleting files", unit="name") as pbar:
            for name in names_to_process:
                # Delete files for this name
                result = self.delete_files_for_name(name)
                
                # Update status
                if (result["metadata"]["status"] in ['success', 'not_found'] and 
                    result["voters"]["status"] in ['success', 'not_found']):
                    self.status["completed"].append(name)
                else:
                    self.status["failed"].append({
                        "name": name,
                        "result": result
                    })
                
                # Save status periodically (every 10 files)
                if len(self.status["completed"]) % 10 == 0:
                    self.save_status()
                
                # Update progress bar
                pbar.update(1)
                
                # Update progress bar description with current stats
                pbar.set_postfix({
                    'M_del': self.metadata_deleted,
                    'V_del': self.voters_deleted,
                    'M_fail': self.metadata_failed,
                    'V_fail': self.voters_failed
                })
        
        # Final save
        self.save_status()
        
        # Print summary
        self.print_summary(len(all_names))
    
    def print_summary(self, total_files):
        """Print deletion summary"""
        print("\n" + "=" * 80)
        print("DELETION SUMMARY")
        print("=" * 80)
        print(f"\nTotal file names processed: {len(self.status['completed'])}/{total_files}")
        print(f"\nMetadata Files:")
        print(f"  ✓ Successfully deleted: {self.metadata_deleted}")
        print(f"  ⚠ Not found: {self.metadata_not_found}")
        print(f"  ✗ Failed to delete: {self.metadata_failed}")
        print(f"\nVoters Files:")
        print(f"  ✓ Successfully deleted: {self.voters_deleted}")
        print(f"  ⚠ Not found: {self.voters_not_found}")
        print(f"  ✗ Failed to delete: {self.voters_failed}")
        print(f"\nTotal Files:")
        print(f"  ✓ Metadata files removed: {self.metadata_deleted}")
        print(f"  ✓ Voters files removed: {self.voters_deleted}")
        print(f"  ✓ Total files removed: {self.metadata_deleted + self.voters_deleted}")
        
        if self.status.get("failed"):
            print(f"\n⚠ Files with errors: {len(self.status['failed'])}")
            print(f"  Check {STATUS_FILE} for details")
        
        print("\n" + "=" * 80)
        print(f"Status saved to: {STATUS_FILE}")
        print("=" * 80)

def main():
    """Main execution function"""
    try:
        deleter = S3FileDeleter()
        deleter.process_deletions()
    except FileNotFoundError:
        print(f"Error: {INVALID_FILE} not found!")
        print("Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

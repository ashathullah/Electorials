import sys
from pathlib import Path
import boto3
from botocore.config import Config as BotoConfig

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config

def check_structure():
    config = Config()
    s3_config = config.s3
    
    # Setup client
    boto_config = BotoConfig(
        connect_timeout=s3_config.connect_timeout,
        read_timeout=s3_config.read_timeout,
        retries={"max_attempts": s3_config.max_retries},
    )
    
    client = boto3.client(
        "s3",
        region_name=s3_config.region,
        aws_access_key_id=s3_config.access_key_id,
        aws_secret_access_key=s3_config.secret_access_key,
        aws_session_token=s3_config.session_token if s3_config.session_token else None,
        config=boto_config
    )
    
    bucket = "264676382451-eci-download"
    prefix = "2026/1/S22/extraction_results/"
    
    print(f"Listing 1 file from {bucket}/{prefix}...")
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
    
    if "Contents" not in response:
        print("No files found.")
        return

    voter_key = None
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith("_voters.csv"):
            voter_key = key
            break
            
    if not voter_key:
        print("No _voters.csv found in first few keys.")
        return
        
    print(f"Checking file: {voter_key}")
    
    # Get last 1KB
    try:
        obj_meta = client.head_object(Bucket=bucket, Key=voter_key)
        size = obj_meta['ContentLength']
        print(f"File size: {size} bytes")
        
        # Read last 1024 bytes
        byte_range = f"bytes={max(0, size-1024)}-{size-1}"
        resp = client.get_object(Bucket=bucket, Key=voter_key, Range=byte_range)
        content = resp['Body'].read().decode('utf-8', errors='ignore')
        
        print("\n--- Last ~1KB Content ---")
        print(content)
        print("-------------------------")
        
        lines = content.strip().split('\n')
        if lines:
            print(f"Last line: {lines[-1]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_structure()

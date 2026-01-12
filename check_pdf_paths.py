#!/usr/bin/env python3
"""Check where PDFs actually exist in S3 and compare with JSON paths"""

import boto3
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-southeast-2')
)

bucket = '264676382451-eci-download'

print("Checking actual PDF locations in S3...")
print("=" * 70)

# Load one sample from your JSON
with open('voters_missing_names.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    sample = data[0]

print(f"\nSample from JSON:")
print(f"  File: {sample['file_name']}")
print(f"  Link: {sample['download_link']}")

# Extract the S3 key from the HTTPS URL
# URL format: https://bucket.s3.region.amazonaws.com/KEY
url = sample['download_link']
if 's3.ap-southeast-2.amazonaws.com/' in url:
    key_from_json = url.split('s3.ap-southeast-2.amazonaws.com/')[1]
    print(f"\n  S3 Key from JSON: {key_from_json}")
    
    # Try to check if this exact file exists
    print(f"\nTest 1: Checking if file exists at this exact path...")
    try:
        response = s3.head_object(Bucket=bucket, Key=key_from_json)
        print(f"  [SUCCESS] File exists! Size: {response['ContentLength']:,} bytes")
    except s3.exceptions.NoSuchKey:
        print(f"  [NOT FOUND] File does not exist at this path")
        
        # Try without URL encoding (replace + with space)
        key_decoded = key_from_json.replace('+', ' ')
        print(f"\nTest 2: Trying with spaces instead of + signs...")
        print(f"  Key: {key_decoded}")
        try:
            response = s3.head_object(Bucket=bucket, Key=key_decoded)
            print(f"  [SUCCESS] File exists! Size: {response['ContentLength']:,} bytes")
            print(f"\n  >>> SOLUTION: The paths in JSON use '+' but S3 has spaces!")
        except s3.exceptions.NoSuchKey:
            print(f"  [NOT FOUND] File does not exist here either")
    except Exception as e:
        print(f"  [ERROR] {e}")

# List what's actually in the Tamil Nadu/Manachanallur folder
print(f"\nTest 3: Listing actual files in Tamil Nadu/Manachanallur...")
try:
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='2026/1/S22/pdfs/Tamil Nadu/Manachanallur/',
        MaxKeys=10
    )
    
    objects = response.get('Contents', [])
    if objects:
        print(f"  Found {response.get('KeyCount', 0)} files (showing first 10):")
        for obj in objects[:10]:
            key = obj['Key']
            filename = key.split('/')[-1]
            print(f"  - {filename}")
    else:
        print(f"  [NOT FOUND] No files found at this prefix")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "=" * 70)

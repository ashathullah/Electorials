#!/usr/bin/env python3
"""Test script to check PDF access in S3"""

import boto3
import os
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

print("Testing S3 PDF access...")
print("=" * 70)

# Test 1: List top-level folders under pdfs/
print("\n1. Checking folders under 2026/1/S22/pdfs/...")
try:
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='2026/1/S22/pdfs/',
        Delimiter='/',
        MaxKeys=10
    )
    
    folders = [p.get('Prefix') for p in response.get('CommonPrefixes', [])]
    print(f"   Found {len(folders)} state folders:")
    for folder in folders[:5]:
        print(f"   - {folder}")
    if len(folders) > 5:
        print(f"   ... and {len(folders) - 5} more")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 2: Check specific PDF path from JSON
print("\n2. Checking specific PDF from your JSON...")
test_key = "2026/1/S22/pdfs/Tamil Nadu/Manachanallur/Tamil Nadu-(S22)_Manachanallur-(AC144)_100.pdf"
print(f"   Key: {test_key}")

try:
    # Try to get metadata (head_object)
    response = s3.head_object(Bucket=bucket, Key=test_key)
    print(f"   [OK] File exists! Size: {response['ContentLength']:,} bytes")
except s3.exceptions.NoSuchKey:
    print("   [ERROR] File does not exist at this path")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 3: List actual PDFs in Tamil Nadu/Manachanallur
print("\n3. Listing PDFs in Tamil Nadu/Manachanallur...")
try:
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='2026/1/S22/pdfs/Tamil Nadu/Manachanallur/',
        MaxKeys=5
    )
    
    objects = response.get('Contents', [])
    print(f"   Found {len(objects)} files (showing max 5):")
    for obj in objects[:5]:
        print(f"   - {obj['Key']} ({obj['Size']:,} bytes)")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

print("\n" + "=" * 70)

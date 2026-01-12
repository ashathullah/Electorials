# PDF Download Script - AWS S3 Authentication Fix

## Problem Detected

The original script was using direct HTTP requests to download PDFs from S3 URLs, which resulted in **403 Forbidden** errors because the S3 bucket requires AWS authentication.

### Original Error:
```
✗ Failed: Tamil Nadu-(S22)_Manachanallur-(AC144)_126.pdf - 
Request failed: 403 Client Error: Forbidden for url: 
https://264676382451-eci-download.s3.ap-southeast-2.amazonaws.com/2026/1/S22/pdfs/...
```

## Solution Implemented

Replaced HTTP-based downloads with **boto3 (AWS SDK)** to use authenticated S3 access.

### Key Changes:

1. **Added AWS SDK Support**
   - Replaced `requests` library with `boto3`
   - Added `python-dotenv` to load AWS credentials from `.env`

2. **URL Parsing**
   - Created `parse_s3_url()` function to extract bucket name and object key from S3 URLs
   - Handles URL-encoded characters properly

3. **Authenticated Downloads**
   - Downloads now use `s3_client.download_file()` with credentials
   - Reads credentials from .env file:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`
     - `AWS_REGION`

4. **Connection Verification**
   - Script tests S3 connection before starting downloads
   - Provides clear error messages if credentials are missing/invalid

5. **Better Error Handling**
   - Distinguishes between different S3 errors (NotFound, AccessDenied, etc.)
   - Doesn't retry on permanent errors (404, NoSuchKey)

## How It Works Now

```python
# Before (HTTP - Failed with 403):
response = requests.get(url)

# After (AWS SDK - Authenticated):
s3_client = boto3.client('s3', aws_access_key_id=..., aws_secret_access_key=...)
s3_client.download_file(bucket, key, output_path)
```

## Testing

### Connection Test:
```bash
python download_missing_voters_pdfs.py
```

Expected output:
```
Loading data from voters_missing_names.json...
Total entries in JSON: 633
Files to download: 593

Initializing AWS S3 client...
✓ AWS S3 connection successful!

Downloading PDFs to missing_voters_pdfs/
```

### Verify Credentials:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('AWS Key ID:', os.getenv('AWS_ACCESS_KEY_ID')[:10] + '...' if os.getenv('AWS_ACCESS_KEY_ID') else 'NOT FOUND')"
```

## Current Status

✅ AWS S3 connection successful  
✅ Script is running and downloading files  
✅ Progress tracking and resume functionality intact  
✅ All existing features preserved  

## Dependencies

- `boto3` - ✅ Already installed (v1.42.16)
- `python-dotenv` - ✅ Already installed (v1.2.1)
- `tqdm` - ✅ Already installed (v4.67.1)

## Notes

- Some PDFs may not exist in S3 (404 errors are expected)
- The script will mark these as "failed" but continue with others
- You can retry failed downloads later with `--retry-failed` flag
- AWS credentials are securely loaded from `.env` file (not hardcoded)

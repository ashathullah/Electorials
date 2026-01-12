#!/usr/bin/env python3
"""Quick test to verify the fix works"""

from urllib.parse import unquote_plus, urlparse

# Sample URL from your JSON
url = "https://264676382451-eci-download.s3.ap-southeast-2.amazonaws.com/2026/1/S22/pdfs/Tamil+Nadu/Manachanallur/Tamil+Nadu-(S22)_Manachanallur-(AC144)_100.pdf"

parsed = urlparse(url)
bucket = parsed.netloc.split('.s3.')[0]
key = unquote_plus(parsed.path.lstrip('/'))

print("Testing URL decoding fix...")
print("=" * 70)
print(f"Original URL: {url}")
print(f"\nExtracted:")
print(f"  Bucket: {bucket}")
print(f"  Key: {key}")
print(f"\nExpected key should have SPACES not + signs")
print(f"Check: {'Tamil Nadu' in key and 'Tamil+Nadu' not in key}")
print("=" * 70)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils.s3_utils import list_s3_objects

def debug_list():
    config = Config()
    # Try listing slightly higher up to see the folder names
    prefix = "2026/1/S22/pdfs/Tamil Nadu/" 
    print(f"Listing '{prefix}'...")
    try:
        keys = list_s3_objects(prefix, config.s3, bucket="264676382451-eci-download", max_keys=20)
        for k in keys:
            print(k)
        if not keys:
            print("No keys found.")
            
        print("-" * 20)
        # Try finding one specific file
        prefix2 = "2026/1/S22/pdfs/Tamil Nadu/Manachanallur/"
        print(f"Listing '{prefix2}'...")
        keys2 = list_s3_objects(prefix2, config.s3, bucket="264676382451-eci-download", max_keys=50)
        for k in keys2:
            print(k, flush=True)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_list()

"""
Script to remove JSON files from invalid_metadata_json directory that are also present in valid_metadata directory.
This helps clean up duplicate metadata files that have been validated and moved to the valid_metadata directory.
"""

import os
from pathlib import Path


def remove_common_invalid_metadata():
    """
    Remove JSON files from invalid_metadata_json that exist in valid_metadata.
    
    This function:
    1. Scans all files in the valid_metadata directory
    2. For each file found in valid_metadata, checks if it exists in invalid_metadata_json
    3. Removes the duplicate file from invalid_metadata_json
    4. Logs all operations for tracking
    """
    # Get the project root directory (2 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define directories
    invalid_dir = project_root / "invalid_metadata_json"
    valid_dir = project_root / "valid_metadata"
    
    # Verify directories exist
    if not invalid_dir.exists():
        print(f"❌ Error: Invalid metadata directory not found: {invalid_dir}")
        return
    
    if not valid_dir.exists():
        print(f"❌ Error: Valid metadata directory not found: {valid_dir}")
        return
    
    print(f"📂 Scanning directories...")
    print(f"   Valid metadata: {valid_dir}")
    print(f"   Invalid metadata: {invalid_dir}")
    print()
    
    # Get all JSON files from valid_metadata
    valid_files = set(f.name for f in valid_dir.glob("*.json"))
    print(f"✓ Found {len(valid_files)} files in valid_metadata")
    
    # Get all JSON files from invalid_metadata_json
    invalid_files = set(f.name for f in invalid_dir.glob("*.json"))
    print(f"✓ Found {len(invalid_files)} files in invalid_metadata_json")
    print()
    
    # Find common files (files that exist in both directories)
    common_files = valid_files.intersection(invalid_files)
    
    if not common_files:
        print("✓ No common files found. No files to remove.")
        return
    
    print(f"🔍 Found {len(common_files)} common files that need to be removed from invalid_metadata_json")
    print()
    
    # Remove common files from invalid_metadata_json
    removed_count = 0
    failed_count = 0
    
    for filename in sorted(common_files):
        file_path = invalid_dir / filename
        try:
            file_path.unlink()
            removed_count += 1
            print(f"  ✓ Removed: {filename}")
        except Exception as e:
            failed_count += 1
            print(f"  ❌ Failed to remove {filename}: {e}")
    
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total files in valid_metadata:        {len(valid_files)}")
    print(f"Total files in invalid_metadata_json: {len(invalid_files)}")
    print(f"Common files found:                   {len(common_files)}")
    print(f"Successfully removed:                 {removed_count}")
    print(f"Failed to remove:                     {failed_count}")
    print(f"Remaining in invalid_metadata_json:   {len(invalid_files) - removed_count}")
    print("=" * 70)


if __name__ == "__main__":
    remove_common_invalid_metadata()

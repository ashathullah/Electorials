"""
Script to fix serial_no in JSON files.
The serial_no should start from 1 and increment sequentially.
"""

import json
import os
from pathlib import Path

def fix_serial_numbers_in_file(file_path):
    """
    Fix serial numbers in a single JSON file to start from 1 and increment sequentially.
    
    Args:
        file_path: Path to the JSON file to fix
        
    Returns:
        Tuple of (filename, original_start, original_end, new_count, changed)
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            voters = json.load(f)
        
        if not voters or not isinstance(voters, list):
            return (file_path.name, None, None, 0, False)
        
        # Get original serial numbers
        original_start = voters[0].get('serial_no') if voters else None
        original_end = voters[-1].get('serial_no') if voters else None
        
        # Check if fix is needed
        needs_fix = False
        for i, voter in enumerate(voters, start=1):
            if voter.get('serial_no') != str(i):
                needs_fix = True
                break
        
        if not needs_fix:
            return (file_path.name, original_start, original_end, len(voters), False)
        
        # Fix the serial numbers
        for i, voter in enumerate(voters, start=1):
            voter['serial_no'] = str(i)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(voters, f, ensure_ascii=False, indent=2)
        
        return (file_path.name, original_start, original_end, len(voters), True)
    
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return (file_path.name, None, None, 0, False)

def main():
    """Main function to fix serial numbers in all JSON files."""
    # Define the directory containing the JSON files
    base_dir = Path(__file__).parent.parent.parent
    json_dir = base_dir / "invalid_voter_sections_json"
    
    if not json_dir.exists():
        print(f"Directory not found: {json_dir}")
        return
    
    print(f"Processing JSON files in: {json_dir}")
    print("=" * 80)
    
    # Get all JSON files
    json_files = sorted(json_dir.glob("*_voters.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print("No JSON files found!")
        return
    
    print(f"Found {total_files} JSON files to process\n")
    
    # Process each file
    fixed_count = 0
    unchanged_count = 0
    error_count = 0
    
    for i, json_file in enumerate(json_files, start=1):
        filename, original_start, original_end, voter_count, changed = fix_serial_numbers_in_file(json_file)
        
        if changed:
            fixed_count += 1
            print(f"[{i}/{total_files}] ✓ FIXED: {filename}")
            print(f"          Original: {original_start} → {original_end} | New: 1 → {voter_count} | Count: {voter_count}")
        elif original_start is None:
            error_count += 1
            print(f"[{i}/{total_files}] ✗ ERROR: {filename}")
        else:
            unchanged_count += 1
            print(f"[{i}/{total_files}] ○ OK: {filename} (already correct: 1 → {voter_count})")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files already correct: {unchanged_count}")
    print(f"  Errors: {error_count}")
    print("=" * 80)
    print("✓ Done!")

if __name__ == "__main__":
    main()

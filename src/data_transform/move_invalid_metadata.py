#!/usr/bin/env python3
"""
Script to identify and move invalid metadata JSON files to a separate directory.
Invalid files are those with fewer than 10 fields (incomplete metadata).
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any


def is_valid_metadata(json_data: Dict[str, Any], min_fields: int = 10) -> bool:
    """
    Check if metadata JSON has sufficient fields to be considered valid.
    
    Args:
        json_data: The parsed JSON data
        min_fields: Minimum number of fields required for valid metadata
    
    Returns:
        True if valid, False if invalid
    """
    # Count non-null fields
    field_count = len(json_data)
    
    # Check if it has minimum required fields
    if field_count < min_fields:
        return False
    
    # Additional check: valid files should have detailed metadata fields
    required_fields = [
        'administrative_address_district',
        'constituency_details_assembly_constituency_name',
        'detailed_elector_summary_net_total_total'
    ]
    
    # Check if at least one of the required fields exists
    has_detailed_fields = any(field in json_data for field in required_fields)
    
    return has_detailed_fields


def process_metadata_files(source_dir: Path, invalid_dir: Path):
    """Process metadata JSON files and move invalid ones to separate directory."""
    # Create invalid directory if it doesn't exist
    invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(source_dir.glob("*_metadata.json"))
    
    if not json_files:
        print(f"No *_metadata.json files found in {source_dir}")
        return
    
    print(f"Found {len(json_files)} metadata JSON files")
    print(f"Checking validity and moving invalid files...\n")
    
    valid_count = 0
    invalid_count = 0
    errors = 0
    
    for json_file in json_files:
        try:
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if valid
            if is_valid_metadata(json_data):
                valid_count += 1
            else:
                # Move to invalid directory
                destination = invalid_dir / json_file.name
                shutil.move(str(json_file), str(destination))
                invalid_count += 1
                
                if invalid_count <= 5:  # Show first 5 invalid files
                    print(f"Moved (invalid): {json_file.name}")
                elif invalid_count == 6:
                    print("... (more invalid files being moved)")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            errors += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Valid files (kept in source): {valid_count}")
    print(f"Invalid files (moved): {invalid_count}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")
    print(f"\nValid files remain in: {source_dir}")
    print(f"Invalid files moved to: {invalid_dir}")


def main():
    """Main function."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define paths
    source_dir = project_root / "extraction_results_metadata_json"
    invalid_dir = project_root / "invalid_metadata_json"
    
    print(f"Source directory: {source_dir}")
    print(f"Invalid directory: {invalid_dir}")
    print()
    
    # Process the files
    process_metadata_files(source_dir, invalid_dir)


if __name__ == "__main__":
    main()

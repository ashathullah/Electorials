#!/usr/bin/env python3
"""
Script to convert *_metadata.csv files to *_metadata.json format.
Reads CSV files from extraction_results directory and saves JSON files
to extraction_result_metadata_json directory.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any


def convert_value(value: str) -> Any:
    """Convert CSV string value to appropriate JSON type."""
    # Handle empty strings
    if value == "" or value is None:
        return None
    
    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def csv_to_json(csv_path: Path) -> Dict[str, Any]:
    """Convert a metadata CSV file to JSON dict."""
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Read the first (and only) row
        for row in reader:
            # Convert values to appropriate types
            json_data = {key: convert_value(value) for key, value in row.items()}
            return json_data
    
    return {}


def process_metadata_files(source_dir: Path, output_dir: Path):
    """Process all *_metadata.csv files and convert them to JSON."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all *_metadata.csv files
    metadata_files = list(source_dir.glob("*_metadata.csv"))
    
    if not metadata_files:
        print(f"No *_metadata.csv files found in {source_dir}")
        return
    
    print(f"Found {len(metadata_files)} metadata CSV files")
    processed = 0
    errors = 0
    
    for csv_file in metadata_files:
        try:
            # Convert CSV to JSON
            json_data = csv_to_json(csv_file)
            
            if not json_data:
                print(f"Warning: No data found in {csv_file.name}")
                continue
            
            # Create output filename: replace _metadata.csv with _metadata.json
            output_filename = csv_file.name.replace("_metadata.csv", "_metadata.json")
            output_path = output_dir / output_filename
            
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(metadata_files)} files...")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            errors += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")


def main():
    """Main function."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define paths
    source_dir = project_root / "extraction_results"
    output_dir = project_root / "extraction_results_metadata_json"
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process the files
    process_metadata_files(source_dir, output_dir)


if __name__ == "__main__":
    main()

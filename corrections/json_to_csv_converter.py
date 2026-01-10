#!/usr/bin/env python3
"""
JSON to CSV Converter for Electoral Roll Metadata
Converts JSON files from corrections/metadata folder to CSV files matching database schema.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def extract_metadata_row(json_data: Dict[str, Any], json_filename: str) -> Dict[str, Any]:
    """
    Extract metadata from JSON and map to database schema columns.
    
    Args:
        json_data: Parsed JSON data
        json_filename: Name of the JSON file
        
    Returns:
        Dictionary with metadata columns matching schema.sql
    """
    # Extract administrative address fields
    admin_addr = json_data.get('administrative_address', {})
    
    # Extract constituency details as JSONB
    constituency_details = json.dumps(json_data.get('constituency_details', {}))
    
    # Extract polling details as JSONB
    polling_details = json.dumps(json_data.get('polling_details', {}))
    
    # Extract detailed elector summary as JSONB
    detailed_elector_summary = json.dumps(json_data.get('detailed_elector_summary', {}))
    
    # Extract authority verification as JSONB
    authority_verification = json.dumps(json_data.get('authority_verification', {}))
    
    # Create PDF name from document_id (remove .json extension and add .pdf)
    document_id = json_data.get('document_id', json_filename.replace('.json', ''))
    pdf_name = f"{document_id}.pdf"
    
    # Build the row dictionary
    row = {
        'document_id': document_id,
        'pdf_name': pdf_name,
        'state': json_data.get('state'),
        'year': json_data.get('electoral_roll_year'),
        'revision_type': json_data.get('revision_type'),
        'qualifying_date': json_data.get('qualifying_date'),
        'publication_date': json_data.get('publication_date'),
        'roll_type': json_data.get('roll_type'),
        'roll_identification': json_data.get('roll_identification'),
        'total_pages': json_data.get('total_pages'),
        'total_voters_extracted': json_data.get('total_voters_extracted'),
        
        # Administrative address fields (promoted from JSONB)
        'town_or_village': admin_addr.get('town_or_village'),
        'main_town_or_village': admin_addr.get('main_town_or_village'),
        'ward_number': admin_addr.get('ward_number'),
        'post_office': admin_addr.get('post_office'),
        'police_station': admin_addr.get('police_station'),
        'taluk_or_block': admin_addr.get('taluk_or_block'),
        'subdivision': admin_addr.get('subdivision'),
        'district': admin_addr.get('district'),
        'pin_code': admin_addr.get('pin_code'),
        'panchayat_name': admin_addr.get('panchayat_name'),
        
        # JSONB fields
        'constituency_details': constituency_details,
        'administrative_address': json.dumps(admin_addr),
        'polling_details': polling_details,
        'detailed_elector_summary': detailed_elector_summary,
        'authority_verification': authority_verification,
        
        'output_identifier': json_data.get('output_identifier'),
        
        # Timestamps (current time for created_at and updated_at)
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    return row


def convert_jsons_to_csv(
    json_dir: Path,
    output_dir: Path,
    metadata_csv_name: str = 'metadata.csv'
):
    """
    Convert all JSON files in a directory to CSV format matching database schema.
    
    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory where CSV files will be saved
        metadata_csv_name: Name of the metadata CSV file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = list(json_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Define metadata CSV columns matching schema.sql
    metadata_columns = [
        'document_id', 'pdf_name', 'state', 'year', 'revision_type',
        'qualifying_date', 'publication_date', 'roll_type', 'roll_identification',
        'total_pages', 'total_voters_extracted', 'created_at', 'updated_at',
        'town_or_village', 'main_town_or_village', 'ward_number', 'post_office',
        'police_station', 'taluk_or_block', 'subdivision', 'district', 'pin_code',
        'panchayat_name', 'constituency_details', 'administrative_address',
        'polling_details', 'detailed_elector_summary', 'authority_verification',
        'output_identifier'
    ]
    
    # Process JSON files and write to CSV
    metadata_csv_path = output_dir / metadata_csv_name
    
    with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metadata_columns)
        writer.writeheader()
        
        for i, json_file in enumerate(json_files, 1):
            try:
                # Load JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Extract metadata row
                row = extract_metadata_row(json_data, json_file.name)
                
                # Write row to CSV
                writer.writerow(row)
                
                if i % 50 == 0:
                    print(f"Processed {i}/{len(json_files)} files...")
                    
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue
    
    print(f"\n✅ Conversion complete!")
    print(f"📄 Metadata CSV: {metadata_csv_path}")
    print(f"   Total records: {len(json_files)}")


def main():
    """Main entry point for the script."""
    # Define paths
    script_dir = Path(__file__).parent
    json_dir = script_dir / 'metadata'
    output_dir = script_dir / 'csv'
    
    # Validate input directory exists
    if not json_dir.exists():
        print(f"❌ Error: JSON directory not found: {json_dir}")
        return
    
    print("=" * 60)
    print("JSON to CSV Converter for Electoral Roll Metadata")
    print("=" * 60)
    print(f"Input directory:  {json_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Convert JSONs to CSV
    convert_jsons_to_csv(json_dir, output_dir)
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

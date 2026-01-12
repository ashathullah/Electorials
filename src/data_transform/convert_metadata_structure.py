"""
Convert metadata JSON files from extraction_results_metadata_json format
to missing_metadata nested structure format.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List


def parse_sections(sections_str: str) -> List[Dict[str, str]]:
    """Parse the sections JSON string into a list of dictionaries."""
    if not sections_str:
        return []
    
    try:
        sections = json.loads(sections_str)
        # Convert to target format
        result = []
        for section in sections:
            result.append({
                "section_number": section.get("street_number", ""),
                "section_name": section.get("street_name", "")
            })
        return result
    except (json.JSONDecodeError, TypeError):
        return []


def convert_metadata_structure(source_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat metadata structure to missing_metadata format.
    
    Args:
        source_data: Flat metadata dictionary from extraction_results_metadata_json
        
    Returns:
        Metadata dictionary in missing_metadata format
    """
    
    # Parse sections
    sections = parse_sections(source_data.get("polling_details_sections", ""))
    
    # Build the structure matching the missing_metadata format
    converted = {
        "language_detected": ["Tamil"],  # Default
        "state": source_data.get("state", ""),
        "electoral_roll_year": source_data.get("electoral_roll_year"),
        "revision_type": source_data.get("revision_type"),
        "qualifying_date": source_data.get("qualifying_date"),
        "publication_date": source_data.get("publication_date"),
        "roll_type": source_data.get("roll_type"),
        "roll_identification": source_data.get("roll_identification"),
        "total_pages": source_data.get("total_pages"),
        "total_voters_extracted": None,
        "page_number_current": 1,
        "output_identifier": source_data.get("output_identifier"),
        "constituency_details": {
            "assembly_constituency_number": source_data.get("constituency_details_assembly_constituency_number"),
            "assembly_constituency_name": source_data.get("constituency_details_assembly_constituency_name"),
            "assembly_reservation_status": source_data.get("constituency_details_assembly_reservation_status"),
            "parliamentary_constituency_number": source_data.get("constituency_details_parliamentary_constituency_number"),
            "parliamentary_constituency_name": source_data.get("constituency_details_parliamentary_constituency_name"),
            "parliamentary_reservation_status": source_data.get("constituency_details_parliamentary_reservation_status"),
            "part_number": source_data.get("constituency_details_part_number")
        },
        "administrative_address": {
            "town_or_village": source_data.get("administrative_address_town_or_village"),
            "ward_number": source_data.get("administrative_address_ward_number"),
            "post_office": source_data.get("Post Office"),
            "police_station": source_data.get("administrative_address_police_station"),
            "taluk_or_block": source_data.get("taluk_or_block"),
            "subdivision": source_data.get("Subdivision"),
            "district": source_data.get("administrative_address_district"),
            "pin_code": source_data.get("administrative_address_pin_code"),
            "panchayat_name": source_data.get("panchayat_name"),
            "main_town_or_village": source_data.get("main_town_or_village")
        },
        "polling_details": {
            "sections": sections,
            "polling_station_number": source_data.get("polling_details_polling_station_number"),
            "polling_station_name": source_data.get("polling_details_polling_station_name"),
            "polling_station_address": source_data.get("polling_details_polling_station_address"),
            "polling_station_type": source_data.get("polling_details_polling_station_type"),
            "auxiliary_polling_station_count": source_data.get("polling_details_auxiliary_polling_station_count")
        },
        "detailed_elector_summary": {
            "serial_number_range": {
                "start": source_data.get("detailed_elector_summary_serial_number_range_start"),
                "end": source_data.get("detailed_elector_summary_serial_number_range_end")
            },
            "net_total": {
                "male": source_data.get("detailed_elector_summary_net_total_male"),
                "female": source_data.get("detailed_elector_summary_net_total_female"),
                "third_gender": source_data.get("detailed_elector_summary_net_total_third_gender"),
                "total": source_data.get("detailed_elector_summary_net_total_total")
            }
        },
        "authority_verification": {
            "designation": None,
            "signature_present": source_data.get("authority_verification_signature_present") == "True"
        },
        "ai_metadata": {
            "provider": "Conversion",
            "model": "metadata_structure_converter",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "extraction_time_sec": source_data.get("timing_total_time_sec", 0.0)
        }
    }
    
    return converted


def convert_file(source_path: Path, target_path: Path) -> bool:
    """
    Convert a single metadata file.
    
    Args:
        source_path: Path to source JSON file
        target_path: Path to target JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read source file
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        # Convert structure
        converted_data = convert_metadata_structure(source_data)
        
        # Write to target file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error converting {source_path.name}: {e}")
        return False


def get_target_filename(source_filename: str) -> str:
    """
    Convert source filename to target filename.
    
    Example: 'Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.json'
    becomes: 'Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1.json'
    """
    if source_filename.endswith('_metadata.json'):
        return source_filename.replace('_metadata.json', '.json')
    return source_filename


def main():
    """Main conversion function."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root / "extraction_results_metadata_json"
    target_dir = project_root / "converted_metadata"
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print("-" * 80)
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files from source directory
    source_files = list(source_dir.glob("*.json"))
    
    if not source_files:
        print("No JSON files found in source directory.")
        return
    
    print(f"Found {len(source_files)} files to convert.")
    print("-" * 80)
    
    # Convert each file
    success_count = 0
    failed_count = 0
    
    for source_file in source_files:
        target_filename = get_target_filename(source_file.name)
        target_file = target_dir / target_filename
        
        # Convert the file (overwrite if exists)
        if convert_file(source_file, target_file):
            print(f"OK: {source_file.name} -> {target_filename}")
            success_count += 1
        else:
            print(f"FAIL: {source_file.name}")
            failed_count += 1
    
    # Summary
    print("-" * 80)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(source_files)}")


if __name__ == "__main__":
    main()

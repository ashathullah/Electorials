"""
Script to remove invalid voters from JSON files in the invalid_voter_sections_json directory.

Invalid voters are identified by:
- Empty or missing 'name' field (primary criterion)
- Empty 'epic_id' or epic_id with less than 5 characters (secondary validation)

The script:
1. Processes all JSON files in the invalid_voter_sections_json directory
2. Filters out invalid voter records
3. Overwrites the original files with cleaned data
4. Generates a detailed report of removed records
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def is_invalid_voter(voter: Dict) -> bool:
    """
    Determine if a voter record is invalid.
    
    Args:
        voter: Dictionary containing voter information
        
    Returns:
        True if the voter is invalid, False otherwise
    """
    # Check if name is empty or missing - this is the primary criterion
    name = voter.get('name', '').strip()
    if not name:
        return True
    
    # Secondary check: if epic_id is suspiciously short (< 5 chars) AND name is empty
    epic_id = voter.get('epic_id', '').strip()
    if not name and len(epic_id) < 5:
        return True
    
    return False


def clean_voter_file(file_path: Path) -> Tuple[int, int, List[Dict]]:
    """
    Clean a single voter JSON file by removing invalid voters.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (original_count, valid_count, removed_voters)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            voters = json.load(f)
        
        if not isinstance(voters, list):
            print(f"Warning: {file_path.name} does not contain a list of voters. Skipping.")
            return 0, 0, []
        
        original_count = len(voters)
        
        # Separate valid and invalid voters
        valid_voters = []
        invalid_voters = []
        
        for voter in voters:
            if is_invalid_voter(voter):
                invalid_voters.append(voter)
            else:
                valid_voters.append(voter)
        
        valid_count = len(valid_voters)
        
        # Only write back if there were changes
        if len(invalid_voters) > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(valid_voters, f, ensure_ascii=False, indent=2)
            print(f"✓ {file_path.name}: Removed {len(invalid_voters)} invalid voters ({original_count} → {valid_count})")
        else:
            print(f"○ {file_path.name}: No invalid voters found ({original_count} voters)")
        
        return original_count, valid_count, invalid_voters
        
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing {file_path.name}: {e}")
        return 0, 0, []
    except Exception as e:
        print(f"✗ Error processing {file_path.name}: {e}")
        return 0, 0, []


def generate_report(report_data: List[Dict], output_path: Path):
    """
    Generate a detailed JSON report of all removed voters.
    
    Args:
        report_data: List of dictionaries containing file info and removed voters
        output_path: Path where the report should be saved
    """
    report = {
        "summary": {
            "total_files_processed": len(report_data),
            "total_files_with_invalid_voters": sum(1 for r in report_data if r['removed_count'] > 0),
            "total_original_voters": sum(r['original_count'] for r in report_data),
            "total_valid_voters": sum(r['valid_count'] for r in report_data),
            "total_removed_voters": sum(r['removed_count'] for r in report_data)
        },
        "files": report_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Report saved to: {output_path}")


def main():
    """Main function to process all voter JSON files."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    invalid_voters_dir = project_root / "invalid_voter_sections_json"
    report_dir = project_root / "reports"
    
    # Create report directory if it doesn't exist
    report_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("INVALID VOTER REMOVAL SCRIPT")
    print("=" * 80)
    print(f"\nProcessing JSON files in: {invalid_voters_dir}")
    print(f"\nCriteria for invalid voters:")
    print("  - Empty or missing 'name' field")
    print("  - Empty 'epic_id' or epic_id with less than 5 characters (when name is also empty)")
    print("\n" + "=" * 80 + "\n")
    
    # Get all JSON files
    json_files = list(invalid_voters_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {invalid_voters_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    # Process each file
    report_data = []
    total_original = 0
    total_valid = 0
    total_removed = 0
    
    for json_file in sorted(json_files):
        original_count, valid_count, removed_voters = clean_voter_file(json_file)
        
        if original_count > 0:
            removed_count = original_count - valid_count
            total_original += original_count
            total_valid += valid_count
            total_removed += removed_count
            
            report_data.append({
                "filename": json_file.name,
                "original_count": original_count,
                "valid_count": valid_count,
                "removed_count": removed_count,
                "removed_voters": removed_voters
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed: {len(report_data)}")
    print(f"Files with invalid voters: {sum(1 for r in report_data if r['removed_count'] > 0)}")
    print(f"Total original voters: {total_original:,}")
    print(f"Total valid voters: {total_valid:,}")
    print(f"Total removed voters: {total_removed:,}")
    print(f"Percentage removed: {(total_removed/total_original*100):.2f}%" if total_original > 0 else "0%")
    print("=" * 80)
    
    # Generate detailed report
    report_path = report_dir / "invalid_voters_removed_report.json"
    generate_report(report_data, report_path)
    
    print(f"\n✓ Processing complete!")


if __name__ == "__main__":
    main()

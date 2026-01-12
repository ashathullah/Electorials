import json
import os
from pathlib import Path

def update_voter_data():
    """
    Updates voter data in valid_voter_sections_json folder from results folder.
    
    For each JSON file in /results:
    - Find corresponding file in /valid_voter_sections_json by adding '_voters' suffix
    - Match voters by serial_no
    - Update: name, relation_name (to father_name or husband_name), and house_no
    """
    
    results_dir = Path("results")
    target_dir = Path("valid_voter_sections_json")
    
    # Counters for statistics
    stats = {
        "files_processed": 0,
        "files_not_found": 0,
        "records_updated": 0,
        "records_not_matched": 0,
        "errors": []
    }
    
    # Get all JSON files from results folder
    result_files = list(results_dir.glob("*.json"))
    print(f"Found {len(result_files)} files in /results folder\n")
    
    for result_file in result_files:
        try:
            # Construct target filename by adding '_voters' before .json
            base_name = result_file.stem  # filename without extension
            target_filename = f"{base_name}_voters.json"
            target_file = target_dir / target_filename
            
            # Check if target file exists
            if not target_file.exists():
                print(f"⚠️  Target file not found: {target_filename}")
                stats["files_not_found"] += 1
                continue
            
            # Load source data (from results)
            with open(result_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            # Load target data (from valid_voter_sections_json)
            with open(target_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            
            # Create a lookup dictionary for source data by serial_no
            source_lookup = {}
            for record in source_data:
                serial_no = str(record.get("serial_no", ""))
                source_lookup[serial_no] = record
            
            # Update target data
            records_updated_in_file = 0
            for target_record in target_data:
                serial_no = str(target_record.get("serial_no", ""))
                
                # Find matching record in source
                if serial_no in source_lookup:
                    source_record = source_lookup[serial_no]
                    
                    # Update name
                    if "name" in source_record:
                        target_record["name"] = source_record["name"]
                    
                    # Update house_no
                    if "house_no" in source_record:
                        target_record["house_no"] = source_record["house_no"]
                    
                    # Update relation_name based on relation_type
                    relation_type = source_record.get("relation_type", "").lower()
                    relation_name = source_record.get("relation_name", "")
                    
                    if relation_type == "father":
                        target_record["father_name"] = relation_name
                    elif relation_type == "husband":
                        target_record["husband_name"] = relation_name
                    elif relation_type == "mother":
                        target_record["mother_name"] = relation_name
                    elif relation_type == "other":
                        target_record["other_name"] = relation_name
                    
                    records_updated_in_file += 1
                else:
                    stats["records_not_matched"] += 1
            
            # Write updated data back to target file
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(target_data, f, ensure_ascii=False, indent=4)
            
            stats["files_processed"] += 1
            stats["records_updated"] += records_updated_in_file
            print(f"✅ Updated {target_filename}: {records_updated_in_file} records updated")
            
        except Exception as e:
            error_msg = f"Error processing {result_file.name}: {str(e)}"
            print(f"❌ {error_msg}")
            stats["errors"].append(error_msg)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed successfully: {stats['files_processed']}")
    print(f"Target files not found: {stats['files_not_found']}")
    print(f"Total records updated: {stats['records_updated']}")
    print(f"Records not matched: {stats['records_not_matched']}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\nError details:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    return stats

if __name__ == "__main__":
    print("Starting voter data update process...\n")
    update_voter_data()
    print("\nProcess completed!")

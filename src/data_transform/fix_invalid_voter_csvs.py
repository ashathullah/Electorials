import csv
from pathlib import Path

def fix_invalid_voter_csv(file_path):
    """
    Fix an invalid voter CSV file by:
    1. Removing the first data record (second line with missing data)
    2. Renumbering all serial numbers to start from 1
    
    Args:
        file_path: Path to the CSV file to fix
        
    Returns:
        tuple: (success: bool, records_processed: int, error_message: str)
    """
    try:
        # Read all records
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            records = list(reader)
        
        if len(records) == 0:
            return False, 0, "No data records found"
        
        # Remove the first invalid record
        records.pop(0)
        
        if len(records) == 0:
            return False, 0, "No valid records left after removing first record"
        
        # Renumber serial_no field starting from 1
        for index, record in enumerate(records, start=1):
            record['serial_no'] = str(index)
        
        # Write back to the same file
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, lineterminator='\r\n')
            writer.writeheader()
            writer.writerows(records)
        
        return True, len(records), ""
        
    except Exception as e:
        return False, 0, str(e)


def fix_all_invalid_voter_csvs():
    """
    Process all CSV files in the invalid_voter_sections directory.
    """
    source_dir = Path("invalid_voter_sections")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ Directory '{source_dir}' does not exist!")
        return
    
    # Find all CSV files (excluding backup files)
    csv_files = [f for f in source_dir.glob("*.csv") if "_backup" not in f.name]
    
    if not csv_files:
        print(f"ℹ️  No CSV files found in '{source_dir}'")
        return
    
    print(f"📁 Found {len(csv_files)} CSV file(s) to fix")
    print("=" * 80)
    
    # Process each file
    success_count = 0
    fail_count = 0
    total_records = 0
    
    for file_path in csv_files:
        success, record_count, error_msg = fix_invalid_voter_csv(file_path)
        
        if success:
            print(f"✅ Fixed: {file_path.name}")
            print(f"   → Removed first invalid record")
            print(f"   → Renumbered {record_count} voter records (1-{record_count})")
            success_count += 1
            total_records += record_count
        else:
            print(f"❌ Failed: {file_path.name}")
            print(f"   → Error: {error_msg}")
            fail_count += 1
    
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully fixed: {success_count} file(s)")
    print(f"   ❌ Failed: {fail_count} file(s)")
    print(f"   📝 Total voter records renumbered: {total_records}")


if __name__ == "__main__":
    fix_all_invalid_voter_csvs()

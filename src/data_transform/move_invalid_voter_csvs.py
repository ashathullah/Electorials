import os
import csv
import shutil
from pathlib import Path

def is_invalid_voter_file(csv_path):
    """
    Check if a voter CSV file is invalid by examining the first data record.
    
    A file is considered invalid if the first record (second line) is missing
    critical details like epic_id, name, father_name, mother_name, etc.
    
    Args:
        csv_path: Path to the CSV file to check
        
    Returns:
        bool: True if the file is invalid, False otherwise
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Read the first data record (second line in the file)
            first_record = next(reader, None)
            
            if not first_record:
                print(f"⚠️  '{csv_path.name}' has no data records")
                return True
            
            # Check critical fields - if most are empty, it's invalid
            critical_fields = ['epic_id', 'name', 'father_name', 'mother_name', 
                             'age', 'gender', 'house_no']
            
            # Count how many critical fields are empty
            empty_count = sum(1 for field in critical_fields 
                            if field in first_record and not first_record[field].strip())
            
            # If 5 or more critical fields are empty, consider it invalid
            # This is a reasonable threshold since invalid files have most fields empty
            if empty_count >= 5:
                print(f"🔍 Invalid file detected: '{csv_path.name}' - {empty_count}/{len(critical_fields)} critical fields empty")
                return True
            
            return False
            
    except Exception as e:
        print(f"❌ Error reading '{csv_path.name}': {e}")
        return False


def move_invalid_voter_csvs():
    """
    Identify and move invalid voter CSV files to invalid_voter_sections directory.
    
    Invalid files are those where the first record has missing critical details.
    """
    # Define source and destination directories
    source_dir = Path("extraction_results")
    dest_dir = Path("invalid_voter_sections")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ Source directory '{source_dir}' does not exist!")
        return
    
    # Find all CSV files (excluding metadata files)
    # We want voter files, which typically don't have "_metadata" in their name
    all_csv_files = list(source_dir.glob("*.csv"))
    voter_csv_files = [f for f in all_csv_files if "_metadata" not in f.name.lower()]
    
    if not voter_csv_files:
        print(f"ℹ️  No voter CSV files found in '{source_dir}'")
        return
    
    print(f"📁 Found {len(voter_csv_files)} voter CSV file(s) to check")
    print("=" * 70)
    
    # Check each file and move invalid ones
    invalid_count = 0
    valid_count = 0
    
    for file_path in voter_csv_files:
        if is_invalid_voter_file(file_path):
            try:
                dest_path = dest_dir / file_path.name
                
                # Check if file already exists in destination
                if dest_path.exists():
                    print(f"⚠️  Skipping '{file_path.name}' - already exists in destination")
                    continue
                
                # Move the file
                shutil.move(str(file_path), str(dest_path))
                print(f"✅ Moved to invalid_voter_sections: {file_path.name}")
                invalid_count += 1
                
            except Exception as e:
                print(f"❌ Error moving '{file_path.name}': {e}")
        else:
            valid_count += 1
    
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   ✅ Valid files: {valid_count}")
    print(f"   ❌ Invalid files moved: {invalid_count}")
    print(f"   📁 Destination: '{dest_dir}'")


if __name__ == "__main__":
    move_invalid_voter_csvs()

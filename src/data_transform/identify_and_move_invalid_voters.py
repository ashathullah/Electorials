import csv
import shutil
import os
import glob

SOURCE_DIR = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\extraction_results"
DEST_DIR = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\invalid_voter_sections"

def is_invalid_voter_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False # Empty file
            
            first_record = next(reader, None)
            if not first_record:
                return False # Only header
            
            # Check if we have enough columns to check (at least 3: serial, epic, name)
            if len(first_record) < 3:
                return True # Malformed
                
            epic_id = first_record[1].strip()
            name = first_record[2].strip()
            
            # User criteria: invalid if first record misses details (only the first)
            # Example shows empty epic_id and empty name
            if not epic_id and not name:
                return True
                
            return False
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def move_invalid_files():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    csv_files = glob.glob(os.path.join(SOURCE_DIR, "*_voters.csv"))
    print(f"Found {len(csv_files)} voter CSV files in {SOURCE_DIR}")
    
    moved_count = 0
    for file_path in csv_files:
        if is_invalid_voter_file(file_path):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(DEST_DIR, file_name)
            
            print(f"Moving invalid file: {file_name}")
            try:
                shutil.move(file_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"Failed to move {file_name}: {e}")
                
    print(f"Moved {moved_count} invalid files to {DEST_DIR}")

if __name__ == "__main__":
    move_invalid_files()
